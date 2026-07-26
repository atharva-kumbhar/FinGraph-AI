"""TigerGraph SEC Financial GraphRAG — deterministic retrieval & SEC chunk loader.

Execution Flow (Pipeline 3):
  1. Extract entities (Company, Executive, Ticker, Auditor, Event, Risk, Metric, Filing, Year) using Python rule-based NLP (NO LLM).
  2. Determine question type (Company, Executive, Auditor, Event, Comparison, Fallback).
  3. Select and execute appropriate TigerGraph query to retrieve graph structure and chunk IDs.
  4. Load the ORIGINAL SEC filing text chunks from data/sp100/chunks.jsonl using matching chunk IDs and entity keywords.
  5. Build readable GraphRAG context including the SEC filing text.
  6. Print detailed Graph Retrieval Validation logs.
  7. Send formatted context + original question to Gemini for final reasoning.

TigerGraph is ONLY for retrieval. Gemini is ONLY for reasoning.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import urllib3

from .config import settings, ROOT_DIR
from .entity_extractor import (
    extract_entities,
    determine_question_type,
    ExtractedEntities,
    COMPANY_MAPPING,
    KNOWN_EXECUTIVES,
)
from .retrievers import RetrievalResult

logger = logging.getLogger("medgraph.graph")


def detect_companies(query: str, known_companies: list[tuple[str, str]] | None = None) -> list[str]:
    """Detect company names and ticker symbols using deterministic Python matching."""
    extracted = extract_entities(query, known_companies)
    if extracted.companies:
        return extracted.companies
    return ["Apple Inc."]


# ─── GraphReasoner ───────────────────────────────────────────────────────────

class GraphReasoner:
    """Deterministic SEC Financial GraphRAG retriever for TigerGraph."""

    def __init__(self) -> None:
        self.tigergraph = TigerGraphClient()

    def stats(self) -> dict[str, Any]:
        return self.tigergraph.stats()

    def reason(self, query: str, retrievals: list[RetrievalResult] | None = None) -> dict[str, Any]:
        logger.info("=" * 80)
        logger.info(f"[GraphRAG] Question received: '{query}'")
        logger.info("=" * 80)

        # ── Step 1: Extract entities using Python (NO LLM) ────────────────────
        known_entities = TigerGraphClient._global_entities_cache or []
        entities = extract_entities(query, known_entities)

        # ── Step 2: Determine Question Type ──────────────────────────────────
        qtype = determine_question_type(entities)

        # ── Step 3: Select and Execute TigerGraph Query ───────────────────────
        import time
        tg_start = time.perf_counter()
        year = entities.years[0] if entities.years else None
        metric = entities.metrics[0] if entities.metrics else None
        keywords = entities.risks + entities.metrics + entities.events + entities.auditors + entities.priority_ordered_list

        all_results = _empty_results(entities.priority_ordered_list)
        all_results["query_type"] = qtype
        all_results["extracted_entities"] = entities.priority_ordered_list

        selected_query_name = ""

        if qtype == "comparison" and len(entities.companies) >= 2:
            comp1 = entities.companies[0]
            comp2 = entities.companies[1]
            selected_query_name = f"compareCompanies('{comp1}', '{comp2}')"
            data = self.tigergraph.compare_companies(comp1, comp2, year, metric)
            for cname, cdata in data.items():
                _merge_company_data(all_results, cdata, cname)
            detected_companies = list(entities.companies)

        elif qtype == "executive":
            exec_name = entities.executives[0]
            selected_query_name = f"retrieveExecutiveContext('{exec_name}')"
            company_name, cdata = self.tigergraph.retrieve_executive_context(exec_name, year, metric)
            _merge_company_data(all_results, cdata, company_name)
            detected_companies = [company_name]
            if len(entities.companies) > 1:
                for cname in entities.companies:
                    if cname not in detected_companies:
                        detected_companies.append(cname)

        elif entities.companies:
            detected_companies = list(entities.companies)
            selected_query_name = f"retrieveContext('{detected_companies[0]}', year={year}, keyword={metric})"
            for company_name in detected_companies:
                cdata = self.tigergraph.retrieve_context(company_name, year, metric)
                _merge_company_data(all_results, cdata, company_name)

        else:  # Fallback
            keyword = entities.priority_ordered_list[0] if entities.priority_ordered_list else query
            selected_query_name = f"searchEntity('{keyword}')"
            company_name, cdata = self.tigergraph.search_entity(keyword, year, metric)
            _merge_company_data(all_results, cdata, company_name)
            detected_companies = [company_name]

        all_results["seed_entities"] = detected_companies
        all_results["selected_query"] = selected_query_name
        all_results["tg_latency_ms"] = int((time.perf_counter() - tg_start) * 1000)

        # ── Step 4: Collect Chunk IDs and Load SEC Filing Text from chunks.jsonl
        matched_chunk_ids = []
        for company in detected_companies:
            cdetails = all_results["company_details"].get(company) or {}
            docs = cdetails.get("documents") or []
            for d in docs:
                cid = d.get("chunk_id") or d.get("id")
                if cid and cid not in matched_chunk_ids:
                    matched_chunk_ids.append(cid)

        chunk_load_start = time.perf_counter()
        loaded_chunks = self.tigergraph.load_sec_text_chunks(
            companies=detected_companies,
            chunk_ids=matched_chunk_ids,
            keywords=keywords,
            max_chunks_per_company=10,
        )
        all_results["loaded_chunks"] = loaded_chunks
        all_results["matched_chunk_ids"] = [c.get("chunk_id") for c in loaded_chunks if c.get("chunk_id")]
        all_results["chunk_load_latency_ms"] = int((time.perf_counter() - chunk_load_start) * 1000)

        # ── Step 5: Convert retrieved SEC text & graph into readable context ──
        formatted_context = self._format_readable_context(
            all_results["company_details"], detected_companies, entities, loaded_chunks
        )
        all_results["formatted_context"] = formatted_context
        all_results["status"] = f"TigerGraph context retrieved via [{qtype}] for: {', '.join(detected_companies)}"

        # ── Step 6: Print Detailed Validation Logs ───────────────────────────
        self._log_validation_info(
            entities=entities,
            qtype=qtype,
            query_name=selected_query_name,
            companies=detected_companies,
            details=all_results["company_details"],
            chunk_ids=all_results["matched_chunk_ids"],
            loaded_chunks=loaded_chunks,
            context=formatted_context,
        )

        return all_results

    def _log_validation_info(
        self,
        entities: ExtractedEntities,
        qtype: str,
        query_name: str,
        companies: list[str],
        details: dict[str, Any],
        chunk_ids: list[str],
        loaded_chunks: list[dict[str, Any]],
        context: str,
    ) -> None:
        matched_docs = []
        matched_execs = []
        matched_metrics = []
        matched_risks = []
        matched_events = []

        for comp in companies:
            cdata = details.get(comp) or {}
            for d in cdata.get("documents") or []:
                did = d.get("chunk_id") or d.get("id") or d.get("filing_type")
                if did and did not in matched_docs:
                    matched_docs.append(did)
            for ex in cdata.get("executives") or []:
                name = ex.get("name") or ex.get("id") if isinstance(ex, dict) else str(ex)
                if name and name not in matched_execs:
                    matched_execs.append(name)
            for m in cdata.get("financial_metrics") or []:
                mname = m.get("metric_name") or m.get("name") or m.get("metric") or m.get("id") if isinstance(m, dict) else str(m)
                if mname and mname not in matched_metrics:
                    matched_metrics.append(mname)
            for r in cdata.get("risks") or []:
                rdesc = r.get("description") or r.get("id") if isinstance(r, dict) else str(r)
                if rdesc and rdesc not in matched_risks:
                    matched_risks.append(rdesc)
            for ev in cdata.get("events") or []:
                edesc = ev.get("description") or ev.get("id") if isinstance(ev, dict) else str(ev)
                if edesc and edesc not in matched_events:
                    matched_events.append(edesc)

        logger.info("=" * 80)
        logger.info("GRAPH RETRIEVAL VALIDATION")
        logger.info("=" * 80)
        logger.info(f"Extracted Entities      : {entities.priority_ordered_list}")
        logger.info(f"Selected TigerGraph Query: {qtype} -> {query_name}")
        logger.info(f"Matched Company         : {companies}")
        logger.info(f"Matched Documents       : {matched_docs[:6]}")
        logger.info(f"Matched Executives      : {matched_execs[:6]}")
        logger.info(f"Matched Metrics         : {matched_metrics[:6]}")
        logger.info(f"Matched Risks           : {matched_risks[:6]}")
        logger.info(f"Matched Events          : {matched_events[:6]}")
        logger.info(f"Matched Chunk IDs       : {chunk_ids[:6]}")
        logger.info(f"Loaded Chunk Count      : {len(loaded_chunks)}")
        logger.info(f"Final Context Length    : {len(context)} characters")
        logger.info("=" * 80)

    def _format_readable_context(
        self,
        company_details: dict[str, dict[str, Any]],
        companies: list[str],
        entities: ExtractedEntities,
        loaded_chunks: list[dict[str, Any]],
    ) -> str:
        blocks = []

        # Index loaded chunks by company
        chunks_by_comp: dict[str, list[dict[str, Any]]] = {}
        for ch in loaded_chunks:
            cname = ch.get("company_name") or ch.get("company") or ""
            matched_c = None
            for comp in companies:
                if comp.lower() in cname.lower() or cname.lower() in comp.lower():
                    matched_c = comp
                    break
            if not matched_c and companies:
                matched_c = companies[0]
            if matched_c:
                if matched_c not in chunks_by_comp:
                    chunks_by_comp[matched_c] = []
                chunks_by_comp[matched_c].append(ch)

        for company in companies:
            data = company_details.get(company) or {}
            if not isinstance(data, dict):
                data = {}
            c_lines = [f"Company:\n{company}\n"]

            # Company Attributes
            comp_attrs = data.get("company_attrs")
            if isinstance(comp_attrs, dict):
                ticker = comp_attrs.get("ticker") or ""
                sector = comp_attrs.get("sector") or ""
                if ticker or sector:
                    c_lines.append(f"Ticker: {ticker} | Sector: {sector}\n")

            # Document Metadata
            docs = data.get("documents") or []
            if isinstance(docs, list) and docs:
                c_lines.append("Document Metadata:")
                for d in docs[:4]:
                    if not isinstance(d, dict):
                        continue
                    ftype = d.get("filing_type") or "Form 10-K"
                    fdate = d.get("filing_date") or d.get("year") or ""
                    cid = d.get("chunk_id") or d.get("id") or ""
                    c_lines.append(f"- {fdate} {ftype} (Chunk ID: {cid})".strip())

            # Relevant SEC Filing Text from data/sp100/chunks.jsonl
            comp_chunks = chunks_by_comp.get(company) or []
            if comp_chunks:
                c_lines.append("\nRelevant SEC Filing Text (loaded from data/sp100/chunks.jsonl):")
                for i, ch in enumerate(comp_chunks[:4], 1):
                    ftype = ch.get("filing_type") or "10-K"
                    fdate = ch.get("filing_date") or ""
                    cid = ch.get("chunk_id") or ""
                    text = (ch.get("text") or ch.get("content") or "").strip()
                    if text:
                        c_lines.append(f"\n[SEC Chunk {i}: {fdate} {ftype} | Chunk ID: {cid}]\n\"{text}\"")

            # Financial Metrics
            metrics = data.get("financial_metrics") or []
            if isinstance(metrics, list) and metrics:
                c_lines.append("\nFinancial Metrics:")
                for m in metrics[:8]:
                    if isinstance(m, dict):
                        name = m.get("metric_name") or m.get("name") or m.get("metric") or m.get("id")
                        val = m.get("value") or m.get("amount") or ""
                        period = m.get("period") or m.get("year") or ""
                    else:
                        name = str(m)
                        val = ""
                        period = ""
                    if name:
                        val_str = f": {val}" if val else ""
                        per_str = f" ({period})" if period else ""
                        c_lines.append(f"- {name}{val_str}{per_str}")

            # Executives
            execs = data.get("executives") or []
            if isinstance(execs, list) and execs:
                c_lines.append("\nExecutives:")
                for ex in execs[:8]:
                    if isinstance(ex, dict):
                        name = ex.get("name") or ex.get("id") or ex.get("Executive")
                        title = ex.get("title") or ""
                    else:
                        name = str(ex)
                        title = ""
                    if name:
                        line = f"- {name} ({title})" if title else f"- {name}"
                        c_lines.append(line)

            # Business Risks
            risks = data.get("risks") or []
            if isinstance(risks, list) and risks:
                c_lines.append("\nBusiness Risks:")
                for r in risks[:4]:
                    desc = r.get("description") or r.get("id") if isinstance(r, dict) else str(r)
                    if desc:
                        c_lines.append(f"- {desc}")

            # Corporate Events
            events = data.get("events") or []
            if isinstance(events, list) and events:
                c_lines.append("\nCorporate Events:")
                for ev in events[:4]:
                    if isinstance(ev, dict):
                        etype = ev.get("event_type") or "Event"
                        desc = ev.get("description") or ev.get("id") or ""
                        edate = ev.get("date") or ""
                    else:
                        etype = "Event"
                        desc = str(ev)
                        edate = ""
                    if desc:
                        edate_str = f" ({edate})" if edate else ""
                        c_lines.append(f"- {etype}: {desc}{edate_str}")

            blocks.append("\n".join(c_lines))

        return "\n\n" + ("=" * 70) + "\n\n".join(blocks)


# ─── Result helpers ──────────────────────────────────────────────────────────

def _empty_results(companies: list[str]) -> dict[str, Any]:
    return {
        "nodes": [], "edges": [], "paths": [],
        "neighbors": {}, "entities": [],
        "seed_entities": companies,
        "extracted_entities": companies,
        "backend": "tigergraph", "status": "", "errors": [],
        "entities_summary": {}, "documents": [], "tg_chunks": [],
        "company_details": {}, "formatted_context": "",
        "loaded_chunks": [], "matched_chunk_ids": [],
    }


def _merge_company_data(
    results: dict[str, Any],
    data: dict[str, Any] | None,
    company_name: str,
) -> None:
    if not isinstance(data, dict):
        data = {}
    results["company_details"][company_name] = data

    comp_attrs = data.get("company_attrs")
    if not isinstance(comp_attrs, dict):
        comp_attrs = {}

    c_node = {
        "id": company_name,
        "label": company_name,
        "type": "Company",
        "attributes": comp_attrs,
    }
    if company_name not in [n["id"] for n in results["nodes"]]:
        results["nodes"].append(c_node)
        results["entities"].append(company_name)

    # Documents
    documents = data.get("documents") or []
    if isinstance(documents, list):
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            did = doc.get("chunk_id") or doc.get("id") or f"doc_{len(results['nodes'])}"
            ftype = doc.get("filing_type") or "Document"
            if did not in [n["id"] for n in results["nodes"]]:
                results["nodes"].append({
                    "id": did,
                    "label": f"{company_name} {ftype}",
                    "type": "Document",
                    "attributes": doc,
                })
            path_str = f"{company_name} -> FILED_BY -> Document({did})"
            if path_str not in results["paths"]:
                results["paths"].append(path_str)
                results["edges"].append({
                    "source": company_name,
                    "target": did,
                    "relation": "FILED_BY",
                    "edge_type": "FILED_BY",
                })
                text = doc.get("text") or doc.get("content") or ""
                results["tg_chunks"].append({
                    "chunk_id": did,
                    "content": text or f"SEC Filing {ftype} for {company_name}.",
                    "source": company_name,
                    "category": ftype,
                    "source_entity": company_name,
                })

    # Executives
    executives = data.get("executives") or []
    if isinstance(executives, list):
        for ex in executives:
            if isinstance(ex, dict):
                eid = ex.get("name") or ex.get("id") or ex.get("Executive")
                title = ex.get("title") or "Executive"
                attrs = ex
            elif isinstance(ex, str):
                eid = ex
                title = "Executive"
                attrs = {"name": ex, "title": title}
            else:
                continue

            if eid and str(eid) not in [n["id"] for n in results["nodes"]]:
                results["nodes"].append({
                    "id": str(eid),
                    "label": f"{eid} ({title})",
                    "type": "Executive",
                    "attributes": attrs,
                })
                path_str = f"{company_name} -> MENTIONS_EXECUTIVE -> {eid}"
                results["paths"].append(path_str)
                results["edges"].append({
                    "source": company_name,
                    "target": str(eid),
                    "relation": "MENTIONS_EXECUTIVE",
                    "edge_type": "MENTIONS_EXECUTIVE",
                })

    # Financial Metrics
    metrics = data.get("financial_metrics") or []
    if isinstance(metrics, list):
        for m in metrics:
            if isinstance(m, dict):
                mname = m.get("metric_name") or m.get("name") or m.get("metric") or m.get("id") or "Financial Metric"
                mval = m.get("value") or m.get("amount") or ""
            elif isinstance(m, str):
                mname = m
                mval = ""
            else:
                continue

            path_str = f"{company_name} -> REPORTS_METRIC -> {mname}: {mval}"
            results["paths"].append(path_str)
            results["edges"].append({
                "source": company_name,
                "target": str(mname),
                "relation": "REPORTS_METRIC",
                "edge_type": "REPORTS_METRIC",
            })

    # Business Risks
    risks = data.get("risks") or []
    if isinstance(risks, list):
        for r in risks:
            if isinstance(r, dict):
                rdesc = r.get("description") or r.get("id") or "Business Risk"
            elif isinstance(r, str):
                rdesc = r
            else:
                continue
            path_str = f"{company_name} -> DISCLOSES_RISK -> {str(rdesc)[:40]}"
            results["paths"].append(path_str)

    # Events
    events = data.get("events") or []
    if isinstance(events, list):
        for ev in events:
            if isinstance(ev, dict):
                edesc = ev.get("description") or ev.get("id") or "Corporate Event"
            elif isinstance(ev, str):
                edesc = ev
            else:
                continue
            path_str = f"{company_name} -> MENTIONS_EVENT -> {str(edesc)[:40]}"
            results["paths"].append(path_str)

    results["entities_summary"][company_name] = {
        "documents": len(documents) if isinstance(documents, list) else 0,
        "executives": len(executives) if isinstance(executives, list) else 0,
        "metrics": len(metrics) if isinstance(metrics, list) else 0,
    }


def _get_company_variants(company: str) -> list[str]:
    c_clean = re.sub(r"['’]?s$", "", company.lower().strip())
    base_names = [c_clean]
    if "apple" in c_clean:
        base_names.extend(["apple", "apple inc", "apple inc.", "aapl"])
    elif "nvidia" in c_clean:
        base_names.extend(["nvidia", "nvidia corp", "nvidia corporation", "nvda"])
    elif "amd" in c_clean or "advanced micro devices" in c_clean:
        base_names.extend(["amd", "advanced micro devices", "advanced micro devices inc"])
    elif "jpmorgan" in c_clean or "chase" in c_clean or "jpm" in c_clean:
        base_names.extend(["jpmorgan", "jpmorgan chase", "jpmorgan chase & co.", "jpmorgan chase bank", "jpm", "chase"])
    elif "exxon" in c_clean or "xom" in c_clean:
        base_names.extend(["exxon", "exxonmobil", "exxon mobil", "xom"])
    elif "coca" in c_clean or "coke" in c_clean:
        base_names.extend(["coca-cola", "coca cola", "the coca-cola company", "coke", "ko"])
    elif "microsoft" in c_clean or "msft" in c_clean:
        base_names.extend(["microsoft", "microsoft corp", "msft"])
    elif "amazon" in c_clean or "amzn" in c_clean:
        base_names.extend(["amazon", "amazon.com", "amzn"])
    elif "netflix" in c_clean or "nflx" in c_clean:
        base_names.extend(["netflix", "netflix inc", "nflx"])
    elif "abbvie" in c_clean or "abbv" in c_clean:
        base_names.extend(["abbvie", "abbv"])
    elif "adobe" in c_clean or "adbe" in c_clean:
        base_names.extend(["adobe", "adbe"])
    elif "verizon" in c_clean or "vz" in c_clean:
        base_names.extend(["verizon", "vz"])
    elif "walmart" in c_clean or "wmt" in c_clean:
        base_names.extend(["walmart", "wmt"])
    elif "caterpillar" in c_clean or "cat" in c_clean:
        base_names.extend(["caterpillar", "cat"])
    elif "chevron" in c_clean or "cvx" in c_clean:
        base_names.extend(["chevron", "cvx"])
    elif "cisco" in c_clean or "csco" in c_clean:
        base_names.extend(["cisco", "csco"])
    elif "costco" in c_clean or "cost" in c_clean:
        base_names.extend(["costco", "cost"])
    elif "amgen" in c_clean or "amgn" in c_clean:
        base_names.extend(["amgen", "amgn"])

    words = [w for w in c_clean.split() if len(w) >= 4 and w not in ("inc.", "corp.", "ltd.", "company")]
    base_names.extend(words)

    return list(set(base_names))


# ─── TigerGraphClient ───────────────────────────────────────────────────────

class TigerGraphClient:
    _global_chunks_index_by_id: dict[str, dict[str, Any]] | None = None
    _global_chunks_index_by_company: dict[str, list[dict[str, Any]]] | None = None
    _global_entities_cache: list[tuple[str, str]] | None = None
    _global_traversal_cache: dict[str, dict[str, Any]] = {}

    def __init__(self) -> None:
        self.host = settings.tg_host.rstrip("/")
        self.graph_name = settings.tg_graph_name
        self.username = settings.tg_username
        self.password = settings.tg_password
        self.verify_ssl = settings.tg_verify_ssl
        self.api_token = settings.tg_api_token
        self.secret = settings.tg_secret
        self.configured = bool(
            self.host and self.graph_name and (self.api_token or self.secret or self.username)
        )
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=25, pool_maxsize=25)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self._entities: list[tuple[str, str]] | None = TigerGraphClient._global_entities_cache
        self._stats: dict[str, Any] | None = None
        self._extracted_data_cache: dict[str, dict[str, Any]] | None = None
        self._chunks_index_by_id: dict[str, dict[str, Any]] | None = TigerGraphClient._global_chunks_index_by_id
        self._chunks_index_by_company: dict[str, list[dict[str, Any]]] | None = TigerGraphClient._global_chunks_index_by_company

    def stats(self) -> dict[str, Any]:
        base = {
            "entities": settings.tg_uploaded_vertices,
            "relationships": settings.tg_uploaded_edges,
            "configured": self.configured,
            "source": "configured_upload_counts",
        }
        if not self.configured:
            return base
        if self._stats is not None:
            return self._stats
        live = dict(base)
        live["source"] = "tigergraph_configured"
        try:
            entities = self.fetch_entities()
            if entities:
                live["entities"] = len(entities)
                live["source"] = "tigergraph_live"
        except Exception as exc:
            live["status"] = str(exc)
        self._stats = live
        return live

    def fetch_entities(self) -> list[tuple[str, str]]:
        """Return [(company_name, vertex_id), ...] from TigerGraph."""
        if TigerGraphClient._global_entities_cache is not None:
            self._entities = TigerGraphClient._global_entities_cache
            return self._entities

        eg = quote(self.graph_name, safe="")
        entities: list[tuple[str, str]] = []

        for vtype in ["Company", "Entity"]:
            path = f"/restpp/graph/{eg}/vertices/{vtype}?limit={settings.tg_vertex_limit}"
            try:
                payload = self._get_json(path)
                results = payload.get("results") or []
                for item in results:
                    vid = str(item.get("v_id") or "")
                    attrs = item.get("attributes") or {}
                    name = attrs.get("name") or attrs.get("company_name") or attrs.get("id") or vid
                    if vid and (name, vid) not in entities:
                        entities.append((name, vid))
                if entities:
                    logger.info(f"[TG] Loaded {len(entities)} '{vtype}' vertices")
                    break
            except Exception as exc:
                logger.debug(f"[TG] Could not fetch vertices for '{vtype}': {exc}")

        TigerGraphClient._global_entities_cache = entities
        self._entities = entities
        return entities

    # ── SEC Text Chunk Loader from data/sp100/chunks.jsonl ──────────────────

    def load_sec_text_chunks(
        self,
        companies: list[str],
        chunk_ids: list[str] | None = None,
        keywords: list[str] | None = None,
        max_chunks_per_company: int = 10,
    ) -> list[dict[str, Any]]:
        """Load original SEC filing text chunks from data/sp100/chunks.jsonl."""
        if TigerGraphClient._global_chunks_index_by_id is None or TigerGraphClient._global_chunks_index_by_company is None:
            by_id: dict[str, dict[str, Any]] = {}
            by_comp: dict[str, list[dict[str, Any]]] = {}
            chunks_path = ROOT_DIR / "data" / "sp100" / "chunks.jsonl"
            if chunks_path.exists():
                with chunks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            cid = row.get("chunk_id") or row.get("id")
                            cname = row.get("company_name") or row.get("company") or ""
                            if cid:
                                by_id[cid] = row

                            cname_low = cname.lower()
                            matched_key = None
                            for existing_key in by_comp:
                                if existing_key.lower() in cname_low or cname_low in existing_key.lower():
                                    matched_key = existing_key
                                    break
                            if not matched_key:
                                matched_key = cname
                                by_comp[matched_key] = []
                            by_comp[matched_key].append(row)
                        except Exception:
                            continue
            TigerGraphClient._global_chunks_index_by_id = by_id
            TigerGraphClient._global_chunks_index_by_company = by_comp

        self._chunks_index_by_id = TigerGraphClient._global_chunks_index_by_id
        self._chunks_index_by_company = TigerGraphClient._global_chunks_index_by_company

        results: list[dict[str, Any]] = []

        # Step 1: Ensure each company gets rich SEC filing text chunks matching keywords/10-Ks
        kws_tokens = []
        for k in (keywords or []):
            if k:
                kws_tokens.extend([w.lower() for w in re.split(r"\W+", k) if len(w) >= 3])
        kws_tokens = list(set(kws_tokens))

        for company in companies:
            variants = _get_company_variants(company)
            comp_chunks = []
            for ckey, clist in (self._chunks_index_by_company or {}).items():
                ckey_low = ckey.lower()
                if any(v in ckey_low or ckey_low in v for v in variants):
                    comp_chunks.extend(clist)

            # Sort company chunks by keyword token match frequency (highest relevance first)
            if kws_tokens:
                comp_chunks.sort(
                    key=lambda r: sum(1 for tok in kws_tokens if tok in (r.get("text") or "").lower()),
                    reverse=True,
                )

            current_comp_matched: list[dict[str, Any]] = []

            # Priority A: Match 10-K chunks containing any key tokens
            if kws_tokens:
                for row in comp_chunks:
                    ftype = (row.get("filing_type") or row.get("form") or "").upper()
                    text_low = (row.get("text") or "").lower()
                    token_match_cnt = sum(1 for tok in kws_tokens if tok in text_low)
                    if (ftype == "10-K" or "10-k" in ftype) and token_match_cnt >= 1 and row not in results:
                        results.append(row)
                        current_comp_matched.append(row)
                        if len(current_comp_matched) >= max_chunks_per_company:
                            break

            # Priority B: Match any chunks containing key tokens
            if len(current_comp_matched) < max_chunks_per_company and kws_tokens:
                for row in comp_chunks:
                    text_low = (row.get("text") or "").lower()
                    token_match_cnt = sum(1 for tok in kws_tokens if tok in text_low)
                    if token_match_cnt >= 1 and row not in results:
                        results.append(row)
                        current_comp_matched.append(row)
                        if len(current_comp_matched) >= max_chunks_per_company:
                            break

            # Priority C: Add graph-traversed chunk IDs for this company
            if len(current_comp_matched) < max_chunks_per_company and chunk_ids:
                for cid in chunk_ids:
                    if cid and cid in self._chunks_index_by_id:
                        row = self._chunks_index_by_id[cid]
                        cname_low = (row.get("company_name") or row.get("company") or "").lower()
                        if any(v in cname_low for v in variants) and row not in results:
                            results.append(row)
                            current_comp_matched.append(row)
                            if len(current_comp_matched) >= max_chunks_per_company:
                                break

            # Priority D: Fill remaining slots with any company SEC chunks
            if len(current_comp_matched) < max_chunks_per_company:
                for row in comp_chunks:
                    if row not in results:
                        results.append(row)
                        current_comp_matched.append(row)
                        if len(current_comp_matched) >= max_chunks_per_company:
                            break

        # Step 3: Global keyword search if no company specified or to enrich multi-doc queries
        if kws_tokens and len(results) < 10 and self._chunks_index_by_id:
            for cid, row in self._chunks_index_by_id.items():
                text_low = (row.get("text") or "").lower()
                if any(kw in text_low for kw in kws_tokens) and row not in results:
                    results.append(row)
                    if len(results) >= 10:
                        break

        return results

    # ── Graph Retrieval Queries ──────────────────────────────────────────────

    def retrieve_context(self, company_name: str, year: str | None = None, keyword: str | None = None) -> dict[str, Any]:
        """retrieveContext(company, year, keyword) query execution."""
        return self.get_company_context(company_name)

    def retrieve_executive_context(
        self, executive_name: str, year: str | None = None, keyword: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """retrieveExecutiveContext(executive) -> Find Company -> retrieveContext(company)."""
        company_name = self._find_company_by_executive(executive_name)
        context = self.retrieve_context(company_name, year, keyword)
        return company_name, context

    def compare_companies(
        self, company1: str, company2: str, year: str | None = None, keyword: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """compareCompanies(company1, company2) -> Parallel retrieve context for both & merge."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(self.retrieve_context, company1, year, keyword)
            f2 = executor.submit(self.retrieve_context, company2, year, keyword)
            c1_data = f1.result()
            c2_data = f2.result()
        return {company1: c1_data, company2: c2_data}

    def search_entity(
        self, keyword: str, year: str | None = None, metric: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """searchEntity(keyword) fallback query execution."""
        company_name = COMPANY_MAPPING.get(keyword.lower(), "Apple Inc.")
        context = self.retrieve_context(company_name, year, metric)
        return company_name, context

    def get_company_context(self, company_name: str) -> dict[str, Any]:
        """Execute getCompanyContext / REST graph traversal on TigerGraph."""
        if self.configured:
            try:
                data = self._traverse_rest_company(company_name)
                if data.get("documents") or data.get("company_attrs"):
                    return data
            except Exception as exc:
                logger.warning(f"[TG] REST traversal call for '{company_name}' failed: {exc}")

        logger.info(f"[TG] Using extracted SEC dataset fallback for '{company_name}'")
        return self._load_from_extracted_dataset(company_name)

    def _find_company_by_executive(self, executive_name: str) -> str:
        """Find company associated with executive via TigerGraph or dataset."""
        if self.configured:
            eg = quote(self.graph_name, safe="")
            try:
                exec_path = f"/restpp/graph/{eg}/vertices/Executive?filter=name%3D%22{quote(executive_name, safe='')}%22"
                payload = self._get_json(exec_path)
                results = payload.get("results") or []
                if results:
                    exec_vid = results[0].get("v_id")
                    if exec_vid:
                        edge_path = f"/restpp/graph/{eg}/edges/Executive/{exec_vid}"
                        e_payload = self._get_json(edge_path)
                        for e in e_payload.get("results") or []:
                            doc_id = e.get("to_id")
                            if doc_id:
                                d_edge_path = f"/restpp/graph/{eg}/edges/Document/{doc_id}"
                                d_payload = self._get_json(d_edge_path)
                                for de in d_payload.get("results") or []:
                                    if de.get("to_type") == "Company":
                                        comp_vid = de.get("to_id")
                                        c_path = f"/restpp/graph/{eg}/vertices/Company/{comp_vid}"
                                        c_payload = self._get_json(c_path)
                                        c_res = c_payload.get("results") or []
                                        if c_res:
                                            c_attrs = c_res[0].get("attributes") or {}
                                            cname = c_attrs.get("name")
                                            if cname:
                                                return cname
            except Exception as exc:
                logger.warning(f"[TG] Executive graph lookup failed: {exc}")

        # Fallback executive mapping
        exec_low = executive_name.lower()
        if "ternus" in exec_low or "cook" in exec_low:
            return "Apple Inc."
        if "jassy" in exec_low or "bezos" in exec_low:
            return "Amazon"
        if "nadella" in exec_low:
            return "Microsoft"
        if "huang" in exec_low:
            return "NVIDIA"
        if "woods" in exec_low:
            return "ExxonMobil"
        if "dimon" in exec_low:
            return "JPMorgan Chase"
        if "mcmillon" in exec_low:
            return "Walmart"
        if "sarandos" in exec_low or "peters" in exec_low:
            return "Netflix, Inc."

        return "Apple Inc."

    def _traverse_rest_company(self, company_name: str) -> dict[str, Any]:
        cache_key = company_name.lower().strip()
        if cache_key in TigerGraphClient._global_traversal_cache:
            return TigerGraphClient._global_traversal_cache[cache_key]

        eg = quote(self.graph_name, safe="")
        cand_path = f"/restpp/graph/{eg}/vertices/Company?filter=name%3D%22{quote(company_name, safe='')}%22"
        try:
            payload = self._get_json(cand_path)
            results = payload.get("results") or []
        except Exception:
            results = []

        comp_vid = None
        company_attrs = {"name": company_name}

        if results:
            comp_vid = results[0].get("v_id")
            company_attrs = results[0].get("attributes") or company_attrs
        else:
            entities = self.fetch_entities()
            for name, vid in entities:
                if company_name.lower() in name.lower() or name.lower() in company_name.lower():
                    comp_vid = vid
                    company_attrs = {"name": name}
                    break

        base_data = self._load_from_extracted_dataset(company_name)

        if not comp_vid:
            TigerGraphClient._global_traversal_cache[cache_key] = base_data
            return base_data

        edge_path = f"/restpp/graph/{eg}/edges/Company/{comp_vid}"
        try:
            edge_payload = self._get_json(edge_path)
            edges = edge_payload.get("results") or []
        except Exception:
            edges = []

        docs = list(base_data.get("documents") or [])
        execs = list(base_data.get("executives") or [])
        metrics = list(base_data.get("financial_metrics") or [])
        risks = list(base_data.get("risks") or [])
        events = list(base_data.get("events") or [])

        seen_ids = set()
        doc_ids_to_fetch = []

        for e in edges[:4]:
            to_id = str(e.get("to_id") or "")
            to_type = e.get("to_type") or ""
            if not to_id or to_id in seen_ids:
                continue
            seen_ids.add(to_id)

            if to_type == "Document":
                doc_item = {"id": to_id, "chunk_id": to_id, "filing_type": "10-K", "company_name": company_name}
                if not any(d.get("chunk_id") == to_id for d in docs):
                    docs.append(doc_item)
                doc_ids_to_fetch.append(to_id)

        if doc_ids_to_fetch:
            from concurrent.futures import ThreadPoolExecutor
            def _fetch_doc_edge(to_id: str):
                try:
                    return to_id, self._get_json(f"/restpp/graph/{eg}/edges/Document/{to_id}")
                except Exception as exc:
                    logger.debug(f"[TG] Document edge traversal failed for {to_id}: {exc}")
                    return to_id, None

            with ThreadPoolExecutor(max_workers=len(doc_ids_to_fetch)) as executor:
                results_list = list(executor.map(_fetch_doc_edge, doc_ids_to_fetch))

            for to_id, d_edges_payload in results_list:
                if not d_edges_payload:
                    continue
                for de in d_edges_payload.get("results") or []:
                    target_id = de.get("to_id")
                    target_type = de.get("to_type")
                    attrs = de.get("attributes") or {}
                    if not target_id or target_id in seen_ids:
                        continue
                    seen_ids.add(target_id)

                    item = {"id": target_id, **attrs}
                    if target_type == "Executive" and item not in execs:
                        execs.append(item)
                    elif target_type == "FinancialMetric" and item not in metrics:
                        metrics.append(item)
                    elif target_type == "Risk" and item not in risks:
                        risks.append(item)
                    elif target_type == "Event" and item not in events:
                        events.append(item)

        res = {
            "company_attrs": company_attrs or base_data.get("company_attrs", {"name": company_name}),
            "documents": docs,
            "executives": execs,
            "financial_metrics": metrics,
            "risks": risks,
            "events": events,
        }
        TigerGraphClient._global_traversal_cache[cache_key] = res
        return res

    def _load_from_extracted_dataset(self, company_name: str) -> dict[str, Any]:
        """Fallback graph data loader from data/sp100/chunks.jsonl / final_extracted.jsonl."""
        if self._extracted_data_cache is None:
            self._extracted_data_cache = {}

            for file_name in ["chunks.jsonl", "final_extracted.jsonl"]:
                p = ROOT_DIR / "data" / "sp100" / file_name
                if not p.exists():
                    continue
                with p.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            row = json.loads(line)
                            cname = row.get("company_name") or row.get("company")
                            if not cname:
                                continue

                            matched_key = None
                            for existing_key in self._extracted_data_cache:
                                if cname.lower() in existing_key.lower() or existing_key.lower() in cname.lower():
                                    matched_key = existing_key
                                    break
                            if not matched_key:
                                matched_key = cname
                                self._extracted_data_cache[matched_key] = {
                                    "company_attrs": {
                                        "name": cname,
                                        "ticker": row.get("company", ""),
                                        "sector": row.get("sector", ""),
                                    },
                                    "documents": [],
                                    "executives": [],
                                    "financial_metrics": [],
                                    "risks": [],
                                    "events": [],
                                }

                            target = self._extracted_data_cache[matched_key]
                            if file_name == "chunks.jsonl":
                                if len(target["documents"]) < 10:
                                    target["documents"].append({
                                        "chunk_id": row.get("chunk_id", ""),
                                        "filing_type": row.get("filing_type", "10-K"),
                                        "filing_date": row.get("filing_date", ""),
                                        "text": (row.get("text") or "")[:500],
                                    })
                            else:
                                llm_out = row.get("llm_output") or {}
                                for ex in llm_out.get("Executive") or []:
                                    if ex and ex not in target["executives"]:
                                        target["executives"].append(ex)
                                for m in llm_out.get("FinancialMetric") or []:
                                    if m and m not in target["financial_metrics"]:
                                        target["financial_metrics"].append(m)
                                for r in llm_out.get("Risk") or []:
                                    if r and r not in target["risks"]:
                                        target["risks"].append(r)
                                for ev in llm_out.get("Event") or []:
                                    if ev and ev not in target["events"]:
                                        target["events"].append(ev)
                        except Exception:
                            continue

        cand_lower = company_name.lower()
        for key, val in self._extracted_data_cache.items():
            if cand_lower in key.lower() or key.lower() in cand_lower:
                return val

        return {
            "company_attrs": {"name": company_name},
            "documents": [],
            "executives": [],
            "financial_metrics": [],
            "risks": [],
            "events": [],
        }

    # ── HTTP Core & Token Auto-Refresh ──────────────────────────────────────

    def _suppress_ssl(self) -> None:
        if not self.verify_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _ensure_token(self) -> str:
        """Fetch fresh JWT token from TigerGraph if missing using TG_SECRET."""
        if self.api_token:
            return self.api_token
        if not self.secret:
            return ""
        try:
            url = f"{self.host}/gsql/v1/tokens"
            self._suppress_ssl()
            resp = self.session.post(
                url,
                json={"secret": self.secret},
                timeout=settings.request_timeout_seconds,
                verify=self.verify_ssl,
            )
            if resp.ok:
                data = resp.json()
                tok = data.get("token")
                if tok:
                    self.api_token = tok
                    logger.info("[TG] Successfully acquired JWT token via secret")
                    return tok
        except Exception as exc:
            logger.warning(f"[TG] Token acquisition via secret failed: {exc}")
        return ""

    def _headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.api_token:
            h["Authorization"] = f"Bearer {self.api_token}"
        return h

    def _get_json(self, path: str, retry: bool = True) -> dict[str, Any]:
        url = f"{self.host}{path}"
        self._suppress_ssl()

        self._ensure_token()

        auth = None
        headers = self._headers()
        if not self.api_token and self.username and self.password:
            auth = requests.auth.HTTPBasicAuth(self.username, self.password)

        try:
            resp = self.session.get(
                url,
                headers=headers,
                auth=auth,
                timeout=min(3.5, settings.request_timeout_seconds),
                verify=self.verify_ssl,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"TigerGraph request failed: {exc}") from exc

        if resp.status_code in (401, 403) and retry and self.secret:
            logger.warning(f"[TG] HTTP {resp.status_code}. Refreshing token and retrying...")
            self.api_token = ""
            self._ensure_token()
            return self._get_json(path, retry=False)

        if not resp.ok:
            text = resp.text
            if "Failed to start workspace" in text or "Auto start is not enabled" in text:
                raise RuntimeError(
                    "TigerGraph workspace is paused on TG Cloud. Please start the instance in TG Cloud Console."
                )
            raise RuntimeError(f"HTTP {resp.status_code}: {text[:300]}")

        data = json.loads(resp.text)
        if data.get("error"):
            if retry and "token" in str(data.get("message", "")).lower() and self.secret:
                self.api_token = ""
                self._ensure_token()
                return self._get_json(path, retry=False)
            raise ValueError(str(data.get("message") or "TigerGraph API error"))
        return data
