"""Three real benchmark pipelines: LLM-only, Basic RAG, and TigerGraph GraphRAG."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import settings
from .corpus import MedicalCorpus
from .evaluator import AccuracyEvaluator, AccuracyResult
from .graph import GraphReasoner
from .llm import GenerationResult, LLMClient
from .metrics import percent_reduction, timed
from .retrievers import HybridRetriever, RetrievalResult


class PipelineService:
    def __init__(self) -> None:
        import threading
        self.corpus = MedicalCorpus()
        self.retriever = HybridRetriever(self.corpus.chunks, dataset_path=self.corpus.dataset_path)
        self.graph = GraphReasoner()
        self.llm = LLMClient()
        self.evaluator = AccuracyEvaluator(self.llm)
        self.abort_event = threading.Event()

        # Pre-warm SEC chunks index and graph dataset in background thread for instant UI responses
        def _warm() -> None:
            import socket
            from medgraph.graph import TigerGraphClient
            try:
                host_str = self.graph.tigergraph.host.replace("https://", "").replace("http://", "").split("/")[0].split(":")[0]
                socket.create_connection((host_str, 443), timeout=0.2)
            except Exception:
                TigerGraphClient._global_unreachable = True
                self.graph.tigergraph.configured = False
            self.graph.tigergraph.load_sec_text_chunks([])
            self.graph.tigergraph._load_from_extracted_dataset("Apple Inc.")

        threading.Thread(target=_warm, daemon=True).start()

    @staticmethod
    def _public_response(pipeline: dict[str, Any]) -> dict[str, Any]:
        return pipeline

    def stop_generation(self) -> None:
        self.abort_event.set()

    def stats(self) -> dict[str, Any]:
        payload = self.corpus.stats(self.graph.stats())
        payload["llm_providers"] = {
            "production": settings.production_provider,
            "llm_only": settings.llm_only_provider,
            "basic_rag": settings.rag_provider,
            "graphrag": settings.graphrag_provider,
            "judge": "gemini",
            "test": settings.test_provider,
            "gemini_model": settings.gemini_model.removeprefix("models/"),
            "nvidia_model": settings.llm_model,
        }
        return payload

    def run_llm_only_api(
        self, query: str, reference_answer: str | None = None, *, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        result = timed(
            lambda: self._run_llm_only(query, reference_answer, provider=provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_basic_rag_api(
        self, query: str, reference_answer: str | None = None, *, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        result = timed(
            lambda: self._run_basic_rag(query, reference_answer, provider=provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_graphrag_api(
        self, query: str, reference_answer: str | None = None, provider: str | None = None, llm_provider: str | None = None
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        effective_provider = provider or llm_provider
        result = timed(
            lambda: self._run_graphrag(query, reference_answer, provider=effective_provider)
        )
        pipeline = self._attach_latency(result.value, result.latency_ms)
        return self._public_response(pipeline)

    def run_metrics_api(self, query: str, reference_answer: str | None = None) -> dict[str, Any]:
        """Return dashboard comparison metrics without duplicating full pipeline payloads."""
        result = self.run_query(query, reference_answer)
        pipelines = result["pipelines"]
        return {
            "query": result["query"],
            "expected_answer": result.get("expected_answer"),
            "expected_answer_source": result.get("expected_answer_source"),
            "pipelines": {
                key: {
                    "label": pipelines[key]["label"],
                    "answer": pipelines[key].get("answer", ""),
                    "error": pipelines[key].get("error"),
                    "metrics": pipelines[key]["metrics"],
                    "graph_paths": (pipelines[key].get("graph") or {}).get("paths", [])
                    if key == "graphrag"
                    else [],
                    "retrieved_chunks": len(pipelines[key].get("evidence") or []),
                }
                for key in ("llm_only", "basic_rag", "graphrag")
            },
            "comparison": result["comparison"],
            "graph": {
                "paths": result.get("graph", {}).get("paths", []),
                "nodes": len(result.get("graph", {}).get("nodes", [])),
                "edges": len(result.get("graph", {}).get("edges", [])),
                "backend": result.get("graph", {}).get("backend"),
                "status": result.get("graph", {}).get("status"),
            },
            "metric_sources": result["metric_sources"],
            "stats": result["stats"],
        }

    def run_query(
        self,
        query: str,
        reference_answer: str | None = None,
        *,
        llm_provider: str | None = None,
    ) -> dict[str, Any]:
        query = self._clean_query(query)
        provider = llm_provider
        reference_answer = self._clean_optional_text(reference_answer)
        self.abort_event.clear()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "llm_only": executor.submit(
                    self._safe_timed,
                    lambda: self._run_llm_only(query, reference_answer, provider=provider),
                    "llm_only",
                    "LLM Only",
                    "red",
                ),
                "basic_rag": executor.submit(
                    self._safe_timed,
                    lambda: self._run_basic_rag(query, reference_answer, provider=provider),
                    "basic_rag",
                    "Basic RAG",
                    "yellow",
                ),
                "graphrag": executor.submit(
                    self._safe_timed,
                    lambda: self._run_graphrag(query, reference_answer, provider=provider),
                    "graphrag",
                    "TigerGraph GraphRAG",
                    "cyan",
                ),
            }
            timed_results = {key: future.result() for key, future in futures.items()}

        pipelines = {
            key: self._attach_latency(result.value, result.latency_ms)
            for key, result in timed_results.items()
        }

        expected_answer = reference_answer
        expected_answer_source = "provided"
        if not expected_answer:
            expected_answer, expected_answer_source = self._generate_expected_answer(query, pipelines)
            if expected_answer:
                self._score_pipelines_against_expected(query, pipelines, expected_answer)

        self._attach_comparison_metrics(pipelines)

        basic_evidence = pipelines["basic_rag"].get("evidence") or []
        graph_payload = pipelines["graphrag"].get("graph") or self._empty_graph()
        return {
            "query": query,
            "expected_answer": expected_answer,
            "expected_answer_source": expected_answer_source,
            "pipelines": pipelines,
            "retrieval": {
                "top_chunks": basic_evidence,
                "entities": sorted({entity for chunk in basic_evidence for entity in chunk.get("entities", [])}),
            },
            "graph": graph_payload,
            "comparison": self._comparison(pipelines),
            "llm_provider": provider,
            "metric_sources": {
                "tokens": "Hosted LLM API usage when returned; otherwise estimated from prompt and answer text.",
                "cost": "Calculated from tokens and MEDGRAPH_PRICE_PER_1K_TOKENS.",
                "latency": "Measured wall-clock time for the real backend pipeline, including retrieval, graph traversal, hosted generation, and judging.",
                "accuracy": "Gemini LLM-as-a-judge PASS/FAIL against the generated expected_answer.",
                "bertscore": "BERTScore F1 between the generated answer and expected_answer when bert-score is installed.",
                "hallucination_risk": "LLM-as-a-judge assessment against retrieved chunks and/or TigerGraph context.",
                "graph": "TigerGraph REST traversal only. No local graph fallback is used.",
                "llm": "LLM-only, Basic RAG, and GraphRAG default to NVIDIA. Gemini generates expected_answer and performs LLM-as-a-judge scoring.",
            },
            "stats": self.stats(),
        }

    def run_benchmark(self, items: list[dict[str, str]]) -> dict[str, Any]:
        rows = []
        aggregate = {
            "llm_only": self._empty_benchmark_bucket(),
            "basic_rag": self._empty_benchmark_bucket(),
            "graphrag": self._empty_benchmark_bucket(),
        }
        for item in items:
            question = str(item.get("question") or "").strip()
            if not question:
                continue
            result = self.run_query(
                question,
                item.get("correct_answer"),
            )
            rows.append(result)
            for key, bucket in aggregate.items():
                metrics = result["pipelines"][key]["metrics"]
                bucket["tokens"] += metrics["tokens"]
                bucket["latency_ms"] += metrics["latency_ms"]
                bucket["cost_usd"] += metrics["cost_usd"]
                if metrics["accuracy"] is not None:
                    bucket["judge_pass_percent"] += metrics["accuracy"]
                    bucket["judge_evaluated"] += 1
                    if metrics["accuracy"] >= 100:
                        bucket["judge_passes"] += 1
                if metrics.get("bertscore_f1") is not None:
                    bucket["bertscore_f1"] += metrics["bertscore_f1"]
                    bucket["bertscore_evaluated"] += 1

        count = max(len(rows), 1)
        for key, bucket in aggregate.items():
            bucket["avg_tokens"] = int(bucket.pop("tokens") / count)
            bucket["avg_latency_ms"] = int(bucket.pop("latency_ms") / count)
            bucket["avg_cost_usd"] = round(bucket.pop("cost_usd") / count, 6)
            bucket["judge_pass_percent"] = (
                round(bucket["judge_pass_percent"] / bucket["judge_evaluated"], 1)
                if bucket["judge_evaluated"]
                else None
            )
            bucket["avg_bertscore_f1"] = (
                round(bucket.pop("bertscore_f1") / bucket["bertscore_evaluated"], 4)
                if bucket["bertscore_evaluated"]
                else None
            )

            # Backward-compatible aliases for older scripts.
            bucket["tokens"] = bucket["avg_tokens"]
            bucket["latency_ms"] = bucket["avg_latency_ms"]
            bucket["cost_usd"] = bucket["avg_cost_usd"]
            bucket["accuracy"] = bucket["judge_pass_percent"]
            bucket["bertscore_f1"] = bucket["avg_bertscore_f1"]

        improvements = self._benchmark_improvements(aggregate)

        return {
            "items_evaluated": len(rows),
            "llm_provider": "per-pipeline defaults",
            "generation_providers": {
                "llm_only": settings.llm_only_provider,
                "basic_rag": settings.rag_provider,
                "graphrag": settings.graphrag_provider,
            },
            "judge_provider": "gemini",
            "aggregate": aggregate,
            "improvements": improvements,
            "token_reduction_percent": improvements["graph_token_reduction_vs_rag_percent"],
            "cost_reduction_percent": improvements["graph_cost_reduction_vs_rag_percent"],
            "accuracy_improvement_percent": improvements["graph_accuracy_improvement_vs_rag_percent"],
            "rows": rows,
        }

    @staticmethod
    def _empty_benchmark_bucket() -> dict[str, Any]:
        return {
            "tokens": 0,
            "latency_ms": 0,
            "cost_usd": 0.0,
            "judge_pass_percent": 0.0,
            "judge_passes": 0,
            "judge_evaluated": 0,
            "bertscore_f1": 0.0,
            "bertscore_evaluated": 0,
        }

    @staticmethod
    def _benchmark_improvements(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
        rag = aggregate["basic_rag"]
        graph = aggregate["graphrag"]
        rag_accuracy = rag.get("judge_pass_percent")
        graph_accuracy = graph.get("judge_pass_percent")
        rag_bert = rag.get("avg_bertscore_f1")
        graph_bert = graph.get("avg_bertscore_f1")
        accuracy_delta = (
            round(graph_accuracy - rag_accuracy, 1)
            if graph_accuracy is not None and rag_accuracy is not None
            else None
        )
        bert_delta = (
            round(graph_bert - rag_bert, 4)
            if graph_bert is not None and rag_bert is not None
            else None
        )
        accuracy_improvement = None
        if accuracy_delta is not None and rag_accuracy not in (None, 0):
            accuracy_improvement = round((accuracy_delta / rag_accuracy) * 100, 1)
        latency_improvement = percent_reduction(
            rag["avg_latency_ms"], graph["avg_latency_ms"]
        )
        bert_improvement = None
        if bert_delta is not None and rag_bert not in (None, 0):
            bert_improvement = round((bert_delta / rag_bert) * 100, 1)
        return {
            "graph_token_reduction_vs_rag_percent": percent_reduction(
                rag["avg_tokens"], graph["avg_tokens"]
            ),
            "graph_cost_reduction_vs_rag_percent": percent_reduction(
                rag["avg_cost_usd"], graph["avg_cost_usd"]
            ),
            "graph_latency_improvement_vs_rag_percent": latency_improvement,
            "graph_accuracy_improvement_vs_rag_percent": accuracy_improvement,
            "graph_accuracy_point_delta_vs_rag": accuracy_delta,
            "graph_bertscore_improvement_vs_rag_percent": bert_improvement,
            "graph_bertscore_delta_vs_rag": bert_delta,
        }

    def _run_llm_only(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        llm_provider = provider or settings.llm_only_provider
        
        import logging
        logger = logging.getLogger("medgraph.pipelines")
        logger.info(f"[LLM Only] Starting pipeline for query: '{query}'")
        
        prompt = f"Answer this corporate finance and SEC filing question directly using your general internal knowledge. State the answer clearly.\n\nQuestion:\n{query}"
        evidence = ""
        retrieved_payload = []
        
        if self.abort_event.is_set():
            raise RuntimeError("LLM generation cancelled by user.")
            
        import time
        llm_start = time.perf_counter()
        logger.info(f"[LLM Only] Calling LLM API ({llm_provider})...")
        try:
            generation = self.llm.generate(
                prompt,
                system=self._financial_system_prompt(),
                provider=llm_provider,
                temperature=0.2,
                max_tokens=850,
            )
            llm_ms = int((time.perf_counter() - llm_start) * 1000)
            warning = f"{llm_provider} generation grounded purely on its own knowledge."
            logger.info(f"[LLM Only] Successfully generated answer.")
        except Exception as exc:
            logger.error(f"[LLM Only] Generation failed: {exc}", exc_info=True)
            result = self._pipeline_error("llm_only", "LLM Only", "red", exc)
            result["warning"] = f"LLM generation failed for {llm_provider}."
            result["provider"] = llm_provider
            result["model"] = settings.llm_model if llm_provider == "nvidia" else None
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            return result

        return self._build_pipeline_result(
            key="llm_only",
            label="LLM Only",
            accent="red",
            generation=generation,
            query=query,
            context=prompt,
            evidence=evidence,
            reference_answer=reference_answer,
            warning=warning,
            extra={
                "evidence": retrieved_payload,
                "retrieved_chunks": retrieved_payload,
                "llm_latency_ms": llm_ms,
            },
        )

    def _run_basic_rag(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        return self._run_basic_rag_with_k(
            query,
            reference_answer,
            provider=provider,
            top_k=settings.rag_retrieval_top_k,
        )

    def _run_basic_rag_with_k(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
        top_k: int,
    ) -> dict[str, Any]:
        import logging
        import time
        from .entity_extractor import extract_entities
        logger = logging.getLogger("medgraph.pipelines")
        logger.info(f"[Basic RAG] Starting pipeline for query: '{query}' (top_k={top_k})")
        
        logger.info("[Basic RAG] Retrieving chunks via FAISS...")
        ret_start = time.perf_counter()
        entities = extract_entities(query)
        if len(entities.companies) >= 2:
            retrievals_dict = {}
            k_per_comp = max(8, top_k // len(entities.companies))
            metric_term = " ".join(entities.metrics + entities.events + entities.risks) or "net income revenue financial performance"
            for comp in entities.companies:
                for r in self.retriever.search(f"{comp} {metric_term}", k=k_per_comp):
                    retrievals_dict[r.chunk_id] = r
            for r in self.retriever.search(query, k=top_k):
                if len(retrievals_dict) >= top_k:
                    break
                retrievals_dict[r.chunk_id] = r
            retrievals = list(retrievals_dict.values())[:top_k]
        else:
            retrievals = self.retriever.search(query, k=top_k)
        retrieval_ms = int((time.perf_counter() - ret_start) * 1000)

        if not retrievals:
            logger.warning("[Basic RAG] FAISS returned no chunks.")
            raise RuntimeError("FAISS retrieval returned no chunks for the query.")
        
        logger.info(f"[Basic RAG] Retrieved {len(retrievals)} chunks.")
        context = self._retrieval_context(retrievals)
        prompt = f"""
Use only the retrieved SEC filing document chunks below to answer the question.
When evidence is incomplete, say what is missing instead of inventing facts.

Question:
{query}

Retrieved SEC filing chunks ({len(retrievals)} vector matches, top-{top_k}):
{context}
"""
        evidence = "\n\n".join(result.content for result in retrievals)
        retrieved_payload = [self._retrieval_to_dict(result) for result in retrievals]
        llm_provider = provider or settings.rag_provider
        
        if self.abort_event.is_set():
            raise RuntimeError("LLM generation cancelled by user.")
            
        logger.info(f"[Basic RAG] Calling LLM API ({llm_provider})...")
        llm_start = time.perf_counter()
        try:
            generation = self.llm.generate(
                prompt,
                system=self._financial_system_prompt(),
                provider=llm_provider,
                temperature=0.2,
                max_tokens=min(8192, 1200 + len(retrievals) * 120),
            )
            llm_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info(f"[Basic RAG] Successfully generated answer.")
        except Exception as exc:
            logger.error(f"[Basic RAG] Generation failed: {exc}", exc_info=True)
            result = self._pipeline_error("basic_rag", "Basic RAG", "yellow", exc)
            result["warning"] = "FAISS retrieval succeeded, but hosted LLM generation failed."
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            return result
        return self._build_pipeline_result(
            key="basic_rag",
            label="Basic RAG",
            accent="yellow",
            generation=generation,
            query=query,
            context=prompt,
            evidence=evidence,
            reference_answer=reference_answer,
            warning=(
                f"SentenceTransformer + FAISS top-{top_k} retrieval + {llm_provider} generation."
            ),
            extra={
                "evidence": retrieved_payload,
                "retrieved_chunks": retrieved_payload,
                "retrieval_latency_ms": retrieval_ms,
                "llm_latency_ms": llm_ms,
            },
        )

    def _run_graphrag(
        self,
        query: str,
        reference_answer: str | None,
        *,
        provider: str | None = None,
    ) -> dict[str, Any]:
        import logging
        import time
        logger = logging.getLogger("medgraph.pipelines")
        logger.info(f"[GraphRAG] Starting pipeline for query: '{query}'")
        
        local_fast_mode = (
            (provider or settings.graphrag_provider) in {"local", "offline"}
        )
        if local_fast_mode:
            logger.info("[GraphRAG] Routing to local fast mode (Basic RAG).")
            result = self._run_basic_rag_with_k(
                query,
                reference_answer,
                provider=provider,
                top_k=settings.graphrag_retrieval_top_k,
            )
            result["label"] = "TigerGraph GraphRAG"
            result["accent"] = "cyan"
            result["warning"] = (
                "Local fast mode routed GraphRAG to compact retrieval; live TigerGraph traversal skipped."
            )
            result["graph"] = {
                **self._empty_graph(),
                "status": "Local fast mode is enabled; TigerGraph traversal skipped.",
                "routing": "local_fast_retrieval",
            }
            result["graph_context"] = result["graph"]["status"]
            return result

        logger.info("[GraphRAG] Retrieving FAISS chunks to seed Graph traversal...")
        ret_start = time.perf_counter()
        retrievals = self.retriever.search(query, k=settings.graphrag_retrieval_top_k)
        retrieval_ms = int((time.perf_counter() - ret_start) * 1000)
        retrieved_payload = [self._retrieval_to_dict(result) for result in retrievals]
        try:
            logger.info("[GraphRAG] Traversing TigerGraph for company context...")
            graph = self.graph.reason(query, retrievals)
            logger.info("[GraphRAG] Traversal complete.")
        except Exception as exc:
            logger.error(f"[GraphRAG] TigerGraph traversal failed: {exc}")
            result = self._pipeline_error("graphrag", "TigerGraph GraphRAG", "cyan", exc)
            result["warning"] = str(exc)
            result["graph"] = {
                **self._empty_graph(),
                "status": f"TigerGraph error: {exc}",
                "routing": "error",
            }
            result["graph_context"] = f"TigerGraph error: {exc}"
            return result

        tg_ms = graph.get("tg_latency_ms", 0)
        chunk_load_ms = graph.get("chunk_load_latency_ms", 0)
        graph_context = graph.get("formatted_context") or self._graph_context(graph)

        prompt = f"""\
Answer the financial question using ONLY the SEC filing text and TigerGraph knowledge graph context below.
The context includes full SEC filing text chunks retrieved from data/sp100/chunks.jsonl via TigerGraph index traversal, along with executives, financial metrics, business risks, and corporate events.

Do NOT use any external knowledge. Rely on the SEC filing text and graph evidence provided below.

Question:
{query}

SEC Filing & GraphRAG Context:
{graph_context}
"""
        logger.info(f"Gemini Prompt:\n{prompt}")

        evidence = graph_context
        llm_provider = provider or settings.graphrag_provider
        
        if self.abort_event.is_set():
            raise RuntimeError("LLM generation cancelled by user.")
            
        logger.info(f"[GraphRAG] LLM request sent to {llm_provider}")
        llm_start = time.perf_counter()
        try:
            generation = self.llm.generate(
                prompt,
                system=self._financial_system_prompt(),
                provider=llm_provider,
                temperature=0.2,
                max_tokens=800,
            )
            llm_ms = int((time.perf_counter() - llm_start) * 1000)
            logger.info(f"[GraphRAG] Answer returned from LLM ({len(generation.answer)} chars)")
        except Exception as exc:
            logger.error(f"[GraphRAG] Generation failed: {exc}", exc_info=True)
            result = self._pipeline_error("graphrag", "TigerGraph GraphRAG", "cyan", exc)
            result["warning"] = "TigerGraph traversal succeeded, but hosted LLM generation failed."
            result["graph"] = graph
            result["graph_context"] = graph_context
            result["evidence"] = retrieved_payload
            result["retrieved_chunks"] = retrieved_payload
            return result
        return self._build_pipeline_result(
            key="graphrag",
            label="TigerGraph GraphRAG",
            accent="cyan",
            generation=generation,
            query=query,
            context=prompt,
            evidence=evidence,
            reference_answer=reference_answer,
            warning=(
                f"TigerGraph deterministic company retrieval + getCompanyContext + {llm_provider} generation."
            ),
            extra={
                "graph": graph,
                "graph_context": graph_context,
                "evidence": graph.get("tg_chunks") or retrieved_payload,
                "retrieved_chunks": graph.get("tg_chunks") or retrieved_payload,
                "retrieval_latency_ms": retrieval_ms,
                "tg_latency_ms": tg_ms,
                "chunk_load_latency_ms": chunk_load_ms,
                "llm_latency_ms": llm_ms,
            },
        )

    def _build_pipeline_result(
        self,
        key: str,
        label: str,
        accent: str,
        generation: GenerationResult,
        query: str,
        context: str,
        evidence: str,
        reference_answer: str | None,
        warning: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        if not reference_answer:
            reference_answer = evidence[:400] if evidence else generation.answer
        accuracy = self.evaluator.score(
            query=query,
            answer=generation.answer,
            evidence=evidence,
            pipeline=key,
            reference_answer=reference_answer,
        )
        result = {
            "label": label,
            "accent": accent,
            "answer": generation.answer,
            "warning": warning,
            "provider": generation.provider,
            "model": generation.model,
            "error": None,
            "metrics": self._metrics(generation, accuracy),
        }
        result.update(extra)
        for lat_key in ("retrieval_latency_ms", "llm_latency_ms", "tg_latency_ms", "chunk_load_latency_ms"):
            if lat_key in extra:
                result["metrics"][lat_key] = extra[lat_key]
        return result

    def _generate_expected_answer(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> tuple[str | None, str]:
        if settings.expected_answer_provider in {"local", "offline", "extractive"}:
            fallback = self._extractive_expected_answer(query, pipelines)
            return fallback, "retrieved_context_fallback"

        if not self.llm.has_provider("gemini"):
            fallback = self._extractive_expected_answer(query, pipelines)
            return fallback, "retrieved_context_fallback"

        context = self._reference_generation_context(query, pipelines)
        prompt = f"""
You are generating a reference financial answer for evaluation purposes only.
Given this financial query, return the ideal concise answer using the supplied
retrieved financial document evidence as the primary source.

Financial query:
{query}

Retrieved financial document chunks, company filings, graph evidence,
and financial metrics available to the benchmark:
{context}

Instructions:
- Return only the expected answer, not JSON.
- Be concise but financially accurate enough for evaluation.
- Prefer specific data from the supplied evidence (financial metrics, filing data, executive info) over generic knowledge.
- Include relevant financial figures, dates, and context when supported by evidence.
- Use factual, data-driven language.
- If the evidence is incomplete, say so briefly instead of filling gaps with generic financial advice.
"""
        try:
            generation = self.llm.generate(
                prompt,
                system=(
                    "You write concise reference financial answers for QA evaluation. "
                    "Ground the answer primarily in retrieved financial document and "
                    "company filing context. Use established knowledge only to connect "
                    "evidence that is already present."
                ),
                provider="gemini",
                temperature=0.0,
                max_tokens=650,
            )
        except Exception as exc:  # pragma: no cover - external integration boundary
            fallback = self._extractive_expected_answer(query, pipelines)
            if fallback:
                return fallback, "retrieved_context_fallback"
            return None, self._friendly_external_error("expected_answer generation failed", exc)

        answer = generation.answer.strip()
        if not answer:
            fallback = self._extractive_expected_answer(query, pipelines)
            if fallback:
                return fallback, "retrieved_context_fallback"
            return None, "expected_answer generation returned an empty response."
        return answer, "gemini_auto_reference"

    def _extractive_expected_answer(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> str | None:
        context = self._reference_generation_context(query, pipelines)
        context = context.split("TigerGraph evidence:", 1)[0]
        if not context.strip():
            return None
        salient = self._salient_reference_sentences(query, context)
        if not salient:
            return None
        return " ".join(salient[:5])

    @staticmethod
    def _salient_reference_sentences(query: str, context: str) -> list[str]:
        import re

        query_terms = set(PipelineService._query_tokens(query))
        priority_terms = {
            "revenue",
            "income",
            "profit",
            "earnings",
            "loss",
            "margin",
            "growth",
            "dividend",
            "filing",
            "executive",
            "risk",
            "metric",
            "quarter",
            "annual",
            "fiscal",
            "balance",
            "asset",
            "liability",
        }
        parts = [
            " ".join(part.split())
            for part in re.split(r"(?<=[.!?])\s+|\n+", context)
            if len(part.split()) >= 5 and "->" not in part
        ]

        def score(sentence: str) -> tuple[int, int, int]:
            lowered = sentence.lower()
            tokens = set(PipelineService._query_tokens(lowered))
            financial_bonus = 5 if any(t in lowered for t in ["net income", "revenue", "earnings per share"]) else 0
            return (
                len(tokens & query_terms) * 2
                + sum(1 for term in priority_terms if term in lowered),
                financial_bonus,
                len(sentence),
            )

        ranked = sorted(parts, key=score, reverse=True)
        selected = []
        seen = set()
        for sentence in ranked:
            key = sentence.lower()
            if key in seen:
                continue
            selected.append(sentence)
            seen.add(key)
            if len(selected) >= 5:
                break
        return selected

    @staticmethod
    def _friendly_external_error(prefix: str, exc: Exception) -> str:
        message = str(exc)
        if "429" in message or "RESOURCE_EXHAUSTED" in message or "quota" in message.lower():
            return f"{prefix}: Gemini quota/rate limit reached"
        return f"{prefix}: external LLM request failed"

    def _score_pipelines_against_expected(
        self,
        query: str,
        pipelines: dict[str, dict[str, Any]],
        expected_answer: str,
    ) -> None:
        for key, pipeline in pipelines.items():
            accuracy = self.evaluator.score(
                query=query,
                answer=pipeline.get("answer", ""),
                evidence=self._pipeline_evidence_text(key, pipeline),
                pipeline=key,
                reference_answer=expected_answer,
            )
            self._apply_accuracy(pipeline, accuracy)

    @staticmethod
    def _apply_accuracy(pipeline: dict[str, Any], accuracy: AccuracyResult) -> None:
        metrics = pipeline["metrics"]
        metrics["accuracy"] = accuracy.pass_rate
        metrics["bertscore_f1"] = accuracy.bertscore_f1
        metrics["judge"] = accuracy.judge
        metrics["judge_rationale"] = accuracy.rationale
        metrics["hallucination_risk"] = accuracy.hallucination_risk
        metrics["accuracy_method"] = accuracy.method

    def _reference_generation_context(
        self, query: str, pipelines: dict[str, dict[str, Any]]
    ) -> str:
        sections = []
        reference_retrievals = self.retriever.search(
            query, k=max(settings.rag_retrieval_top_k, 12)
        )
        reference_chunks = [
            self._retrieval_to_dict(result)
            for result in self._rank_reference_retrievals(query, reference_retrievals)
        ]
        if reference_chunks:
            sections.append("Priority ACR/recommendation chunks:")
            for chunk in reference_chunks[:8]:
                sections.append(
                    f"[{chunk.get('chunk_id', 'chunk')} | {chunk.get('source', '')} | "
                    f"{chunk.get('category', '')}]\n"
                    f"{self._compress(str(chunk.get('content') or chunk.get('preview') or ''), 1000)}"
                )

        basic_chunks = pipelines["basic_rag"].get("retrieved_chunks") or []
        if basic_chunks:
            sections.append("Basic RAG top chunks:")
            for chunk in basic_chunks[: settings.rag_retrieval_top_k]:
                sections.append(
                    f"[{chunk.get('chunk_id', 'chunk')}] "
                    f"{self._compress(str(chunk.get('content') or chunk.get('preview') or ''), 700)}"
                )

        graph = pipelines["graphrag"].get("graph") or {}
        graph_context = pipelines["graphrag"].get("graph_context") or ""
        paths = graph.get("paths") or []
        if paths or graph_context:
            sections.append("TigerGraph evidence:")
            sections.extend(str(path) for path in paths[:8])
            if graph_context:
                sections.append(self._compress(str(graph_context), 1600))

        if not sections:
            sections.append("No retrieved context was available; rely on established medical knowledge.")
        return "\n".join(sections)

    @staticmethod
    def _rank_reference_retrievals(
        query: str, retrievals: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        query_terms = set(PipelineService._query_tokens(query))
        financial_terms = {
            "revenue",
            "income",
            "profit",
            "earnings",
            "filing",
            "10-k",
            "10-q",
            "sec",
            "annual",
            "quarter",
            "fiscal",
            "balance",
            "cash flow",
            "dividend",
            "executive",
            "risk",
        }

        def score(result: RetrievalResult) -> tuple[float, float]:
            text = f"{result.source} {result.category} {result.content}".lower()
            tokens = set(PipelineService._query_tokens(text))
            value = result.similarity * 8.0
            value += len(tokens & query_terms) * 0.8
            value += sum(1.0 for term in financial_terms if term in text)
            return value, result.similarity

        return sorted(retrievals, key=score, reverse=True)

    @staticmethod
    def _query_tokens(text: str) -> list[str]:
        import re

        return [
            token.lower()
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", text or "")
            if len(token) >= 3
        ]

    @staticmethod
    def _pipeline_evidence_text(key: str, pipeline: dict[str, Any]) -> str:
        if key == "graphrag":
            graph_context = pipeline.get("graph_context")
            if graph_context:
                return str(graph_context)
            graph = pipeline.get("graph") or {}
            return "\n".join(str(path) for path in graph.get("paths") or [])

        chunks = pipeline.get("retrieved_chunks") or pipeline.get("evidence") or []
        if isinstance(chunks, list):
            return "\n\n".join(
                str(chunk.get("content") or chunk.get("preview") or chunk)
                if isinstance(chunk, dict)
                else str(chunk)
                for chunk in chunks
            )
        return str(chunks or "")

    def _safe_timed(
        self,
        fn: Callable[[], dict[str, Any]],
        key: str,
        label: str,
        accent: str,
    ):
        try:
            return timed(fn)
        except Exception as exc:
            return timed(lambda: self._pipeline_error(key, label, accent, exc))

    def _pipeline_error(
        self, key: str, label: str, accent: str, exc: Exception
    ) -> dict[str, Any]:
        graph = self._empty_graph() if key == "graphrag" else None
        result: dict[str, Any] = {
            "label": label,
            "accent": accent,
            "answer": "",
            "warning": "Pipeline did not run successfully. No answer was generated.",
            "error": str(exc),
            "provider": None,
            "model": None,
            "evidence": [],
            "retrieved_chunks": [],
            "metrics": {
                "tokens": 0,
                "latency_ms": 0,
                "cost_usd": 0.0,
                "accuracy": None,
                "bertscore_f1": None,
                "judge": "NOT_EVALUATED",
                "judge_rationale": str(exc),
                "hallucination_risk": "UNKNOWN",
                "accuracy_method": "pipeline failed",
                "token_source": "none",
                "token_reduction_vs_basic": None,
                "cost_reduction_vs_basic": None,
            },
        }
        if graph is not None:
            result["graph"] = graph
            result["graph_context"] = ""
        return result

    @staticmethod
    def _metrics(generation: GenerationResult, accuracy: AccuracyResult) -> dict[str, Any]:
        return {
            "tokens": generation.tokens_used,
            "latency_ms": 0,
            "cost_usd": generation.cost_usd,
            "accuracy": accuracy.pass_rate,
            "bertscore_f1": accuracy.bertscore_f1,
            "judge": accuracy.judge,
            "judge_rationale": accuracy.rationale,
            "hallucination_risk": accuracy.hallucination_risk,
            "accuracy_method": accuracy.method,
            "token_source": generation.token_source,
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "token_reduction_vs_basic": None,
            "cost_reduction_vs_basic": None,
        }

    @staticmethod
    def _attach_latency(result: dict[str, Any], measured_latency_ms: int) -> dict[str, Any]:
        result["metrics"]["latency_ms"] = measured_latency_ms
        return result

    @staticmethod
    def _attach_comparison_metrics(pipelines: dict[str, dict[str, Any]]) -> None:
        basic_tokens = pipelines["basic_rag"]["metrics"]["tokens"]
        graph_tokens = pipelines["graphrag"]["metrics"]["tokens"]
        basic_cost = pipelines["basic_rag"]["metrics"]["cost_usd"]
        graph_cost = pipelines["graphrag"]["metrics"]["cost_usd"]
        pipelines["graphrag"]["metrics"]["token_reduction_vs_basic"] = percent_reduction(
            basic_tokens, graph_tokens
        )
        pipelines["graphrag"]["metrics"]["cost_reduction_vs_basic"] = percent_reduction(
            basic_cost, graph_cost
        )

    @staticmethod
    def _comparison(pipelines: dict[str, dict[str, Any]]) -> dict[str, Any]:
        successful = {
            key: value
            for key, value in pipelines.items()
            if not value.get("error") and value["metrics"]["tokens"] > 0
        }
        winner = None
        if successful:
            winner = min(successful.items(), key=lambda row: row[1]["metrics"]["tokens"])[0]
        return {
            "token_reduction_percent": pipelines["graphrag"]["metrics"][
                "token_reduction_vs_basic"
            ],
            "cost_reduction_percent": pipelines["graphrag"]["metrics"][
                "cost_reduction_vs_basic"
            ],
            "winner": winner,
        }

    @staticmethod
    def _public_response(pipeline: dict[str, Any]) -> dict[str, Any]:
        metrics = pipeline["metrics"]
        payload = {
            "answer": pipeline["answer"],
            "tokens_used": metrics["tokens"],
            "latency": metrics["latency_ms"],
            "cost": metrics["cost_usd"],
            "error": pipeline.get("error"),
            "provider": pipeline.get("provider"),
            "model": pipeline.get("model"),
            "accuracy_score": metrics.get("accuracy"),
            "hallucination_risk": metrics.get("hallucination_risk"),
            "pipeline": pipeline,
        }
        if pipeline.get("retrieved_chunks") is not None:
            payload["retrieved_chunks"] = pipeline.get("retrieved_chunks", [])
        if pipeline.get("graph") is not None:
            graph = pipeline["graph"]
            payload.update(
                {
                    "graph_nodes": graph.get("nodes", []),
                    "graph_edges": graph.get("edges", []),
                    "graph_context": pipeline.get("graph_context", ""),
                }
            )
        return payload

    @staticmethod
    def _retrieval_context(retrievals: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[{result.chunk_id} | {result.source} | score={result.similarity}]\n{result.content}"
            for result in retrievals
        )

    def _full_dataset_context(self) -> str:
        return "\n\n".join(
            f"[{chunk.chunk_id} | {chunk.source} | {chunk.category}]\n{chunk.content}"
            for chunk in self.corpus.chunks
        )

    def _dataset_prompt_payload(self, query: str, chunks: list[Any], label: str) -> tuple[str, str, list[dict[str, Any]]]:
        context = "\n\n".join(
            f"[{chunk.chunk_id} | {chunk.source} | {chunk.category}]\n{chunk.content}"
            for chunk in chunks
        )
        prompt = f"""
Answer this financial question using the {label} financial dataset context below.
When the dataset context is incomplete, say what is missing instead of inventing facts.

Question:
{query}

Financial dataset context ({len(chunks)} chunks):
{context}
"""
        return prompt, context, [self._chunk_to_dict(chunk) for chunk in chunks]

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message or "read timed out" in message

    def _graph_context(self, graph: dict[str, Any]) -> str:
        """Format TigerGraph financial data as context for the LLM."""
        sections: list[str] = []

        # Company overview
        companies = graph.get("companies", {})
        if companies:
            sections.append("=== Companies ===")
            for name, summary in companies.items():
                sections.append(
                    f"{name}: {summary.get('documents', 0)} documents, "
                    f"{summary.get('executives', 0)} executives, "
                    f"{summary.get('risks', 0)} risks, "
                    f"{summary.get('events', 0)} events, "
                    f"{summary.get('financial_metrics', 0)} financial metrics"
                )

        # Financial metrics
        metrics = graph.get("financial_metrics", [])
        if metrics:
            sections.append("\n=== Financial Metrics ===")
            for m in metrics[:20]:
                metric_name = m.get("metric") or m.get("metric_id") or "metric"
                value = m.get("value") or ""
                sections.append(f"- {metric_name}: {value}")

        # Executives
        executives = graph.get("executives", [])
        if executives:
            sections.append("\n=== Executives ===")
            for e in executives[:15]:
                name = e.get("name") or e.get("executive_id") or "executive"
                title = e.get("title") or ""
                sections.append(f"- {name}" + (f" ({title})" if title else ""))

        # Risks
        risks = graph.get("risks", [])
        if risks:
            sections.append("\n=== Risk Disclosures ===")
            for r in risks[:10]:
                risk_text = r.get("description") or r.get("name") or r.get("text") or ""
                if not risk_text:
                    # Try first string attribute
                    for v in r.values():
                        if isinstance(v, str) and v:
                            risk_text = v
                            break
                if risk_text:
                    sections.append(f"- {self._compress(risk_text, 300)}")

        # Events
        events = graph.get("events", [])
        if events:
            sections.append("\n=== Events ===")
            for ev in events[:10]:
                ev_text = ev.get("description") or ev.get("name") or ev.get("text") or ""
                if not ev_text:
                    for v in ev.values():
                        if isinstance(v, str) and v:
                            ev_text = v
                            break
                if ev_text:
                    sections.append(f"- {self._compress(ev_text, 300)}")

        # Document summaries
        documents = graph.get("documents", [])
        if documents:
            sections.append(f"\n=== Documents ({len(documents)} filings) ===")
            for d in documents[:10]:
                company = d.get("company_name") or ""
                ticker = d.get("ticker") or ""
                filing = d.get("filing_type") or ""
                date = d.get("filing_date") or ""
                parts = [p for p in [company, ticker, filing, date] if p]
                sections.append(f"- {' | '.join(parts)}")

        # Graph paths
        paths = graph.get("paths") or []
        if paths:
            sections.append(f"\n=== Graph Paths ({len(paths)}) ===")
            for path in paths[:15]:
                sections.append(f"  {path}")

        if sections:
            return "\n".join(sections)

        entities = ", ".join(graph.get("entities") or [])
        status = graph.get("status") or "No graph data returned."
        return f"{status}\nMatched entities: {entities}"

    def _rank_graph_paths(
        self,
        query: str,
        edges: list[dict[str, Any]],
        retrievals: list[RetrievalResult],
    ) -> list[dict[str, Any]]:
        """Rank graph edges by relevance to the query using vector similarity."""
        candidates = []
        retrieval_similarity = max((result.similarity for result in retrievals), default=0.0)
        for edge in edges:
            source = str(edge.get("source") or "")
            target = str(edge.get("target") or "")
            relation = str(edge.get("relation") or "")
            if not target:
                continue
            content = f"{source} {relation} {target}"
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "relation": relation,
                    "edge_type": edge.get("edge_type") or relation,
                    "path": f"{source} -> {relation} -> {target}",
                    "content": content,
                    "retrieval_similarity": retrieval_similarity,
                    "final_score": 1.0,
                }
            )
        if not candidates:
            return []

        # Rank by vector similarity to query
        try:
            ranked_by_vector = self.retriever.rank_texts(
                query,
                candidates,
                text_key="content",
                limit=max(len(candidates), settings.graphrag_graph_chunk_limit),
            )
            for item in ranked_by_vector:
                vector_score = float(item.get("similarity") or item.get("retrieval_similarity") or 0.0)
                item["vector_similarity_score"] = round(vector_score, 4)
                item["final_score"] = round(vector_score, 4)
            ranked_by_vector.sort(key=lambda item: item["final_score"], reverse=True)
            return ranked_by_vector[:settings.graphrag_graph_chunk_limit]
        except Exception:
            return candidates[:settings.graphrag_graph_chunk_limit]

    @staticmethod
    def _short_label(value: str) -> str:
        """Shorten a graph node label for display."""
        clean = " ".join(value.split())
        return PipelineService._compress(clean, 96)

    @staticmethod
    def _is_simple_query(query: str) -> bool:
        lowered = query.lower().strip()
        multi_hop_terms = {
            "compare",
            "complex",
            "multiple",
            "history",
            "risk",
            "relationship",
            "correlation",
            "trend",
            "across",
            "versus",
        }
        if any(term in lowered for term in multi_hop_terms):
            return False
        if lowered.count("?") > 1:
            return False
        if len([token for token in lowered.split() if token]) > 12:
            return False
        simple_starts = ("what is", "define", "who is", "when is", "how much")
        return lowered.startswith(simple_starts)

    @staticmethod
    def _financial_system_prompt(grounding_instruction: str = "") -> str:
        return (
            "You are FinGraph AI, an expert SEC filing and corporate finance analyst. "
            "Answer questions using SEC filings including 10-K, 10-Q, 8-K and DEF 14A documents. "
            "Use only the retrieved context. Report exact financial values, dates, executives, business risks, filings and corporate events whenever available."
        )

    @staticmethod
    def _clean_query(query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        return query

    @staticmethod
    def _clean_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @staticmethod
    def _compress(text: str, limit: int = 260) -> str:
        clean = " ".join(text.split())
        if len(clean) <= limit:
            return clean
        return clean[:limit].rsplit(" ", 1)[0] + "..."

    @staticmethod
    def _retrieval_to_dict(result: RetrievalResult) -> dict[str, Any]:
        return {
            "chunk_id": result.chunk_id,
            "source": result.source,
            "category": result.category,
            "content": result.content,
            "preview": PipelineService._compress(result.content, 260),
            "similarity": result.similarity,
            "entities": result.entities,
        }

    @staticmethod
    def _chunk_to_dict(chunk: Any) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "category": chunk.category,
            "content": chunk.content,
            "preview": PipelineService._compress(chunk.content, 260),
            "similarity": 1.0,
            "entities": chunk.entities,
        }

    def _rank_graph_chunks(self, query: str, tg_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = [self._normalize_graph_chunk(chunk) for chunk in tg_chunks]
        return self.retriever.rank_texts(
            query,
            normalized,
            limit=settings.graphrag_graph_chunk_limit,
        )

    @staticmethod
    def _normalize_graph_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
        attrs = chunk.get("attributes") or {}
        content = attrs.get("content") or chunk.get("content") or attrs.get("text") or ""
        chunk_id = attrs.get("chunk_id") or chunk.get("chunk_id") or chunk.get("v_id") or "graph_chunk"
        source = attrs.get("source") or chunk.get("source") or "TigerGraph"
        category = attrs.get("category") or chunk.get("category") or "graph"
        return {
            **chunk,
            "attributes": attrs,
            "chunk_id": str(chunk_id),
            "source": str(source),
            "category": str(category),
            "content": str(content),
        }

    @staticmethod
    def _graph_chunk_to_dict(chunk: dict[str, Any]) -> dict[str, Any]:
        content = str(chunk.get("content") or "")
        return {
            "chunk_id": chunk.get("chunk_id") or "graph_chunk",
            "source": chunk.get("source") or "TigerGraph",
            "category": chunk.get("category") or "graph",
            "content": content,
            "preview": PipelineService._compress(content, 260),
            "similarity": chunk.get("similarity"),
            "entities": [],
        }

    @staticmethod
    def _empty_graph() -> dict[str, Any]:
        return {
            "nodes": [],
            "edges": [],
            "paths": [],
            "neighbors": {},
            "entities": [],
            "extracted_entities": [],
            "backend": "tigergraph",
            "status": "No TigerGraph traversal was completed.",
            "errors": [],
        }
