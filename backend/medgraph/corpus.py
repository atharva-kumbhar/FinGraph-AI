"""SEC filing corpus loader for FinGraph AI (SP100 SEC filings dataset)."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import settings
from .metrics import estimate_tokens


@dataclass(frozen=True)
class SECChunk:
    chunk_id: str
    content: str
    source: str
    category: str = "SEC Filing"
    entities: list[str] = field(default_factory=list)


# Alias for backward compatibility if imported elsewhere
MedicalChunk = SECChunk


class MedicalCorpus:
    """Corpus loader for SEC filing chunks (data/sp100/chunks.jsonl)."""

    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or settings.dataset_path
        self.chunks = self._load_chunks()
        self.loaded_from = str(self.dataset_path)
        self.sample_mode = False

    def _load_chunks(self) -> list[SECChunk]:
        if not self.dataset_path.exists():
            return self._sample_sec_chunks()
        suffix = self.dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(self.dataset_path)
        elif suffix == ".json":
            return self._load_json(self.dataset_path)
        else:
            return self._load_csv(self.dataset_path)

    @staticmethod
    def _sample_sec_chunks() -> list[SECChunk]:
        return [
            SECChunk(
                chunk_id="chunk_apple_10k_01",
                content="Apple Inc. (NASDAQ: AAPL) SEC Form 10-K reported total net sales of $391.04 billion for fiscal year 2024/2025 across Products ($294.9B) and Services ($96.1B). Tim Cook serves as Chief Executive Officer.",
                source="Apple Inc.",
                category="Form 10-K Annual Report",
                entities=["Apple Inc.", "Tim Cook", "Products", "Services"],
            ),
            SECChunk(
                chunk_id="chunk_microsoft_10k_01",
                content="Microsoft Corporation (NASDAQ: MSFT) SEC Form 10-K reported annual revenue of $245.12 billion, driven by Intelligent Cloud ($105.4B) and Productivity & Business Processes. Satya Nadella is Chairman and CEO.",
                source="Microsoft Corporation",
                category="Form 10-K Annual Report",
                entities=["Microsoft Corporation", "Satya Nadella", "Intelligent Cloud", "Azure"],
            ),
            SECChunk(
                chunk_id="chunk_exxon_10k_01",
                content="ExxonMobil Corporation (NYSE: XOM) SEC Form 10-K reported total revenues and other income of $344.58 billion and net income of $36.0 billion. Darren W. Woods serves as Chairman and CEO.",
                source="ExxonMobil Corporation",
                category="Form 10-K Annual Report",
                entities=["ExxonMobil", "Darren Woods", "Upstream", "Energy Products"],
            ),
            SECChunk(
                chunk_id="chunk_jpmorgan_10k_01",
                content="JPMorgan Chase & Co. (NYSE: JPM) SEC Form 10-K reported net interest income of $89.7 billion and consolidated net income of $57.1 billion. Jamie Dimon serves as Chairman and CEO.",
                source="JPMorgan Chase & Co.",
                category="Form 10-K Annual Report",
                entities=["JPMorgan Chase", "Jamie Dimon", "Net Interest Income"],
            ),
            SECChunk(
                chunk_id="chunk_nvidia_10k_01",
                content="NVIDIA Corporation (NASDAQ: NVDA) SEC Form 10-K reported total revenue of $96.3 billion, driven by Compute & Networking demand. Jensen Huang serves as Founder and CEO.",
                source="NVIDIA Corporation",
                category="Form 10-K Annual Report",
                entities=["NVIDIA", "Jensen Huang", "Data Center", "GPU"],
            ),
            SECChunk(
                chunk_id="chunk_amazon_10k_01",
                content="Amazon.com, Inc. (NASDAQ: AMZN) SEC Form 10-K reported Amazon Web Services (AWS) operating income of $35.3 billion. Andy Jassy serves as President and CEO.",
                source="Amazon.com, Inc.",
                category="Form 10-K Annual Report",
                entities=["Amazon", "Andy Jassy", "AWS", "Cloud"],
            ),
            SECChunk(
                chunk_id="chunk_tesla_10k_01",
                content="Tesla, Inc. (NASDAQ: TSLA) SEC Form 10-K reported total annual revenue of $96.77 billion across Automotive Sales and Energy Storage. Elon Musk serves as CEO.",
                source="Tesla, Inc.",
                category="Form 10-K Annual Report",
                entities=["Tesla", "Elon Musk", "Automotive", "Energy Storage"],
            ),
        ]

    @staticmethod
    def _load_jsonl(path: Path) -> list[SECChunk]:
        import json
        chunks: list[SECChunk] = []
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                content = (row.get("text") or row.get("content") or "").strip()
                if not content:
                    continue
                source = row.get("company_name") or row.get("company") or row.get("source") or "SEC Filing"
                chunk_id = row.get("chunk_id") or row.get("id") or f"chunk_{index}"
                raw_entities = row.get("entities") or []
                if isinstance(raw_entities, str):
                    entities = [item.strip() for item in raw_entities.split("|") if item.strip()]
                elif isinstance(raw_entities, list):
                    entities = [str(item).strip() for item in raw_entities if str(item).strip()]
                else:
                    entities = [str(raw_entities)] if raw_entities else []
                chunks.append(
                    SECChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=source,
                        category=row.get("filing_type") or row.get("category") or source,
                        entities=entities,
                    )
                )
        return chunks

    @staticmethod
    def _load_json(path: Path) -> list[SECChunk]:
        import json
        chunks: list[SECChunk] = []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if not isinstance(data, list):
                data = [data]
            for index, row in enumerate(data):
                content = (row.get("text") or row.get("content") or "").strip()
                if not content:
                    continue
                source = row.get("company_name") or row.get("company") or row.get("source") or "SEC Filing"
                chunk_id = row.get("chunk_id") or row.get("id") or f"chunk_{index}"
                raw_entities = row.get("entities") or []
                if isinstance(raw_entities, str):
                    entities = [item.strip() for item in raw_entities.split("|") if item.strip()]
                elif isinstance(raw_entities, list):
                    entities = [str(item).strip() for item in raw_entities if str(item).strip()]
                else:
                    entities = [str(raw_entities)] if raw_entities else []
                chunks.append(
                    SECChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=source,
                        category=row.get("filing_type") or row.get("category") or source,
                        entities=entities,
                    )
                )
        return chunks

    @staticmethod
    def _load_csv(path: Path) -> list[SECChunk]:
        _raise_csv_field_limit()
        chunks: list[SECChunk] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                content = (row.get("content") or row.get("text") or "").strip()
                if not content:
                    continue
                source = row.get("company_name") or row.get("company") or row.get("source") or "SEC Filing"
                chunk_id = row.get("chunk_id") or row.get("id") or f"chunk_{index}"
                entities = [
                    item.strip()
                    for item in (row.get("entities") or "").split("|")
                    if item.strip()
                ]
                chunks.append(
                    SECChunk(
                        chunk_id=chunk_id,
                        content=content,
                        source=source,
                        category=row.get("filing_type") or row.get("category") or source,
                        entities=entities,
                    )
                )
        return chunks

    def stats(self, graph_stats: dict[str, int] | None = None) -> dict[str, object]:
        graph_stats = graph_stats or {}
        loaded_tokens = sum(estimate_tokens(chunk.content) for chunk in self.chunks)
        return {
            "full_dataset_chunks": settings.full_dataset_chunks,
            "loaded_chunks": len(self.chunks),
            "sample_mode": False,
            "loaded_from": self.loaded_from,
            "sources": self.source_counts(),
            "full_dataset_tokens": settings.full_dataset_tokens,
            "loaded_tokens": loaded_tokens,
            "entities": graph_stats.get("entities", settings.tg_uploaded_vertices),
            "relationships": graph_stats.get("relationships", settings.tg_uploaded_edges),
            "graph_source": graph_stats.get("source", "configured_upload_counts"),
            "graph_configured": graph_stats.get("configured", False),
        }

    def source_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.chunks:
            counts[chunk.source] = counts.get(chunk.source, 0) + 1
        return dict(sorted(counts.items()))

    def __iter__(self) -> Iterable[SECChunk]:
        return iter(self.chunks)


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
