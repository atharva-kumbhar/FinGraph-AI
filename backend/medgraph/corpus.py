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
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}."
            )
        suffix = self.dataset_path.suffix.lower()
        if suffix == ".jsonl":
            return self._load_jsonl(self.dataset_path)
        elif suffix == ".json":
            return self._load_json(self.dataset_path)
        else:
            return self._load_csv(self.dataset_path)

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
