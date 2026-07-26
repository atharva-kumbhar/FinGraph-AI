"""Singleton access to the real MedGraph pipeline orchestrator."""

from __future__ import annotations

from medgraph.pipelines import PipelineService

_service: PipelineService | None = None


def get_pipeline_service() -> PipelineService:
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
