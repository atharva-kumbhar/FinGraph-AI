"""Production API routes for MedGraph AI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from medgraph.config import settings
from services.pipeline_service import get_pipeline_service


router = APIRouter(prefix="/api")
service = get_pipeline_service()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Medical question to analyze")
    reference_answer: str | None = Field(
        default=None,
        description=(
            "Optional ground-truth override. When omitted, /api/query auto-generates "
            "expected_answer with Gemini."
        ),
    )


class BenchmarkItem(BaseModel):
    question: str
    correct_answer: str | None = None


class BenchmarkRequest(BaseModel):
    items: list[BenchmarkItem]


def _query_payload(body: QueryRequest) -> dict[str, Any]:
    return {"query": body.query.strip(), "reference_answer": body.reference_answer}


@router.get("/health")
def health() -> dict[str, Any]:
    from medgraph.config import settings

    return {"ok": True, "app": settings.app_name}


@router.get("/stats")
def stats() -> dict[str, Any]:
    return service.stats()


@router.get("/schema")
def schema() -> dict[str, Any]:
    return {
        "vertices": ["Company", "Document", "Executive", "Risk", "Event", "FinancialMetric"],
        "edges": [
            "FILED_BY",
            "MENTIONS_EXECUTIVE",
            "DISCLOSES_RISK",
            "MENTIONS_EVENT",
            "REPORTS_METRIC",
        ],
        "patterns": [
            "Company <-[FILED_BY]- Document",
            "Document -[MENTIONS_EXECUTIVE]-> Executive",
            "Document -[DISCLOSES_RISK]-> Risk",
            "Document -[MENTIONS_EVENT]-> Event",
            "Document -[REPORTS_METRIC]-> FinancialMetric",
        ],
        "installed_queries": ["getCompanyContext"],
    }


@router.post("/llm")
@router.post("/llm-only")
def llm_only(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_llm_only_api(body.query, body.reference_answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/rag")
@router.post("/basic-rag")
def basic_rag(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_basic_rag_api(body.query, body.reference_answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/graphrag")
def graphrag(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_graphrag_api(body.query, body.reference_answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/query")
def query(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_query(body.query, body.reference_answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stop")
def stop_generation() -> dict[str, Any]:
    try:
        service.stop_generation()
        return {"status": "cancelled"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pipeline/{pipeline_name}")
def pipeline(pipeline_name: str, body: QueryRequest) -> dict[str, Any]:
    """Run one dashboard pipeline and return the full card payload."""
    try:
        name = pipeline_name.strip().lower().replace("-", "_")
        if name in {"llm", "llm_only"}:
            result = service.run_llm_only_api(body.query, body.reference_answer)
        elif name in {"rag", "basic_rag"}:
            result = service.run_basic_rag_api(body.query, body.reference_answer)
        elif name in {"graph", "graphrag"}:
            result = service.run_graphrag_api(body.query, body.reference_answer)
        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        return result["pipeline"]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/metrics")
def metrics(body: QueryRequest) -> dict[str, Any]:
    """Run all three pipelines and return comparison metrics for the dashboard."""
    try:
        return service.run_metrics_api(body.query, body.reference_answer)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/benchmark")
def benchmark(body: BenchmarkRequest) -> dict[str, Any]:
    """Batch evaluation using each production pipeline and Gemini as judge."""
    try:
        items = [item.model_dump() for item in body.items]
        return service.run_benchmark(items)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/test/query")
def test_query(body: QueryRequest) -> dict[str, Any]:
    """Run all three pipelines with the configured test provider for side-by-side testing."""
    try:
        return service.run_query(
            body.query,
            body.reference_answer,
            llm_provider=settings.test_provider,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/test/llm")
def test_llm(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_llm_only_api(body.query, body.reference_answer, llm_provider=settings.test_provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/test/rag")
def test_rag(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_basic_rag_api(body.query, body.reference_answer, llm_provider=settings.test_provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/test/graphrag")
def test_graphrag(body: QueryRequest) -> dict[str, Any]:
    try:
        return service.run_graphrag_api(body.query, body.reference_answer, llm_provider=settings.test_provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
