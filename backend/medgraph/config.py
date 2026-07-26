"""Runtime configuration for MedGraph AI.

The Colab notebook contains live service credentials. This app deliberately
keeps those out of source control and reads integrations from environment
variables instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
ROOT_DATASET_PATH = ROOT_DIR / "data" / "sp100" / "chunks.jsonl"
DATA_DIR_DATASET_PATH = ROOT_DIR / "data" / "sp100" / "chunks.jsonl"


def _load_dotenv() -> None:
    """Load repo-local .env values without requiring python-dotenv."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


def _default_dataset_path() -> Path:
    configured = os.getenv("MEDGRAPH_DATASET_PATH")
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return configured_path
        for base in (ROOT_DIR, ROOT_DIR / "data"):
            cand = base / configured
            if cand.exists():
                return cand
    for base in (ROOT_DIR / "data" / "sp100", ROOT_DIR / "sp100", ROOT_DIR / "data", ROOT_DIR):
        for name in ("chunks.jsonl", "final_extracted.jsonl"):
            candidate = base / name
            if candidate.exists():
                return candidate
    return ROOT_DATASET_PATH


@dataclass(frozen=True)
class Settings:
    app_name: str = "FinGraph AI"
    host: str = os.getenv("MEDGRAPH_HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", os.getenv("MEDGRAPH_PORT", "8000")))
    full_dataset_chunks: int = int(os.getenv("MEDGRAPH_TOTAL_CHUNKS", "50487"))
    full_dataset_tokens: int = int(os.getenv("MEDGRAPH_TOTAL_TOKENS", "15000000"))
    dataset_path: Path = _default_dataset_path()
    frontend_dir: Path = ROOT_DIR / "frontend"
    benchmark_path: Path = ROOT_DIR / "data" / "ground_truth_questions.json"
    triplets_dir: Path = ROOT_DIR / "data" / "triplets"
    price_per_1k_tokens_usd: float = float(
        os.getenv("MEDGRAPH_PRICE_PER_1K_TOKENS", "0.0008")
    )
    request_timeout_seconds: int = max(25, int(os.getenv("MEDGRAPH_REQUEST_TIMEOUT_SECONDS", "8")))
    nvidia_request_timeout_seconds: int = int(
        os.getenv("MEDGRAPH_NVIDIA_TIMEOUT_SECONDS", "120")
    )
    retrieval_top_k: int = int(os.getenv("MEDGRAPH_RETRIEVAL_TOP_K", "50"))
    rag_retrieval_top_k: int = int(os.getenv("MEDGRAPH_RAG_RETRIEVAL_TOP_K", "25"))
    graphrag_retrieval_top_k: int = int(
        os.getenv("MEDGRAPH_GRAPH_RETRIEVAL_TOP_K", "10")
    )
    graphrag_graph_chunk_limit: int = int(
        os.getenv("MEDGRAPH_GRAPH_CHUNK_LIMIT", "2")
    )
    embedding_model: str = os.getenv(
        "MEDGRAPH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    retrieval_backend: str = os.getenv("MEDGRAPH_RETRIEVAL_BACKEND", "keyword").lower()
    embedding_batch_size: int = int(os.getenv("MEDGRAPH_EMBEDDING_BATCH_SIZE", "64"))
    index_dir: Path = Path(
        os.getenv("MEDGRAPH_INDEX_DIR", str(ROOT_DIR / "data" / "faiss_index"))
    )

    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = os.getenv(
        "NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"
    )
    llm_model: str = os.getenv("MEDGRAPH_LLM_MODEL", "meta/llama-3.3-70b-instruct")
    production_provider: str = os.getenv("MEDGRAPH_PRODUCTION_PROVIDER", "auto").lower()
    llm_only_provider: str = os.getenv("MEDGRAPH_LLM_ONLY_PROVIDER", "nvidia").lower()
    rag_provider: str = os.getenv("MEDGRAPH_RAG_PROVIDER", "nvidia").lower()
    graphrag_provider: str = os.getenv("MEDGRAPH_GRAPHRAG_PROVIDER", "nvidia").lower()
    test_provider: str = os.getenv("MEDGRAPH_TEST_PROVIDER", "auto").lower()
    generation_provider: str = os.getenv(
        "MEDGRAPH_GENERATION_PROVIDER", production_provider
    ).lower()
    expected_answer_provider: str = os.getenv(
        "MEDGRAPH_EXPECTED_ANSWER_PROVIDER", "gemini"
    ).lower()

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_base_url: str = os.getenv(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
    )
    gemini_model: str = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_judge_model: str = os.getenv("HF_JUDGE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    enable_llm_judge: bool = os.getenv("MEDGRAPH_ENABLE_LLM_JUDGE", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    enable_bertscore: bool = os.getenv("MEDGRAPH_ENABLE_BERTSCORE", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    judge_provider: str = os.getenv("MEDGRAPH_JUDGE_PROVIDER", "gemini").lower()

    tg_host: str = os.getenv("TG_HOST", "")
    tg_graph_name: str = os.getenv("TG_GRAPH_NAME", "database1")
    tg_username: str = os.getenv("TG_USERNAME", "")
    tg_password: str = os.getenv("TG_PASSWORD", "")
    tg_secret: str = os.getenv("TG_SECRET", "")
    tg_api_token: str = os.getenv("TG_API_TOKEN", "")
    tg_vertex_type: str = os.getenv("TG_VERTEX_TYPE", "Company")
    tg_edge_type: str = os.getenv("TG_EDGE_TYPE", "FILED_BY")
    tg_relationship_edge_type: str = os.getenv("TG_RELATIONSHIP_EDGE_TYPE", "FILED_BY")
    tg_mentioned_edge_type: str = os.getenv("TG_MENTIONED_EDGE_TYPE", "MENTIONS_EXECUTIVE")
    tg_chunk_vertex_type: str = os.getenv("TG_CHUNK_VERTEX_TYPE", "Document")
    tg_chunk_entity_limit: int = int(os.getenv("TG_CHUNK_ENTITY_LIMIT", "2"))
    tg_chunk_fetch_limit: int = int(os.getenv("TG_CHUNK_FETCH_LIMIT", "3"))
    tg_verify_ssl: bool = os.getenv("TG_VERIFY_SSL", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    tg_uploaded_vertices: int = int(os.getenv("TG_UPLOADED_VERTICES", "4872"))
    tg_uploaded_edges: int = int(os.getenv("TG_UPLOADED_EDGES", "5678"))
    tg_vertex_limit: int = int(os.getenv("TG_VERTEX_LIMIT", "500"))
    tg_max_edges: int = int(os.getenv("TG_MAX_EDGES", "12"))


settings = Settings()
