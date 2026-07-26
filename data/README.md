# MedGraph AI Data

| Asset | Path | Description |
|-------|------|-------------|
| RAG corpus | `../final_medical_rag_dataset.csv` | 13,807 medical chunks (`chunk_id`, `content`, `source`) |
| FAISS cache | `faiss_index/` | Built on first RAG/GraphRAG query |
| Ground truth | `ground_truth_questions.json` | Optional benchmark Q&A |
| Triplets (optional) | `triplets/` | `drug_triplets.csv`, `research_triplets.csv`, `clean_acr_partial.csv` for graph build |

Place triplet CSVs under `data/triplets/` when rebuilding the TigerGraph graph from the Colab notebook in `notebooks/`.
