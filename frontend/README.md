# MedGraph AI Frontend

Static dashboard for comparing LLM-only, FAISS RAG, and TigerGraph GraphRAG pipelines.

## Local (served by backend)

```powershell
cd ..\backend
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Vercel

1. Set project root to `frontend/`.
2. Add environment variable `MEDGRAPH_API_BASE` pointing to your Render backend URL (no trailing slash).
3. Inject at build time by adding to `index.html` before `config.js`:

```html
<script>window.MEDGRAPH_API_BASE = "https://your-render-service.onrender.com";</script>
```

Or edit `config.js` directly for a static deploy.

## Hugging Face Spaces (static)

Create a static Space with `frontend/` as the app directory and set `MEDGRAPH_API_BASE` in the Space README custom HTML or `config.js`.
