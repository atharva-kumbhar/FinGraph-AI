FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY backend/ /app/backend/
COPY data/ /app/data/
COPY frontend/ /app/frontend/

ENV PYTHONPATH=/app/backend
ENV FINGRAPH_HOST=0.0.0.0
ENV FINGRAPH_DATASET_PATH=/app/data/sp100/chunks.jsonl
ENV FINGRAPH_INDEX_DIR=/app/data/faiss_index

WORKDIR /app/backend
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
