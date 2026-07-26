FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files explicitly first
COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . /app

EXPOSE 8000
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
