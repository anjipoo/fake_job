# =====================================================
# Dockerfile — Fake Job Detection System
# Multi-stage build for small production image
# =====================================================

FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── Install Python deps ─────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ─── Copy project ────────────────────────────────
COPY . .

# Create directories
RUN mkdir -p models datasets uploads reports explainability

# ─── Expose ports ────────────────────────────────
EXPOSE 8000
EXPOSE 8501

# ─── Default command: start both services ────────
# In production use separate containers or a process manager
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0"]
