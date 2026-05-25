#!/usr/bin/env bash
# =====================================================
# run.sh — Quick launcher for the Fake Job Detector
# Usage: bash run.sh [train|api|ui|all]
# =====================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=== Fake Job Detection System ==="
echo ""

case "${1:-all}" in

  # ── Prepare data only ──────────────────────────
  data)
    echo "[1/1] Preparing dataset …"
    python dataset/prepare_data.py
    echo "Done."
    ;;

  # ── Train model ────────────────────────────────
  train)
    echo "[1/2] Preparing dataset …"
    python dataset/prepare_data.py
    echo "[2/2] Training DistilBERT …"
    python models/train_model.py
    echo "Training complete. Model saved to models/distilbert_fake_job/"
    ;;

  # ── Start FastAPI only ─────────────────────────
  api)
    echo "Starting FastAPI on http://0.0.0.0:8000 …"
    echo "Docs at: http://localhost:8000/docs"
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    ;;

  # ── Start Streamlit only ───────────────────────
  ui)
    echo "Starting Streamlit on http://localhost:8501 …"
    streamlit run frontend/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --browser.gatherUsageStats false
    ;;

  # ── Run everything ─────────────────────────────
  all)
    echo "Starting FastAPI (background) …"
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    API_PID=$!

    echo "Starting Streamlit (foreground) …"
    streamlit run frontend/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0 \
      --browser.gatherUsageStats false

    # Cleanup on exit
    kill $API_PID 2>/dev/null || true
    ;;

  *)
    echo "Usage: bash run.sh [data|train|api|ui|all]"
    exit 1
    ;;
esac
