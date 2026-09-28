# Multi-stage Minimal Dockerfile for VeriShield AI (Phase 13)
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final Lean Production Image
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    PORT=8000

# Copy installed wheels from builder
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY . .

# Expose FastAPI Web Dashboard port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start Uvicorn Server
CMD ["python", "run.py"]
