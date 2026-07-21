FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MOODLENS_DB_PATH=/app/runtime/moodlens.sqlite3 \
    MOODLENS_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 moodlens \
    && mkdir -p /app/runtime /app/artifacts \
    && chown -R moodlens:moodlens /app

USER moodlens
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["sh", "-c", "test -f /app/runtime/moodlens.sqlite3 || python scripts/ingest.py --source demo; streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
