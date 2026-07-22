FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    CULTURAL_MOOD_TRACKER_DB_PATH=/app/runtime/cultural_mood_tracker.sqlite3 \
    CULTURAL_MOOD_TRACKER_DLT_DB_PATH=/app/runtime/cultural_mood_tracker_pipeline.duckdb \
    CULTURAL_MOOD_TRACKER_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 culturaltracker \
    && mkdir -p /app/runtime /app/artifacts \
    && chown -R culturaltracker:culturaltracker /app

USER culturaltracker
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["sh", "-c", "test -f /app/runtime/cultural_mood_tracker.sqlite3 || python scripts/ingest.py --source demo; streamlit run Cultural_Mood_Tracker.py --server.address=0.0.0.0 --server.port=8501"]
