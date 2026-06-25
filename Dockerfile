FROM python:3.12-slim

# psycopg2-binary ships wheels, so no build toolchain needed.
WORKDIR /code
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements-postgres.txt
COPY . .

EXPOSE 8000
# Production serving: gunicorn supervising uvicorn workers (multi-core, crash-resilient).
# Worker count, timeouts, etc. come from gunicorn_conf.py (env-overridable; set WEB_CONCURRENCY).
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
