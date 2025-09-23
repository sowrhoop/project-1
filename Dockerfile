FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

USER appuser
ENV PORT=8080
EXPOSE 8080

CMD ["/bin/sh", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]

