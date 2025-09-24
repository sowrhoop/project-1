FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 10001 -s /usr/sbin/nologin appuser

COPY index.html ./

USER appuser
ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/" || exit 1

CMD ["/bin/sh", "-lc", "python -m http.server ${PORT:-8080} --directory /app --bind 0.0.0.0"]
