FROM node:20-alpine

WORKDIR /app

RUN apk add --no-cache curl

RUN addgroup -g 10001 app \
 && adduser -D -u 10001 -G app appuser

COPY package.json package-lock.json* ./
RUN npm ci || npm install --no-audit --no-fund

COPY . .

USER appuser
ENV PORT=8080
EXPOSE 8080

# Healthcheck for consistency with project-1
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/healthz" || exit 1

CMD ["node", "server.js"]
