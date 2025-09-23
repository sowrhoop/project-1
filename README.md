# Project 1 (FastAPI)

Simple FastAPI service. Default port 8080.

## Build & Run

```sh
# Build
docker build -t project-1:dev .

# Run
docker run --rm -p 8080:8080 project-1:dev

# Test
curl http://localhost:8080/
```

## Container Image (GHCR)

This repo ships with a GitHub Action that builds and pushes the image to GHCR:
- `ghcr.io/<owner>/project-1:latest`
- `ghcr.io/<owner>/project-1:<sha>`

