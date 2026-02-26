# websearch

Self-hosted web search and page content extraction API for LLM tool-use.

SearXNG meta-search engine + FastAPI retriever with trafilatura text extraction and optional Playwright JS rendering.

## Quick Start

```bash
git clone https://github.com/hbanaharis/websearch.git
cd websearch
docker-compose up -d
```

- **Search API:** http://localhost:8089
- **SearXNG UI:** http://localhost:8088

## Endpoints

### POST /search

Search the web and optionally retrieve full page content from top results.

```bash
curl -X POST http://localhost:8089/search \
  -H "Content-Type: application/json" \
  -d '{"query": "MTHFR folate metabolism", "num_results": 5, "retrieve_top": 2}'
```

### POST /retrieve

Fetch and extract content from a single URL.

```bash
curl -X POST http://localhost:8089/retrieve \
  -H "Content-Type: application/json" \
  -d '{"url": "https://pubmed.ncbi.nlm.nih.gov/35235964/"}'
```

### GET /healthz

Health check.

## Full API Documentation

See [API.md](API.md) for complete endpoint reference, response schemas, LLM tool definitions, and usage examples.

## Stack

- **SearXNG** — Meta-search aggregating Google, Bing, DuckDuckGo, Brave, PubMed, Google Scholar
- **FastAPI** — Async Python API server
- **trafilatura** — Best-in-class web content extraction
- **Playwright** — Headless Chromium for JS-rendered pages
- **selectolax** — Fast HTML parsing for links and images
