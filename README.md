# websearch

Self-hosted web search and page content extraction API for LLM tool-use.

SearXNG meta-search engine + FastAPI retriever with semantic re-ranking, PDF extraction, domain-specific metadata, disk caching, and Bearer token authentication.

## Quick Start

```bash
git clone https://github.com/hbanaharis/websearch.git
cd websearch
docker-compose up -d
```

- **Search API:** http://localhost:8089
- **SearXNG UI:** http://localhost:8088

## Authentication

Set `API_KEYS` in docker-compose.yml to enable Bearer token auth. If empty, auth is disabled.

```bash
curl -X POST http://localhost:8089/search \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "MTHFR folate metabolism"}'
```

## Endpoints

### POST /search

Search the web with semantic re-ranking and optionally retrieve full page content from top results.

```bash
curl -X POST http://localhost:8089/search \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "MTHFR folate metabolism", "num_results": 5, "retrieve_top": 2}'
```

### POST /retrieve

Fetch and extract content from a single URL (HTML or PDF).

```bash
curl -X POST http://localhost:8089/retrieve \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://pubmed.ncbi.nlm.nih.gov/35235964/"}'
```

### POST /search/batch

Run up to 10 search queries concurrently.

```bash
curl -X POST http://localhost:8089/search/batch \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["MTHFR", "APOE4", "CYP2D6"], "retrieve_top": 1}'
```

### GET /jobs/{job_id}

Check async job status (when using `webhook_url`).

### GET /healthz

Health check (no auth required).

## Features

- **Disk cache** — search results (1h TTL) and page content (24h TTL) cached to disk
- **Semantic re-ranking** — hits re-ordered by PubMedBERT cosine similarity
- **Smart text chunking** — selects most relevant paragraphs for LLM context budget
- **PDF extraction** — automatic detection and text extraction via pypdf/pymupdf
- **Domain extractors** — structured metadata for PubMed, PMC, Wikipedia, arXiv
- **Citation extraction** — DOIs, PMIDs, and references parsed from HTML
- **Quality scoring** — 0.0-1.0 composite score (length, sentences, boilerplate, paywall)
- **Content dedup** — SimHash fingerprinting detects near-duplicate pages
- **Rate limiting** — per-domain (2) and global (10) concurrent fetch limits
- **Async webhooks** — return 202 + POST results to your endpoint
- **Batch search** — up to 10 concurrent queries in one call
- **Image OCR** — Tesseract OCR on downloaded images

## Full API Documentation

See [API.md](API.md) for complete endpoint reference, response schemas, LLM tool definitions, and usage examples.

## Stack

- **SearXNG** — Meta-search aggregating Google, Bing, DuckDuckGo, Brave, PubMed, Google Scholar
- **FastAPI** — Async Python API server
- **trafilatura** — Best-in-class web content extraction
- **Playwright** — Headless Chromium for JS-rendered pages
- **selectolax** — Fast HTML parsing for links and images
- **pypdf + pymupdf** — PDF text extraction
- **PubMedBERT** — Semantic embeddings via embed-proxy
- **Tesseract** — Image OCR
