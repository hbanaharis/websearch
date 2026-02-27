# Web Search & Retriever API v0.2.0

Self-hosted web search and page content extraction for LLM tool-use. Powered by SearXNG (meta-search) and a FastAPI retriever with trafilatura text extraction, semantic re-ranking, PDF extraction, domain-specific metadata, and disk caching.

## Network Access

| Route | URL |
|-------|-----|
| LAN (from GPU server) | `http://192.168.15.26:8089` |
| Docker internal | `http://websearch_retriever:8080` |
| SearXNG UI (debug) | `http://192.168.15.26:8088` |

## Authentication

All endpoints except `/healthz`, `/docs`, `/openapi.json`, and `/redoc` require a Bearer token.

**Header:** `Authorization: Bearer <API_KEY>`

Authentication is controlled by the `API_KEYS` environment variable (comma-separated list of valid keys). If `API_KEYS` is empty or unset, authentication is disabled.

| Status | Meaning |
|--------|---------|
| 401 | Missing or malformed Authorization header |
| 403 | Invalid API key |

```bash
# Authenticated request
curl -s -X POST http://192.168.15.26:8089/search \
  -H "Authorization: Bearer fe5e611508dc9e000e96e94d9e5fa9a246df874bd6076b60" \
  -H "Content-Type: application/json" \
  -d '{"query": "MTHFR"}'

# Health check (no auth required)
curl -s http://192.168.15.26:8089/healthz
```

---

## Endpoints

### POST /search

Search the web and optionally retrieve + extract full page content from the top results. Results are semantically re-ranked by default.

**Request:**

```json
{
  "query": "MTHFR C677T folate metabolism",
  "num_results": 8,
  "retrieve_top": 3,
  "language": "en",
  "safesearch": 1,
  "time_range": null,
  "site": null,
  "render_js": false,
  "download_images": false,
  "allow_domains": null,
  "no_cache": false,
  "max_chars": 4000,
  "full_text": false,
  "rerank": true,
  "min_quality": 0.0,
  "webhook_url": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | *required* | Search query |
| `num_results` | int | 8 | Number of search hits to return (1-20) |
| `retrieve_top` | int | 3 | How many top hits to fetch + extract full text (0-10). Set to 0 for search hits only. |
| `language` | string | `"en"` | Search language code |
| `safesearch` | int | 1 | Safe search level: 0=off, 1=moderate, 2=strict |
| `time_range` | string | null | Restrict results by age: `"day"`, `"week"`, `"month"`, `"year"` |
| `site` | string | null | Restrict to a domain, e.g. `"nih.gov"`, `"pubmed.ncbi.nlm.nih.gov"` |
| `render_js` | bool | false | Use headless Chromium for retrieved pages (slower, needed for JS-heavy sites) |
| `download_images` | bool | false | Download images from retrieved pages to `/data/media` |
| `allow_domains` | list | null | If set, only retrieve pages from these domains |
| `no_cache` | bool | false | Bypass disk cache — force fresh search and retrieval |
| `max_chars` | int | 4000 | Maximum characters of extracted text per page (100-100000). Semantic chunking selects the most relevant paragraphs. |
| `full_text` | bool | false | Return full extracted text, ignoring `max_chars` |
| `rerank` | bool | true | Re-rank search hits by semantic similarity to query (via embed-proxy). Falls back to original order if embed-proxy is unavailable. |
| `min_quality` | float | 0.0 | Minimum quality score (0.0-1.0) for retrieved pages. Pages below threshold are excluded. |
| `webhook_url` | string | null | If set, returns 202 immediately and POSTs the result to this URL when done. |

**Response:**

```json
{
  "query": "MTHFR C677T folate metabolism",
  "hits": [
    {
      "title": "MTHFR Gene Variant and Folic Acid Facts | CDC",
      "url": "https://www.cdc.gov/folic-acid/data-research/mthfr/index.html",
      "snippet": "People with an MTHFR gene variant can process all types of folate...",
      "engine": "bing"
    }
  ],
  "retrieved": [
    {
      "requested_url": "https://www.cdc.gov/folic-acid/...",
      "final_url": "https://www.cdc.gov/folic-acid/...",
      "canonical_url": "https://www.cdc.gov/folic-acid/...",
      "title": "MTHFR Gene Variant and Folic Acid Facts | CDC",
      "site_name": "Folic Acid",
      "fetched_at": "2026-02-26T07:27:28.456632+00:00",
      "http_status": 200,
      "content_type": "text/html",
      "text": "Key points\n- People with an MTHFR gene variant...",
      "links": ["https://..."],
      "images": [],
      "metadata": null,
      "references": null,
      "quality_score": 0.903,
      "deduplicated": false
    }
  ]
}
```

The `hits` array always contains search result metadata. The `retrieved` array is present when `retrieve_top > 0` and contains full page extractions for the top hits. Duplicate retrieved pages (detected via SimHash) are marked with `deduplicated: true`.

---

### POST /retrieve

Fetch a single URL and extract its content. Supports HTML pages and PDF documents.

**Request:**

```json
{
  "url": "https://pubmed.ncbi.nlm.nih.gov/35235964/",
  "render_js": false,
  "download_images": false,
  "max_images": 20,
  "allow_domains": null,
  "no_cache": false,
  "max_chars": 4000,
  "full_text": false,
  "webhook_url": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | *required* | URL to fetch and extract |
| `render_js` | bool | false | Use headless Chromium (for JS-rendered pages like SPAs) |
| `download_images` | bool | false | Download images to container `/data/media` with SHA-256 dedup. Also triggers OCR on images >= 200x200px. |
| `max_images` | int | 20 | Maximum images to return in response (0-100) |
| `allow_domains` | list | null | If set, request is rejected unless URL matches one of these domains |
| `no_cache` | bool | false | Bypass the 24-hour page cache |
| `max_chars` | int | 4000 | Max text characters returned. Uses semantic chunking to select the most relevant paragraphs when the embed-proxy is available; falls back to head truncation otherwise. |
| `full_text` | bool | false | Return the full extracted text, ignoring `max_chars` |
| `webhook_url` | string | null | If set, returns 202 immediately and POSTs the result to this URL when done. |

**Response:**

```json
{
  "requested_url": "https://pubmed.ncbi.nlm.nih.gov/35235964/",
  "final_url": "https://pubmed.ncbi.nlm.nih.gov/35235964/",
  "canonical_url": "https://pubmed.ncbi.nlm.nih.gov/35235964/",
  "title": "Article Title - PubMed",
  "site_name": "PubMed",
  "fetched_at": "2026-02-26T07:30:00.000000+00:00",
  "http_status": 200,
  "content_type": "text/html",
  "text": "Full extracted article text...",
  "links": ["https://..."],
  "images": [
    {
      "src_url": "https://cdn.example.com/fig1.jpg",
      "alt": "Figure 1",
      "caption": null,
      "width": 800,
      "height": 600,
      "sha256": null,
      "local_path": null,
      "ocr_text": null
    }
  ],
  "metadata": {
    "type": "pubmed",
    "pmid": "35235964",
    "doi": "10.1234/example",
    "journal": "Nature Genetics",
    "year": "2024/01/15",
    "authors": ["Smith J", "Jones A"],
    "abstract": "Background: ..."
  },
  "references": [
    {
      "title": "Reference text...",
      "doi": "10.1234/ref1",
      "pmid": "12345678",
      "year": 2023,
      "url": "https://..."
    }
  ],
  "quality_score": 0.903,
  "deduplicated": false
}
```

**New response fields (v0.2.0):**

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | object\|null | Domain-specific structured data (see Domain Extractors below) |
| `references` | array\|null | Extracted citations/references with DOIs, PMIDs, years, URLs |
| `quality_score` | float\|null | Content quality score 0.0-1.0 (length, sentence quality, boilerplate, paywall detection) |
| `deduplicated` | bool | `true` if this page was identified as a near-duplicate of another retrieved page (via SimHash) |
| `images[].ocr_text` | string\|null | OCR-extracted text from downloaded images (only when `download_images=true` and image >= 200x200px) |

---

### POST /search/batch

Run multiple search queries concurrently. Each query runs through the full pipeline (cache, re-rank, retrieve, dedup).

**Request:**

```json
{
  "queries": ["MTHFR C677T", "APOE4 alzheimer", "CYP2D6 metabolism"],
  "num_results": 5,
  "retrieve_top": 2,
  "language": "en",
  "safesearch": 1,
  "time_range": null,
  "site": null,
  "render_js": false,
  "download_images": false,
  "rerank": true,
  "no_cache": false,
  "max_chars": 4000,
  "full_text": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `queries` | list[string] | *required* | 1-10 search queries to run concurrently |
| *(other fields)* | | | Same as `/search` (except no `webhook_url`) |

**Response:**

```json
{
  "results": [
    { "query": "MTHFR C677T", "hits": [...], "retrieved": [...] },
    { "query": "APOE4 alzheimer", "hits": [...], "retrieved": [...] },
    { "query": "CYP2D6 metabolism", "hits": [...], "retrieved": [...] }
  ]
}
```

---

### GET /jobs/{job_id}

Check the status of an async webhook job (created when `webhook_url` is set on `/search` or `/retrieve`).

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "created_at": "2026-02-26T07:30:00.000000+00:00",
  "result": { "...": "full search or retrieve response" },
  "error": null
}
```

| Status value | Meaning |
|-------------|---------|
| `pending` | Job created, not yet started |
| `processing` | Currently running |
| `completed` | Done, result available |
| `failed` | Error occurred |
| `webhook_sent` | Result delivered to webhook URL |

Jobs are automatically cleaned up after 1 hour.

---

### GET /healthz

Health check (no authentication required).

**Response:**

```json
{
  "ok": true,
  "version": "0.2.0",
  "auth_enabled": true,
  "searxng_url": "http://searxng:8080",
  "embed_proxy": {
    "url": "http://embed_proxy:8085",
    "available": true
  },
  "cache": {
    "search_mb": 0.2,
    "pages_mb": 1.5,
    "search_ttl_s": 3600,
    "pages_ttl_s": 86400
  },
  "active_jobs": 0
}
```

---

## Features

### F1: Disk Cache with TTL

All search and retrieve results are cached to disk using SHA-256 keys.

| Cache | TTL | Max Size | Directory |
|-------|-----|----------|-----------|
| Search results | 1 hour | 500 MB | `/data/cache/search/` |
| Page content | 24 hours | 5 GB | `/data/cache/pages/` |

- Atomic writes (write `.tmp` then `os.rename`)
- Background cleanup every 30 minutes (evicts expired + enforces size limits)
- Page cache stores full text; `max_chars` chunking is re-applied on cache reads
- Use `no_cache: true` to bypass

### F2: PDF Extraction

PDFs are automatically detected via `Content-Type: application/pdf` and extracted using:
1. **pypdf** (primary) - fast text extraction
2. **pymupdf** (fallback) - handles scanned/complex PDFs

Text is returned page-by-page with `[Page N]` markers. Metadata includes `{"type": "pdf", "pages": N}`.

### F3: Smart Text Chunking

When text exceeds `max_chars`, the system selects the most relevant paragraphs:
1. Splits text into paragraphs
2. Embeds each paragraph + query via embed-proxy (PubMedBERT)
3. Selects highest-scoring paragraphs up to the character budget
4. Returns them in original document order

Falls back to head truncation if embed-proxy is unavailable.

### F4: Semantic Re-ranking

When `rerank: true` (default), search hits are re-ordered by semantic similarity:
1. Embeds query + all hit title+snippet texts via embed-proxy
2. Computes cosine similarity
3. Returns hits sorted by relevance

Falls back to SearXNG's original ranking if embed-proxy is unavailable.

### F5: Domain-Specific Extractors

Structured metadata is extracted for known domains:

| Domain | Fields |
|--------|--------|
| **PubMed** | pmid, doi, journal, year, authors, abstract |
| **PMC** | pmid, doi, journal, authors, sections (abstract, introduction, methods, results, discussion, conclusion) |
| **Wikipedia** | lead paragraph, infobox key-value pairs, categories |
| **arXiv** | doi, authors, abstract, pdf_link, categories |

### F6: Batch Search

`POST /search/batch` runs up to 10 queries concurrently via `asyncio.gather`. Each query gets the full pipeline treatment (cache, rerank, retrieve, dedup).

### F7: Citation/Reference Extraction

References are extracted from HTML reference sections (`#references`, `.ref-list`, `#citation-list`) and text DOIs:
- DOI regex: `10.\d{4,9}/[^\s]+`
- PMID regex: `PMID: \d{6,9}`
- Year, URL, and title are also captured when available

### F8: Content Deduplication (SimHash)

Retrieved pages are fingerprinted using 64-bit SimHash (3-word shingles). Pages with hamming distance <= 3 are marked as `deduplicated: true`. The original (first-seen) page is kept unmarked.

### F9: Per-Domain Rate Limiting

Outbound fetches are throttled to prevent hammering individual sites:
- **Global limit:** 10 concurrent fetches
- **Per-domain limit:** 2 concurrent fetches per domain
- Implemented via `asyncio.Semaphore`

### F10: Async Webhook Mode

Set `webhook_url` on any `/search` or `/retrieve` request to get async processing:
1. Returns `202 Accepted` with `{"job_id": "..."}` immediately
2. Processes the request in the background
3. POSTs the full result to your webhook URL (3 retries with exponential backoff)
4. Poll `GET /jobs/{job_id}` for status

### F11: Quality Scoring

Each retrieved page gets a quality score (0.0-1.0) based on:
- **Text length** (30%): sigmoid curve, 0.5 at 500 chars, 0.9 at 2000
- **Sentence quality** (30%): avg sentence length 20-200 chars, minimum 3 sentences
- **Boilerplate detection** (20%): cookie notices, paywall prompts, login walls
- **Paywall detection** (20%): known paywall domains (nature.com, sciencedirect.com, wiley.com, springer.com, cell.com, etc.) with < 300 chars text

Use `min_quality` on `/search` to filter out low-quality pages.

### F12: Image OCR

When `download_images: true`, images >= 200x200px are processed with Tesseract OCR. Extracted text is returned in `images[].ocr_text`.

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Bad request (empty query, domain not in allowlist) |
| 401 | Missing Authorization header |
| 403 | Invalid API key |
| 404 | Job not found (for `/jobs/{job_id}`) |
| 502 | Upstream fetch failed (site returned 4xx/5xx or connection error) |
| 504 | Fetch timed out (default 20s) |

Error body:

```json
{"detail": "Fetch timed out"}
```

---

## Usage Examples

### curl

```bash
API_KEY="fe5e611508dc9e000e96e94d9e5fa9a246df874bd6076b60"
API_URL="http://192.168.15.26:8089"

# Search with auto-retrieval of top 3 results
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "CYP2D6 poor metabolizer drug interactions", "num_results": 5, "retrieve_top": 2}'

# Search hits only (no page retrieval, no re-ranking)
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "vitamin D receptor polymorphism", "retrieve_top": 0, "rerank": false}'

# Restrict to PubMed, get full text
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "APOE4 alzheimer risk", "site": "pubmed.ncbi.nlm.nih.gov", "retrieve_top": 2, "full_text": true}'

# Retrieve a PDF
curl -s -X POST "$API_URL/retrieve" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://arxiv.org/pdf/2301.00001", "full_text": true}'

# Retrieve with domain metadata extraction
curl -s -X POST "$API_URL/retrieve" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://pubmed.ncbi.nlm.nih.gov/35235964/"}'

# Smart chunking (1000 chars, most relevant paragraphs)
curl -s -X POST "$API_URL/retrieve" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/DNA", "max_chars": 1000}'

# Batch search (3 queries concurrently)
curl -s -X POST "$API_URL/search/batch" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["MTHFR", "APOE4", "CYP2D6"], "num_results": 5, "retrieve_top": 2}'

# Bypass cache
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "MTHFR", "no_cache": true}'

# Quality filter (only high-quality pages)
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "gene therapy CRISPR", "min_quality": 0.5, "retrieve_top": 5}'

# Recent results only (last week)
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene therapy 2026", "time_range": "week", "retrieve_top": 1}'

# Async with webhook
curl -s -X POST "$API_URL/search" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "pharmacogenomics", "webhook_url": "http://10.0.1.56:8000/webhook"}'

# Check job status
curl -s "$API_URL/jobs/550e8400-e29b-41d4-a716-446655440000" \
  -H "Authorization: Bearer $API_KEY"

# Health check (no auth)
curl -s "$API_URL/healthz"
```

### Python

```python
import httpx

API_URL = "http://192.168.15.26:8089"
API_KEY = "fe5e611508dc9e000e96e94d9e5fa9a246df874bd6076b60"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# Search + retrieve
resp = httpx.post(f"{API_URL}/search", json={
    "query": "MTHFR folate supplementation",
    "num_results": 5,
    "retrieve_top": 2,
    "max_chars": 2000,
}, headers=HEADERS, timeout=60)
data = resp.json()

for hit in data["hits"]:
    print(f"[{hit['engine']}] {hit['title']}")
    print(f"  {hit['url']}")

if data.get("retrieved"):
    for page in data["retrieved"]:
        print(f"\n--- {page['title']} (quality: {page['quality_score']}) ---")
        if page.get("metadata"):
            print(f"  Metadata: {page['metadata']}")
        if page.get("references"):
            print(f"  References: {len(page['references'])} found")
        print(page["text"][:500])

# Batch search
resp = httpx.post(f"{API_URL}/search/batch", json={
    "queries": ["MTHFR", "APOE4", "CYP2D6"],
    "retrieve_top": 1,
}, headers=HEADERS, timeout=120)
for result in resp.json()["results"]:
    print(f"\nQuery: {result['query']} — {len(result['hits'])} hits")
```

### LLM Tool Definition (Anthropic format)

```json
{
  "name": "web_search",
  "description": "Search the web and retrieve full page content. Returns search hits with titles, URLs, and snippets, plus semantically ranked extracted text from the top results. Supports PubMed, arXiv, Wikipedia metadata extraction and PDF documents.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Search query"
      },
      "num_results": {
        "type": "integer",
        "default": 5,
        "description": "Number of search hits (1-20)"
      },
      "retrieve_top": {
        "type": "integer",
        "default": 2,
        "description": "How many top results to fetch and extract full text (0-10)"
      },
      "site": {
        "type": "string",
        "description": "Optional: restrict to domain (e.g. 'nih.gov', 'pubmed.ncbi.nlm.nih.gov')"
      },
      "time_range": {
        "type": "string",
        "enum": ["day", "week", "month", "year"],
        "description": "Optional: restrict by recency"
      },
      "max_chars": {
        "type": "integer",
        "default": 4000,
        "description": "Max text chars per page (100-100000). Uses semantic chunking."
      },
      "full_text": {
        "type": "boolean",
        "default": false,
        "description": "Return full text ignoring max_chars"
      },
      "min_quality": {
        "type": "number",
        "default": 0.0,
        "description": "Minimum quality score 0.0-1.0"
      }
    },
    "required": ["query"]
  }
}
```

### LLM Tool Handler (Python)

```python
import httpx

WEBSEARCH_URL = "http://192.168.15.26:8089"
WEBSEARCH_KEY = "fe5e611508dc9e000e96e94d9e5fa9a246df874bd6076b60"

def handle_web_search(query: str, num_results: int = 5, retrieve_top: int = 2,
                      site: str = None, time_range: str = None,
                      max_chars: int = 4000, min_quality: float = 0.0) -> str:
    """Call the web search API and format results for LLM context."""
    payload = {
        "query": query,
        "num_results": num_results,
        "retrieve_top": retrieve_top,
        "max_chars": max_chars,
        "min_quality": min_quality,
    }
    if site:
        payload["site"] = site
    if time_range:
        payload["time_range"] = time_range

    resp = httpx.post(
        f"{WEBSEARCH_URL}/search", json=payload,
        headers={"Authorization": f"Bearer {WEBSEARCH_KEY}"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    parts = [f"## Search results for: {data['query']}\n"]

    for i, hit in enumerate(data["hits"], 1):
        parts.append(f"{i}. **{hit['title']}**")
        parts.append(f"   URL: {hit['url']}")
        if hit.get("snippet"):
            parts.append(f"   {hit['snippet']}")
        parts.append("")

    if data.get("retrieved"):
        parts.append("---\n## Retrieved page content\n")
        for page in data["retrieved"]:
            if page.get("deduplicated"):
                continue  # Skip near-duplicate pages
            parts.append(f"### {page['title']}")
            parts.append(f"Source: {page['final_url']}")
            parts.append(f"Quality: {page.get('quality_score', 'N/A')}")
            if page.get("metadata"):
                meta = page["metadata"]
                if meta.get("doi"):
                    parts.append(f"DOI: {meta['doi']}")
                if meta.get("pmid"):
                    parts.append(f"PMID: {meta['pmid']}")
            parts.append("")
            parts.append(page["text"])
            if page.get("references"):
                parts.append(f"\n**References:** {len(page['references'])} citations found")
            parts.append("\n---\n")

    return "\n".join(parts)
```

---

## Architecture

```
              LLM (GPU server 10.0.1.56)
                       |
                  HTTP POST /search
                  + Authorization: Bearer <key>
                       |
                       v
  +-----------------------------------------------------+
  |  websearch_retriever (port 8089)                    |
  |  FastAPI + trafilatura + Playwright + pypdf         |
  |                                                     |
  |  1. Check disk cache (SHA-256 key)                  |
  |  2. Query SearXNG for search hits                   |
  |  3. Re-rank hits by semantic similarity             |
  |  4. Fetch top-k pages (parallel, rate-limited)      |
  |  5. Extract text, metadata, references, quality     |
  |  6. Smart-chunk text for LLM context budget         |
  |  7. Deduplicate near-identical pages (SimHash)      |
  |  8. Cache results + return                          |
  +-----------------------------------------------------+
            |                        |
   HTTP /search?format=json    POST /embeddings
            |                        |
            v                        v
  +--------------------+   +-------------------+
  | searxng (8088)     |   | embed-proxy (8085)|
  | Google, Bing, DDG, |   | PubMedBERT 768d   |
  | Brave, PubMed,     |   | Basic auth        |
  | Scholar, Wikipedia |   +-------------------+
  +--------------------+
            |
        Public web
```

## Search Engines

SearXNG aggregates results from multiple engines. Enabled by default:

| Engine | Shortcut | Notes |
|--------|----------|-------|
| Google | g | General web |
| Bing | b | General web |
| DuckDuckGo | ddg | Privacy-focused |
| Brave | br | Independent index |
| PubMed | pm | Biomedical literature |
| Google Scholar | gs | Academic papers |
| Wikipedia | wp | Encyclopedia |

SearXNG settings: `docker/searxng/settings.yml`

## Docker Services

| Service | Container | Internal Port | External Port | Image |
|---------|-----------|---------------|---------------|-------|
| `searxng` | `searxng` | 8080 | 8088 | `searxng/searxng:latest` |
| `websearch` | `websearch_retriever` | 8080 | 8089 | Custom (Python 3.12 + Chromium + Tesseract) |
| `embed-proxy` | `embed_proxy` | 8085 | 8085 | Custom (PubMedBERT) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEYS` | *(empty)* | Comma-separated Bearer token keys. Empty = no auth. |
| `SEARXNG_URL` | `http://searxng:8080` | SearXNG base URL |
| `EMBED_PROXY_URL` | `http://embed_proxy:8085` | Embed-proxy URL for semantic features |
| `EMBED_PROXY_USER` | `admin` | Embed-proxy Basic auth user |
| `EMBED_PROXY_PASS` | `MyGene@3146!` | Embed-proxy Basic auth password |
| `FETCH_TIMEOUT` | `20` | HTTP fetch timeout in seconds |
| `IMAGE_MAX_BYTES` | `8000000` | Max image download size (8MB) |
| `SEARCH_CACHE_TTL` | `3600` | Search cache TTL in seconds (1h) |
| `PAGE_CACHE_TTL` | `86400` | Page cache TTL in seconds (24h) |

### Rebuild after code changes

```bash
cd /home/harry/Projects/dnaiq_co_php/docker

# main.py is volume-mounted — just restart
docker restart websearch_retriever

# If Dockerfile or requirements.txt changed — full rebuild
docker-compose build websearch && docker-compose up -d websearch

# SearXNG settings changes — restart
docker restart searxng
```

### Logs

```bash
docker logs websearch_retriever --tail 50 -f
docker logs searxng --tail 50 -f
```

## Files

```
docker/
├── searxng/
│   ├── settings.yml          # Engine config, rate limits, JSON output
│   └── limiter.toml          # Bot detection disabled (local use)
├── websearch/
│   ├── Dockerfile            # Python 3.12 + Playwright Chromium + Tesseract OCR
│   ├── requirements.txt      # FastAPI, trafilatura, playwright, pypdf, pymupdf, etc.
│   ├── main.py               # FastAPI service (volume-mounted)
│   ├── data/                 # Cache + downloaded media (SHA-256 deduped)
│   │   ├── cache/search/     # Search result cache (1h TTL, 500MB max)
│   │   ├── cache/pages/      # Page content cache (24h TTL, 5GB max)
│   │   └── media/            # Downloaded images
│   └── API.md                # This file
└── docker-compose.yml        # Service definitions (searxng + websearch + embed-proxy)
```

## Performance Notes

- **Cached search:** ~30ms (60x faster than cold)
- **Cold search + retrieve (3 pages):** 3-8s typical
- **Static fetch** (default): ~1-3s per page via httpx
- **JS render** (`render_js: true`): ~5-15s per page via headless Chromium (persistent browser instance reused)
- **PDF extraction:** ~1-3s for typical 20-page paper
- **Parallel retrieval:** `retrieve_top` pages fetched concurrently via `asyncio.gather`
- **Rate limiting:** max 2 concurrent per domain, 10 global
- **SearXNG retry:** 3 attempts with exponential backoff on search failures
- **Semantic re-ranking:** adds ~200-500ms (embed-proxy call)
