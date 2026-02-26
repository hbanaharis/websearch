# Web Search & Retriever API

Self-hosted web search and page content extraction for LLM tool-use. Powered by SearXNG (meta-search) and a FastAPI retriever with trafilatura text extraction.

## Network Access

| Route | URL |
|-------|-----|
| LAN (from GPU server) | `http://192.168.15.26:8089` |
| Docker internal | `http://websearch_retriever:8080` |
| SearXNG UI (debug) | `http://192.168.15.26:8088` |

No authentication required (LAN-only service).

## Endpoints

### POST /search

Search the web and optionally retrieve + extract full page content from the top results.

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
  "allow_domains": null
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
      "requested_url": "https://www.cdc.gov/folic-acid/data-research/mthfr/index.html",
      "final_url": "https://www.cdc.gov/folic-acid/data-research/mthfr/index.html",
      "canonical_url": "https://www.cdc.gov/folic-acid/data-research/mthfr/index.html",
      "title": "MTHFR Gene Variant and Folic Acid Facts | CDC",
      "site_name": "Folic Acid",
      "fetched_at": "2026-02-26T07:27:28.456632+00:00",
      "http_status": 200,
      "content_type": "text/html",
      "text": "Key points\n- People with an MTHFR gene variant can process all types of folate...",
      "links": ["https://www.cdc.gov/...", "..."],
      "images": [{"src_url": "https://...", "alt": "...", "width": 600, "height": 400}]
    }
  ]
}
```

The `hits` array always contains search result metadata (title, URL, snippet, engine). The `retrieved` array is only present when `retrieve_top > 0` and contains full page extractions for the top hits.

---

### POST /retrieve

Fetch a single URL and extract its content. Use this when you already have a URL (e.g. from a citation or known resource).

**Request:**

```json
{
  "url": "https://pubmed.ncbi.nlm.nih.gov/35235964/",
  "render_js": false,
  "download_images": false,
  "max_images": 20,
  "allow_domains": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `url` | string | *required* | URL to fetch and extract |
| `render_js` | bool | false | Use headless Chromium (for JS-rendered pages like SPAs) |
| `download_images` | bool | false | Download images to container `/data/media` with SHA-256 dedup |
| `max_images` | int | 20 | Maximum images to return in response (0-100) |
| `allow_domains` | list | null | If set, request is rejected unless URL matches one of these domains |

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
  "links": ["https://...", "..."],
  "images": [
    {
      "src_url": "https://cdn.example.com/fig1.jpg",
      "alt": "Figure 1",
      "caption": null,
      "width": 800,
      "height": 600,
      "sha256": null,
      "local_path": null
    }
  ]
}
```

---

### GET /healthz

Health check.

**Response:**

```json
{"ok": true, "searxng_url": "http://searxng:8080"}
```

---

## Error Responses

| Status | Meaning |
|--------|---------|
| 400 | Bad request (empty query, domain not in allowlist) |
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
# Search with auto-retrieval of top 3 results
curl -s -X POST http://192.168.15.26:8089/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CYP2D6 poor metabolizer drug interactions", "num_results": 5, "retrieve_top": 2}'

# Search hits only (no page retrieval)
curl -s -X POST http://192.168.15.26:8089/search \
  -H "Content-Type: application/json" \
  -d '{"query": "vitamin D receptor polymorphism", "retrieve_top": 0}'

# Restrict to PubMed
curl -s -X POST http://192.168.15.26:8089/search \
  -H "Content-Type: application/json" \
  -d '{"query": "APOE4 alzheimer risk", "site": "pubmed.ncbi.nlm.nih.gov", "retrieve_top": 2}'

# Retrieve a single known URL
curl -s -X POST http://192.168.15.26:8089/retrieve \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.nature.com/articles/s41588-024-01234-5"}'

# Retrieve a JS-heavy page with Playwright
curl -s -X POST http://192.168.15.26:8089/retrieve \
  -H "Content-Type: application/json" \
  -d '{"url": "https://app.example.com/dashboard", "render_js": true}'

# Recent results only (last week)
curl -s -X POST http://192.168.15.26:8089/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CRISPR gene therapy 2026", "time_range": "week", "retrieve_top": 1}'
```

### Python

```python
import httpx

API = "http://192.168.15.26:8089"

# Search + retrieve
resp = httpx.post(f"{API}/search", json={
    "query": "MTHFR folate supplementation",
    "num_results": 5,
    "retrieve_top": 2,
})
data = resp.json()

for hit in data["hits"]:
    print(f"[{hit['engine']}] {hit['title']}")
    print(f"  {hit['url']}")

if data.get("retrieved"):
    for page in data["retrieved"]:
        print(f"\n--- {page['title']} ---")
        print(page["text"][:500])
```

### LLM Tool Definition (Anthropic format)

```json
{
  "name": "web_search",
  "description": "Search the web and retrieve full page content. Returns search hits with titles, URLs, and snippets, plus extracted text from the top results.",
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
        "description": "Optional: restrict to domain (e.g. 'nih.gov')"
      },
      "time_range": {
        "type": "string",
        "enum": ["day", "week", "month", "year"],
        "description": "Optional: restrict by recency"
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

def handle_web_search(query: str, num_results: int = 5, retrieve_top: int = 2,
                      site: str = None, time_range: str = None) -> str:
    """Call the web search API and format results for LLM context."""
    payload = {
        "query": query,
        "num_results": num_results,
        "retrieve_top": retrieve_top,
    }
    if site:
        payload["site"] = site
    if time_range:
        payload["time_range"] = time_range

    resp = httpx.post(f"{WEBSEARCH_URL}/search", json=payload, timeout=60)
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
            parts.append(f"### {page['title']}")
            parts.append(f"Source: {page['final_url']}")
            parts.append(f"Fetched: {page['fetched_at']}\n")
            parts.append(page["text"])
            parts.append("\n---\n")

    return "\n".join(parts)
```

---

## Architecture

```
              LLM (GPU server 10.0.1.56)
                       |
                  HTTP POST /search
                       |
                       v
  +--------------------------------------------+
  |  websearch_retriever (port 8089)           |
  |  FastAPI + trafilatura + Playwright        |
  |                                            |
  |  1. Queries SearXNG for search hits        |
  |  2. Fetches top-k pages (parallel)         |
  |  3. Extracts clean text via trafilatura    |
  |  4. Returns hits + full page content       |
  +--------------------------------------------+
                       |
              HTTP GET /search?format=json
                       |
                       v
  +--------------------------------------------+
  |  searxng (port 8088)                       |
  |  Meta-search: Google, Bing, DuckDuckGo,    |
  |  Brave, PubMed, Google Scholar, Wikipedia  |
  +--------------------------------------------+
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
| `websearch` | `websearch_retriever` | 8080 | 8089 | Custom (Python 3.12 + Chromium) |

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
│   ├── Dockerfile            # Python 3.12 + Playwright Chromium
│   ├── requirements.txt      # FastAPI, trafilatura, playwright, etc.
│   ├── main.py               # FastAPI service (volume-mounted)
│   ├── data/                 # Downloaded media (SHA-256 deduped)
│   └── API.md                # This file
└── docker-compose.yml        # Service definitions (searxng + websearch)
```

## Performance Notes

- **Static fetch** (default): ~1-3s per page via httpx
- **JS render** (`render_js: true`): ~5-15s per page via headless Chromium (persistent browser instance reused)
- **Parallel retrieval**: `retrieve_top` pages are fetched concurrently via `asyncio.gather`
- **SearXNG retry**: 3 attempts with exponential backoff on search failures
- **Typical /search with retrieve_top=3**: 3-8 seconds total
