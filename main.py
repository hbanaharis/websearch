from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag

import httpx
import numpy as np
import trafilatura
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from selectolax.parser import HTMLParser
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger("websearch")
logging.basicConfig(level=logging.INFO)

# ======================================================================
# CONFIG
# ======================================================================

FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "20"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122 Safari/537.36",
)
IMAGE_MAX_BYTES = int(os.getenv("IMAGE_MAX_BYTES", "8000000"))
DATA_DIR = os.getenv("DATA_DIR", "/data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

# Embed-proxy (PubMedBERT 768-dim)
EMBED_PROXY_URL = os.getenv("EMBED_PROXY_URL", "http://embed_proxy:8085")
EMBED_PROXY_USER = os.getenv("EMBED_PROXY_USER", "admin")
EMBED_PROXY_PASS = os.getenv("EMBED_PROXY_PASS", "MyGene@3146!")

# Cache
CACHE_DIR_SEARCH = os.path.join(DATA_DIR, "cache", "search")
CACHE_DIR_PAGES = os.path.join(DATA_DIR, "cache", "pages")
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "3600"))        # 1 hour
PAGE_CACHE_TTL = int(os.getenv("PAGE_CACHE_TTL", "86400"))           # 24 hours
CACHE_MAX_SEARCH_MB = 500
CACHE_MAX_PAGES_MB = 5000

# Rate limiting
MAX_CONCURRENT_GLOBAL = 10
MAX_CONCURRENT_PER_DOMAIN = 2

# Authentication — comma-separated API keys (empty = no auth required)
API_KEYS_RAW = os.getenv("API_KEYS", "")
API_KEYS: set = {k.strip() for k in API_KEYS_RAW.split(",") if k.strip()} if API_KEYS_RAW.strip() else set()
AUTH_ENABLED = len(API_KEYS) > 0

for d in [MEDIA_DIR, CACHE_DIR_SEARCH, CACHE_DIR_PAGES]:
    os.makedirs(d, exist_ok=True)

# ======================================================================
# PERSISTENT PLAYWRIGHT BROWSER
# ======================================================================

_browser = None
_playwright = None


async def _get_browser():
    global _browser, _playwright
    if _browser is None or not _browser.is_connected():
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
    return _browser


# ======================================================================
# AUTHENTICATION
# ======================================================================

_NO_AUTH_PATHS = {"/healthz", "/docs", "/openapi.json", "/redoc"}


async def verify_api_key(request: Request):
    """Bearer token auth. Skipped for health/docs endpoints and if API_KEYS is empty."""
    if not AUTH_ENABLED:
        return
    if request.url.path in _NO_AUTH_PATHS:
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization: Bearer <key>")
    token = auth[7:].strip()
    if token not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ======================================================================
# MODELS
# ======================================================================

class Reference(BaseModel):
    title: Optional[str] = None
    authors: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None


class ImageItem(BaseModel):
    src_url: str
    alt: Optional[str] = None
    caption: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: Optional[str] = None
    local_path: Optional[str] = None
    ocr_text: Optional[str] = None                    # F12


class RetrieveRequest(BaseModel):
    url: str = Field(..., description="URL to retrieve")
    render_js: bool = Field(False, description="Use Playwright if true")
    download_images: bool = Field(False, description="Download images to /data/media")
    max_images: int = Field(20, ge=0, le=100, description="Cap images returned")
    allow_domains: Optional[List[str]] = Field(None, description="Allowlist of domains")
    no_cache: bool = Field(False, description="Bypass cache")                           # F1
    max_chars: int = Field(4000, ge=100, le=100000, description="Max text chars")       # F3
    full_text: bool = Field(False, description="Return full text ignoring max_chars")   # F3
    webhook_url: Optional[str] = Field(None, description="Async: POST result here")     # F10


class RetrieveResponse(BaseModel):
    requested_url: str
    final_url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    site_name: Optional[str] = None
    fetched_at: str
    http_status: int
    content_type: Optional[str] = None
    text: str
    links: List[str]
    images: List[ImageItem]
    metadata: Optional[Dict[str, Any]] = None          # F5
    references: Optional[List[Reference]] = None       # F7
    quality_score: Optional[float] = None              # F11
    deduplicated: bool = False                         # F8


class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(8, ge=1, le=20)
    language: str = Field("en")
    safesearch: int = Field(1, ge=0, le=2)
    time_range: Optional[str] = Field(None, description="day/week/month/year")
    site: Optional[str] = Field(None, description="Restrict to domain")
    retrieve_top: int = Field(3, ge=0, le=10)
    render_js: bool = Field(False)
    download_images: bool = Field(False)
    allow_domains: Optional[List[str]] = Field(None)
    no_cache: bool = Field(False)                                          # F1
    max_chars: int = Field(4000, ge=100, le=100000)                        # F3
    full_text: bool = Field(False)                                         # F3
    rerank: bool = Field(True, description="Semantic re-ranking via embeddings")  # F4
    min_quality: float = Field(0.0, ge=0.0, le=1.0)                       # F11
    webhook_url: Optional[str] = Field(None)                               # F10


class SearchHit(BaseModel):
    title: Optional[str] = None
    url: str
    snippet: Optional[str] = None
    engine: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    retrieved: Optional[List[RetrieveResponse]] = None


class BatchSearchRequest(BaseModel):
    queries: List[str] = Field(..., min_length=1, max_length=10)
    num_results: int = Field(5, ge=1, le=20)
    retrieve_top: int = Field(2, ge=0, le=5)
    language: str = Field("en")
    safesearch: int = Field(1, ge=0, le=2)
    time_range: Optional[str] = None
    site: Optional[str] = None
    render_js: bool = Field(False)
    download_images: bool = Field(False)
    rerank: bool = Field(True)
    no_cache: bool = Field(False)
    max_chars: int = Field(4000, ge=100, le=100000)
    full_text: bool = Field(False)


class BatchSearchResponse(BaseModel):
    results: List[SearchResponse]


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ======================================================================
# F1: DISK CACHE WITH TTL
# ======================================================================

def _cache_key_search(query: str, num_results: int, language: str,
                      safesearch: int, time_range: Optional[str],
                      site: Optional[str]) -> str:
    raw = json.dumps({
        "q": query.strip().lower(), "n": num_results, "l": language,
        "s": safesearch, "t": time_range or "", "site": site or "",
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_key_url(url: str) -> str:
    return hashlib.sha256(url.strip().encode()).hexdigest()


def _cache_read(cache_dir: str, key: str, ttl: int) -> Optional[dict]:
    path = os.path.join(cache_dir, f"{key}.json")
    try:
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > ttl:
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_write(cache_dir: str, key: str, data: dict) -> None:
    path = os.path.join(cache_dir, f"{key}.json")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, default=str)
        os.rename(tmp, path)
    except Exception as e:
        logger.warning("Cache write failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _cache_dir_size_mb(cache_dir: str) -> float:
    total = 0
    try:
        for f in os.listdir(cache_dir):
            fp = os.path.join(cache_dir, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    except Exception:
        pass
    return total / (1024 * 1024)


async def _cache_cleanup_loop():
    while True:
        await asyncio.sleep(1800)  # 30 min
        try:
            for cache_dir, ttl, max_mb in [
                (CACHE_DIR_SEARCH, SEARCH_CACHE_TTL, CACHE_MAX_SEARCH_MB),
                (CACHE_DIR_PAGES, PAGE_CACHE_TTL, CACHE_MAX_PAGES_MB),
            ]:
                now = time.time()
                files = []
                for fname in os.listdir(cache_dir):
                    fp = os.path.join(cache_dir, fname)
                    if os.path.isfile(fp):
                        mtime = os.path.getmtime(fp)
                        if now - mtime > ttl * 2:
                            os.unlink(fp)
                        else:
                            files.append((mtime, fp, os.path.getsize(fp)))
                # Enforce size limit
                files.sort(reverse=True)  # newest first
                cumulative = 0
                for _, fp, sz in files:
                    cumulative += sz
                    if cumulative > max_mb * 1024 * 1024:
                        os.unlink(fp)
        except Exception as e:
            logger.warning("Cache cleanup error: %s", e)


# ======================================================================
# F9: PER-DOMAIN RATE LIMITING
# ======================================================================

_domain_semaphores: Dict[str, asyncio.Semaphore] = {}
_global_fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GLOBAL)


def _get_domain_semaphore(domain: str) -> asyncio.Semaphore:
    if domain not in _domain_semaphores:
        _domain_semaphores[domain] = asyncio.Semaphore(MAX_CONCURRENT_PER_DOMAIN)
    return _domain_semaphores[domain]


@asynccontextmanager
async def _rate_limited(url: str):
    domain = urlparse(url).netloc.lower()
    dsem = _get_domain_semaphore(domain)
    async with _global_fetch_semaphore:
        async with dsem:
            yield


# ======================================================================
# EMBED-PROXY CLIENT (F3, F4)
# ======================================================================

_embed_available: Optional[bool] = None
_embed_checked_at: float = 0


async def _check_embed_proxy() -> bool:
    global _embed_available, _embed_checked_at
    if _embed_available is not None and time.time() - _embed_checked_at < 60:
        return _embed_available
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{EMBED_PROXY_URL}/health")
            data = r.json()
            _embed_available = data.get("status") == "ok"
    except Exception:
        _embed_available = False
    _embed_checked_at = time.time()
    return _embed_available


async def embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    if not await _check_embed_proxy():
        return None
    auth = base64.b64encode(f"{EMBED_PROXY_USER}:{EMBED_PROXY_PASS}".encode()).decode()
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                f"{EMBED_PROXY_URL}/embeddings",
                json={"texts": texts},
                headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
            )
            r.raise_for_status()
            return r.json().get("embeddings")
    except Exception as e:
        logger.warning("Embed-proxy call failed: %s", e)
        return None


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ======================================================================
# URL HELPERS
# ======================================================================

def _normalize_url(u: str, base: str) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if re.match(r"^(javascript:|mailto:|tel:|data:)", u, re.I):
        return None
    abs_u = urljoin(base, u)
    abs_u, _ = urldefrag(abs_u)
    parsed = urlparse(abs_u)
    if parsed.scheme not in ("http", "https"):
        return None
    return urlunparse(parsed)


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _check_allow_domains(url: str, allow_domains: Optional[List[str]]):
    if not allow_domains:
        return
    d = _domain(url)
    ok = any(d == a.lower() or d.endswith("." + a.lower()) for a in allow_domains)
    if not ok:
        raise HTTPException(status_code=400, detail=f"Domain not allowed: {d}")


# ======================================================================
# FETCH (with rate limiting)
# ======================================================================

async def fetch_static(url: str) -> Tuple[bytes, str, int, Optional[str], Dict[str, str]]:
    """Fetch URL, return raw bytes + metadata."""
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf"}
    async with _rate_limited(url):
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers,
        ) as client:
            r = await client.get(url)
            ct = r.headers.get("content-type", "")
            return r.content, str(r.url), r.status_code, ct, dict(r.headers)


async def fetch_rendered(url: str) -> Tuple[str, str, int, Optional[str]]:
    async with _rate_limited(url):
        browser = await _get_browser()
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()
        try:
            resp = await page.goto(url, wait_until="networkidle", timeout=int(FETCH_TIMEOUT * 1000))
            html = await page.content()
            final_url = page.url
            status = resp.status if resp else 0
            ct = resp.headers.get("content-type") if resp else None
            return html, final_url, status, ct
        finally:
            await context.close()


# ======================================================================
# F2: PDF EXTRACTION
# ======================================================================

def _extract_pdf_pypdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.warning("pypdf extraction failed: %s", e)
        return None


def _extract_pdf_pymupdf(pdf_bytes: bytes) -> Optional[str]:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.warning("pymupdf extraction failed: %s", e)
        return None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text = _extract_pdf_pypdf(pdf_bytes)
    if not text:
        text = _extract_pdf_pymupdf(pdf_bytes)
    return (text or "").strip()


# ======================================================================
# HTML EXTRACTION
# ======================================================================

def extract_title_site(parser: HTMLParser) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    title = site_name = canonical = None
    tnode = parser.css_first("title")
    if tnode:
        title = tnode.text(strip=True)[:300]
    og_site = parser.css_first('meta[property="og:site_name"]')
    if og_site and og_site.attributes.get("content"):
        site_name = og_site.attributes["content"][:200]
    canon = parser.css_first('link[rel="canonical"]')
    if canon and canon.attributes.get("href"):
        canonical = canon.attributes["href"]
    return title, site_name, canonical


def extract_links(parser: HTMLParser, base_url: str, cap: int = 500) -> List[str]:
    out, seen = [], set()
    for a in parser.css("a"):
        href = a.attributes.get("href")
        nu = _normalize_url(href, base_url)
        if not nu or nu in seen:
            continue
        seen.add(nu)
        out.append(nu)
        if len(out) >= cap:
            break
    return out


def _best_from_srcset(srcset: str) -> Optional[str]:
    parts = [p.strip() for p in srcset.split(",") if p.strip()]
    if not parts:
        return None
    best = parts[-1].split()[0]
    best_w = -1
    for p in parts:
        toks = p.split()
        if not toks:
            continue
        u = toks[0]
        if len(toks) > 1 and toks[1].endswith("w"):
            try:
                w = int(toks[1][:-1])
                if w > best_w:
                    best_w = w
                    best = u
            except ValueError:
                pass
    return best


def extract_images(parser: HTMLParser, base_url: str, cap: int = 50) -> List[Dict[str, Any]]:
    imgs: List[Dict[str, Any]] = []
    seen: set = set()
    for img in parser.css("img"):
        src = img.attributes.get("src") or ""
        srcset = img.attributes.get("srcset") or ""
        if srcset and (not src or src.startswith("data:")):
            candidate = _best_from_srcset(srcset)
            if candidate:
                src = candidate
        nu = _normalize_url(src, base_url)
        if not nu or nu in seen:
            continue
        seen.add(nu)
        alt = img.attributes.get("alt")
        w = img.attributes.get("width")
        h = img.attributes.get("height")
        try:
            w_i = int(w) if w else None
        except ValueError:
            w_i = None
        try:
            h_i = int(h) if h else None
        except ValueError:
            h_i = None
        imgs.append({"src_url": nu, "alt": alt, "width": w_i, "height": h_i})
        if len(imgs) >= cap:
            break
    for sel in ['meta[property="og:image"]', 'meta[name="twitter:image"]', 'meta[property="twitter:image"]']:
        node = parser.css_first(sel)
        if node and node.attributes.get("content"):
            nu = _normalize_url(node.attributes["content"], base_url)
            if nu and nu not in seen:
                seen.add(nu)
                imgs.append({"src_url": nu, "alt": None})
                if len(imgs) >= cap:
                    break
    return imgs


def extract_main_text(html: str, url: str) -> str:
    downloaded = trafilatura.extract(
        html, url=url, include_comments=False, include_tables=True,
        favor_recall=True, deduplicate=True,
    )
    return (downloaded or "").strip()


# ======================================================================
# F5: DOMAIN-SPECIFIC EXTRACTORS
# ======================================================================

def _detect_domain_type(url: str) -> Optional[str]:
    d = _domain(url)
    if "pubmed.ncbi.nlm.nih.gov" in d:
        return "pubmed"
    if "pmc.ncbi.nlm.nih.gov" in d or "ncbi.nlm.nih.gov/pmc" in url:
        return "pmc"
    if "wikipedia.org" in d:
        return "wikipedia"
    if "arxiv.org" in d:
        return "arxiv"
    return None


def _meta_content(parser: HTMLParser, selector: str) -> Optional[str]:
    node = parser.css_first(selector)
    if node and node.attributes.get("content"):
        return node.attributes["content"].strip()
    return None


def _meta_contents(parser: HTMLParser, selector: str) -> List[str]:
    out = []
    for node in parser.css(selector):
        c = node.attributes.get("content")
        if c:
            out.append(c.strip())
    return out


def extract_pubmed_metadata(parser: HTMLParser, url: str) -> dict:
    return {
        "type": "pubmed",
        "pmid": _meta_content(parser, 'meta[name="citation_pmid"]'),
        "doi": _meta_content(parser, 'meta[name="citation_doi"]'),
        "journal": _meta_content(parser, 'meta[name="citation_journal_title"]'),
        "year": _meta_content(parser, 'meta[name="citation_date"]'),
        "authors": _meta_contents(parser, 'meta[name="citation_author"]'),
        "abstract": _extract_abstract_pubmed(parser),
    }


def _extract_abstract_pubmed(parser: HTMLParser) -> Optional[str]:
    for sel in ["div.abstract-content", "div#abstract", "div#enc-abstract"]:
        node = parser.css_first(sel)
        if node:
            return node.text(strip=True)[:5000]
    return None


def extract_pmc_metadata(parser: HTMLParser, url: str) -> dict:
    sections = {}
    for h2 in parser.css("h2"):
        heading = h2.text(strip=True).lower()
        parent = h2.parent
        if parent:
            text = parent.text(strip=True)[:5000]
            for key in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
                if key in heading:
                    sections[key] = text
                    break
    return {
        "type": "pmc",
        "pmid": _meta_content(parser, 'meta[name="citation_pmid"]'),
        "doi": _meta_content(parser, 'meta[name="citation_doi"]'),
        "journal": _meta_content(parser, 'meta[name="citation_journal_title"]'),
        "authors": _meta_contents(parser, 'meta[name="citation_author"]'),
        "sections": sections,
    }


def extract_wikipedia_metadata(parser: HTMLParser, url: str) -> dict:
    lead = None
    content = parser.css_first("div.mw-parser-output")
    if content:
        for p in content.css("p"):
            text = p.text(strip=True)
            if len(text) > 50:
                lead = text[:2000]
                break
    # Infobox
    infobox = {}
    ib = parser.css_first("table.infobox")
    if ib:
        for tr in ib.css("tr"):
            th = tr.css_first("th")
            td = tr.css_first("td")
            if th and td:
                infobox[th.text(strip=True)[:100]] = td.text(strip=True)[:500]
    # Categories
    cats = []
    catlinks = parser.css_first("div#mw-normal-catlinks")
    if catlinks:
        for a in catlinks.css("a"):
            t = a.text(strip=True)
            if t and t != "Categories":
                cats.append(t)
    return {"type": "wikipedia", "lead": lead, "infobox": infobox or None, "categories": cats or None}


def extract_arxiv_metadata(parser: HTMLParser, url: str) -> dict:
    abstract = None
    ab = parser.css_first("blockquote.abstract")
    if ab:
        abstract = ab.text(strip=True).replace("Abstract:", "").strip()[:5000]
    authors = _meta_contents(parser, 'meta[name="citation_author"]')
    pdf_link = None
    for a in parser.css("a"):
        href = a.attributes.get("href", "")
        if "/pdf/" in href:
            pdf_link = _normalize_url(href, url)
            break
    return {
        "type": "arxiv",
        "doi": _meta_content(parser, 'meta[name="citation_doi"]'),
        "authors": authors,
        "abstract": abstract,
        "pdf_link": pdf_link,
        "categories": _meta_content(parser, 'meta[name="citation_arxiv_id"]'),
    }


def extract_domain_metadata(parser: HTMLParser, url: str) -> Optional[Dict[str, Any]]:
    dtype = _detect_domain_type(url)
    if dtype == "pubmed":
        return extract_pubmed_metadata(parser, url)
    elif dtype == "pmc":
        return extract_pmc_metadata(parser, url)
    elif dtype == "wikipedia":
        return extract_wikipedia_metadata(parser, url)
    elif dtype == "arxiv":
        return extract_arxiv_metadata(parser, url)
    return None


# ======================================================================
# F7: CITATION / REFERENCE EXTRACTION
# ======================================================================

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"'<>\]]+)")
_PMID_RE = re.compile(r"\bPMID:\s*(\d{6,9})")


def extract_references(parser: HTMLParser, text: str) -> Optional[List[dict]]:
    refs = _extract_references_html(parser)
    # Also extract standalone DOIs from text
    dois_in_text = set(_DOI_RE.findall(text))
    ref_dois = {r.get("doi") for r in refs if r.get("doi")}
    for doi in dois_in_text - ref_dois:
        refs.append({"doi": doi.rstrip(".")})
    return refs if refs else None


def _extract_references_html(parser: HTMLParser) -> List[dict]:
    refs = []
    # Look for reference list containers
    ref_section = None
    for sel in ["section#references", "div#references", "ol.references",
                "div.ref-list", "section.ref-list", "div#citation-list"]:
        ref_section = parser.css_first(sel)
        if ref_section:
            break
    if not ref_section:
        return refs
    # Parse individual references
    for li in ref_section.css("li, div.citation, div.ref"):
        text = li.text(strip=True)[:500]
        if len(text) < 10:
            continue
        ref: dict = {}
        # DOI
        doi_m = _DOI_RE.search(text)
        if doi_m:
            ref["doi"] = doi_m.group(1).rstrip(".")
        # PMID
        pmid_m = _PMID_RE.search(text)
        if pmid_m:
            ref["pmid"] = pmid_m.group(1)
        # Year
        year_m = re.search(r"\b(19|20)\d{2}\b", text)
        if year_m:
            ref["year"] = int(year_m.group(0))
        # URL from <a> tags
        a = li.css_first("a[href]")
        if a:
            href = a.attributes.get("href", "")
            if href.startswith("http"):
                ref["url"] = href
        # Use full text as title fallback
        if not ref.get("doi") and not ref.get("url"):
            ref["title"] = text[:200]
        elif not ref.get("title"):
            ref["title"] = text[:200]
        refs.append(ref)
    return refs


# ======================================================================
# F11: QUALITY SCORING
# ======================================================================

PAYWALL_DOMAINS = {
    "nature.com", "sciencedirect.com", "onlinelibrary.wiley.com",
    "link.springer.com", "cell.com", "tandfonline.com", "jstor.org",
    "academic.oup.com", "journals.sagepub.com",
}

BOILERPLATE_MARKERS = [
    "subscribe to read", "access denied", "sign in to view",
    "create an account", "purchase this article", "cookie policy",
    "accept cookies", "we use cookies", "javascript is disabled",
    "enable javascript", "please verify", "captcha",
]


def compute_quality_score(text: str, url: str) -> float:
    if not text:
        return 0.0
    # Length score: sigmoid, 0.5 at 500 chars, 0.9 at 2000
    length_score = 1.0 / (1.0 + math.exp(-0.003 * (len(text) - 500)))
    # Sentence quality
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    sent_count = len(sentences)
    if sent_count == 0:
        sentence_score = 0.0
    else:
        avg_len = sum(len(s) for s in sentences) / sent_count
        sentence_score = 1.0 if 20 < avg_len < 200 else 0.5
        if sent_count < 3:
            sentence_score *= 0.5
    # Boilerplate penalty
    text_lower = text.lower()
    boilerplate_hits = sum(1 for m in BOILERPLATE_MARKERS if m in text_lower)
    boilerplate_score = max(0.0, 1.0 - boilerplate_hits * 0.2)
    # Paywall penalty
    paywall_score = 1.0
    d = _domain(url)
    if any(pw in d for pw in PAYWALL_DOMAINS) and len(text) < 300:
        paywall_score = 0.1
    # Weighted composite
    score = (length_score * 0.3 + sentence_score * 0.3 +
             boilerplate_score * 0.2 + paywall_score * 0.2)
    return round(min(1.0, max(0.0, score)), 3)


# ======================================================================
# F12: IMAGE OCR
# ======================================================================

def _ocr_image(image_path: str, min_size: int = 200) -> Optional[str]:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(image_path)
        w, h = img.size
        if w < min_size or h < min_size:
            return None
        text = pytesseract.image_to_string(img).strip()
        return text if len(text) > 5 else None
    except Exception:
        return None


# ======================================================================
# F3: TEXT CHUNKING FOR LLM CONTEXT
# ======================================================================

def split_paragraphs(text: str, min_length: int = 50) -> List[str]:
    raw = re.split(r'\n{2,}', text)
    paras = []
    buf = ""
    for p in raw:
        p = p.strip()
        if not p:
            continue
        if len(p) < min_length and buf:
            buf += "\n" + p
        else:
            if buf:
                paras.append(buf)
            buf = p
    if buf:
        paras.append(buf)
    return paras


async def smart_truncate(text: str, query: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    paras = split_paragraphs(text)
    if len(paras) <= 1:
        return text[:max_chars]
    # Try semantic ranking
    embeddings = await embed_texts([query] + paras)
    if embeddings and len(embeddings) == len(paras) + 1:
        query_emb = embeddings[0]
        scored = []
        for i, emb in enumerate(embeddings[1:]):
            sim = cosine_similarity(query_emb, emb)
            scored.append((sim, i))
        scored.sort(reverse=True)
        selected = []
        total = 0
        for _, idx in scored:
            p = paras[idx]
            if total + len(p) > max_chars:
                break
            selected.append((idx, p))
            total += len(p)
        # Return in original order
        selected.sort(key=lambda x: x[0])
        return "\n\n".join(p for _, p in selected)
    # Fallback: head truncation
    return text[:max_chars]


# ======================================================================
# F4: SEMANTIC RE-RANKING
# ======================================================================

async def rerank_hits(hits: List[SearchHit], query: str) -> List[SearchHit]:
    if not hits:
        return hits
    texts = [f"{h.title or ''} {h.snippet or ''}".strip() for h in hits]
    embeddings = await embed_texts([query] + texts)
    if not embeddings or len(embeddings) != len(texts) + 1:
        return hits  # fallback: original order
    query_emb = embeddings[0]
    scored = []
    for i, emb in enumerate(embeddings[1:]):
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, i))
    scored.sort(reverse=True)
    return [hits[idx] for _, idx in scored]


# ======================================================================
# F8: SIMHASH CONTENT DEDUPLICATION
# ======================================================================

def _simhash(text: str, hashbits: int = 64) -> int:
    words = text.lower().split()
    if len(words) < 3:
        return 0
    v = [0] * hashbits
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i + 3])
        h = int(hashlib.md5(shingle.encode()).hexdigest(), 16)
        for j in range(hashbits):
            if h & (1 << j):
                v[j] += 1
            else:
                v[j] -= 1
    fingerprint = 0
    for j in range(hashbits):
        if v[j] > 0:
            fingerprint |= (1 << j)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate_responses(responses: List[RetrieveResponse], threshold: int = 3) -> List[RetrieveResponse]:
    if len(responses) <= 1:
        return responses
    hashes = [_simhash(r.text) for r in responses]
    for i in range(len(responses)):
        if responses[i].deduplicated:
            continue
        for j in range(i + 1, len(responses)):
            if responses[j].deduplicated:
                continue
            if hashes[i] != 0 and hashes[j] != 0:
                if _hamming_distance(hashes[i], hashes[j]) <= threshold:
                    responses[j].deduplicated = True
    return responses


# ======================================================================
# F10: ASYNC WEBHOOK / JOB STORE
# ======================================================================

_jobs: Dict[str, dict] = {}


async def _post_webhook(url: str, payload: dict, retries: int = 3):
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(url, json=payload)
                if r.status_code < 400:
                    return
        except Exception:
            pass
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt)
    logger.warning("Webhook delivery failed after %d attempts: %s", retries, url)


async def _run_job(job_id: str, coro, webhook_url: str):
    _jobs[job_id]["status"] = "processing"
    try:
        result = await coro
        result_dict = result.model_dump() if hasattr(result, "model_dump") else result
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result_dict
        await _post_webhook(webhook_url, {"job_id": job_id, "status": "completed", "result": result_dict})
        _jobs[job_id]["status"] = "webhook_sent"
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
        await _post_webhook(webhook_url, {"job_id": job_id, "status": "failed", "error": str(e)})


async def _job_cleanup_loop():
    while True:
        await asyncio.sleep(300)  # 5 min
        now = time.time()
        expired = [jid for jid, j in _jobs.items() if now - j.get("created_ts", 0) > 3600]
        for jid in expired:
            _jobs.pop(jid, None)


def _start_async_job(coro, webhook_url: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_ts": time.time(),
        "result": None,
        "error": None,
    }
    asyncio.create_task(_run_job(job_id, coro, webhook_url))
    return job_id


# ======================================================================
# IMAGE DOWNLOAD
# ======================================================================

async def download_image(url: str) -> Tuple[Optional[str], Optional[str]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}
    try:
        async with _rate_limited(url):
            async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers) as client:
                r = await client.get(url)
                if r.status_code != 200:
                    return None, None
                ctype = r.headers.get("content-type", "")
                if not ctype.startswith("image/"):
                    return None, None
                content = r.content
                if len(content) > IMAGE_MAX_BYTES:
                    return None, None
    except Exception:
        return None, None
    h = hashlib.sha256(content).hexdigest()
    ext = "img"
    if "jpeg" in ctype or "jpg" in ctype:
        ext = "jpg"
    elif "png" in ctype:
        ext = "png"
    elif "webp" in ctype:
        ext = "webp"
    elif "gif" in ctype:
        ext = "gif"
    local_path = os.path.join(MEDIA_DIR, f"{h}.{ext}")
    if not os.path.exists(local_path):
        with open(local_path, "wb") as f:
            f.write(content)
    return h, local_path


# ======================================================================
# LIFESPAN
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    cache_task = asyncio.create_task(_cache_cleanup_loop())
    job_task = asyncio.create_task(_job_cleanup_loop())
    yield
    cache_task.cancel()
    job_task.cancel()
    global _browser, _playwright
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()


app = FastAPI(
    title="Web Retriever",
    version="0.2.0",
    lifespan=lifespan,
    dependencies=[Depends(verify_api_key)],
)


# ======================================================================
# ENDPOINTS
# ======================================================================

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    _check_allow_domains(req.url, req.allow_domains)

    # F10: webhook mode
    if req.webhook_url:
        # Create a copy without webhook_url to avoid recursion
        inner_req = req.model_copy(update={"webhook_url": None})
        job_id = _start_async_job(retrieve(inner_req), req.webhook_url)
        return Response(
            status_code=202,
            content=json.dumps({"job_id": job_id, "status": "pending"}),
            media_type="application/json",
        )

    # F1: cache check
    cache_key = _cache_key_url(req.url)
    if not req.no_cache:
        cached = _cache_read(CACHE_DIR_PAGES, cache_key, PAGE_CACHE_TTL)
        if cached:
            # Apply chunking to cached text
            if not req.full_text and len(cached.get("text", "")) > req.max_chars:
                cached["text"] = await smart_truncate(cached["text"], req.url, req.max_chars)
            return RetrieveResponse(**cached)

    # Fetch
    is_pdf = False
    try:
        if req.render_js:
            html_str, final_url, status, ct = await fetch_rendered(req.url)
            raw_bytes = html_str.encode()
        else:
            raw_bytes, final_url, status, ct, _headers = await fetch_static(req.url)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Fetch timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")

    if status and status >= 400:
        raise HTTPException(status_code=502, detail=f"Upstream returned status {status}")

    # F2: PDF detection
    if ct and "application/pdf" in ct:
        is_pdf = True
        text = extract_pdf_text(raw_bytes)
        title = None
        site_name = None
        canonical = None
        links = []
        raw_imgs = []
        metadata = {"type": "pdf", "pages": text.count("[Page ")}
        refs = extract_references(HTMLParser(""), text)
    else:
        html_str = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else raw_bytes
        parser = HTMLParser(html_str)
        title, site_name, canonical = extract_title_site(parser)
        if canonical:
            canonical = _normalize_url(canonical, final_url)
        text = extract_main_text(html_str, final_url)
        links = extract_links(parser, final_url)
        raw_imgs = extract_images(parser, final_url, cap=max(req.max_images * 2, 20))
        # F5: domain metadata
        metadata = extract_domain_metadata(parser, final_url)
        # F7: references
        refs = extract_references(parser, text)

    # F11: quality score
    quality = compute_quality_score(text, final_url)

    # F3: chunking
    full_text_saved = text  # keep full for caching
    if not req.full_text and len(text) > req.max_chars:
        text = await smart_truncate(text, req.url, req.max_chars)

    # Images + F12: OCR
    images: List[ImageItem] = []
    if not is_pdf:
        for item in raw_imgs[:req.max_images]:
            img = ImageItem(**item)
            if req.download_images:
                sha, lp = await download_image(img.src_url)
                img.sha256 = sha
                img.local_path = lp
                if lp:
                    img.ocr_text = _ocr_image(lp)
            images.append(img)

    fetched_at = datetime.now(timezone.utc).isoformat()

    response = RetrieveResponse(
        requested_url=req.url,
        final_url=final_url,
        canonical_url=canonical,
        title=title,
        site_name=site_name,
        fetched_at=fetched_at,
        http_status=status or 200,
        content_type=ct,
        text=text,
        links=links,
        images=images,
        metadata=metadata,
        references=refs,
        quality_score=quality,
    )

    # F1: cache write (store full text, not truncated)
    cache_data = response.model_dump()
    cache_data["text"] = full_text_saved
    _cache_write(CACHE_DIR_PAGES, cache_key, cache_data)

    return response


# -------- search --------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
async def searxng_search(
    query: str, *, num: int, lang: str, safesearch: int,
    time_range: Optional[str], site: Optional[str],
) -> List[Dict[str, Any]]:
    q = query.strip()
    if site:
        q = f"site:{site} {q}"
    params = {
        "q": q, "format": "json", "language": lang,
        "safesearch": str(safesearch), "count": str(num),
    }
    if time_range:
        params["time_range"] = time_range
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers) as client:
        r = await client.get(f"{SEARXNG_URL.rstrip('/')}/search", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])


async def _retrieve_one(hit_url: str, render_js: bool, download_images: bool,
                        allow_domains: Optional[List[str]], max_chars: int,
                        full_text: bool, no_cache: bool) -> Optional[RetrieveResponse]:
    try:
        return await retrieve(RetrieveRequest(
            url=hit_url, render_js=render_js, download_images=download_images,
            max_images=10, allow_domains=allow_domains, max_chars=max_chars,
            full_text=full_text, no_cache=no_cache,
        ))
    except Exception as e:
        logger.warning("Retrieve failed for %s: %s", hit_url, e)
        return None


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    # F10: webhook mode
    if req.webhook_url:
        inner_req = req.model_copy(update={"webhook_url": None})
        job_id = _start_async_job(search(inner_req), req.webhook_url)
        return Response(
            status_code=202,
            content=json.dumps({"job_id": job_id, "status": "pending"}),
            media_type="application/json",
        )

    # F1: search cache
    cache_key = _cache_key_search(
        req.query, req.num_results, req.language, req.safesearch, req.time_range, req.site
    )
    if not req.no_cache:
        cached = _cache_read(CACHE_DIR_SEARCH, cache_key, SEARCH_CACHE_TTL)
        if cached:
            return SearchResponse(**cached)

    results = await searxng_search(
        req.query, num=req.num_results, lang=req.language,
        safesearch=req.safesearch, time_range=req.time_range, site=req.site,
    )

    hits: List[SearchHit] = []
    seen: set = set()
    for r in results:
        url = r.get("url")
        if not url:
            continue
        norm = _normalize_url(url, url)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hits.append(SearchHit(
            title=r.get("title"), url=norm,
            snippet=r.get("content"), engine=r.get("engine"),
        ))
        if len(hits) >= req.num_results:
            break

    # F4: semantic re-ranking
    if req.rerank and len(hits) > 1:
        hits = await rerank_hits(hits, req.query)

    # Parallel retrieval
    retrieved: List[RetrieveResponse] = []
    if req.retrieve_top > 0 and hits:
        tasks = [
            _retrieve_one(h.url, req.render_js, req.download_images,
                          req.allow_domains, req.max_chars, req.full_text, req.no_cache)
            for h in hits[:req.retrieve_top]
        ]
        results_list = await asyncio.gather(*tasks)
        retrieved = [r for r in results_list if r is not None]

        # F11: quality filter
        if req.min_quality > 0:
            retrieved = [r for r in retrieved if (r.quality_score or 0) >= req.min_quality]

        # F8: dedup
        if len(retrieved) > 1:
            retrieved = deduplicate_responses(retrieved)

    response = SearchResponse(
        query=req.query, hits=hits, retrieved=retrieved or None,
    )

    # F1: cache write
    _cache_write(CACHE_DIR_SEARCH, cache_key, response.model_dump())

    return response


# F6: Batch search
@app.post("/search/batch", response_model=BatchSearchResponse)
async def search_batch(req: BatchSearchRequest) -> BatchSearchResponse:
    async def _one(q: str) -> SearchResponse:
        return await search(SearchRequest(
            query=q, num_results=req.num_results, retrieve_top=req.retrieve_top,
            language=req.language, safesearch=req.safesearch, time_range=req.time_range,
            site=req.site, render_js=req.render_js, download_images=req.download_images,
            rerank=req.rerank, no_cache=req.no_cache, max_chars=req.max_chars,
            full_text=req.full_text,
        ))

    tasks = [_one(q) for q in req.queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    valid = []
    for r in results:
        if isinstance(r, SearchResponse):
            valid.append(r)
        else:
            logger.warning("Batch query failed: %s", r)
            valid.append(SearchResponse(query="error", hits=[]))
    return BatchSearchResponse(results=valid)


# F10: Job status
@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(
        job_id=job["job_id"], status=job["status"],
        created_at=job["created_at"], result=job.get("result"),
        error=job.get("error"),
    )


# Health check — extended
@app.get("/healthz", dependencies=[])
async def healthz():
    embed_ok = await _check_embed_proxy()
    search_cache_mb = _cache_dir_size_mb(CACHE_DIR_SEARCH)
    page_cache_mb = _cache_dir_size_mb(CACHE_DIR_PAGES)
    return {
        "ok": True,
        "version": "0.2.0",
        "auth_enabled": AUTH_ENABLED,
        "searxng_url": SEARXNG_URL,
        "embed_proxy": {"url": EMBED_PROXY_URL, "available": embed_ok},
        "cache": {
            "search_mb": round(search_cache_mb, 1),
            "pages_mb": round(page_cache_mb, 1),
            "search_ttl_s": SEARCH_CACHE_TTL,
            "pages_ttl_s": PAGE_CACHE_TTL,
        },
        "active_jobs": len(_jobs),
    }
