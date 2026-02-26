from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse, urldefrag

import httpx
import trafilatura
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from selectolax.parser import HTMLParser

logger = logging.getLogger("websearch")

# -------- config --------
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "20"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122 Safari/537.36",
)
IMAGE_MAX_BYTES = int(os.getenv("IMAGE_MAX_BYTES", "8000000"))  # 8 MB
DATA_DIR = os.getenv("DATA_DIR", "/data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://searxng:8080")

os.makedirs(MEDIA_DIR, exist_ok=True)

# -------- persistent Playwright browser --------
_browser = None
_playwright = None


async def _get_browser():
    """Reuse a single Chromium instance across requests."""
    global _browser, _playwright
    if _browser is None or not _browser.is_connected():
        from playwright.async_api import async_playwright
        _playwright = await async_playwright().start()
        _browser = await _playwright.chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
    return _browser


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown: close persistent browser
    global _browser, _playwright
    if _browser:
        await _browser.close()
    if _playwright:
        await _playwright.stop()


app = FastAPI(title="Web Retriever", version="0.1.0", lifespan=lifespan)


# -------- request/response models --------
class RetrieveRequest(BaseModel):
    url: str = Field(..., description="URL to retrieve")
    render_js: bool = Field(False, description="Use Playwright if true")
    download_images: bool = Field(False, description="Download images to /data/media")
    max_images: int = Field(20, ge=0, le=100, description="Cap images returned")
    allow_domains: Optional[List[str]] = Field(None, description="Allowlist of domains")


class ImageItem(BaseModel):
    src_url: str
    alt: Optional[str] = None
    caption: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    sha256: Optional[str] = None
    local_path: Optional[str] = None


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


class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(8, ge=1, le=20)
    language: str = Field("en")
    safesearch: int = Field(1, ge=0, le=2)
    time_range: Optional[str] = Field(None, description="day/week/month/year")
    site: Optional[str] = Field(None, description="Restrict to domain, e.g. nih.gov")
    retrieve_top: int = Field(3, ge=0, le=10, description="How many top links to fetch+extract")
    render_js: bool = Field(False, description="Use Playwright for fetched pages")
    download_images: bool = Field(False, description="Download images for fetched pages")
    allow_domains: Optional[List[str]] = Field(None, description="Allowlist for retrieved pages")


class SearchHit(BaseModel):
    title: Optional[str] = None
    url: str
    snippet: Optional[str] = None
    engine: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
    retrieved: Optional[List[RetrieveResponse]] = None


# -------- URL helpers --------
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


# -------- fetch --------
async def fetch_static(url: str) -> Tuple[str, str, int, Optional[str], Dict[str, str]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT,
        follow_redirects=True,
        headers=headers,
    ) as client:
        r = await client.get(url)
        ct = r.headers.get("content-type")
        return r.text, str(r.url), r.status_code, ct, dict(r.headers)


async def fetch_rendered(url: str) -> Tuple[str, str, int, Optional[str]]:
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


# -------- extraction --------
def extract_title_site(parser: HTMLParser) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    title = None
    site_name = None
    canonical = None

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
    out = []
    seen: set[str] = set()
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
    seen: set[str] = set()

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

    # og:image / twitter:image
    for sel in [
        'meta[property="og:image"]',
        'meta[name="twitter:image"]',
        'meta[property="twitter:image"]',
    ]:
        node = parser.css_first(sel)
        if node and node.attributes.get("content"):
            nu = _normalize_url(node.attributes["content"], base_url)
            if nu and nu not in seen:
                seen.add(nu)
                imgs.append({"src_url": nu, "alt": None})
                if len(imgs) >= cap:
                    break

    return imgs


async def download_image(url: str) -> Tuple[Optional[str], Optional[str]]:
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*"}
    try:
        async with httpx.AsyncClient(
            timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers
        ) as client:
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


def extract_main_text(html: str, url: str) -> str:
    downloaded = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_recall=True,
        deduplicate=True,
    )
    return (downloaded or "").strip()


# -------- endpoints --------
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    _check_allow_domains(req.url, req.allow_domains)

    try:
        if req.render_js:
            html, final_url, status, ct = await fetch_rendered(req.url)
        else:
            html, final_url, status, ct, _headers = await fetch_static(req.url)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Fetch timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")

    if status and status >= 400:
        raise HTTPException(status_code=502, detail=f"Upstream returned status {status}")

    parser = HTMLParser(html)
    title, site_name, canonical = extract_title_site(parser)

    if canonical:
        canonical = _normalize_url(canonical, final_url)

    text = extract_main_text(html, final_url)
    links = extract_links(parser, final_url)
    raw_imgs = extract_images(parser, final_url, cap=max(req.max_images * 2, 20))
    images: List[ImageItem] = []

    for item in raw_imgs[: req.max_images]:
        img = ImageItem(**item)
        if req.download_images:
            sha, lp = await download_image(img.src_url)
            img.sha256 = sha
            img.local_path = lp
        images.append(img)

    fetched_at = datetime.now(timezone.utc).isoformat()

    return RetrieveResponse(
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
    )


# -------- search --------
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4))
async def searxng_search(
    query: str,
    *,
    num: int,
    lang: str,
    safesearch: int,
    time_range: Optional[str],
    site: Optional[str],
) -> List[Dict[str, Any]]:
    q = query.strip()
    if site:
        q = f"site:{site} {q}"

    params = {
        "q": q,
        "format": "json",
        "language": lang,
        "safesearch": str(safesearch),
        "count": str(num),
    }
    if time_range:
        params["time_range"] = time_range

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(
        timeout=FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        r = await client.get(f"{SEARXNG_URL.rstrip('/')}/search", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("results", [])


async def _retrieve_one(hit_url: str, render_js: bool, download_images: bool,
                        allow_domains: Optional[List[str]]) -> Optional[RetrieveResponse]:
    """Retrieve a single URL, returning None on failure."""
    try:
        return await retrieve(
            RetrieveRequest(
                url=hit_url,
                render_js=render_js,
                download_images=download_images,
                max_images=10,
                allow_domains=allow_domains,
            )
        )
    except Exception as e:
        logger.warning("Retrieve failed for %s: %s", hit_url, e)
        return None


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    results = await searxng_search(
        req.query,
        num=req.num_results,
        lang=req.language,
        safesearch=req.safesearch,
        time_range=req.time_range,
        site=req.site,
    )

    hits: List[SearchHit] = []
    seen: set[str] = set()
    for r in results:
        url = r.get("url")
        if not url:
            continue
        norm = _normalize_url(url, url)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        hits.append(
            SearchHit(
                title=r.get("title"),
                url=norm,
                snippet=r.get("content"),
                engine=r.get("engine"),
            )
        )
        if len(hits) >= req.num_results:
            break

    # Parallel retrieval of top-k results
    retrieved: List[RetrieveResponse] = []
    if req.retrieve_top > 0 and hits:
        tasks = [
            _retrieve_one(h.url, req.render_js, req.download_images, req.allow_domains)
            for h in hits[: req.retrieve_top]
        ]
        results_list = await asyncio.gather(*tasks)
        retrieved = [r for r in results_list if r is not None]

    return SearchResponse(
        query=req.query,
        hits=hits,
        retrieved=retrieved or None,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True, "searxng_url": SEARXNG_URL}
