"""Web search + page fetch.
Tavily is the PRIMARY provider (via TAVILY_API_KEY).
Falls back to DuckDuckGo and Wikipedia on failure, rate limit, or timeout.
Records provider metadata for every query."""
import os
import threading
import requests
import trafilatura

UA = {"User-Agent": "Mozilla/5.0 (compatible; DeeptraceBot/1.0)"}

_tavily_lock = threading.Lock()
_tavily_credits_used = 0


def get_tavily_credits_used() -> int:
    with _tavily_lock:
        return _tavily_credits_used


def _tavily(query: str, n: int, search_depth: str = "basic") -> list[dict]:
    global _tavily_credits_used
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    cap = int(os.getenv("TAVILY_MONTHLY_CAP", "0") or "0")
    with _tavily_lock:
        if cap > 0 and _tavily_credits_used >= cap:
            return []

    r = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": search_depth or "basic",
            "max_results": n,
            "include_raw_content": False,
            "include_answer": False,
        },
        headers=UA,
        timeout=12,
    )
    if r.status_code != 200:
        return []

    data = r.json()
    with _tavily_lock:
        _tavily_credits_used += 1

    out = []
    for item in data.get("results", []):
        url = item.get("url", "")
        if url:
            out.append({
                "title": item.get("title", "") or url,
                "url": url,
                "snippet": item.get("content", "") or "",
                "provider": "tavily",
                "raw_content": item.get("raw_content", "") or "",
            })
    return out


def _ddg(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    out = []
    for r in DDGS().text(query, max_results=n) or []:
        if r.get("href"):
            out.append({
                "title": r.get("title", ""),
                "url": r["href"],
                "snippet": r.get("body", ""),
                "provider": "ddg",
            })
    return out


def _wikipedia(query: str, n: int) -> list[dict]:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": n},
        headers=UA,
        timeout=10,
    )
    if r.status_code != 200:
        return []
    out = []
    for item in r.json().get("query", {}).get("search", []):
        title = item["title"]
        snippet = item.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", "")
        out.append({
            "title": title,
            "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
            "snippet": snippet,
            "provider": "wikipedia",
        })
    return out


def web_search(query: str, n: int = 6, *, search_depth: str = "basic") -> list[dict]:
    """Search using Tavily first, falling back to DuckDuckGo, then Wikipedia."""
    providers = []
    if os.getenv("TAVILY_API_KEY"):
        providers.append(_tavily)
    providers.extend([_ddg, _wikipedia])

    for fn in providers:
        try:
            if fn == _tavily:
                res = fn(query, n, search_depth=search_depth)
            else:
                res = fn(query, n)
            if res:
                return res
        except Exception:
            continue
    return []


def _tavily_extract(url: str, max_chars: int = 20000) -> str:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return ""
    try:
        r = requests.post(
            "https://api.tavily.com/extract",
            json={"api_key": api_key, "urls": [url]},
            headers=UA,
            timeout=12,
        )
        if r.status_code == 200:
            results = r.json().get("results", [])
            if results and results[0].get("raw_content"):
                return results[0]["raw_content"][:max_chars]
    except Exception:
        pass
    return ""


def fetch_page(url: str, max_chars: int = 20000) -> str:
    """Fetch page text using trafilatura, falling back to Tavily extract."""
    try:
        r = requests.get(url, headers=UA, timeout=12)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", "text/html"):
            text = trafilatura.extract(r.text, include_comments=False, include_tables=True) or ""
            if text:
                return text[:max_chars]
    except Exception:
        pass

    # Secondary extraction via Tavily if direct fetch returned empty
    if os.getenv("TAVILY_API_KEY"):
        extracted = _tavily_extract(url, max_chars)
        if extracted:
            return extracted

    return ""
