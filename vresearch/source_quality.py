"""Source-quality weighting. Higher weight = more trustworthy. Used to (a) choose which
pages to fetch first, (b) order evidence shown to the drafter, (c) break ties when
two sources contradict each other. The logic is deliberately simple and visible."""
from urllib.parse import urlparse

PRIMARY_SUFFIXES = (".gov", ".edu", ".int", ".mil")
PRIMARY_DOMAINS = {
    "sec.gov", "iea.org", "oecd.org", "worldbank.org", "imf.org", "europa.eu", "nature.com",
    "science.org", "arxiv.org", "nih.gov", "who.int", "energy.gov", "bls.gov", "census.gov",
}
REPUTABLE_NEWS = {
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "economist.com", "cnbc.com", "theguardian.com", "washingtonpost.com",
    "mckinsey.com", "bcg.com", "spglobal.com", "statista.com", "techcrunch.com", "wired.com",
}
LOW_QUALITY = {
    "medium.com", "substack.com", "reddit.com", "quora.com", "blogspot.com", "wordpress.com",
    "linkedin.com", "facebook.com", "x.com", "twitter.com", "pinterest.com", "tumblr.com",
}


def _host(url: str) -> str:
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def _matches(host: str, domains: set[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def score_source(url: str, kind: str = "web") -> float:
    if kind == "document":
        return 0.9  # user-provided material
    host = _host(url)
    if _matches(host, PRIMARY_DOMAINS) or host.endswith(PRIMARY_SUFFIXES):
        return 1.0
    if _matches(host, REPUTABLE_NEWS):
        return 0.85
    if host.endswith("wikipedia.org"):
        return 0.6
    if _matches(host, LOW_QUALITY):
        return 0.3
    if host.endswith(".org"):
        return 0.65
    return 0.5


def tier(score: float) -> str:
    return "primary" if score >= 0.95 else "reputable" if score >= 0.8 else "general" if score >= 0.5 else "low"
