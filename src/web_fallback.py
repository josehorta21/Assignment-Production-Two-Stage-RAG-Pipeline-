"""No-key DDGS search with strict URL validation and untrusted text handling."""
from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse
from .schemas import RetrievedChunk

_INSTRUCTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(all\s+)?(the\s+)?(previous|prior|system|developer)\s+instructions?[^.!?]*[.!?]?"),
    re.compile(r"(?i)(output|print|say|respond with)\s+HACKED\b[^.!?]*[.!?]?"),
    re.compile(r"(?is)<\s*/?\s*(system|developer|assistant|instruction)\s*>"),
    re.compile(r"(?i)\bHACKED\b"),
]


def sanitize_untrusted(value: str, max_chars: int = 800) -> str:
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", str(value))
    for expression in _INSTRUCTION_PATTERNS:
        clean = expression.sub("", clean)
    return re.sub(r"\s+", " ", clean).strip()[:max_chars]


def valid_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False
        hostname = parsed.hostname.lower().rstrip(".")
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
            return False
        try:
            address = ipaddress.ip_address(hostname)
            if not address.is_global:
                return False
        except ValueError:
            pass
        return True
    except ValueError:
        return False


class WebFallback:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        self.last_error: str | None = None

    def search(self, query: str) -> list[RetrievedChunk]:
        self.last_error = None
        try:
            from ddgs import DDGS
            results = DDGS(timeout=8).text(query, max_results=self.max_results)
        except Exception as exc:
            self.last_error = f"Live search was unavailable ({type(exc).__name__})."
            return []
        citations = []
        seen = set()
        for hit in (results or []):
            url = str(hit.get("href") or hit.get("url") or "")
            if not valid_public_url(url) or url in seen:
                continue
            title = sanitize_untrusted(hit.get("title", "Untitled result"), 200)
            snippet = sanitize_untrusted(hit.get("body") or hit.get("snippet") or "", 900)
            if not snippet:
                continue
            seen.add(url)
            citations.append(RetrievedChunk(
                chunk_id=f"web:{len(citations) + 1}", title=title,
                content=snippet, url=url, source="web", score=1.0,
            ))
        return citations
