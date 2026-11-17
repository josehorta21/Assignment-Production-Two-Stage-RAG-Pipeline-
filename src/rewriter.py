"""Contextual rewriting with exact-token preservation and fail-safe local LLM."""
from __future__ import annotations

import re
from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, instruction: str, max_new_tokens: int = 100) -> str: ...


_PRONOUNS = re.compile(r"\b(it|its|that|those|them|they|their|this|these)\b", re.I)
_IDENTIFIER = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:[_./:-][a-zA-Z0-9]+)+\b|\b[a-z]+[A-Z][a-zA-Z0-9]*\b")
_IGNORABLE = re.compile(r"^(hey|hi|hello)[,!]?\s*|^(can you please|could you please|please)\s+", re.I)


def _context_subject(history: list[str]) -> str:
    for message in reversed(history):
        msg = message.strip()
        if not msg:
            continue
        # Prefer the most recent proper multi-word technical component.
        matches = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){1,5}\b", msg)
        if matches:
            return matches[-1]
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_:-]*", msg)
        return " ".join(tokens[-min(8, len(tokens)):])
    return ""


def deterministic_rewrite(query: str, history: list[str] | None = None) -> str:
    clean = " ".join(query.strip().split())
    if not clean:
        raise ValueError("Query must not be blank")
    clean = _IGNORABLE.sub("", clean)
    if _PRONOUNS.search(clean) and history:
        topic = _context_subject(history)
        if topic:
            clean = _PRONOUNS.sub(lambda match: topic if match.group(0).lower() in {"it", "that", "this", "them", "these", "those", "they"} else f"{topic}'s", clean, count=1)
    return clean


class QueryRewriter:
    def __init__(self, llm: TextGenerator | None = None, mode: str = "deterministic"):
        self.llm, self.mode = llm, mode

    def rewrite(self, query: str, history: list[str] | None = None) -> str:
        history = history or []
        base = deterministic_rewrite(query, history)
        if self.mode != "local" or self.llm is None:
            return base
        prompt = (
            "Rewrite the question into ONE concise standalone technical search query. "
            "Keep all exact configuration names, flags, numbers, and error codes unchanged. "
            "Do not answer or invent details.\n"
            f"PREVIOUS CONTEXT: {' '.join(history[-2:])[:280]}\n"
            f"QUESTION: {base[:280]}\nREWRITTEN QUERY:"
        )
        try:
            suggestion = self.llm.generate(prompt, max_new_tokens=65).strip()
            suggestion = suggestion.splitlines()[0].strip(" \"'`\t")
            # The local model is fallible; never lose exact identifiers.
            flags = set(_IDENTIFIER.findall(base))
            if not suggestion or len(suggestion) > 350 or any(flag not in suggestion for flag in flags):
                return base
            # Require the rewrite to retain a majority of the original technical terms.
            relevant = {t.lower() for t in re.findall(r"[A-Za-z0-9_-]{4,}", base)
                        if t.lower() not in {"what", "does", "which", "that", "this", "about", "where", "when", "with", "system", "requirements"}}
            rewritten_terms = {t.lower() for t in re.findall(r"[A-Za-z0-9_-]{4,}", suggestion)}
            if relevant and len(relevant & rewritten_terms) / len(relevant) < 0.55:
                return base
            # Prevent a language model from replacing the subject in a follow-up.
            if history and _PRONOUNS.search(query) and _context_subject(history).lower() not in suggestion.lower():
                return base
            return suggestion
        except Exception:
            return base
