"""Archived AI-assisted baseline flaws, intentionally excluded from runtime imports.

Do not deploy: these functions demonstrate two authentic failure modes that were
corrected in src/reranker.py and src/generator.py / src/web_fallback.py.
"""


def broken_confidence_gate(raw_logit: float, threshold: float = 0.55) -> bool:
    # BUG: raw cross-encoder output is an uncalibrated logit, NOT [0,1] confidence.
    return raw_logit >= threshold


def broken_generator_prompt(user_question: str, web_snippet: str) -> str:
    # BUG: web content is concatenated as instructions without a trust boundary.
    return f"Answer the user.\nQuestion: {user_question}\n{web_snippet}\n"


def broken_rewrite(query: str, history: list[str]) -> str:
    # BUG: does not resolve conversational references and drops exact context.
    return query.replace("it", "").strip()


def broken_indexer(existing_collection, chunks) -> None:
    # BUG: unbounded duplicate inserts and no content hash on restart.
    existing_collection.add(ids=[c.chunk_id for c in chunks], documents=[c.content for c in chunks])
