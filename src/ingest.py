"""Header-aware Markdown loading with stable section IDs and original evidence."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from .schemas import RetrievedChunk


_SECTION = re.compile(r"^(#{1,3})\s+(.+)$")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "root"


def read_markdown_documents(folder: str | Path) -> list[RetrievedChunk]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Documentation folder not found: {root}")
    output: list[RetrievedChunk] = []
    for file in sorted(root.rglob("*.md")):
        document = file.read_text(encoding="utf-8")
        sections: list[tuple[str, str, int]] = []
        title = file.stem.replace("_", " ")
        hierarchy: dict[int, str] = {}
        lines: list[str] = []
        start_line = 1
        current_path = "overview"
        inside_fence = False
        fence_marker = ""

        def flush() -> None:
            nonlocal lines
            body = "\n".join(lines).strip()
            if body:
                sections.append((current_path, body, start_line))
            lines = []

        for lineno, line in enumerate(document.splitlines(), 1):
            fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)
            if fence_match:
                marker = fence_match.group(1)
                if not inside_fence:
                    inside_fence, fence_marker = True, marker[:3]
                elif marker.startswith(fence_marker):
                    inside_fence = False
                lines.append(line)
                continue
            matched = None if inside_fence else _SECTION.match(line)
            if matched:
                flush()
                depth, header = len(matched.group(1)), matched.group(2).strip()
                hierarchy = {k: v for k, v in hierarchy.items() if k < depth}
                hierarchy[depth] = header
                if depth == 1:
                    title = header
                current_path = "/".join(_slug(hierarchy[level]) for level in sorted(hierarchy))
                start_line = lineno
                lines = [line]
            else:
                lines.append(line)
        flush()
        for idx, (path, content, line) in enumerate(sections):
            stable_id = f"{file.name}::{path}"
            # Some malformed files repeat a heading; suffix such cases deterministically.
            if any(c.chunk_id == stable_id for c in output):
                stable_id = f"{stable_id}::{idx}"
            output.append(RetrievedChunk(
                chunk_id=stable_id,
                content=f"Document: {title}\nSection: {path}\n{content}",
                source="internal_doc", title=title,
                metadata={"file": file.name, "header_path": path, "line": line},
            ))
    if not output:
        raise ValueError(f"No Markdown sections found in {root}")
    return output


def corpus_fingerprint(folder: str | Path, model_name: str, chunk_version: str = "headers-v2") -> str:
    hash_input = hashlib.sha256(f"{model_name}|{chunk_version}|".encode())
    root = Path(folder)
    for file in sorted(root.rglob("*.md")):
        hash_input.update(str(file.relative_to(root)).encode())
        hash_input.update(file.read_bytes())
    return hash_input.hexdigest()
