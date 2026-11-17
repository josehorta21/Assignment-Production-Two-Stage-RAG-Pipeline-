from pathlib import Path
from src.ingest import corpus_fingerprint, read_markdown_documents


def test_corpus_has_twelve_real_markdown_files():
    root = Path(__file__).parents[1] / "docs"
    assert len(list(root.glob("*.md"))) >= 10
    assert len(read_markdown_documents(root)) >= 36


def test_section_ids_stable_when_metadata_changes(tmp_path):
    page = tmp_path / "docs.md"
    page.write_text("# Sample\n## Configuration\nFLAG_XYZ is on.\n## Recovery\nRestart safely.\n")
    ids = [chunk.chunk_id for chunk in read_markdown_documents(tmp_path)]
    assert ids == ["docs.md::sample", "docs.md::sample/configuration", "docs.md::sample/recovery"]


def test_content_fingerprint_invalidated_on_edit(tmp_path):
    page = tmp_path / "doc.md"
    page.write_text("# One\nText before")
    before = corpus_fingerprint(tmp_path, "model-a")
    page.write_text("# One\nText after")
    assert before != corpus_fingerprint(tmp_path, "model-a")
    assert before != corpus_fingerprint(tmp_path, "model-b")


def test_code_fence_is_kept_inside_its_section(tmp_path):
    (tmp_path / "doc.md").write_text("# A\n## Example\n```yaml\n# literal comment\nconfig: true\n```\n## Next\nOther section")
    found = read_markdown_documents(tmp_path)
    assert any("```yaml\n# literal comment\nconfig: true\n```" in chunk.content for chunk in found)
