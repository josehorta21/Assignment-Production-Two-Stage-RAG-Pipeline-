# Verification Status - Week 4 Two-Stage RAG

**Prepared on:** November 16, 2026

## Verified locally in the artifact preparation environment

- 26 passing deterministic automated tests in `results/pytest_verified.txt`
- Python source imports/compiles for tested modules and test doubles
- 12 complete, original fictional Markdown technical handbook pages
- 20 human-authored routing labels (12 internal and 8 out of scope)
- Readable Streamlit app source, Docker configurations, and code audit with diff examples

## Requires a real macOS / Linux / Windows run

- `uv sync --extra dev`
- First model downloads and bi-encoder Chroma indexing
- Real cross-encoder rank/score inspection
- Live DDGS web results
- Streamlit UI screenshots
- `uv run python -m scripts.calibrate_threshold` to select a measured cutoff
- `uv run python -m scripts.run_smoke` to capture live routing traces

The rubric asks for a production pipeline, not only unit-test mocks. This file explicitly prevents a dry-run from being mislabeled as evidence of live model performance.
