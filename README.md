# Northstar Adaptive RAG Workbench

**Week 4 Hands-On: Production-Ready Two-Stage RAG with Query Rewriting, Cross-Encoder Re-Ranking, Confidence Gating & Live Web Fallback**  
**Student:** José Antonio Horta Herrera · **Course:** Artificial Intelligence / AI Systems Engineering · **Term:** Fall 2026

An end-to-end, **API-key-free**, two-stage Retrieval-Augmented Generation (RAG) project designed for a fictional internal cloud engineering handbook. It transforms multi-turn questions, retrieves up to **K1 = 20** candidates with normalized MiniLM vectors in **ChromaDB**, re-ranks candidates with **MS MARCO MiniLM CrossEncoder**, gates internal answers with a **sigmoid-transformed logit** (locally calibrated threshold `0.35`), and queries **DDGS** for live external evidence when the internal material is insufficient or freshness-sensitive. The Streamlit UI exposes the decisions, rank shifts, scores, sources, citations, and execution trace.

> **Verification and integrity statement (Nov. 16, 2026):** This project contains **12 synthetic technical documents** and **20 labeled calibration questions**. A user-provided macOS terminal screenshot verifies a **real local MiniLM + MS MARCO CrossEncoder calibration run** that selected `0.35` from the evaluated grid. Its results are **19/20 correct routes (95%) on the same calibration set**, not independent test accuracy. The updated project has **28 passing deterministic QA tests** in the preparation environment (recorded in `results/pytest_updated.txt`). Live DDGS results, FLAN-T5 synthesis quality, and production deployment have not been independently verified; show these honestly in your recorded demonstration.

## 1. Architecture

```mermaid
flowchart LR
  U["User question + prior turns"] --> R["Query Rewriter<br/>Local FLAN-T5 / deterministic fallback"]
  R --> V["Stage 1: MiniLM bi-encoder<br/>Chroma cosine ANN<br/>K1 = 20"]
  V --> X["Stage 2: MS MARCO cross-encoder<br/>Query-passage pairwise scoring<br/>Top K2 = 5"]
  X --> C["Confidence = sigmoid(top raw logit)"]
  C --> G{"Score >= threshold<br/>and not freshness-sensitive?"}
  G -->|Yes| D["Internal evidence<br/>source chunk IDs"]
  G -->|No| W["DDGS live search<br/>validate URLs and isolate snippets"]
  W --> E["External evidence<br/>web URLs"]
  D --> S["Grounded synthesis<br/>optional local FLAN-T5<br/>extractive evidence always retained"]
  E --> S
  S --> O["Cited answer + execution trace<br/>Streamlit dashboard"]
  T["20 human-labeled route cases"] --> A["Threshold evaluation<br/>false-internal vs unnecessary-web"]
  C -.-> A
```

### Engineering decisions

| Layer | Selected implementation | Key consideration |
|---|---|---|
| Query transformation | Local `google/flan-t5-small`, with deterministic rewrite fallback | Multi-turn pronoun expansion and exact flag preservation |
| First-stage dense retriever | `sentence-transformers/all-MiniLM-L6-v2` + Chroma persistent HNSW cosine collection | `K1=20` candidates, unit-normalized embeddings, hash-based index rebuild |
| Second-stage re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | True pairwise attention over query/document, rank changes visible |
| Confidence gate | Stable `sigmoid(logit)` with **local selected cutoff `0.35`** | Chosen from 20 training/evaluation labels; this sigmoid is **not a calibrated probability** |
| External fallback | `ddgs` Python metasearch, 3 previews | No key needed; HTTP(S) URL validation, prompt-injection filtering |
| Response generator | Local FLAN-T5 with evidence-checked output, plus always-cited extractive excerpts | Unsupported generated claims are omitted; citations must remain traceable |
| User interface | Streamlit | 20-candidate table, rerank shifts, score meter, routing reason, citation explorers |
| Packaging | `uv`, local model cache, Docker Compose | No OpenAI/Tavily/Cohere API key or paid infrastructure |

**Important:** A high sigmoid value measures a transformed scoring logit, **not** the statistical probability that a passage is relevant. Domain-specific evaluation and manual inspection are required; heuristic prompt-injection defenses also cannot guarantee immunity against adversarial inputs.

## 2. Quick start (macOS, Windows, or Linux)

Prerequisites: **Python 3.11 or 3.12**, `uv`, and an internet connection for the initial package/model downloads and for live web fallback. Start all commands from the project root.

```bash
# Install uv if necessary: https://docs.astral.sh/uv/getting-started/installation/
uv python install 3.11
uv sync --extra dev

# Optional configuration: copy .env.example to .env
cp .env.example .env

# Verify deterministic QA; no network calls needed
uv run pytest -v

# Launch interactive dashboard
uv run streamlit run src/app.py
# Open http://localhost:8501
```

On Windows PowerShell, use `Copy-Item .env.example .env` instead of `cp`. `uv sync` resolves dependencies and downloads Python packages. The first live query may download weights for MiniLM, MS MARCO, and FLAN-T5. `HF_TOKEN` is **optional**, used only if Hugging Face rate limits anonymous model downloads. No paid API key is required.

**Important for existing local installations:** if you created `.env` before this update, edit its `CONFIDENCE_THRESHOLD` line to `0.35` yourself. The application honors the existing `.env` over the new source-code default. Restart Streamlit after editing `.env` to reload the settings. Keep `.env` out of Git.

### Configurable settings

The optional `.env` file supports:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LOCAL_LLM_MODEL=google/flan-t5-small
REWRITE_MODE=local
GENERATION_MODE=local
CONFIDENCE_THRESHOLD=0.35
K1=20
K2=5
WEB_RESULTS=3
DOCS_DIR=docs
VECTOR_STORE_DIR=.rag_index
```

For a lighter demonstration, switch `REWRITE_MODE=deterministic` and `GENERATION_MODE=extractive`: the bi-encoder and cross-encoder remain **real**, but local FLAN generation is disabled. The dashboard also offers an extractive-only toggle.

### Docker option (local container demonstration)

```bash
docker compose up --build
# Open http://localhost:8501
```

Docker Compose persists `./.model_cache` and `./.rag_index` across container restarts. The first build and model download require access to package and Hugging Face registries. Docker manifests by themselves do not establish that a deployment was successfully executed; demonstrate your local running container in your recorded walkthrough if you choose this option.

## 3. Repository layout

```text
production-rag-pipeline/
├── pyproject.toml, .env.example, .python-version, .gitignore
├── README.md, ai_prompt_log.md, Dockerfile, docker-compose.yml
├── docs/                       # 12 original Markdown technical handbook files
├── data/eval_cases.json        # 12 internal + 8 out-of-domain routing cases
├── src/
│   ├── schemas.py              # RetrievedChunk and RAGResponse contracts
│   ├── config.py               # Environment configuration
│   ├── ingest.py               # Stable section IDs and code-fence-safe splitting
│   ├── local_llm.py            # Optional FLAN-T5 rewriting and synthesis
│   ├── rewriter.py             # Conversation-aware query transformation
│   ├── retriever.py            # Normalized MiniLM embeddings + Chroma cache
│   ├── reranker.py             # Raw cross-encoder logits + stable sigmoid
│   ├── calibration.py          # False-internal and unnecessary-web rate accounting
│   ├── web_fallback.py         # DDGS search, URL and snippet controls
│   ├── generator.py            # Evidence-first response + explicit citations
│   ├── pipeline.py             # Branching orchestration + telemetry
│   ├── factory.py              # Lazy construction of model-backed components
│   └── app.py                  # Streamlit workbench
├── scripts/
│   ├── run_smoke.py            # Optional live model / web integration example
│   └── calibrate_threshold.py  # Real score collection and threshold comparison
├── audit/initial_draft.py      # Archived imperfect AI-assisted initial draft
├── tests/                      # 28 deterministic tests across seven test modules (incl. config)
├── results/pytest_verified.txt # Archived original 26-test QA output
├── results/pytest_updated.txt  # Re-run on updated source: 28 tests
├── VIDEO_SCRIPT.md             # 5–7 minute demo narration
├── GUIA_ENTREGA_ES.md          # Delivery steps in Spanish
└── CANVAS_SUBMISSION.md        # Fill in live GitHub/YouTube links after recording
```

## 4. Knowledge base and query scenarios

`docs/` contains **12 synthetic Northstar Engineering Reference pages** explaining rolling deployment, connection pooling, retry budgets, idempotency, queue visibility, circuit breakers, rate limits, JWT rotation, backup recovery, tenant isolation, vector indexing, and pod disruption. Each page includes an operational overview, a configuration section, code-fenced example, runbook, and trade-off notes.

**These are fictional training documents.** Examples such as `maxUnavailable defaults to 25% in Rolling Update Deployment` describe the Northstar demonstration system only; they are not asserted as real Kubernetes, AWS, or database defaults. The original documentation was authored for this educational project family.

Sample queries to demonstrate the three required behaviors:

1. **Internal exact match:** `What is the maxUnavailable default in Rolling Update Deployment?`
2. **Conceptual internal:** `How do retry budgets prevent cascading overload during a partial outage?`
3. **Multi-turn:** first ask `Tell me about Kubernetes Worker Nodes`, then `What are its system requirements?` (shows anaphora resolution; if Northstar lacks Kubernetes worker node requirements, the engine should not pretend otherwise).
4. **Fresh web fallback:** `What is the latest AWS outage announced today?` (freshness guard routes to web regardless of deceptively high internal similarity).
5. **Forced web demo:** enable the explicit sidebar switch if the chosen out-of-scope query happens to pass the provisional threshold; disclose the override in your video.

The UI displays `original_query`, `transformed_query`, **stage-1 cosine similarity**, **stage-2 raw logit and sigmoid score**, old/new rank positions, confidence threshold, route reason, cited snippets, and downloadable JSON trace. There is no secret hidden reranker or fabricated status result.

## 5. Measured threshold calibration (Nov. 16, 2026)

The initial engineering threshold was `0.55`. A **real local macOS calibration** of the actual `sentence-transformers/all-MiniLM-L6-v2` bi-encoder and `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker evaluated **20 manually labeled queries** (`data/eval_cases.json`): 12 expected to use `INTERNAL_RAG` and 8 expected to use `WEB_FALLBACK`. No live web requests are made by the calibration script.

Run:

```bash
uv run python -m scripts.calibrate_threshold
# Writes results/calibration_results.json (no longer excluded by `.gitignore` in the updated package)
```

The script evaluated thresholds **0.35, 0.40, 0.45, ... 0.95**. It uses a weighted routing error with **3 points for an external query incorrectly accepted as internal** and **1 point for an internal query unnecessarily sent to web**. It breaks ties by accuracy, then the lower threshold.

### Results from the user's local terminal output

| Indicator | Selected gate: **0.35** | Previous gate: **0.55** |
|---|---:|---:|
| Evaluated queries | 20 | 20 |
| Correct route classifications | **19/20** | 18/20 |
| Route accuracy, same labeled set | **95.0%** | 90.0% |
| External queries incorrectly routed internally | **0/8** | 0/8 |
| Internal queries unnecessarily routed to web | **1/12** | 2/12 |
| False-internal rate | **0.0%** | 0.0% |
| False-web rate | **8.3%** | 16.7% |
| Weighted error (3 × false-internal + false-web) | **1** | 2 |

**Failure analysis:** The internal question **I03 — “What causes connection pool socket starvation?”** received a top sigmoid score of approximately **0.114**, so it was routed to the web at `0.35`. The internal question **I12 — “How are HNSW vector indexes tuned?”** scored approximately **0.521** and was saved from unnecessary web routing when the cutoff was lowered from `0.55` to `0.35`. All eight externally labeled questions received `0.000` to three decimal places in the terminal display and were routed to `WEB_FALLBACK`.

This **95% is in-sample routing accuracy**, not proof of retrieval accuracy, factual correctness, or generalization. It is based on a small, fictional internal corpus with deliberately out-of-domain external examples; a production threshold would require a separate held-out dataset, harder near-domain negatives, freshness cases, and periodic recalibration. The sigmoid output is **not** a calibrated probability of relevance. Freshness-sensitive queries independently trigger web fallback regardless of confidence.

The user's Mac generated the full-precision per-query evidence at `results/calibration_results.json`; **commit that exact file** rather than recreating values from the screenshot. The source package now stops excluding this report in `.gitignore`.

To use the selected threshold:

```dotenv
CONFIDENCE_THRESHOLD=0.35
```

If you already have `.env`, update it and restart the application. The sidebar threshold control can also be used for comparative demonstrations, but it does not rewrite `.env`. Re-run calibration with a held-out labeled set before treating `0.35` as a stable production policy.

## 6. Engineering audit (human-in-the-loop AI)

`ai_prompt_log.md` records the AI generation request, the imperfect baseline (archived in `audit/initial_draft.py`), a four-item audit, exact corrections, impact, and corresponding automated tests. The most serious defects and their fixes:

1. **Raw logit compared against `0.55`:** replaced with explicitly raw `CrossEncoder` outputs and a stable sigmoid before threshold evaluation.
2. **Untrusted web text appended to instructions:** replaced with validated URL search previews, sanitization, data-only boundaries, and extractive citations whose claims are tied to retrieved snippets.
3. **Naive rewriting damaged exact keys / pronouns:** replaced with context-explicit pronoun resolution and defensive validation of model rewrites.
4. **Re-embedding and duplicate indexing on reload:** replaced with SHA-256 document+model fingerprints, persisted Chroma collection and rebuild-on-change logic.

The deterministic test suite checks these corrected behaviors, but it **does not prove a production-grade jailbreak guarantee** or eliminate the need for live model integration tests.

## 7. Verified QA and limitations

```bash
uv run pytest -v
# Packaged evidence: 28 passed, 0 failed (deterministic tests, updated source package)

# Live integration checks (run locally; requires models and web access)
uv run python -m scripts.run_smoke
# Writes results/live_smoke_results.json
```

| Test focus | Verification status |
|---|---|
| Pronoun rewriting, empty input, exact identifiers, local timeout | Deterministic automated tests passed |
| Sigmoid `-3/0/+3`, extreme values and threshold routing | Deterministic automated tests passed |
| Cross-encoder pairwise reorder using known stub logits | Deterministic automated tests passed |
| Internal route + document chunk citation | Deterministic automated tests passed |
| Low-confidence and freshness-triggered web routing + URL citations | Deterministic automated tests passed with web test doubles |
| Prompt injection, invalid URLs, empty web evidence, unsupported LLM draft | Deterministic automated tests passed |
| Ingestion, code fences, stable source IDs, cache invalidation fingerprint | Deterministic automated tests passed |
| Real MiniLM + Chroma corpus indexing | Local Mac calibration invoked the actual retriever; independently verify full UI/search outputs |
| Real MS MARCO score distribution and empirical calibration | **Executed on user’s Mac; gate selected at 0.35 (19/20 on calibration set)** |
| Live DDGS result and clickable citation link | **Needs a visible live fallback/citation check in the recorded demo** |
| Local FLAN-T5 response quality and Streamlit UI | Streamlit launched on user’s Mac; generation/citation quality still requires review |

Practical caveats: (a) remote web search is best effort and may be rate-limited; (b) snippet citation is not verification of a full external article; (c) CrossEncoder logits are not calibrated likelihoods; (d) regex-based injection filtering is defense in depth, not an absolute prevention mechanism; (e) this prototype has no authentication, adversarial web crawling or commercial SLA.

## 8. Mac launch fix and source-code update

The original `src/app.py` produced `ModuleNotFoundError: No module named 'src'` when launched directly as a Streamlit entrypoint. The updated file now inserts the repository root into `sys.path` **before importing from `src`**. `src/config.py` also resolves relative `docs/`, `.rag_index/`, and `.env` against the repository root, regardless of the shell's current directory. The locally selected threshold is the new source default (`0.35`), while existing `.env` overrides are respected. `tests/test_config.py` adds two regression tests for the calibrated default and path handling.



### References

- [Sentence Transformers: CrossEncoder scores and sigmoid](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
- [Chroma database documentation](https://docs.trychroma.com/)
- [DDGS package and text search API](https://pypi.org/project/ddgs/)
- [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/)

*This project is an educational prototype; use live-calibrated measurements and security review before production deployment.*
