"""Empirically calibrate the gate without firing web requests or inventing labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from src.calibration import choose_threshold
from src.factory import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate sigmoid transformed cross-encoder gate")
    parser.add_argument("--dataset", default="data/eval_cases.json")
    args = parser.parse_args()
    labeled = json.loads(Path(args.dataset).read_text(encoding="utf-8"))["cases"]
    engine = build_pipeline()
    observations = []
    print(f"Evaluating {len(labeled)} ground-truth route labels (no web requests)...")
    for case in labeled:
        candidates = engine.retriever.search(case["query"], k=engine.k1)
        reranked = engine.reranker.rerank(case["query"], candidates, k2=engine.k2)
        score = reranked[0].score if reranked else 0.0
        observations.append({"id": case["id"], "query": case["query"],
                             "expected": case["route"], "score": round(score, 6)})
        print(f"{case['id']:>3} {score:.3f} {case['route']}: {case['query']}")
    choices = [round(x / 100, 2) for x in range(35, 96, 5)]
    calibration = choose_threshold(observations, choices)
    result = {"model": engine.reranker.model.model_name_or_path if hasattr(engine.reranker.model, "model_name_or_path") else "ms-marco-MiniLM-L-6-v2",
              "method": "sigmoid of raw logits; not a calibrated posterior probability",
              "costs": {"wrong_internal": 3, "unnecessary_web": 1},
              "observations": observations, **calibration}
    output = Path("results/calibration_results.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSelected gate={calibration['selected']['threshold']:.2f}; details saved to {output}")
    print("Update CONFIDENCE_THRESHOLD in .env and retest with a held-out labeled set.")

if __name__ == "__main__":
    main()
