"""Optional live integration smoke test; requires downloaded models and web access."""
from __future__ import annotations
import json
from pathlib import Path
from src.factory import build_pipeline


def main() -> None:
    engine = build_pipeline()
    results = []
    for query in [
        "What is the maxUnavailable default in Rolling Update Deployment?",
        "What is the queue visibility timeout?",
        "What is the latest AWS outage announced today?",
    ]:
        response = engine.run(query)
        results.append(response.model_dump(mode="json"))
        print(f"{response.route_taken} score={response.confidence_score:.3f} query={query}")
    Path("results").mkdir(exist_ok=True)
    Path("results/live_smoke_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved results/live_smoke_results.json")

if __name__ == "__main__":
    main()
