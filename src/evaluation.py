"""Run the reference inspection pipeline on every dataset image."""

from __future__ import annotations

from pathlib import Path

from src.config import STATUS_FAIL, STATUS_PASS, STATUS_REVIEW, list_dataset_images
from src.inspection import inspect_path


def actual_label(split: str) -> str:
    return "GOOD" if split == "good" else "BAD"


def predicted_label(status: str) -> str:
    if status == STATUS_PASS:
        return "GOOD"
    if status == STATUS_FAIL:
        return "BAD"
    return "REVIEW"


def evaluate_all(params: dict | None = None) -> dict:
    rows = []
    images = list_dataset_images()
    for split, paths in (("good", images["good"]), ("bad", images["not_good"])):
        for path in paths:
            r = inspect_path(path, params=params)
            pred = predicted_label(r["status"])
            rows.append(
                {
                    "filename": path.name,
                    "actual": actual_label(split),
                    "predicted": pred,
                    "status": r["status"],
                    "score": float(r.get("score") or 0),
                    "match_score": float(r["match_score"]),
                    "best_reference": r.get("best_reference") or "-",
                    "nearest_label": r.get("nearest_label") or "-",
                    "why": r.get("why") or "",
                    "n_blobs": int(r.get("n_blobs") or 0),
                    "mean_ssim": float(r.get("mean_ssim") or 0),
                }
            )
    good_ok = sum(1 for x in rows if x["actual"] == "GOOD" and x["predicted"] == "GOOD")
    good_bad = sum(1 for x in rows if x["actual"] == "GOOD" and x["predicted"] == "BAD")
    bad_ok = sum(1 for x in rows if x["actual"] == "BAD" and x["predicted"] == "BAD")
    bad_good = sum(1 for x in rows if x["actual"] == "BAD" and x["predicted"] == "GOOD")
    review = sum(1 for x in rows if x["predicted"] == "REVIEW")
    return {
        "rows": rows,
        "good_correctly_accepted": good_ok,
        "good_incorrectly_rejected": good_bad,
        "bad_correctly_rejected": bad_ok,
        "bad_incorrectly_accepted": bad_good,
        "review_count": review,
        "n": len(rows),
    }


def format_table(report: dict) -> str:
    lines = ["Filename | Actual | Predicted | Score | Nearest photo"]
    lines.append("-" * 72)
    for r in report["rows"]:
        lines.append(
            f"{r['filename']} | {r['actual']} | {r['predicted']} | {r['score']:.2f} | {r['best_reference'] or '-'}"
        )
    lines.append("")
    lines.append(f"Good correctly accepted:     {report['good_correctly_accepted']}")
    lines.append(f"Good incorrectly rejected:   {report['good_incorrectly_rejected']}")
    lines.append(f"Bad correctly rejected:      {report['bad_correctly_rejected']}")
    lines.append(f"Bad incorrectly accepted:    {report['bad_incorrectly_accepted']}")
    lines.append(f"Review count:                {report['review_count']}")
    return "\n".join(lines)
