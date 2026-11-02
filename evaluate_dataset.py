"""Evaluate every GOOD and BAD photo. Do not hide failures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_all, format_table  # noqa: E402


def main() -> int:
    report = evaluate_all()
    print(format_table(report))
    print()
    print(
        "Proof-of-concept inspection system. Current sample count is too small "
        "for production validation. Production deployment requires controlled "
        "imaging conditions and a significantly larger validated dataset."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
