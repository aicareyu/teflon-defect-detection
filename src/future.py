"""
Extension points for a later production inspection cell.

The live demo runs one task only: surface_black_mark on white plastic.
Other checks (Teflon presence/orientation, scratch, crack, missing component)
must be added as separate inspectors — not mixed into one classifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.config import FUTURE_DEFECT_TYPES


@dataclass
class FutureDefectTaxonomy:
    """Image-level or region-level labels that can be added later."""

    types: List[str] = field(default_factory=lambda: list(FUTURE_DEFECT_TYPES))


@dataclass
class MultiCameraStation:
    camera_ids: List[str] = field(
        default_factory=lambda: ["cam_top", "cam_side", "cam_thread"]
    )


@dataclass
class PlcIoMap:
    """Placeholder for OK/NG discrete signals."""

    ok_address: str = "Q0.0"
    ng_address: str = "Q0.1"
    ready_address: str = "I0.0"
    inhibit_on_review: bool = True


def emit_ok_ng_signal(status: str) -> Optional[str]:
    """
    Map inspection status to a future PLC write.

    PASS -> OK, FAIL -> NG, REVIEW REQUIRED -> no automatic fire (inhibit).
    """
    if status == "PASS":
        return "OK"
    if status == "FAIL":
        return "NG"
    return None


def production_counter_fields() -> dict:
    return {
        "parts_inspected": 0,
        "parts_pass": 0,
        "parts_fail": 0,
        "parts_review": 0,
        "yield_percent": None,
    }
