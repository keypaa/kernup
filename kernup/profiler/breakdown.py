"""Operation time breakdown stage for profile flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BreakdownSummary:
    bottlenecks: dict[str, float]


def run_breakdown(dry_run: bool) -> BreakdownSummary:
    """Return placeholder bottleneck percentages in dry-run mode."""
    if dry_run:
        return BreakdownSummary(
            bottlenecks={
                "linear_projections": 0.0,
                "attention": 0.0,
                "rmsnorm": 0.0,
                "other": 0.0,
            }
        )

    raise NotImplementedError("Real profiler breakdown is not implemented yet")
