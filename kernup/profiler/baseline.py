"""Baseline benchmark stage for profile flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from kernup.types import BaselineResult


@dataclass(frozen=True)
class BaselineResultSummary:
    seq_len_128: BaselineResult
    seq_len_512: BaselineResult
    seq_len_2048: BaselineResult

    def to_dict(self) -> dict[str, object]:
        return {
            "seq_len_128": asdict(self.seq_len_128),
            "seq_len_512": asdict(self.seq_len_512),
            "seq_len_2048": asdict(self.seq_len_2048),
        }


def run_baseline(dry_run: bool) -> BaselineResultSummary:
    """Return placeholder benchmark metrics in dry-run mode."""
    if dry_run:
        zero = BaselineResult(
            tok_s=0.0,
            ttft_ms=0.0,
            latency_ms=0.0,
            vram_used_gb=0.0,
            seq_len=0,
        )
        return BaselineResultSummary(seq_len_128=zero, seq_len_512=zero, seq_len_2048=zero)

    raise NotImplementedError("Real baseline benchmarking is not implemented yet")
