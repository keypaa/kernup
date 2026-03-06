"""Self-healing retries for phase 2 validation failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HealAttempt:
    attempt: int
    success: bool
    code: str
    error: str | None = None


def heal_with_retries(
    initial_code: str,
    validator: Callable[[str], tuple[bool, str | None]],
    fixer: Callable[[str, str], str],
    max_attempts: int = 3,
) -> list[HealAttempt]:
    """Try to repair invalid kernel code using validator feedback."""
    attempts: list[HealAttempt] = []
    code = initial_code

    ok, err = validator(code)
    attempts.append(HealAttempt(attempt=0, success=ok, code=code, error=err))
    if ok:
        return attempts

    for idx in range(1, max_attempts + 1):
        if err is None:
            err = "Unknown validation error"
        code = fixer(code, err)
        ok, err = validator(code)
        attempts.append(HealAttempt(attempt=idx, success=ok, code=code, error=err))
        if ok:
            break

    return attempts
