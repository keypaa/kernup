"""Genetic search loop for phase 1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from kernup.phase1.hyperparams import HyperparameterConfig, create_initial_population, mutate_config
from kernup.phase1.scoring import score_config
from kernup.types import KernelScore


@dataclass(frozen=True)
class Phase1SearchResult:
    best: KernelScore
    best_config: HyperparameterConfig
    history_best_tok_s: list[float]
    evaluations: list[KernelScore]
    stopped_on_plateau: bool


def _objective(score: KernelScore, target: str) -> float:
    if target == "latency":
        return -score.latency_ms
    if target == "balanced":
        return score.tok_s - (score.latency_ms / 20.0)
    return score.tok_s


def _tournament_select(
    pool: list[tuple[HyperparameterConfig, KernelScore]],
    rng: random.Random,
    target: str,
    k: int = 3,
) -> HyperparameterConfig:
    candidates = [rng.choice(pool) for _ in range(k)]
    winner = max(candidates, key=lambda pair: _objective(pair[1], target))
    return winner[0]


def run_phase1_search(
    iterations: int,
    population: int,
    plateau_window: int,
    plateau_threshold: float,
    target: str,
    cache_dir: Path,
    gpu_compute_capability: str,
    dry_run: bool,
    hf_model: str = "",
    prompt_text: str = "Write a short summary of GPU kernel optimization best practices.",
    max_new_tokens: int = 32,
    warmup_runs: int = 1,
    measure_runs: int = 2,
    start_generation: int = 0,
    seed: int = 42,
) -> Phase1SearchResult:
    rng = random.Random(seed)
    configs = create_initial_population(population, rng)
    scored: list[tuple[HyperparameterConfig, KernelScore]] = []
    evaluations: list[KernelScore] = []

    for cfg in configs:
        score = score_config(
            cfg,
            generation=start_generation,
            cache_dir=cache_dir,
            target=target,
            gpu_compute_capability=gpu_compute_capability,
            dry_run=dry_run,
            hf_model=hf_model,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measure_runs=measure_runs,
        )
        scored.append((cfg, score))
        evaluations.append(score)

    best_cfg, best_score = max(scored, key=lambda pair: _objective(pair[1], target))
    history_best = [best_score.tok_s]
    plateau = False

    for generation in range(start_generation + 1, start_generation + iterations + 1):
        next_population: list[HyperparameterConfig] = [best_cfg]
        while len(next_population) < population:
            parent = _tournament_select(scored, rng, target)
            child = mutate_config(parent, rng)
            next_population.append(child)

        scored = []
        for cfg in next_population:
            score = score_config(
                cfg,
                generation=generation,
                cache_dir=cache_dir,
                target=target,
                gpu_compute_capability=gpu_compute_capability,
                dry_run=dry_run,
                hf_model=hf_model,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                warmup_runs=warmup_runs,
                measure_runs=measure_runs,
            )
            scored.append((cfg, score))
            evaluations.append(score)

        gen_best_cfg, gen_best_score = max(scored, key=lambda pair: _objective(pair[1], target))
        if _objective(gen_best_score, target) > _objective(best_score, target):
            best_cfg, best_score = gen_best_cfg, gen_best_score

        history_best.append(best_score.tok_s)

        if len(history_best) > plateau_window:
            recent = history_best[-1]
            past = history_best[-1 - plateau_window]
            improvement = 0.0 if past == 0 else (recent - past) / abs(past)
            if improvement < plateau_threshold:
                plateau = True
                break

    return Phase1SearchResult(
        best=best_score,
        best_config=best_cfg,
        history_best_tok_s=history_best,
        evaluations=evaluations,
        stopped_on_plateau=plateau,
    )
