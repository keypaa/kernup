"""Genetic search loop for phase 1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from statistics import mean, pstdev

from kernup.phase1.hyperparams import HyperparameterConfig, create_initial_population, mutate_config, random_config
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
    objective,
    k: int = 3,
) -> HyperparameterConfig:
    candidates = [rng.choice(pool) for _ in range(k)]
    winner = max(candidates, key=lambda pair: objective(pair[0], pair[1]))
    return winner[0]


def _config_key(config: HyperparameterConfig) -> str:
    return json.dumps(config.as_dict(), sort_keys=True)


def _evaluate_config(
    cfg: HyperparameterConfig,
    generation: int,
    cache_dir: Path,
    target: str,
    gpu_compute_capability: str,
    dry_run: bool,
    hf_model: str,
    prompt_text: str,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
) -> KernelScore:
    return score_config(
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


def _refine_top_configs(
    scored: list[tuple[HyperparameterConfig, KernelScore]],
    stability_penalty: dict[str, float],
    generation: int,
    top_k: int,
    cache_dir: Path,
    target: str,
    gpu_compute_capability: str,
    dry_run: bool,
    hf_model: str,
    prompt_text: str,
    max_new_tokens: int,
    warmup_runs: int,
    measure_runs: int,
    evaluations: list[KernelScore],
) -> None:
    ranked = sorted(scored, key=lambda pair: _objective(pair[1], target), reverse=True)
    for cfg, _ in ranked[:top_k]:
        samples = [
            _evaluate_config(
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
            for _ in range(2)
        ]
        evaluations.extend(samples)

        avg_score = KernelScore(
            tok_s=round(mean([s.tok_s for s in samples]), 3),
            ttft_ms=round(mean([s.ttft_ms for s in samples]), 3),
            latency_ms=round(mean([s.latency_ms for s in samples]), 3),
            vram_used_gb=round(mean([s.vram_used_gb for s in samples]), 3),
            generation=generation,
            config=cfg.as_dict(),
        )
        key = _config_key(cfg)
        stability_penalty[key] = round(pstdev([s.tok_s for s in samples]), 6)

        for idx, (candidate_cfg, _old) in enumerate(scored):
            if candidate_cfg == cfg:
                scored[idx] = (candidate_cfg, avg_score)
                break


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
    search_mode: str = "standard",
    max_refine_top_k: int = 2,
    max_refine_warmup_runs: int = 2,
    max_refine_measure_runs: int = 6,
    max_stability_penalty: float = 0.15,
    max_restarts: int = 1,
    start_generation: int = 0,
    seed: int = 42,
) -> Phase1SearchResult:
    rng = random.Random(seed)
    configs = create_initial_population(population, rng)
    scored: list[tuple[HyperparameterConfig, KernelScore]] = []
    evaluations: list[KernelScore] = []
    stability_penalty_by_cfg: dict[str, float] = {}

    def adjusted_objective(cfg: HyperparameterConfig, score: KernelScore) -> float:
        base = _objective(score, target)
        if search_mode != "max":
            return base
        key = _config_key(cfg)
        return base - (max_stability_penalty * stability_penalty_by_cfg.get(key, 0.0))

    for cfg in configs:
        score = _evaluate_config(
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

    if search_mode == "max" and not dry_run:
        _refine_top_configs(
            scored=scored,
            stability_penalty=stability_penalty_by_cfg,
            generation=start_generation,
            top_k=max(1, min(max_refine_top_k, len(scored))),
            cache_dir=cache_dir,
            target=target,
            gpu_compute_capability=gpu_compute_capability,
            dry_run=dry_run,
            hf_model=hf_model,
            prompt_text=prompt_text,
            max_new_tokens=max_new_tokens,
            warmup_runs=max_refine_warmup_runs,
            measure_runs=max_refine_measure_runs,
            evaluations=evaluations,
        )

    best_cfg, best_score = max(scored, key=lambda pair: adjusted_objective(pair[0], pair[1]))
    history_best = [best_score.tok_s]
    plateau = False
    restarts_left = max_restarts

    for generation in range(start_generation + 1, start_generation + iterations + 1):
        next_population: list[HyperparameterConfig] = [best_cfg]
        while len(next_population) < population:
            parent = _tournament_select(scored, rng, adjusted_objective)
            child = mutate_config(parent, rng)
            next_population.append(child)

        scored = []
        for cfg in next_population:
            score = _evaluate_config(
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

        if search_mode == "max" and not dry_run:
            _refine_top_configs(
                scored=scored,
                stability_penalty=stability_penalty_by_cfg,
                generation=generation,
                top_k=max(1, min(max_refine_top_k, len(scored))),
                cache_dir=cache_dir,
                target=target,
                gpu_compute_capability=gpu_compute_capability,
                dry_run=dry_run,
                hf_model=hf_model,
                prompt_text=prompt_text,
                max_new_tokens=max_new_tokens,
                warmup_runs=max_refine_warmup_runs,
                measure_runs=max_refine_measure_runs,
                evaluations=evaluations,
            )

        gen_best_cfg, gen_best_score = max(scored, key=lambda pair: adjusted_objective(pair[0], pair[1]))
        if adjusted_objective(gen_best_cfg, gen_best_score) > adjusted_objective(best_cfg, best_score):
            best_cfg, best_score = gen_best_cfg, gen_best_score

        history_best.append(best_score.tok_s)

        if len(history_best) > plateau_window:
            recent = history_best[-1]
            past = history_best[-1 - plateau_window]
            improvement = 0.0 if past == 0 else (recent - past) / abs(past)
            if improvement < plateau_threshold:
                if search_mode == "max" and restarts_left > 0:
                    restarts_left -= 1
                    configs = [best_cfg]
                    while len(configs) < population:
                        configs.append(mutate_config(best_cfg, rng) if rng.random() < 0.5 else random_config(rng))

                    scored = []
                    for cfg in configs:
                        score = _evaluate_config(
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
                    continue

                plateau = True
                break

    return Phase1SearchResult(
        best=best_score,
        best_config=best_cfg,
        history_best_tok_s=history_best,
        evaluations=evaluations,
        stopped_on_plateau=plateau,
    )
