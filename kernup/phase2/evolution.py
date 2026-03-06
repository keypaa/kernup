"""Evolutionary loop for phase 2 kernel search."""

from __future__ import annotations

from dataclasses import dataclass
import random

from kernup.phase2.generator import GenerationRequest, KernelGenerator
from kernup.phase2.pipeline import Phase2PipelineResult, run_phase2_validation_pipeline


@dataclass(frozen=True)
class Phase2Evaluation:
    generation: int
    mutation_type: str
    pipeline: Phase2PipelineResult


@dataclass(frozen=True)
class Phase2EvolutionResult:
    best: Phase2Evaluation
    evaluations: list[Phase2Evaluation]
    history_best_tok_s: list[float]
    stopped_on_plateau: bool


def _objective(eval_result: Phase2Evaluation, target: str) -> float:
    bench = eval_result.pipeline.benchmark
    if bench is None:
        return 0.0
    if target == "latency":
        return -bench.latency_ms
    if target == "balanced":
        return bench.tok_s - (bench.latency_ms / 20.0)
    return bench.tok_s


def run_phase2_evolution(
    iterations: int,
    population: int,
    plateau_window: int,
    plateau_threshold: float,
    target: str,
    dry_run: bool,
    max_healing_attempts: int = 3,
    seed: int = 101,
) -> Phase2EvolutionResult:
    rng = random.Random(seed)
    generator = KernelGenerator(dry_run=dry_run, seed=seed)

    mutation_weights = [0.6, 0.3, 0.1]
    mutation_labels = ["light", "medium", "heavy"]

    def pick_mutation() -> str:
        return rng.choices(mutation_labels, weights=mutation_weights, k=1)[0]

    reference_kernel = "import triton\n\n@triton.jit\ndef kernel_seed(x):\n    return x\n"
    best_kernel = reference_kernel

    evaluations: list[Phase2Evaluation] = []

    # Generation 0
    for _ in range(population):
        mutation = pick_mutation()
        candidate = generator.generate(
            GenerationRequest(
                reference_kernel=reference_kernel,
                previous_best_kernel=best_kernel,
                mutation_type=mutation,
                generation=0,
            )
        )
        pipeline = run_phase2_validation_pipeline(
            kernel_code=candidate,
            target=target,
            dry_run=dry_run,
            max_healing_attempts=max_healing_attempts,
        )
        evaluations.append(Phase2Evaluation(generation=0, mutation_type=mutation, pipeline=pipeline))

    best = max(evaluations, key=lambda item: _objective(item, target))
    history_best = [best.pipeline.benchmark.tok_s if best.pipeline.benchmark else 0.0]

    stopped_on_plateau = False
    for gen in range(1, iterations + 1):
        if len(history_best) > 5:
            # adaptive mutation: increase heavy mutation to escape local minima
            mutation_weights = [0.5, 0.25, 0.25]

        for _ in range(population):
            mutation = pick_mutation()
            candidate = generator.generate(
                GenerationRequest(
                    reference_kernel=reference_kernel,
                    previous_best_kernel=best_kernel,
                    mutation_type=mutation,
                    generation=gen,
                )
            )
            pipeline = run_phase2_validation_pipeline(
                kernel_code=candidate,
                target=target,
                dry_run=dry_run,
                max_healing_attempts=max_healing_attempts,
            )
            current = Phase2Evaluation(generation=gen, mutation_type=mutation, pipeline=pipeline)
            evaluations.append(current)
            if _objective(current, target) > _objective(best, target):
                best = current
                best_kernel = current.pipeline.final_code

        history_best.append(best.pipeline.benchmark.tok_s if best.pipeline.benchmark else 0.0)
        if len(history_best) > plateau_window:
            current = history_best[-1]
            past = history_best[-1 - plateau_window]
            improvement = 0.0 if past == 0 else (current - past) / abs(past)
            if improvement < plateau_threshold:
                stopped_on_plateau = True
                break

    return Phase2EvolutionResult(
        best=best,
        evaluations=evaluations,
        history_best_tok_s=history_best,
        stopped_on_plateau=stopped_on_plateau,
    )
