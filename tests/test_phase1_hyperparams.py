from dataclasses import replace
import random

import pytest

from kernup.phase1.hyperparams import (
    HyperparameterConfig,
    create_initial_population,
    mutate_config,
    validate_config,
)


def _sample_config() -> HyperparameterConfig:
    return HyperparameterConfig(
        block_size=64,
        num_warps=4,
        num_stages=2,
        kv_strategy="full",
        tensor_padding=128,
        split_k=2,
        prefetch_distance=2,
        l2_cache_hint="none",
    )


def test_validate_config_accepts_valid_values() -> None:
    validate_config(_sample_config())


def test_validate_config_rejects_invalid_block_size() -> None:
    bad = replace(_sample_config(), block_size=999)
    with pytest.raises(ValueError):
        validate_config(bad)


def test_create_population_respects_requested_size() -> None:
    population = create_initial_population(6, random.Random(7))
    assert len(population) == 6


def test_mutate_config_returns_valid_config() -> None:
    rng = random.Random(11)
    mutated = mutate_config(_sample_config(), rng)
    validate_config(mutated)
