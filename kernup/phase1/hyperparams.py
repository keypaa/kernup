"""Hyperparameter definitions and mutation helpers for phase 1."""

from __future__ import annotations

from dataclasses import dataclass
import random

BLOCK_SIZE_VALUES = (16, 32, 64, 128, 256)
NUM_WARPS_VALUES = (1, 2, 4, 8, 16)
NUM_STAGES_VALUES = (1, 2, 3, 4, 5)
KV_STRATEGY_VALUES = ("sliding", "full", "chunked")
PADDING_VALUES = (64, 128, 256)
SPLIT_K_VALUES = (1, 2, 4, 8)
PREFETCH_VALUES = (1, 2, 3)
L2_HINT_VALUES = ("evict_first", "evict_last", "none")


@dataclass(frozen=True)
class HyperparameterConfig:
    block_size: int
    num_warps: int
    num_stages: int
    kv_strategy: str
    tensor_padding: int
    split_k: int
    prefetch_distance: int
    l2_cache_hint: str

    def as_dict(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "kv_strategy": self.kv_strategy,
            "tensor_padding": self.tensor_padding,
            "split_k": self.split_k,
            "prefetch_distance": self.prefetch_distance,
            "l2_cache_hint": self.l2_cache_hint,
        }


def random_config(rng: random.Random) -> HyperparameterConfig:
    return HyperparameterConfig(
        block_size=rng.choice(BLOCK_SIZE_VALUES),
        num_warps=rng.choice(NUM_WARPS_VALUES),
        num_stages=rng.choice(NUM_STAGES_VALUES),
        kv_strategy=rng.choice(KV_STRATEGY_VALUES),
        tensor_padding=rng.choice(PADDING_VALUES),
        split_k=rng.choice(SPLIT_K_VALUES),
        prefetch_distance=rng.choice(PREFETCH_VALUES),
        l2_cache_hint=rng.choice(L2_HINT_VALUES),
    )


def validate_config(config: HyperparameterConfig) -> None:
    if config.block_size not in BLOCK_SIZE_VALUES:
        raise ValueError("Invalid block_size")
    if config.num_warps not in NUM_WARPS_VALUES:
        raise ValueError("Invalid num_warps")
    if config.num_stages not in NUM_STAGES_VALUES:
        raise ValueError("Invalid num_stages")
    if config.kv_strategy not in KV_STRATEGY_VALUES:
        raise ValueError("Invalid kv_strategy")
    if config.tensor_padding not in PADDING_VALUES:
        raise ValueError("Invalid tensor_padding")
    if config.split_k not in SPLIT_K_VALUES:
        raise ValueError("Invalid split_k")
    if config.prefetch_distance not in PREFETCH_VALUES:
        raise ValueError("Invalid prefetch_distance")
    if config.l2_cache_hint not in L2_HINT_VALUES:
        raise ValueError("Invalid l2_cache_hint")


def create_initial_population(population_size: int, rng: random.Random) -> list[HyperparameterConfig]:
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    return [random_config(rng) for _ in range(population_size)]


def mutate_config(parent: HyperparameterConfig, rng: random.Random) -> HyperparameterConfig:
    fields = list(parent.as_dict().keys())
    selected = rng.choice(fields)
    values = parent.as_dict()

    if selected == "block_size":
        values[selected] = rng.choice(BLOCK_SIZE_VALUES)
    elif selected == "num_warps":
        values[selected] = rng.choice(NUM_WARPS_VALUES)
    elif selected == "num_stages":
        values[selected] = rng.choice(NUM_STAGES_VALUES)
    elif selected == "kv_strategy":
        values[selected] = rng.choice(KV_STRATEGY_VALUES)
    elif selected == "tensor_padding":
        values[selected] = rng.choice(PADDING_VALUES)
    elif selected == "split_k":
        values[selected] = rng.choice(SPLIT_K_VALUES)
    elif selected == "prefetch_distance":
        values[selected] = rng.choice(PREFETCH_VALUES)
    else:
        values[selected] = rng.choice(L2_HINT_VALUES)

    child = HyperparameterConfig(**values)
    validate_config(child)
    return child
