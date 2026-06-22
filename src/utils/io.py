from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Paths:
    raw: Path
    processed: Path
    cache_tmdb: Path


@dataclass(frozen=True)
class TMDBCfg:
    api_key_env: str
    sleep_sec: float
    timeout_sec: int
    max_retries: int
    backoff_sec: float


@dataclass(frozen=True)
class FiltersCfg:
    min_user_ratings: int
    min_item_ratings: int
    iterative_filtering: bool


@dataclass(frozen=True)
class Settings:
    dataset: str
    paths: Paths
    tmdb: TMDBCfg
    clean_year_regex: str
    filters: FiltersCfg


def load_settings(settings_path: str = "src/config/settings.yaml", dataset: str | None = None) -> Settings:
    with open(settings_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = dataset or os.environ.get("CINEREC_DATASET") or cfg["dataset"]

    try:
        dataset_cfg = cfg["datasets"][dataset]
    except KeyError:
        raise ValueError(
            f"Dataset non configurato: {dataset!r}. Disponibili: {sorted(cfg['datasets'])}"
        )

    raw = Path(dataset_cfg["raw"])
    processed = Path(dataset_cfg["processed"])
    cache_tmdb = Path(cfg["paths"]["cache_tmdb"])

    processed.mkdir(parents=True, exist_ok=True)
    cache_tmdb.mkdir(parents=True, exist_ok=True)

    tmdb = TMDBCfg(
        api_key_env=cfg["tmdb"]["api_key_env"],
        sleep_sec=float(cfg["tmdb"]["sleep_sec"]),
        timeout_sec=int(cfg["tmdb"]["timeout_sec"]),
        max_retries=int(cfg["tmdb"]["max_retries"]),
        backoff_sec=float(cfg["tmdb"]["backoff_sec"]),
    )

    filters = FiltersCfg(
        min_user_ratings=int(cfg["filters"]["min_user_ratings"]),
        min_item_ratings=int(cfg["filters"]["min_item_ratings"]),
        iterative_filtering=bool(cfg["filters"]["iterative_filtering"]),
    )

    return Settings(
        dataset=dataset,
        paths=Paths(raw=raw, processed=processed, cache_tmdb=cache_tmdb),
        tmdb=tmdb,
        clean_year_regex=cfg["movies"]["clean_year_regex"],
        filters=filters,
    )