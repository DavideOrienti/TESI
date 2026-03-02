from __future__ import annotations
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
class Settings:
    dataset: str
    paths: Paths
    tmdb: TMDBCfg
    clean_year_regex: str

def load_settings(settings_path: str = "src/config/settings.yaml") -> Settings:
    with open(settings_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    raw = Path(cfg["paths"]["raw_small" if dataset == "small" else "raw_20m"])
    processed = Path(cfg["paths"]["processed_small" if dataset == "small" else "processed_20m"])
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

    return Settings(
        dataset=dataset,
        paths=Paths(raw=raw, processed=processed, cache_tmdb=cache_tmdb),
        tmdb=tmdb,
        clean_year_regex=cfg["movies"]["clean_year_regex"],
    )