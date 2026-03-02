from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
import requests

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class TMDBClient:
    api_key: str
    cache_dir: Path
    timeout_sec: int
    max_retries: int
    backoff_sec: float
    sleep_sec: float

    base_url: str = "https://api.themoviedb.org/3"
    image_base: str = "https://image.tmdb.org/t/p/original"

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _get_json_cached(self, cache_name: str, url: str, params: dict) -> dict | None:
        p = self._cache_path(cache_name)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(url, params=params, timeout=self.timeout_sec)
                r.raise_for_status()
                data = r.json()
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                time.sleep(self.sleep_sec)
                return data
            except Exception as e:
                last_err = e
                time.sleep(self.backoff_sec * attempt)

        # Se fallisce: non creare cache “vuota”, così puoi riprovare in futuro
        return None

    def movie_details(self, tmdb_id: int, language: str) -> dict | None:
        url = f"{self.base_url}/movie/{tmdb_id}"
        params = {"api_key": self.api_key, "language": language}
        return self._get_json_cached(f"movie_{tmdb_id}_{language}.json", url, params)

    def movie_credits(self, tmdb_id: int) -> dict | None:
        url = f"{self.base_url}/movie/{tmdb_id}/credits"
        params = {"api_key": self.api_key}
        return self._get_json_cached(f"credits_{tmdb_id}.json", url, params)

    def poster_url_from_details(self, details: dict | None) -> str:
        if not details:
            return ""
        poster_path = details.get("poster_path") or ""
        if not poster_path:
            return ""
        return f"{self.image_base}{poster_path}"

def build_client(cache_dir: Path, api_key_env: str, timeout_sec: int, max_retries: int, backoff_sec: float, sleep_sec: float) -> TMDBClient:
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing TMDB API key. Set env var {api_key_env}.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    return TMDBClient(
        api_key=api_key,
        cache_dir=cache_dir,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        sleep_sec=sleep_sec,
    )