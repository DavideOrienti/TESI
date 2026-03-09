from __future__ import annotations
import hashlib

def normalize_name(name: str) -> str:
    name = str(name).strip()
    name = " ".join(name.split())
    return name

def stable_int_id(namespace: str, value: str) -> int:
    """
    ID deterministico (32-bit) da stringa.
    Nota: collisioni possibili ma rare su questa scala; per essere più sicuri puoi usare 64-bit.
    """
    s = f"{namespace}::{normalize_name(value).lower()}".encode("utf-8")
    h = hashlib.sha1(s).hexdigest()[:8]  # 32-bit
    return int(h, 16)