from __future__ import annotations
import hashlib

def normalize_name(name: str) -> str:
    name = str(name).strip()
    name = " ".join(name.split())
    return name

def stable_int_id(namespace: str, value: str) -> int:
    """ID deterministico 64-bit da stringa tramite SHA-256.
    Usa 64-bit (invece di 32-bit) per ridurre la probabilità di collisione
    secondo il birthday problem: sicuro fino a ~2^31 entità (~2 miliardi)."""
    s = f"{namespace}::{normalize_name(value).lower()}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()[:16]  # 64-bit
    return int(h, 16) % (2**63)             # safe positive int64