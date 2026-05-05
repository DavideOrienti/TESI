"""
Seed globale per la riproducibilità degli esperimenti.

Importare RANDOM_STATE da questo modulo in tutti gli script
che richiedono un seed deterministico.
"""

RANDOM_STATE = 42
"""
Seed globale usato in tutti gli esperimenti per garantire riproducibilità.

Usato in:
- TruncatedSVD (Fase_1/10_matrix_factorization.py, random_state=RANDOM_STATE)
- Eventuali split random future
- Inizializzazioni di modelli stocastici

Riferimento:
    Importare con: from src.config.random_state import RANDOM_STATE
"""
