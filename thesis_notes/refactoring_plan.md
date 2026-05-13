# Piano Di Refactoring Per La Tesi

## Obiettivo

Trasformare il progetto da prototipo sperimentale a repository modulare, riproducibile e difendibile in una tesi magistrale.

## Fase 1 - Documentazione E Riproducibilita

- creare README principale;
- aggiungere `.env.example`;
- documentare architettura, pipeline e recommender;
- separare dipendenze base e ML opzionali;
- rafforzare `.gitignore`.

Impatto tesi: supporta capitoli su architettura, setup sperimentale e riproducibilita.

## Fase 2 - Recommender Core

Estrarre logica comune in:

```text
src/recommenders/popularity.py
src/recommenders/collaborative.py
src/recommenders/content_based.py
src/recommenders/hybrid.py
```

Gli script sperimentali dovranno importare questi moduli invece di duplicare la logica.

Impatto tesi: rende chiara la distinzione tra modelli confrontati.

Stato: completata per il primo nucleo.

Creati:

- `src/recommenders/popularity.py`;
- `src/recommenders/collaborative.py`;
- `src/recommenders/content_based.py`;
- `src/recommenders/hybrid.py`.

Il backend usa gia questi moduli per recommendation serving, fallback popularity, SVD score, scoring content e ranking hybrid.

Script offline riallineati:

- `src/pipeline/Fase_1/08_popularity_baseline.py`;
- `src/pipeline/Fase_1/09b_itemknn_cf_improved.py`;
- `src/pipeline/Fase_1/10_matrix_factorization.py`.
- `src/pipeline/Fase_2a/14a_evaluate_content_baseline_v2.py`.
- `src/pipeline/Fase_3/15_hybrid_weighted_eval.py`;
- `src/pipeline/Fase_3/17_hybrid_svd_content_eval.py`;
- `src/pipeline/Fase_4/18c_evaluate_content_ablation.py`.

## Fase 3 - Evaluation Framework

Creare `src/evaluation` con:

- split per utente;
- metriche top-N;
- runner comparativo;
- salvataggio summary;
- coverage e ablation study.

Impatto tesi: rafforza la validita sperimentale.

Stato: parzialmente completata.

Creati:

- `src/evaluation/metrics.py`;
- `src/utils/eval.py` come shim compatibile.

Implementate metriche top-N, coverage, novelty, RMSE e MAE. Da completare: runner comparativo unico, manifest esperimenti e salvataggio standardizzato dei risultati.

## Fase 4 - Data Pipeline E TMDB

Separare:

- cleaning MovieLens;
- merge tag/link;
- client TMDB;
- parser TMDB;
- build people tables.

Impatto tesi: rende il dataset arricchito un contributo tracciabile.

## Fase 5 - Backend

Separare route, service e repository:

```text
src/webapp/backend/routes/
src/webapp/backend/services/
src/webapp/backend/repositories/
```

Spostare import CSV e caricamento artefatti fuori dai punti critici dell'avvio applicazione.

Impatto tesi: migliora la qualita ingegneristica del prototipo.

## Fase 6 - Explainability

Creare spiegazioni deterministiche e post-hoc, poi mantenere LLM come strato opzionale.

Impatto tesi: permette di discutere trasparenza, fiducia e trade-off.

## Fase 7 - Multimodalita

Isolare poster, CLIP e phash in `src/multimodal`.

Impatto tesi: prepara sviluppi futuri senza appesantire il core del sistema.

## Primo Refactoring Tecnico Consigliato

Completato: la logica hybrid e stata estratta in `src/recommenders/hybrid.py`, insieme ai moduli core per popularity, collaborative, content-based ed evaluation.

## Prossimo Refactoring Tecnico Consigliato

Aggiornare progressivamente gli script sperimentali rimasti in `Fase_3` e `Fase_4`, con priorita a:

- `Fase_3/16_hybrid_with_popularity_eval.py`, per consolidare il confronto con popularity;
- `Fase_4/24_coverage_novelty_eval.py`, per centralizzare metriche beyond-accuracy;
- `Fase_4/25_coldstart_analysis.py`, per rendere piu chiara l'analisi cold-start.

## Test Introdotti

La prima suite di regressione copre:

- popularity baseline e fallback;
- collaborative filtering helper;
- content-based neighbors e ranking;
- hybrid ranking;
- metriche di evaluation.

Comando:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_popularity tests.test_collaborative tests.test_content_based tests.test_metrics tests.test_hybrid
```
