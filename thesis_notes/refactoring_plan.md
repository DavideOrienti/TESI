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

## Fase 3 - Evaluation Framework

Creare `src/evaluation` con:

- split per utente;
- metriche top-N;
- runner comparativo;
- salvataggio summary;
- coverage e ablation study.

Impatto tesi: rafforza la validita sperimentale.

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

Estrarre da `src/webapp/backend/recommender_service.py` la logica hybrid riusabile in `src/recommenders/hybrid.py`, mantenendo il backend come chiamante. Questo riduce duplicazione e collega direttamente sperimentazione offline e web app.

