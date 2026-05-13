# CineRec - Movie Recommendation System

Sistema di raccomandazione per film sviluppato come progetto di tesi magistrale in Ingegneria Informatica.

Il progetto usa MovieLens come base dati principale e lo arricchisce con metadati TMDB per costruire e confrontare modelli collaborative filtering, content-based e hybrid. Include inoltre una web app dimostrativa con utenti, rating, preferiti, raccomandazioni personalizzate, spiegazioni e prime funzionalita multimodali basate su poster.

## Obiettivi

- costruire una pipeline dati riproducibile da MovieLens a dataset arricchito;
- implementare baseline e modelli di raccomandazione confrontabili;
- valutare i modelli con metriche top-N;
- esporre il sistema tramite web app;
- preparare estensioni explainable, multimodali e context-aware.

## Stato Del Progetto

Il repository contiene gia:

- pipeline offline organizzata per fasi in `src/pipeline`;
- arricchimento TMDB con cache locale;
- artefatti deploy in `data/deploy_artifacts`;
- modelli popularity, ItemKNN, SVD, content-based e hybrid;
- evaluation sperimentale e ablation study;
- backend Flask in `src/webapp/backend`;
- frontend React/Vite in `src/webapp/frontend`;
- ricerca semantica testuale e ricerca visuale tramite poster.

La fase corrente del lavoro e la trasformazione del prototipo in repository modulare, documentato e difendibile in tesi.

## Struttura Attuale

```text
TESI/
+-- data/
|   +-- raw/                  # dataset MovieLens originali, non versionati
|   +-- processed/            # output sperimentali, non versionati
|   +-- deploy_artifacts/     # artefatti usati dalla web app
|   +-- posters/              # poster scaricati da TMDB
+-- cache/tmdb/               # cache chiamate TMDB, non versionata
+-- docs/                     # documentazione tecnica
+-- notebooks/                # analisi esplorative
+-- src/
|   +-- config/               # configurazione pipeline
|   +-- pipeline/             # script offline numerati per fasi
|   +-- recommenders/         # funzioni condivise di scoring
|   +-- utils/                # utility comuni
|   +-- webapp/
|       +-- backend/          # API Flask, auth, DB, servizi recommender
|       +-- frontend/         # interfaccia React/Vite
+-- requirements.txt          # dipendenze backend/base
+-- requirements-ml.txt       # dipendenze sperimentali opzionali
+-- render.yaml               # deploy backend su Render
+-- run_backend.py            # avvio backend locale
```

## Pipeline Sperimentale

La pipeline offline e organizzata in fasi:

- `Fase_0`: pulizia MovieLens, merge tag/link, arricchimento TMDB, tabelle persone, audit dati;
- `Fase_1`: preparazione rating, split train/validation/test, popularity baseline, ItemKNN, SVD;
- `Fase_2` e `Fase_2a`: embedding testuali, similarita content-based, baseline content;
- `Fase_3`: modelli hybrid e ricerca dei pesi;
- `Fase_4`: analisi per segmenti, popolarita item, ablation study;
- `Fase_5`: ricerca semantica, poster, embedding visuali CLIP, phash.

## Setup Locale

Creare un ambiente virtuale Python 3.11:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Per eseguire gli script sperimentali con embedding o analisi avanzate:

```powershell
pip install -r requirements-ml.txt
```

Creare un file `.env` partendo da `.env.example` e valorizzare le chiavi necessarie.

## Avvio Backend

```powershell
python run_backend.py
```

Il backend legge gli artefatti in `data/deploy_artifacts`. In produzione il comando usato e definito in `Procfile` e `render.yaml`.

## Avvio Frontend

```powershell
cd src\webapp\frontend
npm install
npm run dev
```

## Componenti Di Raccomandazione

- **Popularity baseline**: ranking dei film piu popolari, usato anche come fallback cold-start.
- **Collaborative filtering**: ItemKNN e SVD su matrice user-item.
- **Content-based**: similarita tra embedding testuali costruiti con generi, tag, overview, attori e regista.
- **Hybrid**: combinazione pesata di collaborative e content-based con normalizzazione dei punteggi.
- **Explainability**: spiegazioni post-hoc basate su film simili valutati positivamente e contributi dei modelli.

## Valutazione

Il progetto include metriche top-N e risultati sperimentali in `data/processed`. Le metriche attualmente usate includono Hit Rate, Precision@K, Recall@K, NDCG@K e MRR@K. Il refactoring previsto consolidera queste funzioni in un modulo `src/evaluation`.

## Roadmap Di Refactoring

1. consolidare documentazione, dipendenze e configurazione;
2. estrarre moduli core in `src/recommenders`;
3. creare un framework evaluation riproducibile;
4. separare pipeline offline, backend services e repository layer;
5. isolare explainability e multimodalita in moduli dedicati;
6. aggiungere test mirati su metriche, scoring e filtraggio film gia visti.

## Note Per La Tesi

La tesi puo presentare il sistema su tre livelli:

- livello dati: costruzione del dataset MovieLens arricchito con TMDB;
- livello algoritmico: confronto tra baseline, collaborative, content e hybrid;
- livello applicativo: web app personalizzata con explainability e possibili estensioni multimodali.
