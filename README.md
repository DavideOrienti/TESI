# CineRec — Sistema di Raccomandazione Film

Tesi magistrale in Ingegneria Informatica. CineRec è un sistema di raccomandazione film che integra il dataset MovieLens con metadati arricchiti via API TMDB. Il progetto comprende una pipeline sperimentale offline che confronta modelli di Collaborative Filtering, Content-Based e ibridi, e una web application interattiva con registrazione utente, raccomandazioni personalizzate, ricerca semantica e ricerca visuale tramite poster.

## Web App

| Servizio | URL |
|----------|-----|
| Frontend (Vercel) | https://tesi-ten.vercel.app |
| Backend API (Render) | https://cinerec-backend.onrender.com |

> Il backend su Render usa il piano gratuito: la prima richiesta dopo inattività può richiedere 30–60 secondi (cold start).

---

## Struttura del Repository

```
CineRec/
├── src/
│   ├── pipeline/           # Pipeline sperimentale offline (Fase_0 – Fase_5)
│   ├── webapp/
│   │   ├── backend/        # API Flask (Python 3.11)
│   │   └── frontend/       # Interfaccia React + Vite
│   ├── recommenders/       # Moduli di scoring (CF, content, hybrid, popularity)
│   ├── evaluation/         # Metriche top-N (metrics.py)
│   ├── utils/              # Utilità condivise (tmdb_client, io, ids, eval)
│   ├── experiments/        # Script di benchmark su dataset di scala diversa
│   └── config/             # Configurazione pipeline (settings.yaml, random_state.py)
├── data/
│   ├── raw/                # Dataset MovieLens originali (non versionati)
│   ├── processed/          # Output intermedi della pipeline (non versionati)
│   ├── deploy_artifacts/   # Artefatti pre-calcolati per la web app (versionati)
│   └── posters/            # Cache poster TMDB (non versionata)
├── tests/                  # Suite di test unitari (unittest)
├── notebooks/              # Notebook esplorativi storici (non parte della pipeline)
├── docs/                   # Documentazione architetturale
├── grafici/                # Figure generate dagli esperimenti
├── requirements.txt        # Dipendenze web app e pipeline base
├── requirements-ml.txt     # Dipendenze aggiuntive per la pipeline ML completa
├── run_backend.py          # Entry point backend locale
└── Procfile                # Comando gunicorn per Render
```

> In `data/deploy_artifacts/` i file `clip_checkpoint.npy`, `clip_checkpoint_idx.json` e `predicted_scores_matrix.csv` sono esclusi da git; vanno generati con la Fase_5.

---

## Requisiti

- **Python**: 3.11.9 (`.python-version` nella root)
- **Node.js**: LTS recente (per il frontend)
- **npm**: incluso con Node.js

### Dipendenze Python principali

| Pacchetto | Versione | Uso |
|-----------|----------|-----|
| flask | 3.1.0 | Web API backend |
| flask-sqlalchemy | 3.1.1 | ORM database |
| PyJWT | 2.10.1 | Autenticazione JWT |
| pandas | 2.1.3 | Manipolazione dati |
| numpy | 1.26.4 | Algebra lineare |
| scikit-learn | 1.3.2 | Modelli ML |
| pyarrow | 19.0.0 | I/O Parquet |
| onnxruntime | 1.26.0 | Inferenza modello MiniLM (ONNX) |
| tokenizers | 0.19.1 | Tokenizzazione testo |
| gunicorn | 23.0.0 | WSGI server produzione |
| psycopg2-binary | 2.9.9 | Connettore PostgreSQL (produzione) |
| groq | 1.2.0 | LLM per chat e spiegazioni |
| Pillow | 10.3.0 | Elaborazione immagini |
| imagehash | 4.3.1 | Perceptual hashing poster |

Dipendenze aggiuntive per la pipeline ML (`requirements-ml.txt`):
- `sentence-transformers==2.7.0` — generazione embedding testuali
- `scipy==1.11.4`
- CLIP — installazione manuale (vedi sezione Pipeline)

---

## Installazione

```bash
# 1. Clonare il repository
git clone <url-repository>
cd TESI

# 2. Creare e attivare il virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Installare le dipendenze base (necessarie per la web app)
pip install -r requirements.txt

# 4. (Solo pipeline ML) installare le dipendenze aggiuntive
pip install -r requirements-ml.txt

# 5. (Solo Fase_5 — embedding visuale) installare PyTorch e CLIP
#    Installare prima PyTorch seguendo https://pytorch.org/get-started/locally/
pip install git+https://github.com/openai/CLIP.git
```

---

## Dataset e Artefatti

### Per la web app

Gli artefatti pre-calcolati sono già versionati in `data/deploy_artifacts/`. Non è necessario scaricare dataset né rieseguire la pipeline per avviare l'applicazione in locale.

```
data/deploy_artifacts/
├── movies_enriched_tmdb.csv       # Catalogo film arricchito (8.7 MB)
├── ratings_train.csv              # Rating di training (1.95 MB)
├── svd_matrix.parquet             # Fattori SVD (6.2 MB)
├── content_top_neighbors_v2.csv   # Vicinanze content-based (13.8 MB)
├── movie_embeddings_index_v2.csv  # Indice embedding film (0.25 MB)
├── search_embeddings_index.csv    # Indice TF-IDF (0.25 MB)
├── search_embeddings_minilm.npy   # Embedding MiniLM ONNX (14.3 MB)
├── poster_phash_index.csv         # Indice perceptual hash poster (0.84 MB)
├── poster_embeddings_index.csv    # Indice embedding CLIP (0.25 MB)
└── minilm_onnx/                   # Modello ONNX per ricerca semantica
```

> `clip_checkpoint.npy` e `poster_embeddings_clip.npy` non sono versionati. La ricerca visuale via CLIP richiede di eseguire la Fase_5 per generarli.

### Per riprodurre la pipeline sperimentale

Scaricare il dataset MovieLens e posizionarlo in `data/raw/`:

| Dataset | URL | Cartella attesa |
|---------|-----|-----------------|
| MovieLens Latest Small | https://grouplens.org/datasets/movielens/latest/ | `data/raw/ml-latest-small/` |
| MovieLens 20M | https://grouplens.org/datasets/movielens/20m/ | `data/raw/ml-20m/` |
| MovieLens 25M | https://grouplens.org/datasets/movielens/25m/ | `data/raw/ml-25m/` |

> L'arricchimento TMDB (Fase_0) richiede una chiave API gratuita: https://www.themoviedb.org/settings/api

---

## Configurazione

Creare un file `.env` nella root del progetto copiando da `.env.example`:

```bash
cp .env.example .env
```

### Variabili d'ambiente backend

| Variabile | Obbligatoria | Default nel codice | Descrizione |
|-----------|-------------|-------------------|-------------|
| `SECRET_KEY` | Sì (in produzione) | `dev-secret-change-me` | Chiave Flask per sessioni |
| `JWT_SECRET_KEY` | Sì (in produzione) | `jwt-dev-secret` | Chiave di firma token JWT |
| `DATABASE_URL` | No | SQLite locale | URL PostgreSQL in produzione |
| `FRONTEND_URL` | No | `http://localhost:5173` | Origine aggiuntiva consentita dal CORS |
| `ADMIN_KEY` | No | `admin-dev-key` | Chiave per `/api/admin/stats` |
| `GROQ_API_KEY` | No | — | Chiave Groq per chat e spiegazioni LLM |
| `RENDER` | No | — | Impostare a `true` su Render |
| `APP_BASE_DIR` | No | root del progetto | Override percorso base artefatti |
| `TMDB_API_KEY` | Solo pipeline | — | API TMDB (Fase_0 offline; non usata dalla web app) |

### Esempio `.env` per sviluppo locale

```dotenv
SECRET_KEY=cambia-questo-valore
JWT_SECRET_KEY=cambia-anche-questo
DATABASE_URL=
FRONTEND_URL=http://localhost:5173
ADMIN_KEY=cambia-admin-key
GROQ_API_KEY=
TMDB_API_KEY=la_tua_chiave_tmdb
```

---

## Pipeline Sperimentale

La pipeline non ha un orchestratore unico. Gli script si eseguono manualmente in ordine numerico con:

```bash
python -m src.pipeline.<Fase_X>.<NN_nome_script>
```

La configurazione globale si trova in `src/config/settings.yaml` (dataset attivo, soglie di filtraggio, rate-limit TMDB).

### Fase_0 — Data Engineering (script 01–05)

**Input**: `data/raw/ml-latest-small/`

```bash
python -m src.pipeline.Fase_0.01_clean_movielens
python -m src.pipeline.Fase_0.02_merge_tags
python -m src.pipeline.Fase_0.02b_merge_links
python -m src.pipeline.Fase_0.03_enrich_tmdb        # richiede TMDB_API_KEY
python -m src.pipeline.Fase_0.04_build_people_tables
python -m src.pipeline.Fase_0.05_data_audit
```

**Output principale**: `data/processed/small/movies_enriched_tmdb.csv`

### Fase_1 — Collaborative Filtering Baselines (script 06–10)

```bash
python -m src.pipeline.Fase_1.06_prepare_ratings
python -m src.pipeline.Fase_1.07_train_test_split
python -m src.pipeline.Fase_1.08_popularity_baseline
python -m src.pipeline.Fase_1.09b_itemknn_cf_improved
python -m src.pipeline.Fase_1.10_matrix_factorization
```

**Output**: split train/val/test, similarity matrix ItemKNN, fattori SVD

### Fase_2a — Embedding Content-Based (script 11a–14a)

```bash
python -m src.pipeline.Fase_2a.11a_build_movie_embeddings_v2
python -m src.pipeline.Fase_2a.12a_build_similarity_matrix_v2
python -m src.pipeline.Fase_2a.13a_test_content_recommender_v2
python -m src.pipeline.Fase_2a.14a_evaluate_content_baseline_v2
```

**Output**: `movie_embeddings_v2.npy`, `content_top_neighbors_v2.csv`

### Fase_3 — Modelli Ibridi (script 15–17)

```bash
python -m src.pipeline.Fase_3.15_hybrid_weighted_eval
python -m src.pipeline.Fase_3.16_hybrid_with_popularity_eval
python -m src.pipeline.Fase_3.17_hybrid_svd_content_eval
```

**Output**: tabelle metriche top-N per ogni combinazione di iperparametri

### Fase_4 — Analisi e Ablation Study (script 17a–25)

```bash
python -m src.pipeline.Fase_4.17a_user_segment_analysis
python -m src.pipeline.Fase_4.17b_item_popularity_analysis
python -m src.pipeline.Fase_4.18a_build_movie_embeddings_ablation
python -m src.pipeline.Fase_4.18b_build_content_neighbors_ablation
python -m src.pipeline.Fase_4.18c_evaluate_content_ablation
python -m src.pipeline.Fase_4.24_coverage_novelty_eval
python -m src.pipeline.Fase_4.25_coldstart_analysis
```

### Fase_5 — Multimodale e Ricerca Visuale (script 20–23)

Richiede PyTorch e CLIP installati (vedi Installazione, punto 5).

```bash
python -m src.pipeline.Fase_5.20_build_search_embeddings
python -m src.pipeline.Fase_5.21_download_posters       # scarica ~9600 poster da TMDB
python -m src.pipeline.Fase_5.22_build_clip_embeddings
python -m src.pipeline.Fase_5.23_build_phash_index
```

**Output**: `poster_embeddings_clip.npy`, `poster_phash_index.csv`

---

## Web App in Locale

La web app richiede **due processi separati** in terminali distinti.

### Backend — Flask (porta 5000)

```bash
# Dalla root del progetto, con .venv attivo
python run_backend.py
```

Al primo avvio il backend importa automaticamente il catalogo film da `data/deploy_artifacts/movies_enriched_tmdb.csv` se il database è vuoto. I log di caricamento modelli usano i prefissi `[search]`, `[visual]`, `[recommender_service]`.

### Frontend — Vite (porta 5173)

```bash
cd src/webapp/frontend
npm install
npm run dev
```

Aprire il browser su **http://localhost:5173**. Il frontend comunica con il backend su `http://localhost:5000`.

### Comando di avvio per produzione (Render)

```bash
gunicorn "src.webapp.backend.app:create_app()" --bind "0.0.0.0:$PORT" --workers 1 --timeout 120
```

---

## Test

```bash
# Tutti i test
python -m unittest discover -s tests -p "test_*.py"

# Singolo modulo
python -m unittest tests.test_collaborative
```

I cinque moduli coprono: `test_collaborative`, `test_content_based`, `test_hybrid`, `test_metrics`, `test_popularity`. Framework: `unittest` (libreria standard Python, nessuna dipendenza aggiuntiva).

---

## Risultati e Grafici

Le figure generate dagli esperimenti vengono salvate in `grafici/`. Le tabelle di metriche (Hit Rate, Precision@K, Recall@K, NDCG@K, MRR@K) sono prodotte da ogni script di evaluation nelle rispettive fasi e stampate su stdout o salvate in `data/processed/`.

---

## Note sui Notebook

I notebook in `notebooks/exploration/` sono materiale esplorativo storico usato durante lo sviluppo della tesi. Non fanno parte della pipeline riproducibile e non sono eseguibili nella configurazione attuale (dipendenze incompatibili con l'ambiente corrente). Tutto il codice valutabile si trova in `src/pipeline/`.

---

## AVVISI SICUREZZA

I seguenti problemi devono essere risolti prima di rendere pubblico il repository.

| Severità | Problema | File | Azione richiesta |
|----------|----------|------|-----------------|
| **CRITICO** | `.env` tracciato in git con chiave TMDB reale | `.env` | Aggiungere `.env` a `.gitignore`; revocare la chiave su themoviedb.org e generarne una nuova |
| **CRITICO** | Chiave TMDB reale usata come placeholder | `src/webapp/.env.example` | Sostituire il valore con `your_tmdb_api_key_here` |
| **MEDIO** | Default insicuri hardcoded nel sorgente | `src/webapp/backend/config.py`, `admin.py` | Impostare `SECRET_KEY`, `JWT_SECRET_KEY` e `ADMIN_KEY` tramite variabili d'ambiente in ogni ambiente di deploy |
| **MEDIO** | Database SQLite con dati di test versionato | `src/webapp/backend/database.db` | Aggiungere `src/webapp/backend/database.db` a `.gitignore` |

> La chiave TMDB esposta è già nella git history. Anche dopo la rimozione dal file, va invalidata e rigenerata su themoviedb.org.
