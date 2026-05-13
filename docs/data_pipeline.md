# Data Pipeline

## Obiettivo

La pipeline dati trasforma MovieLens in un dataset arricchito con informazioni TMDB, adatto a modelli collaborative, content-based, hybrid e multimodali.

## Input

- `data/raw/ml-latest-small`
- `data/raw/ml-20m`
- API TMDB, tramite `TMDB_API_KEY`

## Output Principali

- `movies_clean.csv`
- `movies_with_tags.csv`
- `movies_with_links.csv`
- `movies_enriched_tmdb.csv`
- `people_actors.csv`
- `people_directors.csv`
- `ratings_train.csv`
- `ratings_val.csv`
- `ratings_test.csv`
- embedding e matrici di similarita
- summary sperimentali e metriche per modello

## Fasi

### Fase 0 - Data Engineering

Pulisce titoli, estrae anno, integra tag e link TMDB, arricchisce i film con overview, poster, attori, regista e popolarita.

### Fase 1 - Collaborative Filtering

Prepara rating, applica filtri minimi per utenti/item, costruisce split per utente e valuta popularity, ItemKNN e SVD.

### Fase 2 - Content-Based

Costruisce rappresentazioni testuali dei film e calcola embedding semantici e vicini piu simili.

### Fase 3 - Hybrid

Combina segnali collaborative e content-based con pesi configurabili e ricerca su validation set.

### Fase 4 - Evaluation E Ablation

Analizza prestazioni per segmenti utente/item e misura il contributo delle feature content-based.

### Fase 5 - Search E Multimodalita

Costruisce embedding per ricerca semantica, scarica poster, genera embedding CLIP e indice phash.

## Migliorie Pianificate

- spostare le funzioni comuni da script numerati a moduli riusabili;
- introdurre CLI unificata per eseguire le fasi;
- salvare manifest degli artefatti prodotti;
- documentare schema colonne per ogni CSV rilevante;
- aggiungere controlli automatici su copertura e valori mancanti.

