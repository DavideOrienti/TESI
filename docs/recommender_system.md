# Recommender System

## Obiettivo

Il sistema confronta modelli di raccomandazione top-N per film e usa il modello hybrid come base della web app.

## Modelli

### Popularity Baseline

Ordina i film per popolarita o numero/media rating. Serve come baseline sperimentale e fallback per utenti nuovi.

Implementazione core:

- `src/recommenders/popularity.py`

Responsabilita:

- costruzione ranking ponderato stile IMDb;
- conversione in score di popolarita;
- raccomandazioni popolari con esclusione dei film gia visti;
- ranking da metadati TMDB per il backend.

### Collaborative Filtering

Sono presenti due linee:

- **ItemKNN**: similarita item-item su matrice user-item, con rating centrati sulla media utente;
- **SVD**: fattorizzazione della matrice user-item e predizione dei punteggi utente-film.

Implementazione core:

- `src/recommenders/collaborative.py`
- `src/recommenders/scoring.py`

Responsabilita:

- costruzione dizionari di vicini ItemKNN;
- scoring item-item collaborativo;
- score SVD per utente;
- fallback per utenti cold-start o item assenti dalla matrice;
- ranking collaborativo con fallback popularity.

Script sperimentali gia migrati al core:

- `src/pipeline/Fase_1/09b_itemknn_cf_improved.py`;
- `src/pipeline/Fase_1/10_matrix_factorization.py`.

### Content-Based

Costruisce embedding testuali usando metadati dei film:

- generi;
- tag;
- overview;
- attori principali;
- regista.

Le raccomandazioni derivano dalla similarita tra candidati e film gia valutati positivamente dall'utente.

Implementazione core:

- `src/recommenders/content_based.py`
- `src/recommenders/scoring.py`

Responsabilita:

- mapping tra indice embedding e `movieId`;
- conversione dei vicini content da indici a MovieLens ID;
- scoring dei candidati basato sui vicini contenutistici;
- ranking content top-k;
- recupero film simili per dettaglio film e web app.

Script sperimentali gia migrati al core:

- `src/pipeline/Fase_2a/14a_evaluate_content_baseline_v2.py`.

### Hybrid

Combina punteggi collaborative e content-based dopo normalizzazione per utente. Il peso viene scelto tramite validation set.

Implementazione core:

- `src/recommenders/hybrid.py`

Responsabilita:

- fusione rating espliciti e preferiti;
- media utente con default cold-start;
- normalizzazione min-max dei punteggi;
- combinazione pesata collaborative/content;
- output con contributi separati per explainability.

Script sperimentali gia migrati al core:

- `src/pipeline/Fase_3/15_hybrid_weighted_eval.py`;
- `src/pipeline/Fase_3/17_hybrid_svd_content_eval.py`.

Nel backend il peso attuale e `GAMMA = 0.7`, selezionato tramite validation set negli esperimenti `Fase_3`.

### Ablation Content-Based

L'ablation study confronta varianti della rappresentazione testuale, ad esempio modello completo, senza tag, senza overview e solo generi. La valutazione usa lo stesso scoring content-based del core, cosi le differenze misurate dipendono dalle feature testuali e non da implementazioni diverse del ranking.

Script sperimentali gia migrati al core:

- `src/pipeline/Fase_4/18c_evaluate_content_ablation.py`.

## Requisiti Funzionali

- escludere film gia visti, valutati o preferiti;
- gestire cold-start utente;
- produrre top-N ordinato;
- normalizzare punteggi eterogenei;
- restituire metadati utili alla spiegazione.

## Evaluation

Metriche usate o pianificate:

- Precision@K;
- Recall@K;
- NDCG@K;
- MAP@K;
- MRR@K;
- Hit Rate@K;
- coverage catalogo;
- analisi per segmento utente e popolarita item.

Implementazione core:

- `src/evaluation/metrics.py`
- `src/utils/eval.py`, mantenuto come shim compatibile per gli script esistenti.

Metriche attualmente implementate:

- Hit Rate@K;
- Precision@K;
- Recall@K;
- NDCG@K;
- MRR@K;
- Average Precision@K per leave-one-out;
- Catalog Coverage@K;
- Novelty@K;
- RMSE;
- MAE.

Nota metodologica: molte pipeline correnti usano valutazione leave-one-out, quindi alcune metriche sono definite rispetto a un singolo item rilevante per utente.

## Test

I moduli core sono coperti da test unitari in:

- `tests/test_popularity.py`;
- `tests/test_collaborative.py`;
- `tests/test_content_based.py`;
- `tests/test_hybrid.py`;
- `tests/test_metrics.py`.

Comando di verifica:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_popularity tests.test_collaborative tests.test_content_based tests.test_metrics tests.test_hybrid
```

## Explainability

Le spiegazioni principali dovrebbero essere deterministiche:

- film simili a quelli valutati positivamente;
- generi condivisi;
- regista o attori ricorrenti;
- contributo collaborative;
- contributo content-based.

Un modello LLM puo essere usato come strato opzionale per rendere la spiegazione piu naturale, ma non deve essere l'unica fonte della spiegazione.

## Limiti Da Discutere In Tesi

- sparsita della matrice user-item;
- cold-start per nuovi utenti e nuovi film;
- dipendenza dalla qualita dei metadati TMDB;
- bias verso film popolari;
- costo computazionale di embedding e modelli multimodali;
- differenza tra metriche offline e soddisfazione reale dell'utente.
