# Recommender System

## Obiettivo

Il sistema confronta modelli di raccomandazione top-N per film e usa il modello hybrid come base della web app.

## Modelli

### Popularity Baseline

Ordina i film per popolarita o numero/media rating. Serve come baseline sperimentale e fallback per utenti nuovi.

### Collaborative Filtering

Sono presenti due linee:

- **ItemKNN**: similarita item-item su matrice user-item, con rating centrati sulla media utente;
- **SVD**: fattorizzazione della matrice user-item e predizione dei punteggi utente-film.

### Content-Based

Costruisce embedding testuali usando metadati dei film:

- generi;
- tag;
- overview;
- attori principali;
- regista.

Le raccomandazioni derivano dalla similarita tra candidati e film gia valutati positivamente dall'utente.

### Hybrid

Combina punteggi collaborative e content-based dopo normalizzazione per utente. Il peso viene scelto tramite validation set.

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

