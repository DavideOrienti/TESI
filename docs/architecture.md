# Architettura Del Sistema

## Visione Generale

Il progetto e composto da due parti complementari:

1. **pipeline offline**, che prepara i dati, costruisce gli artefatti e valuta i modelli;
2. **web app**, che usa gli artefatti generati offline per fornire raccomandazioni personalizzate.

Questa separazione e importante per la tesi: la parte sperimentale resta riproducibile e misurabile, mentre la web app dimostra l'integrazione del sistema in un prototipo utente.

## Flusso Dati

```text
MovieLens raw
  -> cleaning e normalizzazione titoli
  -> merge tag e link TMDB
  -> enrichment TMDB
  -> dataset arricchito
  -> split rating train/validation/test
  -> training/evaluation recommender
  -> artefatti deploy
  -> backend Flask
  -> frontend React
```

## Moduli Attuali

- `src/pipeline`: script offline numerati per fasi sperimentali.
- `src/recommenders`: core dei modelli popularity, collaborative, content-based e hybrid.
- `src/evaluation`: metriche di valutazione offline.
- `src/utils`: utility per configurazione, ID, TMDB e compatibilita con script esistenti.
- `src/webapp/backend`: API Flask, autenticazione, DB e servizi applicativi.
- `src/webapp/frontend`: interfaccia React.
- `data/deploy_artifacts`: artefatti necessari al backend in produzione.

## Moduli Target

```text
src/
+-- data_pipeline/
+-- tmdb/
+-- database/
+-- recommenders/
+-- evaluation/
+-- explainability/
+-- multimodal/
+-- utils/
```

La migrazione deve essere incrementale: gli script numerati possono restare come wrapper compatibili, mentre la logica comune viene progressivamente estratta in moduli testabili.

## Moduli Core Estratti

```text
src/recommenders/
+-- popularity.py       # baseline e fallback cold-start
+-- collaborative.py    # ItemKNN, SVD helpers, fallback collaborativo
+-- content_based.py    # vicini content, scoring e similar movies
+-- hybrid.py           # fusione score collaborative/content
+-- scoring.py          # funzioni atomiche di scoring e normalizzazione

src/evaluation/
+-- metrics.py          # metriche top-N, coverage, novelty, RMSE, MAE
```

Il backend `src/webapp/backend/recommender_service.py` ora carica artefatti e delega la logica algoritmica a questi moduli. Questo riduce accoppiamento tra web app e sperimentazione.

Anche le prime baseline offline sono state riallineate al core:

- `Fase_1/08_popularity_baseline.py` usa `src/recommenders/popularity.py`;
- `Fase_1/09b_itemknn_cf_improved.py` usa `src/recommenders/collaborative.py` e `src/recommenders/popularity.py`;
- `Fase_1/10_matrix_factorization.py` usa `src/recommenders/collaborative.py` e `src/recommenders/popularity.py`.
- `Fase_2a/14a_evaluate_content_baseline_v2.py` usa `src/recommenders/content_based.py`.
- `Fase_3/15_hybrid_weighted_eval.py` usa `src/recommenders/collaborative.py`, `content_based.py` e `hybrid.py`.
- `Fase_3/17_hybrid_svd_content_eval.py` usa `src/recommenders/collaborative.py`, `content_based.py` e `hybrid.py`.
- `Fase_4/18c_evaluate_content_ablation.py` usa `src/recommenders/content_based.py`.

## Confini Architetturali

- La pipeline offline produce dataset, embedding, matrici e metriche.
- Il backend non dovrebbe rigenerare artefatti pesanti all'avvio.
- Il backend dovrebbe caricare artefatti gia pronti e gestire utenti, rating, preferiti e API.
- Le route Flask dovrebbero delegare a servizi applicativi.
- Le query al database dovrebbero essere isolate in repository layer.
- Explainability e multimodalita dovrebbero essere moduli separati e opzionali.

## Rischi Tecnici

- duplicazione residua tra alcuni script sperimentali e i nuovi moduli core;
- dipendenze pesanti non sempre disponibili in produzione;
- dati e artefatti molto grandi;
- valutazione ancora distribuita in piu script, anche se le metriche sono centralizzate;
- import dati nel database legato all'avvio dell'app.

## Test Di Regressione

I primi test unitari coprono il core recommender ed evaluation:

```text
tests/test_popularity.py
tests/test_collaborative.py
tests/test_content_based.py
tests/test_hybrid.py
tests/test_metrics.py
```

Questi test non sostituiscono la valutazione sperimentale, ma proteggono le invarianti di base: esclusione dei film gia visti, ranking deterministico, fallback e formule metriche.

## Criterio Di Successo

Il repository finale deve permettere di:

- riprodurre la pipeline principale;
- confrontare i modelli con metriche chiare;
- avviare la web app;
- spiegare le scelte ingegneristiche e sperimentali nella tesi.
