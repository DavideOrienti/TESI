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
- `src/utils`: utility per configurazione, ID, TMDB e metriche.
- `src/recommenders`: funzioni condivise di scoring e normalizzazione.
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

## Confini Architetturali

- La pipeline offline produce dataset, embedding, matrici e metriche.
- Il backend non dovrebbe rigenerare artefatti pesanti all'avvio.
- Il backend dovrebbe caricare artefatti gia pronti e gestire utenti, rating, preferiti e API.
- Le route Flask dovrebbero delegare a servizi applicativi.
- Le query al database dovrebbero essere isolate in repository layer.
- Explainability e multimodalita dovrebbero essere moduli separati e opzionali.

## Rischi Tecnici

- duplicazione tra script sperimentali e backend;
- dipendenze pesanti non sempre disponibili in produzione;
- dati e artefatti molto grandi;
- valutazione distribuita in piu script;
- import dati nel database legato all'avvio dell'app.

## Criterio Di Successo

Il repository finale deve permettere di:

- riprodurre la pipeline principale;
- confrontare i modelli con metriche chiare;
- avviare la web app;
- spiegare le scelte ingegneristiche e sperimentali nella tesi.
