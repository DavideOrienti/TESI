
# Stato di implementazione

## Implementato e funzionante

- **Step 01** — `01_clean_movielens.py`: pulizia titoli, estrazione anno, filtro rating
- **Step 02** — `02_merge_tags.py` / `02_merge_tags_tf-idf.py` / `02b_merge_links.py`: integrazione tag e link TMDB
- **Step 03** — `03_enrich_tmdb.py`: arricchimento via API TMDB (attori, regista, overview, poster)
- **Step 04** — `04_build_people_tables.py`: costruzione tabelle attori e registi
- **Step 05** — `05_data_audit.py`: audit di copertura e qualità del dataset
- **Step 06** — `06_prepare_ratings.py` (Fase 1): preparazione rating per il training

## Pianificato / da implementare

- **Import database SQLAlchemy** (`src/webapp/import_movielens.py`): importazione dei CSV prodotti in un database SQLite tramite i modelli definiti in `src/webapp/models.py`
- **Web app** (`src/webapp/`): interfaccia FastAPI per il serving delle raccomandazioni; struttura presente ma non integrata con la pipeline
- **Explainability**: giustificazione delle raccomandazioni all'utente (es. "consigliato perché hai visto X")

---

# FASE 0 — DATA ENGINEERING

## 1. Obiettivo della fase

La fase di **Data Engineering** ha lo scopo di costruire un dataset arricchito e coerente a partire dal dataset originale **MovieLens**.

Il dataset MovieLens contiene principalmente:

* film
* utenti
* rating
* generi
* tag

Tuttavia non contiene molte informazioni semantiche utili per un sistema di raccomandazione avanzato, come:

* attori
* regista
* descrizione del film
* popolarità
* poster

Per questo motivo il dataset è stato **arricchito utilizzando le API di The Movie Database (TMDB)**.

Il risultato finale della fase 0 è un dataset strutturato che contiene:

* metadati dei film
* informazioni semantiche
* relazioni tra film, attori, registi e generi

Questo dataset sarà utilizzato nelle fasi successive per costruire i modelli di raccomandazione.

---

# 2. Dataset utilizzati

## MovieLens

Il dataset principale utilizzato è **MovieLens**, che fornisce:

* movieId
* titolo
* generi
* rating utenti
* tag

Questi dati rappresentano la base del sistema di raccomandazione.

---

## TMDB (The Movie Database)

Per arricchire il dataset sono state utilizzate le **API TMDB** per recuperare:

* attori principali
* regista
* popolarità
* descrizione del film
* poster

L'arricchimento è stato effettuato tramite chiamate HTTP alle API TMDB.
Ad esempio nel codice viene effettuata una richiesta del tipo:

```python
url = f"{BASE_URL}/movie/{tmdb_id}"
params = {'api_key': API_KEY, 'language': 'it-IT'}
```

che restituisce i metadati del film. 

---

# 3. Pipeline di Data Engineering

La pipeline di preprocessing è composta da diversi step sequenziali.

## Step 1 — Pulizia titolo e estrazione anno

Il primo step consiste nell’estrarre l'anno dal titolo del film.

Nel dataset MovieLens infatti il titolo è nel formato:

```
Toy Story (1995)
```

Utilizzando una **regex** viene estratto l'anno e separato dal titolo.

Esempio di codice:

```python
pattern = r'\((\d{4})\)$'
df['anno'] = df['title'].str.extract(pattern)
```

Successivamente il titolo viene ripulito rimuovendo l'anno. 

Questo permette di ottenere due colonne separate:

* `title`
* `anno`

---

# Step 2 — Integrazione dei tag

MovieLens fornisce anche un dataset di **tag assegnati dagli utenti ai film**.

I tag vengono raggruppati per film e concatenati in una singola stringa.

```python
df_tag_grouped = df_tag.groupby('movieId')['tag'].apply(
    lambda tags: ', '.join(tags.astype(str))
)
```

Successivamente vengono uniti al dataset dei film tramite `movieId`. 

Questo arricchimento è utile per:

* analisi semantica
* modelli content-based
* embedding testuali

---

# Step 3 — Arricchimento tramite TMDB

Per ogni film viene utilizzato il campo `tmdbId` per interrogare le API TMDB.

Le informazioni recuperate includono:

* attori principali
* regista
* popolarità
* descrizione del film

Il codice utilizza due endpoint principali:

### Credits API

```
/movie/{tmdb_id}/credits
```

per recuperare:

* cast
* crew

Da questi dati vengono estratti:

* primi 5 attori
* regista

### Details API

```
/movie/{tmdb_id}
```

per recuperare:

* popolarità
* overview

Questo processo è implementato nella funzione:

```
process_movie()
```

che restituisce:

```
attori
regista
popolarita
descrizione
```



Per evitare problemi con il rate limit delle API viene utilizzata una pausa:

```
time.sleep(0.25)
```

---

# Step 4 — Recupero poster dei film

Il dataset è stato ulteriormente arricchito con l'URL del poster del film.

Il poster viene ottenuto tramite la proprietà `poster_path` restituita dalle API TMDB.

```python
poster_url = f"{IMAGE_BASE_URL}{data['poster_path']}"
```

Questo URL consente di scaricare direttamente l'immagine del poster. 

Il poster sarà utile nelle fasi successive per:

* sistemi multimodali
* riconoscimento immagini
* visual recommendation

---

# Step 5 — Descrizione in lingua inglese

Oltre alla descrizione italiana, è stata recuperata anche la descrizione in inglese.

Questo permette di utilizzare modelli NLP più avanzati (es. transformer).

La richiesta alle API utilizza il parametro:

```
language = en-US
```

per ottenere la descrizione inglese. 

---

# Step 6 — Estrazione attori e registi

Una volta ottenuti i metadati TMDB, vengono estratti gli **attori e i registi unici** presenti nel dataset.

Il codice scorre tutti i film e costruisce due insiemi:

* attori
* registi

utilizzando set per evitare duplicati.

```python
attori_scritti = set()
registi_scritti = set()
```

Successivamente vengono salvati nei file:

```
attori.csv
registi.csv
```



---

# Step 7 — Generazione ID per attori e registi

Per poter gestire correttamente le relazioni nel database è stato necessario assegnare un ID univoco a ciascun attore e regista.

Questo viene fatto enumerando le righe dei file CSV.

```python
for idx, attore in enumerate(attori, start=1):
    attore['attore_id'] = idx
```

In questo modo ogni attore ha un identificatore univoco. 

Un processo analogo viene applicato ai registi. 

---

# Step 8 — Importazione nel database

Dopo la costruzione dei dataset intermedi, i dati vengono importati in un database SQLite tramite SQLAlchemy.

Le principali entità del database sono:

* Film
* Attore
* Regista
* Genere

Sono inoltre presenti tabelle relazionali:

* FilmAttori
* FilmGeneri

Il processo di importazione legge i CSV generati e crea le relazioni tra le entità.

Ad esempio:

```python
db.session.add(FilmAttori(
    movie_id=nuovo_film.movieId,
    attore_id=attore_id
))
```

Questo consente di modellare la relazione molti-a-molti tra film e attori. 

---

# 4. Dataset finale

Al termine della fase di Data Engineering il dataset finale contiene:

### Informazioni sui film

* movieId
* title
* anno
* generi
* descrizione
* descrizione inglese
* popolarità
* poster_url
* tag

### Informazioni sui registi

* regista_id
* nome

### Informazioni sugli attori

* attore_id
* nome

### Relazioni

* FilmAttori
* FilmGeneri

---

# 5. Risultato della fase

La Fase 0 produce un dataset strutturato e arricchito che rappresenta la base per lo sviluppo del sistema di raccomandazione.

In particolare, i dati ottenuti permettono di implementare:

* sistemi **content-based**
* sistemi **collaborative filtering**
* sistemi **ibridi**
* modelli **multimodali**

Questa fase è fondamentale perché la qualità dei dati influisce direttamente sulle prestazioni del sistema di raccomandazione.

---

# 6. Considerazioni

L'integrazione con TMDB ha permesso di trasformare il dataset MovieLens in un dataset molto più ricco dal punto di vista semantico.

Tuttavia, l'utilizzo di API esterne introduce alcune criticità:

* rate limit delle API
* dati mancanti per alcuni film
* dipendenza da servizi esterni

Per mitigare questi problemi sono state adottate alcune strategie:

* salvataggi parziali durante l'elaborazione
* gestione degli errori nelle richieste HTTP
* uso di valori placeholder per dati mancanti

---

# 7. Output della fase

I principali file prodotti dalla pipeline sono:

```
movies.csv
movies_con_info_tmdb.csv
movies_con_descrizione_en.csv
movies_con_poster_final.csv
attori.csv
registi.csv