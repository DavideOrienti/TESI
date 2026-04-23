# notebooks/exploration — Reference storici

Questa cartella contiene notebook esplorativi usati durante lo sviluppo
della tesi ma **non eseguibili** nella pipeline finale.

Non fanno parte del codice valutabile della tesi.

---

## Contenuto

| File | Cosa fa | Perché non è in pipeline |
|------|---------|--------------------------|
| `fase_2a_content_refinement.ipynb` | Esperimenti iniziali sulla costruzione degli embedding content-based (Fase 2a) | Dipende da `sentence_transformers==2.2.2` incompatibile con `huggingface_hub` attuale; gli step funzionanti sono stati migrati in `src/pipeline/Fase_2a/` come script `.py` |

---

## Pipeline eseguibile

Tutto il codice che produce risultati per la tesi si trova in:

```
src/pipeline/
├── Fase_0/   — data engineering e arricchimento TMDB
├── Fase_1/   — baseline CF (popularity, ItemKNN, SVD)
├── Fase_2/   — costruzione embedding film
├── Fase_2a/  — content-based recommender (versione finale)
├── Fase_3/   — modelli ibridi (weighted, with-popularity)
└── Fase_4/   — analisi e ablation study
```
