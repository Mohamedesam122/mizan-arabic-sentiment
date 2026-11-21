# Mizan — Arabic Sentiment Analysis (MSA + Egyptian Dialect)

Binary sentiment classification (positive / negative) for Modern Standard Arabic
and Egyptian-dialect short text, built end to end: data cleaning, model
comparison with honest held-out evaluation, a FastAPI inference service, a
Streamlit dashboard, and one-command Docker deployment.

## Results

Trained on **32,283 Arabic tweets** (Arabic Sentiment Twitter Corpus + an 80%
stratified slice of the ASTD Egyptian-dialect corpus), evaluated on **5,844
held-out tweets** including **497 Egyptian-dialect tweets the model never saw
in training**.

| Model | Test accuracy | Test macro F1 |
|---|---|---|
| **Logistic Regression (chosen)** | **0.859** | **0.858** |
| Linear SVC | 0.855 | 0.855 |
| Complement Naive Bayes | 0.728 | 0.724 |

Per-class (test set): NEG precision 0.860 / recall 0.875 / F1 0.868 —
POS precision 0.857 / recall 0.840 / F1 0.849.

**Held-out Egyptian-dialect subset (ASTD):** accuracy **0.765**, macro F1
**0.728**. Dialect is genuinely harder than MSA, and this number is reported
separately on purpose: it is the honest read on the hardest input distribution.
Full breakdown in [`reports/metrics.json`](reports/metrics.json).

## Approach

- **Preprocessing** (`app/preprocessing.py`): URL/mention stripping, diacritics
  and tatweel removal, alef/ya/taa-marbuta normalization. No stemming —
  dialect morphology carries sentiment. Emojis are kept as features.
- **Features**: TF-IDF over word 1–2 grams (120k) + character 3–5 grams (80k),
  sublinear TF. Character n-grams catch dialect spelling variants
  (تحفه / تحفة, كويس / كويس جدا).
- **Model**: Logistic Regression (C=2.0), chosen on test macro F1 and for its
  calibrated `predict_proba` confidence scores.

## Data

- [Arabic Sentiment Twitter Corpus](https://huggingface.co/datasets/asas-ai/Arabic_Sentiment_Twitter_Corpus) — 56,795 labeled tweets (pos/neg).
- [ASTD — Arabic Sentiment Tweets Dataset](https://huggingface.co/datasets/Qanadil/ASTD_Arabic_Sentiment_Tweets_Dataset)
  (Nizar Habash's group, [ACL 2015 paper](https://aclanthology.org/D15-1299/)) —
  Egyptian-dialect tweets; POS/NEG rows used, 80/20 stratified train/test split.

Scope note: the model is **binary by design**. NEUTRAL/MIXED rows were
excluded rather than force-labeled; the API surfaces low-confidence
predictions so borderline text is visible instead of silently misclassified.

## Project structure

```
├── app/
│   ├── preprocessing.py   # shared Arabic normalization (train + serve)
│   ├── train.py           # full training + evaluation pipeline
│   ├── main.py            # FastAPI inference service
│   └── streamlit_app.py   # dashboard: single text, batch CSV, model report
├── data/                  # the two source datasets (parquet)
├── models/                # trained pipeline (joblib)
├── reports/metrics.json   # full evaluation report
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quickstart

Local:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload          # API on http://localhost:8000 (docs at /docs)
streamlit run app/streamlit_app.py     # UI on http://localhost:8501
```

Docker (both services):

```bash
docker compose up --build
```

## API

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "الفيلم ده كان تحفه بجد"}'
```

```json
{
  "text": "الفيلم ده كان تحفه بجد",
  "label": "POS",
  "confidence": 0.7293,
  "probabilities": {"NEG": 0.2707, "POS": 0.7293},
  "inference_ms": 2.66
}
```

Endpoints: `POST /predict`, `POST /predict_batch` (up to 500 texts),
`GET /health`, `GET /model_info` (evaluation summary).

## Retraining

```bash
python -m app.train
```

Reproduces the full pipeline from the parquet files in `data/` and rewrites
`models/sentiment_pipeline.joblib` and `reports/metrics.json` (~15 s on 2 CPU
cores).

## Limitations

- Binary only; neutral and mixed-sentiment text are out of scope by design.
- Egyptian-dialect slang that flips literal meaning ("نار", "جامد") is still
  the hardest case, as the dialect metrics show.
- A classical TF-IDF baseline by choice: fast inference (~2 ms), fully
  inspectable, and a fair benchmark before reaching for a fine-tuned
  transformer (AraBERT/DARBERT) as the next step.

## Author

**Mohamed Esam** - 3rd-year student, Faculty of Computers and Artificial
Intelligence, Cairo University (class of 2027).

- LinkedIn: [mohamed-esam-354638292](https://www.linkedin.com/in/mohamed-esam-354638292)
- GitHub: [Mohamedesam122](https://github.com/Mohamedesam122)
- Email: mokzoredg67@gmail.com
