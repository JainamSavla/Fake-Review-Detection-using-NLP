# Fake Review Detector (Streamlit)

A single-page web app built from the `Fake_Reviews_Exploration_Modelling.ipynb` notebook.

- **Review checker tab:** paste a review, get fake (CG) / genuine (OR) with a confidence score,
  and see which words pushed the decision.
- **Project insights tab:** label balance, review length, ratings, most telling words,
  and the comparison of all four models.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501. The trained models are already included
(`model_lr.pkl`, `model_xgb.pkl`, `tfidf_vectorizer.pkl`, `insights.json`), so you do **not** need the CSV to run the app.

## Re-train (optional)

```bash
python train.py fake_reviews_dataset.csv
```

This repeats the notebook pipeline (same cleaning, TF-IDF with 5000 uni+bigram features,
70/30 stratified split, `random_state=15`) and rewrites the four files above.

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `train.py` | Re-creates the notebook pipeline and saves models + numbers for the app |
| `textutils.py` | The notebook's `clean()` function (NLTK stopword list embedded, no download needed) |
| `model_lr.pkl`, `model_xgb.pkl`, `tfidf_vectorizer.pkl` | Trained models and vectorizer |
| `insights.json` | Pre-computed numbers for the Insights tab |

## Note

The model separates computer-generated from original reviews **in this dataset**.
It is a signal, not proof, and it cannot verify purchases or catch real people writing paid reviews.
