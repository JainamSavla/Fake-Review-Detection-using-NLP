"""Re-creates the notebook pipeline, saves the models and the numbers the app shows.

Usage:  python train.py fake_reviews_dataset.csv
Outputs (in this folder): tfidf_vectorizer.pkl, model_lr.pkl, model_xgb.pkl, insights.json
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from textutils import clean, STOPWORDS

csv_path = sys.argv[1] if len(sys.argv) > 1 else "fake_reviews_dataset.csv"
data = pd.read_csv(csv_path)
data["text_"] = data["text_"].fillna("")
data["length"] = data["text_"].apply(len)
data["word_count"] = data["text_"].apply(lambda x: len(x.split()))
print("Dataset:", data.shape)

# ---------------- exploration numbers for the Insights tab ----------------
ins = {"n_reviews": int(len(data))}
ins["label_counts"] = {k: int(v) for k, v in data["label"].value_counts().items()}
ins["rating_by_label"] = {
    str(float(r)): {k: int(v) for k, v in row.items()}
    for r, row in pd.crosstab(data["rating"], data["label"]).iterrows()
}
ins["cg_share_by_category"] = {
    k: round(float(v) * 100, 2)
    for k, v in (data["label"].eq("CG").groupby(data["category"]).mean()).items()
}
ins["length_means"] = {
    lab: {"length": round(float(g["length"].mean()), 1),
          "word_count": round(float(g["word_count"].mean()), 1)}
    for lab, g in data.groupby("label")
}
bins = np.arange(0, 201, 10)
ins["wordcount_hist"] = {
    "bins": [int(b) for b in bins[:-1]],
    **{lab: np.histogram(g["word_count"].clip(upper=199), bins=bins)[0].tolist()
       for lab, g in data.groupby("label")},
}

cv_tmp = CountVectorizer(stop_words="english", min_df=20)
Xc = cv_tmp.fit_transform(data["text_"])
vocab = np.array(cv_tmp.get_feature_names_out())
is_cg = (data["label"] == "CG").values
cg_freq = np.asarray(Xc[is_cg].sum(axis=0)).ravel() + 1
or_freq = np.asarray(Xc[~is_cg].sum(axis=0)).ravel() + 1
ratio = np.log(cg_freq / or_freq)
ins["cg_specific_words"] = [str(w) for w in vocab[np.argsort(-ratio)[:20]]]
ins["or_specific_words"] = [str(w) for w in vocab[np.argsort(ratio)[:20]]]

# a few real example reviews for the "try an example" buttons
rng = np.random.RandomState(7)
short = data[(data["word_count"] >= 8) & (data["word_count"] <= 40)]
ins["examples"] = {
    lab: [str(t) for t in short[short["label"] == lab]["text_"].sample(4, random_state=rng).tolist()]
    for lab in ["CG", "OR"]
}

# ---------------- preprocessing + modelling (same as the notebook) ----------------
corpus = data["text_"].apply(clean)
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(corpus)
y = (data["label"] == "CG").astype(int).values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=15, stratify=y)
print("Train:", X_train.shape, "Test:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=15),
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, n_jobs=-1, random_state=15),
    "Decision Tree": DecisionTreeClassifier(random_state=15),
}
ins["models"] = {}
for name, m in models.items():
    m.fit(X_train, y_train)
    preds = m.predict(X_test)
    rep = classification_report(y_test, preds, output_dict=True)
    ins["models"][name] = {
        "train_acc": round(float(m.score(X_train, y_train)), 4),
        "test_acc": round(float(m.score(X_test, y_test)), 4),
        "precision_fake": round(rep["1"]["precision"], 4),
        "recall_fake": round(rep["1"]["recall"], 4),
        "f1_fake": round(rep["1"]["f1-score"], 4),
        "confusion": confusion_matrix(y_test, preds).tolist(),
    }
    print(f"{name:20s} train={ins['models'][name]['train_acc']:.4f} test={ins['models'][name]['test_acc']:.4f}")

cv = cross_val_score(LogisticRegression(max_iter=1000), X_train, y_train, cv=5)
ins["lr_cv"] = {"mean": round(float(cv.mean()), 4), "std": round(float(cv.std()), 4)}

model_lr, model_xgb = models["Logistic Regression"], models["XGBoost"]
feat = np.array(tfidf.get_feature_names_out())
coef = model_lr.coef_[0]
ins["lr_top_fake"] = [[str(feat[i]), round(float(coef[i]), 3)] for i in np.argsort(-coef)[:20]]
ins["lr_top_genuine"] = [[str(feat[i]), round(float(coef[i]), 3)] for i in np.argsort(coef)[:20]]

# ---------------- save ----------------
pickle.dump(tfidf, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(model_lr, open("model_lr.pkl", "wb"))
pickle.dump(model_xgb, open("model_xgb.pkl", "wb"))
json.dump(ins, open("insights.json", "w"), indent=1)
print("Saved model files and insights.json")
