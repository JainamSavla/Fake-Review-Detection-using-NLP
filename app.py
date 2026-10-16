"""Fake Review Detector - single-page Streamlit app.

Run:  streamlit run app.py
"""
import json
import pickle
import random
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from textutils import clean, STOPWORDS, stemmer

HERE = Path(__file__).parent
FAKE_COLOR = "#e5484d"
REAL_COLOR = "#2f9e63"

st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")


# ------------------------------------------------------------------ loading
@st.cache_resource
def load_models():
    tfidf = pickle.load(open(HERE / "tfidf_vectorizer.pkl", "rb"))
    lr = pickle.load(open(HERE / "model_lr.pkl", "rb"))
    xgb = pickle.load(open(HERE / "model_xgb.pkl", "rb"))
    return tfidf, lr, xgb, np.array(tfidf.get_feature_names_out())


@st.cache_data
def load_insights():
    return json.load(open(HERE / "insights.json"))


tfidf, model_lr, model_xgb, FEATURES = load_models()
ins = load_insights()
MODELS = {"Logistic Regression": model_lr, "XGBoost": model_xgb}


# ------------------------------------------------------------------ helpers
def predict(text: str, model) -> float:
    """Probability that the review is computer-generated (fake)."""
    return float(model.predict_proba(tfidf.transform([clean(text)]))[0][1])


def contributions(text: str) -> dict:
    """Per-feature push towards fake (+) or genuine (-) from the Logistic Regression."""
    coo = tfidf.transform([clean(text)]).tocoo()
    coef = model_lr.coef_[0]
    return {FEATURES[j]: float(coef[j] * v) for j, v in zip(coo.col, coo.data)}


def highlight(text: str, contrib: dict) -> str:
    """HTML of the review with words tinted by how strongly they pushed the decision."""
    unigram = {k: v for k, v in contrib.items() if " " not in k}
    peak = max([abs(v) for v in unigram.values()] + [1e-9])
    out = []
    for tok in re.findall(r"[A-Za-z]+|[^A-Za-z]+", text):
        if tok.isalpha() and tok.lower() not in STOPWORDS:
            c = unigram.get(stemmer.stem(tok.lower()))
            if c is not None and abs(c) > 0.02 * peak:
                rgb = "229,72,77" if c > 0 else "47,158,99"
                alpha = 0.18 + 0.55 * min(abs(c) / peak, 1)
                out.append(f'<span style="background:rgba({rgb},{alpha:.2f});'
                           f'border-radius:4px;padding:1px 3px">{escape(tok)}</span>')
                continue
        out.append(escape(tok))
    return "".join(out).replace("\n", "<br>")


def set_example(kind: str):
    st.session_state.review_text = random.choice(ins["examples"][kind])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODELS), help="Both were trained on the same data.")
    m = ins["models"][model_name]
    st.metric("Test accuracy", f"{m['test_acc']*100:.1f}%")
    st.caption(f"Trained on {int(ins['n_reviews'] * 0.7):,} reviews, "
               f"tested on {int(ins['n_reviews'] * 0.3):,} unseen ones.")
    with st.expander("How reliable is this?"):
        st.write(
            "The model learns the difference between the **computer-generated (CG)** and "
            "**original (OR)** reviews in this dataset. It is right about "
            f"{ins['models']['Logistic Regression']['test_acc']*100:.0f}% of the time on "
            "held-out data from that same dataset. Treat it as a signal, not proof: "
            "it can't verify a purchase, and reviews written by real people for pay "
            "may look genuine to it."
        )

# ------------------------------------------------------------------ header
st.title("🕵️ Fake Review Detector")
st.write("Paste a product review and see whether it looks **computer-generated (fake)** "
         "or **genuine**, and which words drove the decision.")

tab_check, tab_insights = st.tabs(["🔍 Review checker", "📊 Project insights"])

# ================================================================== checker
with tab_check:
    c1, c2, c3 = st.columns(3)
    c1.button("Try a genuine example", on_click=set_example, args=("OR",), width="stretch")
    c2.button("Try a fake example", on_click=set_example, args=("CG",), width="stretch")
    c3.button("Clear", on_click=lambda: st.session_state.update(review_text=""), width="stretch")

    text = st.text_area("Review text", key="review_text", height=150,
                        placeholder="e.g. Arrived late and the zipper broke after two weeks...")
    st.caption("Click outside the box or press Ctrl+Enter to update the result.")

    if text.strip():
        n_words = len(text.split())
        p_fake = predict(text, MODELS[model_name])
        is_fake = p_fake > 0.5
        conf = max(p_fake, 1 - p_fake)
        color = FAKE_COLOR if is_fake else REAL_COLOR

        st.markdown(
            f"""<div style="border-left:6px solid {color};padding:12px 16px;border-radius:6px;
            background:{color}18;margin:8px 0 4px">
            <div style="font-size:1.5rem;font-weight:700;color:{color}">
            {'Looks computer-generated (fake)' if is_fake else 'Looks genuine'}</div>
            <div>Confidence: <b>{conf*100:.0f}%</b> &nbsp;·&nbsp; {model_name}</div></div>""",
            unsafe_allow_html=True)
        st.progress(p_fake, text=f"Probability fake: {p_fake*100:.1f}%")
        if n_words < 5:
            st.warning("Very short reviews give unreliable results. Try a longer one.")

        contrib = contributions(text)
        if contrib:
            st.subheader("Why?")
            st.markdown(
                f'<span style="background:rgba(229,72,77,.5);padding:1px 6px;border-radius:4px">'
                f'pushes towards fake</span> &nbsp; '
                f'<span style="background:rgba(47,158,99,.5);padding:1px 6px;border-radius:4px">'
                f'pushes towards genuine</span>', unsafe_allow_html=True)
            st.markdown(f'<div style="line-height:2;margin-top:8px">{highlight(text, contrib)}</div>',
                        unsafe_allow_html=True)

            top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            df = pd.DataFrame(top, columns=["term", "push"]).sort_values("push")
            df["direction"] = np.where(df["push"] > 0, "towards fake", "towards genuine")
            st.bar_chart(df, x="term", y="push", color="direction", horizontal=True,
                         x_label="", y_label="push towards fake (+) / genuine (-)")
            st.caption("Terms are shown in their stemmed form (e.g. “love” and “loved” both "
                       "become “love”). The word-level explanation always comes from the "
                       "Logistic Regression model, even if XGBoost is selected.")
        else:
            st.info("None of the words in this review are in the model's vocabulary.")

# ================================================================== insights
with tab_insights:
    best = max(ins["models"].items(), key=lambda kv: kv[1]["test_acc"])
    k1, k2, k3 = st.columns(3)
    k1.metric("Reviews", f"{ins['n_reviews']:,}")
    k2.metric("Fake / genuine", "50% / 50%")
    k3.metric("Best model", f"{best[1]['test_acc']*100:.1f}%", best[0], delta_color="off")

    st.subheader("Model comparison")
    comp = pd.DataFrame(ins["models"]).T[["train_acc", "test_acc", "precision_fake", "recall_fake", "f1_fake"]]
    comp.columns = ["Train acc", "Test acc", "Precision (fake)", "Recall (fake)", "F1 (fake)"]
    comp = comp.sort_values("Test acc", ascending=False)
    st.bar_chart(comp[["Test acc"]], horizontal=True, color="#3b82f6", y_label="Test accuracy")
    st.dataframe(comp.style.format("{:.3f}"), width="stretch")
    st.caption("Random Forest and Decision Tree score ~100% on the training data but much lower "
               "on unseen data, which is a sign of overfitting. Logistic Regression generalises best "
               f"(5-fold CV accuracy {ins['lr_cv']['mean']*100:.1f}% ± {ins['lr_cv']['std']*100:.1f}).")

    cm = np.array(ins["models"][model_name]["confusion"])
    st.markdown(f"**Confusion matrix: {model_name}** (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(cm, index=["Actual genuine", "Actual fake"],
                              columns=["Predicted genuine", "Predicted fake"])
                 .style.background_gradient(cmap="Blues", axis=None), width="stretch")

    st.subheader("What the data looks like")
    h = ins["wordcount_hist"]
    hist = pd.DataFrame({"Fake (CG)": h["CG"], "Genuine (OR)": h["OR"]},
                        index=[f"{b}-{b+9}" if b < 190 else "190+" for b in h["bins"]])
    st.markdown("**Review length (words) by label**")
    st.bar_chart(hist, color=[FAKE_COLOR, REAL_COLOR], stack=False,
                 x_label="words per review", y_label="reviews")
    lm = ins["length_means"]
    st.caption(f"Fake reviews are shorter on average: {lm['CG']['word_count']:.0f} words "
               f"vs {lm['OR']['word_count']:.0f} for genuine ones.")

    st.markdown("**Rating by label**")
    r = pd.DataFrame(ins["rating_by_label"]).T.rename(columns={"CG": "Fake (CG)", "OR": "Genuine (OR)"})
    r.index = [f"{int(float(i))}★" for i in r.index]
    st.bar_chart(r, color=[FAKE_COLOR, REAL_COLOR], stack=False, y_label="reviews")
    st.caption("Ratings are almost identical for fake and genuine reviews, and every product "
               "category is split 50/50, so the review **text** is what carries the signal.")

    st.subheader("Words that give it away")
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Most fake-specific words**")
        st.write(", ".join(ins["cg_specific_words"]))
    with w2:
        st.markdown("**Most genuine-specific words**")
        st.write(", ".join(ins["or_specific_words"]))

    tf = pd.DataFrame(ins["lr_top_fake"][:12], columns=["term", "weight"])
    tg = pd.DataFrame(ins["lr_top_genuine"][:12], columns=["term", "weight"])
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top terms → fake** (Logistic Regression)")
        st.bar_chart(tf.sort_values("weight"), x="term", y="weight", color=FAKE_COLOR, horizontal=True)
    with t2:
        st.markdown("**Top terms → genuine**")
        st.bar_chart(tg.sort_values("weight", ascending=False), x="term", y="weight",
                     color=REAL_COLOR, horizontal=True)
