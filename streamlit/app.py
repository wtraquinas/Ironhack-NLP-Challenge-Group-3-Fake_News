import streamlit as st
import joblib
import numpy as np
from pathlib import Path

from preprocessing import preprocess

###############################################################
# PAGE CONFIG
###############################################################

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

###############################################################
# LOAD MODEL
###############################################################

@st.cache_resource
def load_models():

    BASE_DIR = Path(__file__).parent

    model = joblib.load(BASE_DIR / "mod4_joblib_svm_ngtfidf.pkl")

    vectorizer = joblib.load(BASE_DIR / "mod4_joblib_svm_ngtfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_models()

###############################################################
# EXAMPLES
###############################################################

examples = {

    "Choose an example...": "",

    "Real News #1":
    """GENEVA (Reuters) - A United Nations human rights panel called on the Russian Federation on Monday to step up prosecutions of racist attacks by ultra-nationalists and neo-Nazis and of hate speech by politicians. Russian authorities must intensify measures to  vigorously combat racist behavior in sports, particularly in football, and ensure that sports regulatory bodies investigate manifestations of racism, xenophobia and intolerance,  the U.N. Committee against Racial Discrimination (CERD) said. Fines or administrative sanctions should be imposed for such cases..""" ,

    "Real News #2":
    """BEIJING (Reuters) - China s Foreign Ministry said on Friday that it opposed North Korea s use of ballistic missiles in violation of United Nations Security Council resolutions. The emphasis on curbing North Korea s missile and nuclear capabilities should not come at the expense of pushing for a peaceful and diplomatic resolution, ministry spokeswoman Hua Chunying said after North Korea fired a missile that flew over Japan s northern Hokkaido island far out into the Pacific Ocean.""" ,

    "Fake News #1":
    """21st Century Wire says While politicians and news media alike are accusing Trump of pushing an  arms race  in his December 22nd Tweet about strengthening America s nuclear capability, the Obama administration made a landmark change to the policies governing space based weapons under the guise of missile defense. Why is the mainstream media silent on what Obama just did? Not only does this potentially damage relationships with other military powers such as Russia and China but is being called a massive waste of money.""" ,

    "Fake News #2":
    """The Obamas pride themselves on being staples in the black community who can do no wrong in the eyes of African Americans. Unfortunately for them, however, more and more black people are turning on the Obamas and seeing them for what they really are. After seeing Michelle Obama s graduation speech at the predominantly black Tuskegee University."""
}

###############################################################
# TITLE
###############################################################

st.title("📰 Fake News Detection using NLP")

st.markdown(
"""
Detect whether a news article is **Real** or **Fake**
using a **Linear Support Vector Machine** trained on
TF-IDF features.
"""
)

st.markdown(
"""
by Antonio Traquinas & Casilda Finat
"""
)

st.divider()

###############################################################
# TWO COLUMNS
###############################################################

left, right = st.columns([2,1])

###############################################################
# LEFT COLUMN
###############################################################

with left:

    #st.subheader("📝 News Article")

###############################################################
# EXAMPLE ARTICLES
###############################################################

    if "article" not in st.session_state:
        st.session_state.article = ""

    st.markdown("### 📝 Try News Examples :")

    st.markdown("#### 🟢 Real Articles")

    c1, c2 = st.columns(2)

    if c1.button("Reuters UN", use_container_width=True):
        st.session_state.article = examples["Real News #1"]
        st.rerun()

    if c2.button("Reuters China", use_container_width=True):
        st.session_state.article = examples["Real News #2"]
        st.rerun()

    st.markdown("#### 🔴 Fake Articles")

    c3, c4 = st.columns(2)

    if c3.button("21st Century Wire", use_container_width=True):
        st.session_state.article = examples["Fake News #1"]
        st.rerun()

    if c4.button("Obamas", use_container_width=True):
        st.session_state.article = examples["Fake News #2"]
        st.rerun()

    ###############################################################
    # TEXT AREA
    ###############################################################

    news = st.text_area(
        "Paste or edit a news article",
        key="article",
        height=350
    )


    col1, col2 = st.columns(2)

    predict = col1.button(
        "🔍 Predict",
        use_container_width=True
    )

    clear = col2.button(
        "🗑 Clear",
        use_container_width=True
    )

    if clear:

        st.session_state.news_text = ""

        st.rerun()

###############################################################
# RIGHT COLUMN
###############################################################

with right:

    st.subheader("Prediction")

    if predict:

        if len(news.strip()) == 0:

            st.warning("Please enter some text.")

        else:

            clean = preprocess(news)

            X = vectorizer.transform([clean])

            prediction = model.predict(X)[0]

            score = float(model.decision_function(X)[0])

            confidence = min(
                abs(score)/3,
                1
            )

            ##################################################

            if prediction == 1:

                st.success("✅ REAL NEWS")

                label = "Real"

            else:

                st.error("🚨 FAKE NEWS")

                label = "Fake"

            ##################################################

            st.metric(

                "Prediction",

                label

            )

            st.metric(

                "Confidence",

                f"{confidence*100:.1f}%"

            )

            st.metric(

                "Decision Score",

                f"{score:.3f}"

            )

            st.progress(confidence)

            ##################################################

            with st.expander("🔎 Preprocessed Text"):

                st.write(clean)

            ##################################################

            st.caption(
                """
                Larger absolute decision scores indicate the
                model is more confident in its prediction.
                """
            )

###############################################################
# FOOTER
###############################################################

st.divider()

st.markdown(
"""
**Model**

- Linear Support Vector Machine (LinearSVC)

- TF-IDF (Unigrams + Bigrams)

- NLP preprocessing:
    - lowercase
    - punctuation removal
    - stopword removal
    - lemmatization
"""
)