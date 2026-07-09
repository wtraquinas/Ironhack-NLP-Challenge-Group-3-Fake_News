import streamlit as st
import joblib
import numpy as np

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

    model = joblib.load("mod4_joblib_svm_ngtfidf.pkl")

    vectorizer = joblib.load("mod4_joblib_svm_ngtfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_models()

###############################################################
# EXAMPLES
###############################################################

examples = {

    "Choose an example...": "",

    "Real News #1":
    """The Federal Reserve announced today that interest rates
    will remain unchanged while inflation continues to slow.""" ,

    "Real News #2":
    """NASA successfully launched a new weather satellite to improve
    climate monitoring across North America.""" ,

    "Fake News #1":
    """Scientists confirm the Earth will become completely dark
    for six days beginning next month.""" ,

    "Fake News #2":
    """Drinking hot lemon water every hour permanently eliminates
    every virus from the human body."""
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

st.divider()

###############################################################
# TWO COLUMNS
###############################################################

left, right = st.columns([2,1])

###############################################################
# LEFT COLUMN
###############################################################

with left:

    st.subheader("📝 News Article")

    selected = st.selectbox(
        "Example Articles",
        list(examples.keys())
    )

    if "news_text" not in st.session_state:
        st.session_state.news_text = ""

    if selected != "Choose an example...":
        st.session_state.news_text = examples[selected]

    news = st.text_area(

        "Paste or edit a news article",

        value=st.session_state.news_text,

        height=350,

        key="article"

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