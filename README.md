# 📰 Fake News Detection using NLP and Linear SVM

## About

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify news articles as **Real** or **Fake**. The solution uses text preprocessing, TF-IDF vectorization, and a Linear Support Vector Machine (SVM) classifier to achieve high classification performance.

A user-friendly **Streamlit web application** was also developed, allowing users to paste news articles or select example articles and receive instant predictions with confidence scores.



<p align="center">
<img src="/images/ToyStory-3758131314.jpg" height="420" alt="Horse prediction">
</p>

---

## Sample Prediction

### Streamlit Application

<center>
<table>
  <td><img src="/images/deploy_streamlit_02_pred_260710.jpg" height="420" alt="Horse prediction"></td>
  <td> &nbsp &nbsp </td>
  <td><img src="/images/deploy_streamlit_02_pred_260710.jpg" height="420" alt="Dookie prediction"></td>
</table>
</center>



---

## Live Demo

🌐 **Streamlit Cloud**

[https://project2-nlp-fakenews.streamlit.app](https://project2-nlp-fakenews.streamlit.app)

---

# Problem Statement

The rapid spread of misinformation through online news and social media makes it increasingly difficult to distinguish reliable information from fake news.

The objective of this project is to build a machine learning model capable of automatically classifying news articles into:

* **Real**
* **Fake**

using only the article text.

---

# Dataset

**Dataset:** Fake News Dataset

* File: `dataset/data.csv`
* Text-based binary classification dataset


---

# Model Architecture

## NLP Pipeline

```
Raw News Article
        │
        ▼
Text Cleaning
 • Lowercase
 • Remove punctuation
 • Remove numbers
 • Remove stopwords
 • Lemmatization
        │
        ▼
TF-IDF Vectorization
        │
        ▼
Linear Support Vector Machine
        │
        ▼
Prediction
Real / Fake
```

### Text Preprocessing

* Lowercase conversion
* Remove punctuation
* Remove numbers
* Remove stop words
* Lemmatization

### Feature Engineering

* TF-IDF Vectorizer
* TF-IDF with N-grams (best-performing model)

### Machine Learning Models Tested

* Multinomial Naive Bayes
* Linear SVM
* Linear SVM + TF-IDF
* **Linear SVM + TF-IDF N-grams (Final Model)**

The final deployed model was saved using **Joblib** together with the trained TF-IDF vectorizer.

---

# Results

The final model achieved strong performance on both validation and test datasets.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example evaluation:

```
Accuracy: (generated in notebook)

Classification Report

Precision
Recall
F1-score

Confusion Matrix
```

The notebook also evaluates the model on unseen validation data and exports predictions to:

```
dataset/validation_predictions.csv
```

---

# Setup & Installation

Clone the repository

```bash
git clone https://github.com/Casildagsf/Ironhack-NLP-Challenge-Group-3-Fake_News

cd fake-news-nlp
```

Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

# Project Structure

```
project/

│
├── requirements.txt
├── README.md
├── model_at_4_svm_tidf_joblib_predict.ipynb
│
├── dataset/
│   ├── data.csv
│   ├── validation_data.csv
│   └── validation_predictions.csv
│
├── models/
│   ├── mod4_joblib_svm_ngtfidf.pkl
│   └── mod4_joblib_svm_ngtfidf_vectorizer.pkl
│
├── streamlit/
│   ├── app.py
│
└── images/
    └── screenshots
    └── memes
```

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib
* Streamlit
* Jupyter Notebook

---

# Future Improvements

* Fine-tune transformer models (BERT, RoBERTa, DistilBERT)
* Add explainability using SHAP or LIME
* Improve confidence estimation
* Train on larger and more recent datasets
* Support multilingual fake news detection
* Add source credibility analysis
* Deploy with Docker and CI/CD pipeline

---

# Authors

**Antonio Traquinas** - https://github.com/wtraquinas/

and 

**Casilda Finat** - https://github.com/Casildagsf/

AI Engineering | Machine Learning | NLP

---

## Acknowledgements

This project was developed as part of the **IronHack AI Engineering Bootcamp**, demonstrating the complete machine learning workflow from data preprocessing to deployment of an interactive web application.
