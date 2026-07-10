# 📰 Fake News Detection using NLP and Linear SVM

<br>

## About

This project applies **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify news articles as **Real** or **Fake**. The solution uses text preprocessing, TF-IDF vectorization, and a Linear Support Vector Machine (SVM) classifier to achieve high classification performance.

A user-friendly **Streamlit web application** was also developed, allowing users to paste news articles or select example articles and receive instant predictions with confidence scores.



<p align="center">
<img src="/images/ToyStory-3758131314.jpg" height="420" alt="Horse prediction">
</p>

<br>

---

## MODEL TRACKING SPREADSHEET

| Model                                        | Feature extraction   | Training model          | Accuracy | test    | Notes                                                                                                                |
|----------------------------------------------|----------------------|-------------------------|----------|---------|----------------------------------------------------------------------------------------------------------------------|
| Base_model                                   | TF-IDF               | Log Regression          | 97.80%   |         | Baseline - start here                                                                                                |
| Base_model                                   | n-grams (TF-IDF)     | Log Regression          | 98%      |         | Baseline - start here                                                                                                |
| Model_cf_2                                   | Embedding - Word2Vec | Linear SVM              | 97%      |         | When increasing vector size from 100 - 300 increased accuracy from 96% - 97%                                         |
| Model_cf_3                                   | BoW                  | Multinomial Naive Bayes | 93%      |         | Decreased accuracy                                                                                                   |
| Model_cf_4                                   | n-grams (TF-IDF)     | Multinomial Naive Bayes | 93%      |         | same accuracy as with model_cf_3                                                                                     |
| Model_cf_5                                   | Embedding - Word2Vec | Log Regression          | 96,80%   |         |                                                                                                                      |
| Model_cf_6                                   | n-grams (TF-IDF)     | Log Regression          | 98,15%   |         | "same as baseline but changing to: max_features=20000, ngram_range=(1, 3) from max_features=5000, ngram_range=(1, 2) - Energy innefficient for the slight increase"                                                                         |
| model_at_2                                   | TF-IDF               | Multinomial Naive Bayes | 93,8%    |         |                                                                                                                      |
| model_at_2                                   | n-grams (TF-IDF)     | Multinomial Naive Bayes | 94,6%    |         |                                                                                                                      |
| model_at_3                                   | Embedding - Glove    | Linear SVM              | 94,6%    |         |                                                                                                                      |
| model_at_4                                   | TF-IDF               | Linear SVM              | 99,3%    |         |                                                                                                                      |
| model_at_4                                   | n-grams (TF-IDF)     | Linear SVM              | 99,4%    | 99,30%  |                                                                                                                      |
| model_at_5                                   | bert-base-uncased    | TL Fine-tuned BERT      | 99,98%   | 100,00% | 20mins on Colab GPU                                                                                                  |

<br>

---

## Sample Prediction

### Streamlit Application

<center>
<table>
  <td><img src="/images/deploy_streamlit_02_pred_260710.jpg" height="420" alt="Horse prediction"></td>
  <td> &nbsp &nbsp </td>
  <td><img src="/images/deploy_streamlit_03_pred_260710.jpg" height="420" alt="Dookie prediction"></td>
</table>
</center>


<br>


---

## Live Demo

🌐 **Streamlit Cloud**

[https://project2-nlp-fakenews.streamlit.app](https://project2-nlp-fakenews.streamlit.app)


<br>


---

# Problem Statement

<p align="center">
<img src="/images/bbc_fake_news.jpg" height="420" alt="BBC">
</p>

The rapid spread of misinformation through online news and social media makes it increasingly difficult to distinguish reliable information from fake news.

The objective of this project is to build a machine learning model capable of automatically classifying news articles into:

* **Real**
* **Fake**

using only the article text.

<br>



---

# Dataset

**Dataset:** Fake News Dataset
- dataset containing news articles with the following columns:

    - label: 0 if the news is fake, 1 if the news is real.
    - title: The headline of the news article.
    - text: The full content of the article.
    - subject: The category or topic of the news.
    - date: The publication date of the article.

* File: `dataset/data.csv`
* Text-based binary classification dataset
  
<br>

<br>

<p align="center">
<img src="/images/ITDonkeySocks.jpg" height="220" alt="Donkey Socks">
</p>

<br>

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


<br>


### Text Preprocessing

* Lowercase conversion
* Remove punctuation
* Remove numbers
* Remove stop words
* Lemmatization


<br>


### Feature Engineering

* TF-IDF Vectorizer
* TF-IDF with N-grams (best-performing model)

### Machine Learning Models Tested

* Multinomial Naive Bayes
* Linear SVM
* Linear SVM + TF-IDF
* **Linear SVM + TF-IDF N-grams (Final Model)**

The final deployed model was saved using **Joblib** together with the trained TF-IDF vectorizer.




<br>

---

# Results

The final model achieved strong performance on both validation and test datasets.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score

<p align="center">
<img src="/images/test_accuracy_lsvm_ngtfidf_20260710.jpg" height="420" alt="test_accuracy_lsvm_ngtfidf">
</p>

The notebook also evaluates the model on unseen validation data and exports predictions to:

```
dataset/validation_predictions.csv
```

<br>

---

## MODEL_AT_5 - Transfer Learning: Fine-tuned BERT (bert-base-uncased)


<p align="center">
<img src="/images/BERT_encoder.jpg" height="420" alt="BERT_encoder">
</p>

We also tested with Transfer Learning - Transformers Model BERT, arriving at **100% accuracy** in Test.
- Unlike TF-IDF (a fixed, non-trainable transform computed once), during trainer.train() the entire 110M-parameter BERT body is updated by backpropagation too, not just the classification head.
- So "feature extraction" and "training" aren't separate stages anymore — the model is simultaneously learning what features matter for fake-news detection and adjusting its language understanding to your specific dataset.
- That's the core meaning of transfer learning here: you start from features learned on a massive general-text pretraining corpus, then adapt them to this task.

<p align="center">
<img src="/images/train_accuracy_finetune_bert_20260710.jpg" height="420" alt="train_accuracy_finetune_bert">
<img src="/images/test_accuracy_finetune_bert_20260710.jpg" height="420" alt="test_accuracy_finetune_bert">
</p>


<br>

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


<br>


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

<br>

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

<br>

---

# Future Improvements

* Fine-tune other faster transformer models (DistilBERT)
* Train on larger and more recent datasets
* Support multilingual fake news detection
* Add source credibility analysis

<br>

---

# Authors

**Antonio Traquinas** - https://github.com/wtraquinas/

&

**Casilda Finat** - https://github.com/Casildagsf/

<br>

---

<p align="center">
<img src="/images/fakenews_fallout.jpg" height="220" alt="fakenews_fallout">
</p>


<br>


## Acknowledgements

Special thanks to the indefatigable **Luis Junco**, always ready to clear any doubt with seemingly infinite patience.

This project was developed as part of the **IronHack AI Engineering Bootcamp**, demonstrating the complete machine learning workflow from data preprocessing to deployment of an interactive web application.

<br>

