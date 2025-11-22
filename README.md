# Disaster_Tweets-Fake_Or_Real
This project identifies whether a tweet describes a real disaster event or is fake/misleading, inspired by the Kaggle Real or Not: Disaster Tweets dataset. The system combines traditional ML (TF-IDF + Random Forest/XGBoost) with Deep Learning (Word2Vec + LSTM) to achieve highly accurate predictions. Baseline ML models achieved ~77% accuracy, while the LSTM model with pretrained embeddings reached 98% accuracy.

A Streamlit dashboard allows users to input any tweet and instantly view the prediction along with explainability, using a TF-IDF–based word-importance visualizer.

🔗 Dataset

Kaggle – Real or Not? Disaster Tweets
https://www.kaggle.com/competitions/nlp-getting-started

🛠 Tech Stack

Python, NumPy, Pandas

Scikit-learn, XGBoost

Gensim Word2Vec

TensorFlow + Keras LSTM

Streamlit

🔄 Workflow

Data Cleaning → TF-IDF + ML Baselines → Word2Vec Embeddings → LSTM Training → Explainability → Streamlit Deployment
