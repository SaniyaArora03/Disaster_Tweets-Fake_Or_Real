import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title=" Fake Tweet Detection Dashboard", layout="centered")
st.title("Fake Tweet Detection")
st.markdown("Enter a tweet below to check if it's **Real or Fake**, and understand why the model made this prediction. 🧩")

# -------------------- Load Models --------------------
@st.cache_resource
def load_models():
    tokenizer = pickle.load(open("tokenizer.pkl", "rb"))
    lstm_model = tf.keras.models.load_model("lstm_model.h5")
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    return tokenizer, lstm_model, tfidf

tokenizer, lstm_model, tfidf = load_models()
MAX_SEQUENCE_LENGTH = 50  # adjust if needed

# -------------------- Helpers --------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text, tokenizer, maxlen=MAX_SEQUENCE_LENGTH):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=maxlen)

def get_word_importance(tweet, tfidf):
    """Use TF-IDF weights as a lightweight importance proxy."""
    words = tweet.split()
    tfidf_features = tfidf.transform([tweet])
    feature_names = tfidf.get_feature_names_out()
    importance = []
    for w in words:
        if w in feature_names:
            idx = np.where(feature_names == w)[0][0]
            importance.append(tfidf_features[0, idx])
        else:
            importance.append(0)
    importance = np.array(importance)
    if np.all(importance == 0):
        importance = np.random.rand(len(words)) * 0.01  # tiny noise to avoid all zero
    return words, importance

def highlight_words(words, importance, label):
    """Render colored text inline with polarity-based coloring."""
    norm = plt.Normalize(importance.min(), importance.max() + 1e-8)
    cmap = plt.cm.get_cmap('RdBu')  # red-blue
    colored_text = ""
    for w, score in zip(words, importance):
        rgba = cmap(norm(score))
        color = mcolors.rgb2hex(rgba)
        # Polarity: red = fake, blue = real
        text_color = "white" if label == "fake" else "black"
        colored_text += f"<span style='background-color:{color}; padding:3px 6px; border-radius:5px; color:{text_color}; margin:2px;'>{w}</span> "
    return colored_text

# -------------------- Input --------------------
tweet = st.text_area("Enter a tweet:", placeholder="e.g. We're shaking...It's an earthquake")

if st.button("Analyze Tweet"):
    if not tweet.strip():
        st.warning("Please enter a tweet first.")
    else:
        clean_tweet = clean_text(tweet)
        seq = preprocess_text(clean_tweet, tokenizer)

        # Prediction
        pred = lstm_model.predict(seq)[0][0]
        label = "real" if pred >= 0.5 else "fake"
        conf = pred if pred >= 0.5 else 1 - pred

        st.subheader(" Prediction Result:")
        emoji = " REAL / GENUINE" if label == "real" else "FAKE or MISLEADING"
        st.markdown(f"**{emoji}**  \nConfidence: **{conf*100:.2f}%**")

        # -------------------- Word Importance Visualization --------------------
        try:
            words, importance = get_word_importance(clean_tweet, tfidf)
            st.markdown("###  Word Importance Visualization")
            st.markdown(highlight_words(words, importance, label), unsafe_allow_html=True)
        except Exception as e:
            st.warning("⚠️ Could not compute reasoning visualization.")
            st.info(f"Reason: {e}")

# -------------------- Footer --------------------
st.markdown("---")

