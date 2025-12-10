import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

with open("nb_model.pkl","rb") as f:
    model = pickle.load(f)

st.title("Movie Review Positive/Negative Class Predictor")
st.write("Bhanu Prakash Reddy ChinnaPashula")

review = st.text_input("review")

import nltk
import re
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def Clean(text):
  regex = "[^A-Za-z\s]"
  text = re.sub(regex," ",text)
  text = text.lower()
  tokens = nltk.word_tokenize(text)
  stop_words = set(stopwords.words('english'))
  filtered_tokens = [word for word in tokens if word not in stop_words]
  lemmatized_tokens = [WordNetLemmatizer().lemmatize(token) for token in filtered_tokens]
  return " ".join(lemmatized_tokens)

if st.button("Predict Review"):
    cleaned_text = Clean(review)
    input_df = pd.DataFrame([{
        "review" : cleaned_text
    }])

    result = model.predict([input_df])
    st.success(f"Predicted Review Class : {result[0]}")