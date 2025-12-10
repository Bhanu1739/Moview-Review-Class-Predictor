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

if st.button("Predict Review"):
    input_df = pd.DataFrame([{
        "review" : review
    }])

    result = model.predict(input_df)[0]
    st.success(f"Predicted Review Class : {result}")