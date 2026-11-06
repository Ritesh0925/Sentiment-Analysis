import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Start the backend with: flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Amazon Alexa Review Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV or TSV file for bulk prediction",
    type=["csv", "tsv"],
)

# Text input for sentiment prediction
user_input = st.text_input("Or enter a single review to predict:", "")

# Prediction button
if st.button("Predict"):
    if uploaded_file is not None:
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        response_bytes = BytesIO(response.content)
        
        is_tsv = uploaded_file.name.lower().endswith(".tsv")
        sep = "\t" if is_tsv else ","
        response_df = pd.read_csv(response_bytes, sep=sep)

        st.success(f"Successfully processed {len(response_df)} reviews!")
        st.dataframe(response_df.head(10))

        ext = "tsv" if is_tsv else "csv"
        st.download_button(
            label=f"Download Predictions (.{ext.upper()})",
            data=response_bytes.getvalue(),
            file_name=f"Predictions.{ext}",
            mime="text/tab-separated-values" if is_tsv else "text/csv",
            key="result_download_button",
        )

    elif user_input.strip() != "":
        response = requests.post(prediction_endpoint, json={"text": user_input})
        result = response.json()
        if "sentiment" in result or "prediction" in result:
            sentiment = result.get("sentiment", result.get("prediction"))
            conf = result.get("confidence", "")
            st.success(f"Predicted Sentiment: **{sentiment}** ({conf}%)")
        else:
            st.error(result.get("error", "Failed to analyze sentiment."))
    else:
        st.warning("Please upload a file or type a review.")
