
import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SocialPulse",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model & TF-IDF
# -----------------------------
model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# Header
# -----------------------------
st.title("📊 SocialPulse")
st.subheader("Social Media Sentiment Analytics Dashboard")

st.markdown(
    "Analyze social media sentiment using Natural Language Processing "
    "and Machine Learning."
)

st.divider()

# -----------------------------
# Project Overview
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Size", "1.6M Tweets")

with col2:
    st.metric("Best Model", "Logistic Regression")

with col3:
    st.metric("Test Accuracy", "80.61%")

st.divider()

# -----------------------------
# Sentiment Prediction
# -----------------------------
st.header("🤖 Sentiment Prediction")

tweet = st.text_area(
    "Enter a social media post:",
    placeholder="Example: I really love this product!"
)

if st.button("🔍 Analyze Sentiment"):

    if tweet.strip() == "":
        st.warning("Please enter a tweet first.")

    else:
        cleaned = clean_text(tweet)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.success("😊 Predicted Sentiment: POSITIVE")
        else:
            st.error("😞 Predicted Sentiment: NEGATIVE")

st.divider()

# -----------------------------
# Model Performance
# -----------------------------
st.header("🏆 Model Performance")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Multinomial Naive Bayes",
        "Linear SVM"
    ],
    "Accuracy": [80.61, 78.53, 80.33],
    "Precision": [79.67, 78.78, 79.20],
    "Recall": [82.21, 78.09, 82.25],
    "F1 Score": [80.92, 78.43, 80.70]
})

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

# Performance chart
fig, ax = plt.subplots(figsize=(10, 5))

x = range(len(results))
width = 0.2

ax.bar(
    [i - 1.5 * width for i in x],
    results["Accuracy"],
    width,
    label="Accuracy"
)

ax.bar(
    [i - 0.5 * width for i in x],
    results["Precision"],
    width,
    label="Precision"
)

ax.bar(
    [i + 0.5 * width for i in x],
    results["Recall"],
    width,
    label="Recall"
)

ax.bar(
    [i + 1.5 * width for i in x],
    results["F1 Score"],
    width,
    label="F1 Score"
)

ax.set_xticks(list(x))
ax.set_xticklabels(results["Model"])
ax.set_ylabel("Score (%)")
ax.set_ylim(70, 90)
ax.set_title("Machine Learning Model Comparison")
ax.legend()

st.pyplot(fig)

st.divider()

# -----------------------------
# Conclusion
# -----------------------------
st.header("📌 Project Conclusion")

st.write(
    "Among the three evaluated machine learning models, Logistic Regression "
    "achieved the highest overall test accuracy of 80.61% and the highest "
    "F1 Score of 80.92%. Therefore, Logistic Regression was selected as "
    "the final sentiment classification model."
)

st.caption(
    "SocialPulse — NLP & Machine Learning based Social Media Sentiment Analytics"
)
