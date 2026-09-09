
import os
import time
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SocialPulse | Sentiment Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "assets")

MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

model, vectorizer = load_model()


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ---------- GLOBAL ---------- */
.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(88, 28, 135, 0.18), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(37, 99, 235, 0.16), transparent 28%),
        #09090f;
    color: #f5f5f5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #11111b 0%, #0b0b12 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #f1f1f1 !important;
}

/* ---------- HERO ---------- */
.hero {
    position: relative;
    padding: 42px 35px;
    margin-bottom: 30px;
    border-radius: 24px;
    overflow: hidden;
    background: linear-gradient(
        120deg,
        #171329,
        #101827,
        #18112b,
        #101827
    );
    background-size: 300% 300%;
    animation: gradientMove 10s ease infinite;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 20px 60px rgba(0,0,0,0.28);
}

.hero:before,
.hero:after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    filter: blur(70px);
    opacity: 0.22;
    animation: floatOrb 7s ease-in-out infinite alternate;
}

.hero:before {
    background: #7c3aed;
    top: -70px;
    left: -50px;
}

.hero:after {
    background: #2563eb;
    bottom: -80px;
    right: -40px;
    animation-delay: 2s;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    margin-top: 10px;
    color: #c7c8d5;
}

.hero-badge {
    display: inline-block;
    margin-bottom: 12px;
    padding: 7px 13px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: #d9d7ff;
    font-size: 0.82rem;
    font-weight: 600;
}

/* ---------- METRIC CARDS ---------- */
.metric-card {
    padding: 23px;
    border-radius: 20px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
    box-shadow: 0 8px 30px rgba(0,0,0,0.18);
    transition: all 0.3s ease;
    animation: fadeUp 0.7s ease both;
}

.metric-card:hover {
    transform: translateY(-7px);
    border-color: rgba(167,139,250,0.45);
    box-shadow: 0 16px 40px rgba(0,0,0,0.30);
}

.metric-label {
    font-size: 0.85rem;
    color: #a9abba;
    margin-bottom: 7px;
}

.metric-value {
    font-size: 1.65rem;
    font-weight: 750;
    color: #ffffff;
}

/* ---------- SECTION TITLE ---------- */
.section-title {
    margin-top: 40px;
    margin-bottom: 18px;
    font-size: 1.65rem;
    font-weight: 750;
    color: #ffffff;
}

.section-description {
    color: #aeb0bf;
    margin-bottom: 20px;
}

/* ---------- GLASS CARD ---------- */
.glass-card {
    padding: 22px;
    border-radius: 20px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    background: rgba(255,255,255,0.05);
}

/* ---------- PREDICTION BOX ---------- */
.prediction-box {
    padding: 26px;
    border-radius: 22px;
    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,0.15),
            rgba(37,99,235,0.10)
        );
    border: 1px solid rgba(167,139,250,0.18);
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
}

/* ---------- BUTTON ---------- */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    padding: 12px 18px;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
    background: linear-gradient(135deg, #6d28d9, #2563eb);
    color: white;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 30px rgba(37,99,235,0.30);
}

/* ---------- TEXT AREA ---------- */
textarea {
    border-radius: 14px !important;
}

/* ---------- ANIMATIONS ---------- */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes floatOrb {
    from { transform: translate(0,0) scale(1); }
    to { transform: translate(35px,25px) scale(1.15); }
}

@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ---------- MOBILE ---------- */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.1rem;
    }

    .hero {
        padding: 30px 24px;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🧠 SocialPulse")
    st.caption("Intelligent Sentiment Analytics")

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Sentiment Analytics",
            "Trend Analysis",
            "Word Analysis",
            "Model Performance",
            "Live Prediction"
        ]
    )

    st.markdown("---")
    st.caption("Sentiment140 Dataset")
    st.caption("1.6M Tweets")


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge">● AI-POWERED SOCIAL MEDIA ANALYTICS</div>
        <div class="hero-title">SocialPulse</div>
        <div class="hero-subtitle">
            Discover sentiment patterns, trends and language insights
            from 1.6 million social media posts.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Dataset Size", "1.6M Tweets"),
    ("Best Model", "Logistic Regression"),
    ("Test Accuracy", "80.61%"),
    ("Sentiment Classes", "2"),
]

for col, (label, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# HELPER
# =========================================================
def show_image(filename, caption=None):
    path = os.path.join(ASSET_DIR, filename)

    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Asset not found: {filename}")


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    st.markdown(
        '<div class="section-title">📊 Project Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'SocialPulse transforms social media text into actionable sentiment insights.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        show_image("sentiment_distribution.png", "Overall Sentiment Distribution")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        show_image("sentiment_percentage.png", "Sentiment Percentage")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# SENTIMENT ANALYTICS
# =========================================================
elif page == "Sentiment Analytics":

    st.markdown(
        '<div class="section-title">💬 Sentiment Analytics</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "sentiment_distribution.png",
            "Positive vs Negative Tweet Distribution"
        )

    with col2:
        show_image(
            "sentiment_percentage.png",
            "Sentiment Percentage"
        )

    st.markdown(
        '<div class="section-title">📏 Tweet Length Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "tweet_length_histogram.png",
            "Tweet Length Distribution"
        )

    with col2:
        show_image(
            "tweet_length_boxplot.png",
            "Tweet Length by Sentiment"
        )


# =========================================================
# TREND ANALYSIS
# =========================================================
elif page == "Trend Analysis":

    st.markdown(
        '<div class="section-title">📈 Sentiment Trend Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Monthly changes in the proportion of positive and negative sentiment.'
        '</div>',
        unsafe_allow_html=True
    )

    show_image(
        "monthly_sentiment_trend.png",
        "Monthly Sentiment Percentage Trend"
    )


# =========================================================
# WORD ANALYSIS
# =========================================================
elif page == "Word Analysis":

    st.markdown(
        '<div class="section-title">🔤 Word & Language Analysis</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.2, 1])

    with col1:
        show_image(
            "top_words.png",
            "Top 20 Most Frequent Words"
        )

    with col2:
        csv_path = os.path.join(ASSET_DIR, "top_words.csv")

        if os.path.exists(csv_path):
            top_words = pd.read_csv(csv_path)
            st.dataframe(
                top_words,
                use_container_width=True,
                hide_index=True
            )

    st.markdown(
        '<div class="section-title">☁️ Sentiment Word Clouds</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        show_image(
            "positive_wordcloud.png",
            "Positive Sentiment Word Cloud"
        )

    with col2:
        show_image(
            "negative_wordcloud.png",
            "Negative Sentiment Word Cloud"
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">🤖 Machine Learning Performance</div>',
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({
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
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section-title">🏆 Model Comparison</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(performance))

    ax.bar(
        [i - 0.3 for i in x],
        performance["Accuracy"],
        width=0.2,
        label="Accuracy"
    )

    ax.bar(
        [i - 0.1 for i in x],
        performance["Precision"],
        width=0.2,
        label="Precision"
    )

    ax.bar(
        [i + 0.1 for i in x],
        performance["Recall"],
        width=0.2,
        label="Recall"
    )

    ax.bar(
        [i + 0.3 for i in x],
        performance["F1 Score"],
        width=0.2,
        label="F1 Score"
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(performance["Model"])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(70, 90)
    ax.set_title("Model Performance Comparison")
    ax.legend()

    st.pyplot(fig, use_container_width=True)


# =========================================================
# LIVE PREDICTION
# =========================================================
elif page == "Live Prediction":

    st.markdown(
        '<div class="section-title">🔮 Live Sentiment Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Enter a social media post and let the trained model classify its sentiment.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)

    text = st.text_area(
        "Enter your tweet / social media post",
        placeholder="Example: I absolutely love this product! It is amazing ❤️",
        height=150
    )

    analyze = st.button("✨ Analyze Sentiment")

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:

        if not text.strip():
            st.warning("Please enter a social media post first.")

        else:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.8)

                vector = vectorizer.transform([text])
                prediction = model.predict(vector)[0]

            if prediction == 1:
                st.success("### 😊 Positive Sentiment")
                st.balloons()
            else:
                st.error("### 😞 Negative Sentiment")


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div style="
    text-align:center;
    margin-top:60px;
    padding:20px;
    color:#8f91a1;
    border-top:1px solid rgba(255,255,255,0.08);
">
    <b>SocialPulse</b> • Intelligent Social Media Sentiment Analytics
    <br>
    Built with Python, Scikit-learn, Pandas, Matplotlib & Streamlit
</div>
""", unsafe_allow_html=True)
