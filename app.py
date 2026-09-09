
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
    page_title="SocialPulse | AI Sentiment Analytics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(BASE_DIR, "assets")
ADVANCED_DIR = os.path.join(BASE_DIR, "advanced")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "logistic_regression_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "tfidf_vectorizer.pkl"
)

TOPIC_MODEL_PATH = os.path.join(
    ADVANCED_DIR,
    "kmeans_topic_model.pkl"
)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    sentiment_model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    topic_model = joblib.load(
        TOPIC_MODEL_PATH
    )

    return (
        sentiment_model,
        vectorizer,
        topic_model
    )


try:
    model, vectorizer, topic_model = load_models()
    model_loaded = True

except Exception as e:
    model_loaded = False
    model_error = str(e)


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
<style>

/* ======================================================
   GLOBAL
   ====================================================== */

.stApp {

    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(124, 58, 237, 0.20),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 5%,
            rgba(37, 99, 235, 0.18),
            transparent 28%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(236, 72, 153, 0.08),
            transparent 30%
        ),
        #07070d;

    color: #f8fafc;
}

.block-container {

    max-width: 1450px;

    padding-top: 1.5rem;
    padding-bottom: 4rem;

}


/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #11111c 0%,
            #0a0a11 100%
        );

    border-right:
        1px solid
        rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {

    color: #eef0f6 !important;
}


/* ======================================================
   HERO
   ====================================================== */

.hero {

    padding: 42px 40px;

    border-radius: 28px;

    margin-bottom: 30px;

    background:
        linear-gradient(
            115deg,
            #19132f,
            #10192d,
            #1c1130,
            #10192d
        );

    background-size: 350% 350%;

    animation:
        heroGradient 12s ease infinite;

    border:
        1px solid
        rgba(255,255,255,0.10);

    box-shadow:
        0 25px 70px
        rgba(0,0,0,0.32);

    overflow: hidden;

    position: relative;

}


.hero::before {

    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    top: -130px;
    left: -80px;

    border-radius: 50%;

    background:
        rgba(124,58,237,0.22);

    filter: blur(65px);

    animation:
        floatingOrb 7s ease-in-out infinite alternate;

}


.hero::after {

    content: "";

    position: absolute;

    width: 260px;
    height: 260px;

    bottom: -150px;
    right: -70px;

    border-radius: 50%;

    background:
        rgba(37,99,235,0.22);

    filter: blur(70px);

    animation:
        floatingOrb 9s ease-in-out infinite alternate-reverse;

}


.hero-content {

    position: relative;

    z-index: 2;

}


.hero-badge {

    display: inline-block;

    padding:
        7px 14px;

    border-radius:
        999px;

    background:
        rgba(255,255,255,0.07);

    border:
        1px solid
        rgba(255,255,255,0.12);

    font-size:
        0.80rem;

    font-weight:
        700;

    letter-spacing:
        0.3px;

    color:
        #d7d3ff;

}


.hero-title {

    margin-top:
        14px;

    font-size:
        3.35rem;

    font-weight:
        900;

    letter-spacing:
        -1.8px;

}


.hero-subtitle {

    margin-top:
        9px;

    max-width:
        850px;

    font-size:
        1.08rem;

    line-height:
        1.7;

    color:
        #c3c7d4;

}


/* ======================================================
   KPI CARDS
   ====================================================== */

.kpi-card {

    padding:
        23px 22px;

    min-height:
        112px;

    border-radius:
        20px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.055),
            rgba(255,255,255,0.025)
        );

    border:
        1px solid
        rgba(255,255,255,0.09);

    box-shadow:
        0 12px 35px
        rgba(0,0,0,0.20);

    transition:
        transform .30s ease,
        box-shadow .30s ease,
        border-color .30s ease;

    animation:
        riseIn .65s ease both;

}


.kpi-card:hover {

    transform:
        translateY(-8px);

    box-shadow:
        0 20px 45px
        rgba(0,0,0,0.34);

    border-color:
        rgba(167,139,250,0.34);

}


.kpi-label {

    color:
        #9ea4b5;

    font-size:
        0.83rem;

    font-weight:
        600;

}


.kpi-value {

    margin-top:
        9px;

    color:
        #ffffff;

    font-size:
        1.55rem;

    font-weight:
        850;

}


/* ======================================================
   SECTION HEADER
   ====================================================== */

.section-heading {

    margin-top:
        42px;

    margin-bottom:
        8px;

    font-size:
        1.65rem;

    font-weight:
        850;

    letter-spacing:
        -0.4px;

}


.section-text {

    color:
        #a8adbd;

    margin-bottom:
        20px;

    line-height:
        1.6;

}


/* ======================================================
   CARDS
   ====================================================== */

.glass-card {

    padding:
        20px;

    border-radius:
        20px;

    background:
        rgba(255,255,255,0.035);

    border:
        1px solid
        rgba(255,255,255,0.08);

    box-shadow:
        0 10px 30px
        rgba(0,0,0,0.16);

    transition:
        transform .30s ease,
        background .30s ease;

}


.glass-card:hover {

    transform:
        translateY(-4px);

    background:
        rgba(255,255,255,0.045);

}


/* ======================================================
   BUTTON
   ====================================================== */

.stButton > button {

    border:
        1px solid
        rgba(167,139,250,0.28) !important;

    border-radius:
        13px !important;

    min-height:
        45px;

    font-weight:
        750 !important;

    color:
        white !important;

    background:
        linear-gradient(
            135deg,
            #6d28d9,
            #2563eb
        ) !important;

    box-shadow:
        0 9px 25px
        rgba(37,99,235,0.18);

    transition:
        all .25s ease !important;

}


.stButton > button:hover {

    transform:
        translateY(-3px)
        scale(1.01);

    box-shadow:
        0 14px 32px
        rgba(37,99,235,0.28);

}


/* ======================================================
   TEXT AREA
   ====================================================== */

textarea {

    border-radius:
        15px !important;

}


/* ======================================================
   RESULT CARD
   ====================================================== */

.prediction-result {

    padding:
        28px;

    margin-top:
        18px;

    text-align:
        center;

    border-radius:
        22px;

    animation:
        resultAppear .45s ease both;

}


.pred-positive {

    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,0.15),
            rgba(34,197,94,0.05)
        );

    border:
        1px solid
        rgba(34,197,94,0.35);

}


.pred-negative {

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,0.15),
            rgba(220,38,38,0.05)
        );

    border:
        1px solid
        rgba(239,68,68,0.35);

}


.pred-title {

    font-size:
        2rem;

    font-weight:
        900;

}


.pred-confidence {

    margin-top:
        8px;

    color:
        #d6d8e2;

    font-size:
        1.05rem;

}


/* ======================================================
   INFO BOX
   ====================================================== */

.info-card {

    padding:
        18px 20px;

    border-radius:
        16px;

    background:
        rgba(59,130,246,0.08);

    border:
        1px solid
        rgba(59,130,246,0.18);

    color:
        #dce7ff;

}


/* ======================================================
   FOOTER
   ====================================================== */

.footer {

    text-align:
        center;

    margin-top:
        60px;

    padding:
        25px;

    color:
        #858a9b;

    border-top:
        1px solid
        rgba(255,255,255,0.08);

}


/* ======================================================
   ANIMATIONS
   ====================================================== */

@keyframes heroGradient {

    0% {
        background-position:
            0% 50%;
    }

    50% {
        background-position:
            100% 50%;
    }

    100% {
        background-position:
            0% 50%;
    }

}


@keyframes floatingOrb {

    from {
        transform:
            translate(0,0)
            scale(1);
    }

    to {
        transform:
            translate(42px,28px)
            scale(1.14);
    }

}


@keyframes riseIn {

    from {
        opacity:
            0;

        transform:
            translateY(15px);
    }

    to {
        opacity:
            1;

        transform:
            translateY(0);
    }

}


@keyframes resultAppear {

    from {
        opacity:
            0;

        transform:
            scale(.95)
            translateY(7px);
    }

    to {
        opacity:
            1;

        transform:
            scale(1)
            translateY(0);
    }

}


/* ======================================================
   MOBILE
   ====================================================== */

@media (max-width: 800px) {

    .hero {

        padding:
            30px 24px;

    }

    .hero-title {

        font-size:
            2.25rem;

    }

    .hero-subtitle {

        font-size:
            .98rem;

    }

}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
with st.sidebar:

    st.markdown("## 🧠 SocialPulse")
    st.caption(
        "AI-Powered Sentiment Analytics"
    )

    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Sentiment Analytics",
            "Trend Analysis",
            "Word Analysis",
            "Topic Intelligence",
            "Anomaly Detection",
            "Model Performance",
            "Live Prediction",
            "Batch Prediction",
            "Model Explainability"
        ]
    )

    st.markdown("---")

    st.caption("Dataset")
    st.caption("Sentiment140 • 1.6M Tweets")

    st.markdown("---")

    st.caption(
        "NLP • ML • Clustering • Explainable AI"
    )


# =========================================================
# HERO
# =========================================================
st.html(
    """
    <div class="hero">
        <div class="hero-content">

            <div class="hero-badge">
                ● AI-POWERED SOCIAL MEDIA ANALYTICS
            </div>

            <div class="hero-title">
                🧠 SocialPulse
            </div>

            <div class="hero-subtitle">
                Intelligent sentiment, trend and topic analytics
                powered by Natural Language Processing and Machine Learning.
            </div>

        </div>
    </div>
    """
)


# =========================================================
# TOP KPI CARDS
# =========================================================
kpi_columns = st.columns(4)

kpis = [
    ("Dataset Size", "1.6M Tweets"),
    ("Best Model", "Logistic Regression"),
    ("Test Accuracy", "80.61%"),
    ("Detected Topics", "6"),
]

for column, (label, value) in zip(
    kpi_columns,
    kpis
):

    with column:

        st.html(
            f"""
            <div class="kpi-card">

                <div class="kpi-label">
                    {label}
                </div>

                <div class="kpi-value">
                    {value}
                </div>

            </div>
            """
        )


# =========================================================
# IMAGE HELPER
# =========================================================
def display_asset(
    filename,
    caption=None
):

    path = os.path.join(
        ASSET_DIR,
        filename
    )

    if os.path.exists(path):

        st.image(
            path,
            caption=caption,
            use_container_width=True
        )

    else:

        st.warning(
            f"Missing asset: {filename}"
        )


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    st.html(
        """
        <div class="section-heading">
            📊 Intelligence Overview
        </div>
        """
    )

    st.markdown(
        """
        SocialPulse analyzes large-scale social media
        text using NLP preprocessing, TF-IDF feature
        engineering, supervised classification and
        unsupervised topic discovery.
        """
    )

    st.markdown("### Sentiment Snapshot")

    left, right = st.columns(2)

    with left:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        display_asset(
            "sentiment_distribution.png",
            "Positive vs Negative Tweet Distribution"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    with right:

        st.markdown(
            '<div class="glass-card">',
            unsafe_allow_html=True
        )

        display_asset(
            "sentiment_percentage.png",
            "Sentiment Percentage"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### 🚀 Intelligent Capabilities")

    a, b, c = st.columns(3)

    with a:

        st.markdown(
            "#### 💬 Sentiment Classification"
        )

        st.write(
            "Predict Positive or Negative sentiment "
            "for new social media posts."
        )

    with b:

        st.markdown(
            "#### 🧩 Topic Discovery"
        )

        st.write(
            "Discover latent discussion topics "
            "using K-Means clustering."
        )

    with c:

        st.markdown(
            "#### 🧠 Explainable AI"
        )

        st.write(
            "Understand which words influence "
            "the Logistic Regression model."
        )


# =========================================================
# SENTIMENT ANALYTICS
# =========================================================
elif page == "Sentiment Analytics":

    st.html(
        """
        <div class="section-heading">
            💬 Sentiment Analytics
        </div>
        """
    )

    st.markdown(
        """
        Explore the overall sentiment composition and
        distribution of tweet lengths.
        """
    )

    left, right = st.columns(2)

    with left:
        display_asset(
            "sentiment_distribution.png",
            "Sentiment Distribution"
        )

    with right:
        display_asset(
            "sentiment_percentage.png",
            "Sentiment Percentage"
        )

    st.html(
        """
        <div class="section-heading">
            📏 Tweet Length Analysis
        </div>
        """
    )

    left, right = st.columns(2)

    with left:

        display_asset(
            "tweet_length_histogram.png",
            "Tweet Length Distribution"
        )

    with right:

        display_asset(
            "tweet_length_boxplot.png",
            "Tweet Length by Sentiment"
        )


# =========================================================
# TREND ANALYSIS
# =========================================================
elif page == "Trend Analysis":

    st.html(
        """
        <div class="section-heading">
            📈 Sentiment Trend Detection
        </div>
        """
    )

    st.markdown(
        """
        Track monthly changes in Positive and Negative
        sentiment proportions across the available
        Sentiment140 time period.
        """
    )

    display_asset(
        "monthly_sentiment_trend.png",
        "Monthly Sentiment Trend"
    )

    monthly_path = os.path.join(
        ASSET_DIR,
        "monthly_percentage.csv"
    )

    if os.path.exists(monthly_path):

        trend_data = pd.read_csv(
            monthly_path
        )

        st.markdown("### 📋 Monthly Summary")

        st.dataframe(
            trend_data,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# WORD ANALYSIS
# =========================================================
elif page == "Word Analysis":

    st.html(
        """
        <div class="section-heading">
            🔤 Word & Language Intelligence
        </div>
        """
    )

    left, right = st.columns(
        [1.15, 1]
    )

    with left:

        display_asset(
            "top_words.png",
            "Top 20 Most Frequent Words"
        )

    with right:

        words_path = os.path.join(
            ASSET_DIR,
            "top_words.csv"
        )

        if os.path.exists(words_path):

            words = pd.read_csv(
                words_path
            )

            st.markdown(
                "### Top Word Frequency"
            )

            st.dataframe(
                words,
                use_container_width=True,
                hide_index=True
            )

    st.html(
        """
        <div class="section-heading">
            ☁️ Positive vs Negative Language
        </div>
        """
    )

    left, right = st.columns(2)

    with left:

        display_asset(
            "positive_wordcloud.png",
            "Positive Sentiment Word Cloud"
        )

    with right:

        display_asset(
            "negative_wordcloud.png",
            "Negative Sentiment Word Cloud"
        )


# =========================================================
# TOPIC INTELLIGENCE
# =========================================================
elif page == "Topic Intelligence":

    st.html(
        """
        <div class="section-heading">
            🧩 Topic Intelligence
        </div>
        """
    )

    st.markdown(
        """
        K-Means clustering identifies six latent topic
        groups from TF-IDF representations of a
        representative sample of tweets.
        """
    )

    topic_chart = os.path.join(
        ADVANCED_DIR,
        "topic_distribution.png"
    )

    topic_keywords = os.path.join(
        ADVANCED_DIR,
        "topic_keywords.csv"
    )

    topic_sentiment_chart = os.path.join(
        ADVANCED_DIR,
        "topic_sentiment.png"
    )

    left, right = st.columns(2)

    with left:

        if os.path.exists(topic_chart):

            st.image(
                topic_chart,
                caption="Topic Distribution",
                use_container_width=True
            )

    with right:

        if os.path.exists(topic_keywords):

            topic_df = pd.read_csv(
                topic_keywords
            )

            st.markdown(
                "### 🔑 Topic Keywords"
            )

            st.dataframe(
                topic_df,
                use_container_width=True,
                hide_index=True
            )

    st.html(
        """
        <div class="section-heading">
            💬 Sentiment × Topic
        </div>
        """
    )

    if os.path.exists(
        topic_sentiment_chart
    ):

        st.image(
            topic_sentiment_chart,
            caption="Sentiment Distribution Across Topics",
            use_container_width=True
        )

    topic_percentage_path = os.path.join(
        ADVANCED_DIR,
        "topic_sentiment_percentage.csv"
    )

    if os.path.exists(
        topic_percentage_path
    ):

        topic_percentage = pd.read_csv(
            topic_percentage_path,
            index_col=0
        )

        st.markdown(
            "### Topic Sentiment Matrix"
        )

        st.dataframe(
            topic_percentage.round(2),
            use_container_width=True
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================
elif page == "Anomaly Detection":

    st.html(
        """
        <div class="section-heading">
            🚨 Sentiment Anomaly Detection
        </div>
        """
    )

    st.markdown(
        """
        The anomaly module flags unusually large monthly
        changes in positive sentiment relative to the
        observed variation.
        """
    )

    anomaly_path = os.path.join(
        ADVANCED_DIR,
        "monthly_anomalies.csv"
    )

    if os.path.exists(anomaly_path):

        anomaly_df = pd.read_csv(
            anomaly_path
        )

        st.dataframe(
            anomaly_df,
            use_container_width=True,
            hide_index=True
        )

        if "Anomaly" in anomaly_df.columns:

            anomaly_flags = (
                anomaly_df["Anomaly"]
                .astype(str)
                .str.lower()
                .eq("true")
            )

            detected = anomaly_df[
                anomaly_flags
            ]

            if len(detected) > 0:

                st.error(
                    f"🚨 {len(detected)} "
                    "potential anomaly period(s) detected."
                )

                st.dataframe(
                    detected,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "✅ No strong anomaly was detected "
                    "under the selected statistical threshold."
                )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.html(
        """
        <div class="section-heading">
            🤖 Machine Learning Performance
        </div>
        """
    )

    performance = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Linear SVM"
        ],

        "Accuracy (%)": [
            80.61,
            78.53,
            80.33
        ],

        "Precision (%)": [
            79.67,
            78.78,
            79.20
        ],

        "Recall (%)": [
            82.21,
            78.09,
            82.25
        ],

        "F1 Score (%)": [
            80.92,
            78.43,
            80.70
        ]

    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.html(
        """
        <div class="section-heading">
            🏆 Model Comparison
        </div>
        """
    )

    fig, ax = plt.subplots(
        figsize=(11, 5.5)
    )

    x = range(
        len(performance)
    )

    metrics = [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)"
    ]

    width = 0.19

    for i, metric in enumerate(
        metrics
    ):

        positions = [
            j +
            (i - 1.5) * width
            for j in x
        ]

        ax.bar(
            positions,
            performance[metric],
            width=width,
            label=metric.replace(
                " (%)",
                ""
            )
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        performance["Model"]
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_ylim(
        70,
        90
    )

    ax.set_title(
        "Machine Learning Model Comparison"
    )

    ax.legend(
        ncols=4
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        **Selected model:** Logistic Regression

        It achieved the highest test accuracy (80.61%)
        and highest F1 score (80.92%) among the three
        evaluated classifiers.
        """
    )


# =========================================================
# LIVE PREDICTION
# =========================================================
elif page == "Live Prediction":

    st.html(
        """
        <div class="section-heading">
            🔮 Live AI Sentiment Prediction
        </div>
        """
    )

    st.markdown(
        """
        Enter a new social media post. The trained
        Logistic Regression classifier returns the
        sentiment together with prediction confidence.
        """
    )

    if not model_loaded:

        st.error(
            f"Model loading failed: {model_error}"
        )

    else:

        text = st.text_area(
            "Social media post",
            placeholder=(
                "Example: "
                "I absolutely love this product! "
                "It works perfectly."
            ),
            height=160
        )

        analyze = st.button(
            "✨ Analyze Sentiment",
            use_container_width=True
        )

        if analyze:

            if not text.strip():

                st.warning(
                    "Please enter a social media post first."
                )

            else:

                with st.spinner(
                    "Analyzing sentiment..."
                ):

                    time.sleep(0.6)

                    vector = vectorizer.transform(
                        [text]
                    )

                    prediction = model.predict(
                        vector
                    )[0]

                    probabilities = model.predict_proba(
                        vector
                    )[0]

                negative_probability = (
                    probabilities[0] * 100
                )

                positive_probability = (
                    probabilities[1] * 100
                )

                if prediction == 1:

                    st.html(
                        f"""
                        <div class="prediction-result pred-positive">

                            <div class="pred-title">
                                😊 POSITIVE SENTIMENT
                            </div>

                            <div class="pred-confidence">
                                Confidence:
                                <b>
                                    {positive_probability:.2f}%
                                </b>
                            </div>

                        </div>
                        """
                    )

                else:

                    st.html(
                        f"""
                        <div class="prediction-result pred-negative">

                            <div class="pred-title">
                                😞 NEGATIVE SENTIMENT
                            </div>

                            <div class="pred-confidence">
                                Confidence:
                                <b>
                                    {negative_probability:.2f}%
                                </b>
                            </div>

                        </div>
                        """
                    )

                st.markdown("### Probability Breakdown")

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Positive Probability",
                        f"{positive_probability:.2f}%"
                    )

                    st.progress(
                        min(
                            positive_probability / 100,
                            1.0
                        )
                    )

                with p2:

                    st.metric(
                        "Negative Probability",
                        f"{negative_probability:.2f}%"
                    )

                    st.progress(
                        min(
                            negative_probability / 100,
                            1.0
                        )
                    )


# =========================================================
# BATCH PREDICTION
# =========================================================
elif page == "Batch Prediction":

    st.html(
        """
        <div class="section-heading">
            📂 Batch Sentiment Prediction
        </div>
        """
    )

    st.markdown(
        """
        Upload a CSV containing a column named **text**
        to classify multiple social media posts at once.
        """
    )

    if not model_loaded:

        st.error(
            f"Model loading failed: {model_error}"
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"]
        )

        if uploaded_file is not None:

            try:

                batch_df = pd.read_csv(
                    uploaded_file
                )

                if "text" not in batch_df.columns:

                    st.error(
                        'CSV must contain a column named "text".'
                    )

                else:

                    texts = (
                        batch_df["text"]
                        .fillna("")
                        .astype(str)
                    )

                    vectors = vectorizer.transform(
                        texts
                    )

                    predictions = model.predict(
                        vectors
                    )

                    probabilities = model.predict_proba(
                        vectors
                    )

                    batch_df["sentiment"] = [
                        "Positive"
                        if value == 1
                        else "Negative"
                        for value in predictions
                    ]

                    batch_df["confidence"] = (
                        probabilities.max(
                            axis=1
                        ) * 100
                    ).round(2)

                    positive_count = int(
                        (
                            batch_df["sentiment"]
                            == "Positive"
                        ).sum()
                    )

                    negative_count = int(
                        (
                            batch_df["sentiment"]
                            == "Negative"
                        ).sum()
                    )

                    st.success(
                        f"{len(batch_df):,} posts processed successfully."
                    )

                    a, b, c = st.columns(3)

                    with a:

                        st.metric(
                            "Total Posts",
                            f"{len(batch_df):,}"
                        )

                    with b:

                        st.metric(
                            "Positive",
                            f"{positive_count:,}"
                        )

                    with c:

                        st.metric(
                            "Negative",
                            f"{negative_count:,}"
                        )

                    st.markdown(
                        "### Prediction Results"
                    )

                    st.dataframe(
                        batch_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    result = pd.Series(
                        {
                            "Positive": positive_count,
                            "Negative": negative_count
                        }
                    )

                    fig, ax = plt.subplots(
                        figsize=(7, 4)
                    )

                    ax.bar(
                        result.index,
                        result.values
                    )

                    ax.set_title(
                        "Batch Sentiment Distribution"
                    )

                    ax.set_ylabel(
                        "Number of Posts"
                    )

                    fig.tight_layout()

                    st.pyplot(
                        fig,
                        use_container_width=True
                    )

                    result_csv = (
                        batch_df
                        .to_csv(index=False)
                        .encode("utf-8")
                    )

                    st.download_button(
                        "⬇️ Download Predictions CSV",
                        result_csv,
                        "socialpulse_predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )

            except Exception as e:

                st.error(
                    f"Could not process the CSV: {e}"
                )


# =========================================================
# MODEL EXPLAINABILITY
# =========================================================
elif page == "Model Explainability":

    st.html(
        """
        <div class="section-heading">
            🧠 Explainable AI
        </div>
        """
    )

    st.markdown(
        """
        Logistic Regression coefficients provide an
        interpretable view of the features that push
        predictions toward Positive or Negative sentiment.
        """
    )

    if not model_loaded:

        st.error(
            f"Model loading failed: {model_error}"
        )

    else:

        try:

            feature_names = (
                vectorizer
                .get_feature_names_out()
            )

            coefficients = model.coef_[0]

            explanation = pd.DataFrame(
                {
                    "Feature": feature_names,
                    "Coefficient": coefficients
                }
            )

            positive_features = (
                explanation
                .sort_values(
                    "Coefficient",
                    ascending=False
                )
                .head(20)
                .reset_index(drop=True)
            )

            negative_features = (
                explanation
                .sort_values(
                    "Coefficient",
                    ascending=True
                )
                .head(20)
                .reset_index(drop=True)
            )

            left, right = st.columns(2)

            with left:

                st.markdown(
                    "### 😊 Positive Indicators"
                )

                st.dataframe(
                    positive_features,
                    use_container_width=True,
                    hide_index=True
                )

            with right:

                st.markdown(
                    "### 😞 Negative Indicators"
                )

                st.dataframe(
                    negative_features,
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(
                """
                <div class="info-card">
                    Positive coefficients push the classifier
                    toward Positive sentiment, while negative
                    coefficients push it toward Negative sentiment.
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"Explainability could not be generated: {e}"
            )


# =========================================================
# FOOTER
# =========================================================
st.html(
    """
    <div class="footer">

        <b>SocialPulse</b>
        • Intelligent Social Media Analytics

        <br><br>

        NLP • TF-IDF • Machine Learning
        • K-Means • Explainable AI

    </div>
    """
)
