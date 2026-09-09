import os
import time
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PAGE
# =========================================================
st.set_page_config(
    page_title="SocialPulse | AI Analytics",
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


model, vectorizer, topic_model = load_models()


# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at 10% 20%,
            rgba(88,28,135,.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(37,99,235,.16),
            transparent 28%
        ),
        #09090f;
    color: #f5f5f5;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #11111b,
            #0b0b12
        );
}

.hero {
    padding: 42px;
    border-radius: 26px;
    margin-bottom: 30px;

    background:
        linear-gradient(
            120deg,
            #171329,
            #101827,
            #18112b,
            #101827
        );

    background-size: 300% 300%;

    animation:
        gradientMove 10s ease infinite;

    border:
        1px solid rgba(
            255,
            255,
            255,
            .10
        );

    box-shadow:
        0 20px 60px
        rgba(0,0,0,.30);
}

.hero-title {
    font-size: 3.2rem;
    font-weight: 850;
}

.hero-subtitle {
    color: #c7c8d5;
    margin-top: 10px;
    font-size: 1.05rem;
}

.metric-card {

    padding: 22px;

    border-radius: 20px;

    background:
        rgba(
            255,
            255,
            255,
            .045
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .09
        );

    transition:
        all .3s ease;

    animation:
        fadeUp .7s ease both;
}

.metric-card:hover {

    transform:
        translateY(-7px)
        scale(1.01);

    box-shadow:
        0 16px 40px
        rgba(0,0,0,.30);
}

.metric-label {
    color: #a9abba;
    font-size: .84rem;
}

.metric-value {
    color: white;
    font-size: 1.6rem;
    font-weight: 800;
    margin-top: 7px;
}

.section-title {

    font-size: 1.7rem;
    font-weight: 800;

    margin-top: 42px;
    margin-bottom: 18px;
}

.glass {

    padding: 24px;

    border-radius: 20px;

    background:
        rgba(
            255,
            255,
            255,
            .035
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .08
        );
}

.result {

    padding: 28px;

    border-radius: 22px;

    text-align: center;

    margin-top: 20px;

    animation:
        resultPop .45s ease;
}

.positive {

    background:
        rgba(
            16,
            185,
            129,
            .14
        );

    border:
        1px solid
        rgba(
            16,
            185,
            129,
            .40
        );
}

.negative {

    background:
        rgba(
            239,
            68,
            68,
            .14
        );

    border:
        1px solid
        rgba(
            239,
            68,
            68,
            .40
        );
}

.big-result {

    font-size: 2rem;
    font-weight: 850;
}

@keyframes gradientMove {

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

@keyframes fadeUp {

    from {
        opacity: 0;
        transform:
            translateY(15px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }
}

@keyframes resultPop {

    from {
        opacity: 0;
        transform:
            scale(.94);
    }

    to {
        opacity: 1;
        transform:
            scale(1);
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🧠 SocialPulse")
    st.caption(
        "Intelligent Social Media Analytics"
    )

    st.markdown("---")

    page = st.radio(
        "Navigation",
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

    st.caption(
        "Dataset: Sentiment140"
    )

    st.caption(
        "1.6M Tweets"
    )


# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">

    <div class="hero-title">
        🧠 SocialPulse
    </div>

    <div class="hero-subtitle">
        Intelligent Social Media Sentiment,
        Topic & Trend Analytics
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# METRICS
# =========================================================
c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Dataset", "1.6M Tweets"),
    ("Best Model", "Logistic Regression"),
    ("Accuracy", "80.61%"),
    ("Topics", "6")
]

for col, (label, value) in zip(
    [c1, c2, c3, c4],
    metrics
):

    with col:

        st.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# IMAGE HELPER
# =========================================================
def show_image(
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
            f"Asset not found: {filename}"
        )


# =========================================================
# OVERVIEW
# =========================================================
if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        '📊 Intelligence Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "SocialPulse combines NLP, "
        "machine learning, clustering "
        "and statistical analysis."
    )

    a, b = st.columns(2)

    with a:

        show_image(
            "sentiment_distribution.png"
        )

    with b:

        show_image(
            "sentiment_percentage.png"
        )

    st.success(
        "✅ Logistic Regression selected "
        "as the best overall classifier."
    )


# =========================================================
# SENTIMENT ANALYTICS
# =========================================================
elif page == "Sentiment Analytics":

    st.markdown(
        '<div class="section-title">'
        '💬 Sentiment Analytics'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        show_image(
            "sentiment_distribution.png"
        )

    with b:

        show_image(
            "sentiment_percentage.png"
        )

    st.markdown(
        '<div class="section-title">'
        '📏 Tweet Length'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        show_image(
            "tweet_length_histogram.png"
        )

    with b:

        show_image(
            "tweet_length_boxplot.png"
        )


# =========================================================
# TREND
# =========================================================
elif page == "Trend Analysis":

    st.markdown(
        '<div class="section-title">'
        '📈 Sentiment Trends'
        '</div>',
        unsafe_allow_html=True
    )

    show_image(
        "monthly_sentiment_trend.png",
        "Monthly Sentiment Trend"
    )

    st.info(
        "Trend analysis shows how positive "
        "and negative sentiment proportions "
        "changed over the available time period."
    )


# =========================================================
# WORD ANALYSIS
# =========================================================
elif page == "Word Analysis":

    st.markdown(
        '<div class="section-title">'
        '🔤 Word Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        show_image(
            "top_words.png"
        )

    with b:

        csv_path = os.path.join(
            ASSET_DIR,
            "top_words.csv"
        )

        if os.path.exists(csv_path):

            words = pd.read_csv(csv_path)

            st.dataframe(
                words,
                use_container_width=True,
                hide_index=True
            )

    st.markdown(
        '<div class="section-title">'
        '☁️ Sentiment Word Clouds'
        '</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:

        show_image(
            "positive_wordcloud.png"
        )

    with b:

        show_image(
            "negative_wordcloud.png"
        )


# =========================================================
# TOPIC INTELLIGENCE
# =========================================================
elif page == "Topic Intelligence":

    st.markdown(
        '<div class="section-title">'
        '🧩 Topic Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Unsupervised K-Means clustering "
        "groups tweets into six latent topics "
        "using TF-IDF representations."
    )

    a, b = st.columns(2)

    with a:

        show_image(
            os.path.join(
                "../advanced",
                "topic_distribution.png"
            )
        )

    # Better direct path display
    topic_chart = os.path.join(
        ADVANCED_DIR,
        "topic_distribution.png"
    )

    if os.path.exists(topic_chart):

        st.image(
            topic_chart,
            use_container_width=True
        )

    with b:

        topic_keywords_path = os.path.join(
            ADVANCED_DIR,
            "topic_keywords.csv"
        )

        if os.path.exists(topic_keywords_path):

            topic_keywords = pd.read_csv(
                topic_keywords_path
            )

            st.dataframe(
                topic_keywords,
                use_container_width=True,
                hide_index=True
            )

    st.markdown(
        '<div class="section-title">'
        '💬 Sentiment × Topic'
        '</div>',
        unsafe_allow_html=True
    )

    sentiment_topic_chart = os.path.join(
        ADVANCED_DIR,
        "topic_sentiment.png"
    )

    if os.path.exists(
        sentiment_topic_chart
    ):

        st.image(
            sentiment_topic_chart,
            use_container_width=True
        )


# =========================================================
# ANOMALY DETECTION
# =========================================================
elif page == "Anomaly Detection":

    st.markdown(
        '<div class="section-title">'
        '🚨 Sentiment Anomaly Detection'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Potential anomalies are identified "
        "when monthly positive-sentiment changes "
        "are unusually large relative to the "
        "observed variation."
    )

    anomaly_path = os.path.join(
        ADVANCED_DIR,
        "monthly_anomalies.csv"
    )

    if os.path.exists(anomaly_path):

        anomaly_df = pd.read_csv(
            anomaly_path,
            index_col=0
        )

        st.dataframe(
            anomaly_df,
            use_container_width=True
        )

        if "Anomaly" in anomaly_df.columns:

            detected = anomaly_df[
                anomaly_df["Anomaly"] == True
            ]

            if len(detected) > 0:

                st.error(
                    f"🚨 {len(detected)} "
                    "potential anomaly period(s) detected."
                )

                st.dataframe(
                    detected,
                    use_container_width=True
                )

            else:

                st.success(
                    "✅ No strong anomaly detected "
                    "under the selected threshold."
                )


# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">'
        '🤖 Machine Learning Performance'
        '</div>',
        unsafe_allow_html=True
    )

    performance = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Multinomial Naive Bayes",
            "Linear SVM"
        ],

        "Accuracy": [
            80.61,
            78.53,
            80.33
        ],

        "Precision": [
            79.67,
            78.78,
            79.20
        ],

        "Recall": [
            82.21,
            78.09,
            82.25
        ],

        "F1 Score": [
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

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    x = range(
        len(performance)
    )

    metrics_to_plot = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    for i, metric in enumerate(
        metrics_to_plot
    ):

        ax.bar(
            [
                j +
                (
                    i - 1.5
                ) * 0.2
                for j in x
            ],
            performance[metric],
            width=.2,
            label=metric
        )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        performance["Model"]
    )

    ax.set_ylim(
        70,
        90
    )

    ax.set_ylabel(
        "Score (%)"
    )

    ax.set_title(
        "Machine Learning Model Comparison"
    )

    ax.legend()

    st.pyplot(
        fig,
        use_container_width=True
    )


# =========================================================
# LIVE PREDICTION
# =========================================================
elif page == "Live Prediction":

    st.markdown(
        '<div class="section-title">'
        '🔮 AI Sentiment Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    text = st.text_area(
        "Enter a social media post",
        placeholder=(
            "Example: "
            "I absolutely love this product!"
        ),
        height=150
    )

    analyze = st.button(
        "✨ Analyze Sentiment"
    )

    if analyze:

        if not text.strip():

            st.warning(
                "Please enter some text."
            )

        else:

            with st.spinner(
                "AI is analyzing the post..."
            ):

                time.sleep(.6)

                vector = vectorizer.transform(
                    [text]
                )

                prediction = model.predict(
                    vector
                )[0]

                probabilities = (
                    model.predict_proba(
                        vector
                    )[0]
                )

            positive = probabilities[1] * 100
            negative = probabilities[0] * 100

            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result positive">

                        <div class="big-result">
                            😊 POSITIVE
                        </div>

                        <p>
                            Confidence:
                            <b>{positive:.2f}%</b>
                        </p>

                        <p>
                            Positive:
                            {positive:.2f}%
                            &nbsp; | &nbsp;
                            Negative:
                            {negative:.2f}%
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result negative">

                        <div class="big-result">
                            😞 NEGATIVE
                        </div>

                        <p>
                            Confidence:
                            <b>{negative:.2f}%</b>
                        </p>

                        <p>
                            Positive:
                            {positive:.2f}%
                            &nbsp; | &nbsp;
                            Negative:
                            {negative:.2f}%
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# BATCH PREDICTION
# =========================================================
elif page == "Batch Prediction":

    st.markdown(
        '<div class="section-title">'
        '📂 Batch Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Upload CSV with a 'text' column",
        type=["csv"]
    )

    if uploaded:

        data = pd.read_csv(
            uploaded
        )

        if "text" not in data.columns:

            st.error(
                "CSV must contain a column named 'text'."
            )

        else:

            texts = (
                data["text"]
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

            data["sentiment"] = [
                "Positive"
                if p == 1
                else "Negative"
                for p in predictions
            ]

            data["confidence"] = (
                probabilities.max(
                    axis=1
                ) * 100
            ).round(2)

            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True
            )

            positive_count = (
                data["sentiment"]
                .eq("Positive")
                .sum()
            )

            negative_count = (
                data["sentiment"]
                .eq("Negative")
                .sum()
            )

            a, b, c = st.columns(3)

            a.metric(
                "Total",
                len(data)
            )

            b.metric(
                "Positive",
                positive_count
            )

            c.metric(
                "Negative",
                negative_count
            )

            result_csv = data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Predictions",
                result_csv,
                "socialpulse_predictions.csv",
                "text/csv"
            )


# =========================================================
# EXPLAINABILITY
# =========================================================
elif page == "Model Explainability":

    st.markdown(
        '<div class="section-title">'
        '🧠 Explainable AI'
        '</div>',
        unsafe_allow_html=True
    )

    features = (
        vectorizer
        .get_feature_names_out()
    )

    coefficients = model.coef_[0]

    explanation = pd.DataFrame({

        "Feature": features,

        "Coefficient": coefficients
    })

    positive = (
        explanation
        .sort_values(
            "Coefficient",
            ascending=False
        )
        .head(20)
    )

    negative = (
        explanation
        .sort_values(
            "Coefficient",
            ascending=True
        )
        .head(20)
    )

    a, b = st.columns(2)

    with a:

        st.markdown(
            "### 😊 Positive Indicators"
        )

        st.dataframe(
            positive,
            use_container_width=True,
            hide_index=True
        )

    with b:

        st.markdown(
            "### 😞 Negative Indicators"
        )

        st.dataframe(
            negative,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "A positive coefficient pushes the "
        "Logistic Regression prediction toward "
        "Positive sentiment; a negative coefficient "
        "pushes it toward Negative sentiment."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:60px;
        padding:22px;
        color:#888b9b;
        border-top:
            1px solid
            rgba(255,255,255,.08);
    ">
        <b>SocialPulse</b>
        • AI-Powered Social Media Analytics
        <br>
        NLP • Machine Learning • Clustering
        • Explainable AI
    </div>
    """,
    unsafe_allow_html=True
)
