
import os
import re
import time
import joblib
import nltk
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SocialPulse",
    page_icon="☀️",
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

TOPIC_KEYWORDS_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_keywords.csv"
)

TOPIC_SENTIMENT_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_sentiment_percentage.csv"
)

TOPIC_DISTRIBUTION_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_distribution.csv"
)

ANOMALY_PATH = os.path.join(
    ADVANCED_DIR,
    "monthly_anomalies.csv"
)

MONTHLY_PATH = os.path.join(
    ASSET_DIR,
    "monthly_percentage.csv"
)

TOP_WORDS_PATH = os.path.join(
    ASSET_DIR,
    "top_words.csv"
)


# =========================================================
# NLTK SETUP
# =========================================================

@st.cache_resource
def setup_nltk():

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)

    try:
        stops = set(stopwords.words("english"))
    except LookupError:
        stops = set()

    return stops, WordNetLemmatizer()


stop_words, lemmatizer = setup_nltk()


# Preserve negations exactly as used during training
negation_words = {
    "no", "not", "nor", "never", "none", "neither",
    "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't",
    "haven't", "hasn't", "hadn't"
}

stop_words = stop_words - negation_words


# =========================================================
# TEXT PREPROCESSING
# Matches training workflow
# =========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    cleaned_words = []

    for word in words:

        if word in stop_words:
            continue

        try:
            word = lemmatizer.lemmatize(word)
        except Exception:
            pass

        cleaned_words.append(word)

    return " ".join(cleaned_words)


# =========================================================
# MODEL LOADING
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

    return sentiment_model, vectorizer, topic_model


try:

    model, vectorizer, topic_model = load_models()

    MODEL_READY = True
    MODEL_ERROR = ""

except Exception as exc:

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# =========================================================
# SUNSET ANALYTICS UI
# =========================================================

st.markdown(
    """
<style>

.stApp {

    background:
        radial-gradient(
            circle at 7% 8%,
            rgba(245,158,11,.15),
            transparent 24%
        ),
        radial-gradient(
            circle at 94% 9%,
            rgba(124,58,237,.12),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #fffaf3 0%,
            #f7f3ec 100%
        );

    color: #241f27;
}


.block-container {

    max-width: 1450px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;

}


/* SIDEBAR */

section[data-testid="stSidebar"] {

    background: #fffdf9;

    border-right:
        1px solid
        rgba(50,40,30,.08);

}

section[data-testid="stSidebar"] * {

    color: #2a2530 !important;

}


/* HERO */

.hero {

    position: relative;
    overflow: hidden;

    padding: 44px;

    border-radius: 30px;

    margin-bottom: 28px;

    background:
        linear-gradient(
            120deg,
            #fff0c9,
            #ffe5db,
            #eee4ff,
            #fff0c9
        );

    background-size: 300% 300%;

    animation:
        sunsetMove 12s ease infinite;

    border:
        1px solid
        rgba(124,58,237,.10);

    box-shadow:
        0 20px 55px
        rgba(76,51,24,.10);

}


.hero-orb {

    position: absolute;

    width: 270px;
    height: 270px;

    right: -100px;
    top: -120px;

    border-radius: 50%;

    background:
        rgba(245,158,11,.16);

    filter: blur(72px);

    animation:
        orbFloat 7s ease-in-out infinite alternate;

}


.hero-content {

    position: relative;
    z-index: 2;

}


.hero-badge {

    display: inline-block;

    padding: 7px 14px;

    border-radius: 999px;

    background:
        rgba(255,255,255,.66);

    color:
        #754219;

    font-size:
        .80rem;

    font-weight:
        800;

}


.hero-title {

    margin-top: 14px;

    font-size: 3.35rem;

    font-weight: 900;

    letter-spacing: -1.8px;

    color: #292329;

}


.hero-subtitle {

    max-width: 880px;

    margin-top: 12px;

    color: #675f66;

    font-size: 1.06rem;

    line-height: 1.7;

}


/* KPI */

.kpi {

    padding: 22px;

    min-height: 110px;

    border-radius: 22px;

    background:
        rgba(255,255,255,.88);

    border:
        1px solid
        rgba(55,43,28,.08);

    box-shadow:
        0 10px 28px
        rgba(55,36,15,.07);

    transition:
        all .28s ease;

    animation:
        rise .6s ease both;

}


.kpi:hover {

    transform:
        translateY(-7px);

    box-shadow:
        0 18px 38px
        rgba(55,36,15,.13);

}


.kpi-label {

    color: #827a80;

    font-size: .82rem;

    font-weight: 700;

}


.kpi-value {

    margin-top: 8px;

    color: #28232a;

    font-size: 1.55rem;

    font-weight: 900;

}


/* SECTION */

.section-title {

    margin-top: 38px;

    margin-bottom: 9px;

    font-size: 1.72rem;

    font-weight: 900;

    color: #28232a;

}


/* CARDS */

.card {

    padding: 22px;

    border-radius: 22px;

    background:
        rgba(255,255,255,.91);

    border:
        1px solid
        rgba(55,43,28,.08);

    box-shadow:
        0 10px 30px
        rgba(55,36,15,.06);

    transition:
        all .28s ease;

}


.card:hover {

    transform:
        translateY(-4px);

    box-shadow:
        0 16px 38px
        rgba(55,36,15,.10);

}


/* INPUT */

.stTextArea textarea {

    background:
        #fffdfa !important;

    color:
        #241f27 !important;

    border:
        1px solid
        #d3cbc3 !important;

    border-radius:
        16px !important;

    font-size:
        1rem !important;

}


.stTextArea textarea::placeholder {

    color:
        #81777e !important;

    opacity:
        1 !important;

}


/* BUTTON */

.stButton > button {

    min-height:
        45px;

    border-radius:
        13px !important;

    background:
        linear-gradient(
            135deg,
            #f59e0b,
            #8b5cf6
        ) !important;

    color:
        #ffffff !important;

    font-weight:
        800 !important;

    border:
        none !important;

    box-shadow:
        0 8px 20px
        rgba(124,58,237,.14);

    transition:
        all .25s ease !important;

}


.stButton > button:hover {

    transform:
        translateY(-3px);

    box-shadow:
        0 14px 30px
        rgba(124,58,237,.20);

}


/* RESULT */

.prediction {

    margin-top:
        18px;

    padding:
        28px;

    border-radius:
        22px;

    text-align:
        center;

    animation:
        resultPop .45s ease both;

}


.pred-positive {

    background:
        linear-gradient(
            135deg,
            #e9f8ef,
            #fbfffc
        );

    border:
        1px solid
        #abdcc1;

}


.pred-negative {

    background:
        linear-gradient(
            135deg,
            #fff0ef,
            #fffafa
        );

    border:
        1px solid
        #ebb5b0;

}


.pred-title {

    font-size:
        2rem;

    font-weight:
        900;

}


.pred-sub {

    margin-top:
        8px;

    color:
        #655d65;

}


/* INSIGHT */

.insight {

    margin-top:
        18px;

    padding:
        18px 20px;

    border-radius:
        17px;

    background:
        #fff8e9;

    border:
        1px solid
        #f0d399;

    color:
        #604720;

}


/* FOOTER */

.footer {

    margin-top:
        60px;

    padding:
        24px;

    text-align:
        center;

    color:
        #8a8287;

    border-top:
        1px solid
        rgba(50,40,30,.08);

}


/* ANIMATIONS */

@keyframes sunsetMove {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }

}


@keyframes orbFloat {

    from {
        transform:
            translate(0,0)
            scale(1);
    }

    to {
        transform:
            translate(-32px,24px)
            scale(1.15);
    }

}


@keyframes rise {

    from {
        opacity: 0;
        transform:
            translateY(12px);
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
            scale(.96);
    }

    to {
        opacity: 1;
        transform:
            scale(1);
    }

}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ☀️ SocialPulse"
    )

    st.caption(
        "Feel the pulse of social media."
    )

    st.markdown("---")

    page = st.radio(
        "Explore",
        [
            "Home",
            "Pulse",
            "Trends",
            "Topics",
            "Language",
            "Models",
            "Predict",
            "Why?"
        ]
    )

    st.markdown("---")

    st.caption(
        "Sentiment140 • 1.6M Tweets"
    )

    st.caption(
        "NLP • TF-IDF • ML • Clustering"
    )


# =========================================================
# HERO
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-orb"></div>

        <div class="hero-content">

            <div class="hero-badge">
                ☀️ AI-POWERED SOCIAL MEDIA INTELLIGENCE
            </div>

            <div class="hero-title">
                SocialPulse
            </div>

            <div class="hero-subtitle">
                Discover what people are saying,
                how sentiment moves, which themes
                dominate the conversation, and why
                the model makes a prediction.
            </div>

        </div>

    </div>
    """
)


# =========================================================
# KPI
# =========================================================

kpi_cols = st.columns(4)

kpis = [
    ("Dataset", "1.6M Tweets"),
    ("Sentiment", "50% / 50%"),
    ("Clusters", "6"),
    ("Best Accuracy", "80.61%")
]

for col, (label, value) in zip(
    kpi_cols,
    kpis
):

    with col:

        st.html(
            f"""
            <div class="kpi">
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
# HELPER: PREDICT
# =========================================================

def predict_text(text):

    cleaned = clean_text(text)

    if not cleaned:
        return None

    vector = vectorizer.transform(
        [cleaned]
    )

    prediction = int(
        model.predict(vector)[0]
    )

    probabilities = model.predict_proba(
        vector
    )[0]

    topic_id = int(
        topic_model.predict(vector)[0]
    ) + 1

    return {
        "cleaned": cleaned,
        "vector": vector,
        "prediction": prediction,
        "probabilities": probabilities,
        "topic_id": topic_id
    }


# =========================================================
# HELPER: DISPLAY PREDICTION
# =========================================================

def display_prediction(result):

    prediction = result["prediction"]

    probabilities = result["probabilities"]

    positive_probability = (
        float(probabilities[1]) * 100
    )

    negative_probability = (
        float(probabilities[0]) * 100
    )

    confidence = (
        positive_probability
        if prediction == 1
        else negative_probability
    )

    topic_id = result["topic_id"]

    if prediction == 1:

        st.html(
            f"""
            <div class="prediction pred-positive">

                <div class="pred-title">
                    😊 POSITIVE
                </div>

                <div class="pred-sub">
                    Model confidence:
                    <b>{confidence:.2f}%</b>
                </div>

            </div>
            """
        )

    else:

        st.html(
            f"""
            <div class="prediction pred-negative">

                <div class="pred-title">
                    😞 NEGATIVE
                </div>

                <div class="pred-sub">
                    Model confidence:
                    <b>{confidence:.2f}%</b>
                </div>

            </div>
            """
        )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Positive Probability",
            f"{positive_probability:.2f}%"
        )

    with b:

        st.metric(
            "Negative Probability",
            f"{negative_probability:.2f}%"
        )

    with c:

        st.metric(
            "Detected Cluster",
            f"Cluster {topic_id}"
        )

    probability_df = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Negative"
            ],
            "Probability": [
                positive_probability,
                negative_probability
            ]
        }
    )

    fig = px.bar(
        probability_df,
        x="Sentiment",
        y="Probability",
        text="Probability",
        title="Prediction Probability"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=370,
        yaxis={
            "title": "Probability (%)",
            "range": [0, 105]
        },
        xaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =========================================================
# HELPER: EXPLAIN CURRENT PREDICTION
# =========================================================

def prediction_contributions(vector):

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    coefficients = model.coef_[0]

    row = vector.tocoo()

    contributions = []

    for idx, value in zip(
        row.col,
        row.data
    ):

        contribution = (
            float(value)
            *
            float(coefficients[idx])
        )

        contributions.append(
            (
                feature_names[idx],
                contribution
            )
        )

    positive = sorted(
        [
            x for x in contributions
            if x[1] > 0
        ],
        key=lambda x: x[1],
        reverse=True
    )[:6]

    negative = sorted(
        [
            x for x in contributions
            if x[1] < 0
        ],
        key=lambda x: x[1]
    )[:6]

    return positive, negative


# =========================================================
# HOME
# =========================================================

if page == "Home":

    st.html(
        """
        <div class="section-title">
            Welcome to SocialPulse
        </div>
        """
    )

    st.write(
        "Explore sentiment, trends, topics and "
        "machine-learning insights — or analyze "
        "a post instantly."
    )

    st.markdown(
        "### ⚡ Quick Demo"
    )

    if "home_text" not in st.session_state:

        st.session_state.home_text = ""

    st.html(
        '<div class="action-card">'
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        if st.button(
            "😊 Happy Example",
            use_container_width=True
        ):

            st.session_state.home_text = (
                "I absolutely love this product! "
                "It is amazing and works perfectly."
            )

    with q2:

        if st.button(
            "😡 Angry Example",
            use_container_width=True
        ):

            st.session_state.home_text = (
                "This service is terrible. "
                "I am extremely disappointed."
            )

    with q3:

        if st.button(
            "🤩 Excited Example",
            use_container_width=True
        ):

            st.session_state.home_text = (
                "What an amazing day! "
                "Everything is going perfectly."
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    home_text = st.text_area(
        "What are people saying?",
        value=st.session_state.home_text,
        placeholder=(
            "Write a social media post here..."
        ),
        height=150,
        key="home_input"
    )

    if st.button(
        "✨ Analyze Sentiment",
        use_container_width=True,
        key="home_analyze"
    ):

        if not MODEL_READY:

            st.error(
                f"Model loading failed: {MODEL_ERROR}"
            )

        elif not home_text.strip():

            st.warning(
                "Please enter a social media post first."
            )

        else:

            with st.spinner(
                "Reading the pulse..."
            ):

                time.sleep(.35)

                result = predict_text(
                    home_text
                )

            if result is None:

                st.error(
                    "The entered text became empty "
                    "after preprocessing."
                )

            else:

                display_prediction(result)


# =========================================================
# PULSE
# =========================================================

elif page == "Pulse":

    st.html(
        """
        <div class="section-title">
            💓 Sentiment Pulse
        </div>
        """
    )

    pulse_df = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Negative"
            ],
            "Tweets": [
                800000,
                800000
            ]
        }
    )

    fig = px.pie(
        pulse_df,
        names="Sentiment",
        values="Tweets",
        hole=.68,
        title="Overall Sentiment Balance"
    )

    fig.update_traces(
        textinfo="percent",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Tweets: %{value:,}<br>"
            "Share: %{percent}"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        height=540,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.html(
        """
        <div class="insight">

            💡 The dataset contains an equal number
            of Positive and Negative tweets:
            <b>800,000 each.</b>

        </div>
        """
    )


# =========================================================
# TRENDS
# =========================================================

elif page == "Trends":

    st.html(
        """
        <div class="section-title">
            📈 Sentiment Trends
        </div>
        """
    )

    if os.path.exists(MONTHLY_PATH):

        monthly = pd.read_csv(
            MONTHLY_PATH
        )

        month_col = monthly.columns[0]

        long_data = monthly.melt(
            id_vars=[month_col],
            var_name="Sentiment",
            value_name="Percentage"
        )

        fig = px.area(
            long_data,
            x=month_col,
            y="Percentage",
            color="Sentiment",
            markers=True,
            title="Monthly Sentiment Movement"
        )

        fig.update_layout(
            height=520,
            hovermode="x unified",
            yaxis_title="Sentiment Share (%)",
            xaxis_title="Month",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.markdown(
        "### 🚨 Unusual Sentiment Activity"
    )

    if os.path.exists(ANOMALY_PATH):

        anomaly_df = pd.read_csv(
            ANOMALY_PATH
        )

        if "Anomaly" in anomaly_df.columns:

            flags = (
                anomaly_df["Anomaly"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )

            detected = anomaly_df[
                flags
            ]

            if len(detected) > 0:

                st.warning(
                    f"{len(detected)} statistical "
                    "anomaly period(s) detected."
                )

                st.dataframe(
                    detected,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.success(
                    "No strong statistical anomaly detected."
                )

        with st.expander(
            "View full anomaly data"
        ):

            st.dataframe(
                anomaly_df,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# TOPICS
# =========================================================

elif page == "Topics":

    st.html(
        """
        <div class="section-title">
            🧩 Topic Universe
        </div>
        """
    )

    st.write(
        "Six latent conversation clusters were "
        "discovered using K-Means on TF-IDF features."
    )

    if os.path.exists(
        TOPIC_DISTRIBUTION_PATH
    ):

        topic_dist = pd.read_csv(
            TOPIC_DISTRIBUTION_PATH
        )

        label_col = topic_dist.columns[0]
        count_col = topic_dist.columns[1]

        fig = px.scatter(
            topic_dist,
            x=label_col,
            y=count_col,
            size=count_col,
            hover_name=label_col,
            title="Discovered Topic Clusters"
        )

        fig.update_layout(
            height=500,
            xaxis_title="Cluster",
            yaxis_title="Tweet Count",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    if os.path.exists(
        TOPIC_KEYWORDS_PATH
    ):

        keywords = pd.read_csv(
            TOPIC_KEYWORDS_PATH
        )

        st.markdown(
            "### 🔑 Cluster Keywords"
        )

        st.dataframe(
            keywords,
            use_container_width=True,
            hide_index=True
        )

    if os.path.exists(
        TOPIC_SENTIMENT_PATH
    ):

        topic_sentiment = pd.read_csv(
            TOPIC_SENTIMENT_PATH
        )

        topic_sentiment = (
            topic_sentiment
            .reset_index()
        )

        first_col = (
            topic_sentiment.columns[0]
        )

        long_topic = topic_sentiment.melt(
            id_vars=[first_col],
            var_name="Sentiment",
            value_name="Percentage"
        )

        fig = px.bar(
            long_topic,
            x=first_col,
            y="Percentage",
            color="Sentiment",
            barmode="group",
            title="Sentiment Mix Across Clusters"
        )

        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# =========================================================
# LANGUAGE
# =========================================================

elif page == "Language":

    st.html(
        """
        <div class="section-title">
            🔤 Language Intelligence
        </div>
        """
    )

    if os.path.exists(
        TOP_WORDS_PATH
    ):

        words = pd.read_csv(
            TOP_WORDS_PATH
        )

        word_col = words.columns[0]
        freq_col = words.columns[1]

        words_sorted = words.sort_values(
            freq_col,
            ascending=True
        )

        fig = px.bar(
            words_sorted,
            x=freq_col,
            y=word_col,
            orientation="h",
            title="Top 20 Most Frequent Words"
        )

        fig.update_layout(
            height=650,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    a, b = st.columns(2)

    with a:

        path = os.path.join(
            ASSET_DIR,
            "positive_wordcloud.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="Positive Language",
                use_container_width=True
            )

    with b:

        path = os.path.join(
            ASSET_DIR,
            "negative_wordcloud.png"
        )

        if os.path.exists(path):

            st.image(
                path,
                caption="Negative Language",
                use_container_width=True
            )


# =========================================================
# MODELS
# =========================================================

elif page == "Models":

    st.html(
        """
        <div class="section-title">
            🤖 Model Arena
        </div>
        """
    )

    performance = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Naive Bayes",
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
            "F1": [
                80.92,
                78.43,
                80.70
            ]
        }
    )

    selected_name = st.selectbox(
        "Explore model",
        performance["Model"]
    )

    selected = performance[
        performance["Model"]
        == selected_name
    ].iloc[0]

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    values = [
        selected["Accuracy"],
        selected["Precision"],
        selected["Recall"],
        selected["F1"]
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=metrics + [metrics[0]],
            fill="toself",
            name=selected_name
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[70,90]
            )
        ),
        height=540,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.html(
        """
        <div class="insight">

            🏆 <b>Best overall:</b>
            Logistic Regression — 80.61% test accuracy
            and 80.92% F1 score.

        </div>
        """
    )


# =========================================================
# PREDICT
# =========================================================

elif page == "Predict":

    st.html(
        """
        <div class="section-title">
            🔮 Predict
        </div>
        """
    )

    st.write(
        "Enter a social media post and get sentiment, "
        "confidence, probability and cluster information."
    )

    if not MODEL_READY:

        st.error(
            f"Model loading failed: {MODEL_ERROR}"
        )

    else:

        text = st.text_area(
            "What are people saying?",
            placeholder=(
                "Example: "
                "I absolutely love this product!"
            ),
            height=170
        )

        if st.button(
            "✨ Analyze Sentiment",
            use_container_width=True,
            key="predict_button"
        ):

            if not text.strip():

                st.warning(
                    "Please enter a social media post."
                )

            else:

                with st.spinner(
                    "Reading the pulse..."
                ):

                    time.sleep(.35)

                    result = predict_text(
                        text
                    )

                if result is None:

                    st.error(
                        "The text became empty after preprocessing."
                    )

                else:

                    display_prediction(
                        result
                    )

                    st.markdown(
                        "### 🧹 Processed Text"
                    )

                    st.code(
                        result["cleaned"]
                    )

                    positive, negative = (
                        prediction_contributions(
                            result["vector"]
                        )
                    )

                    st.markdown(
                        "### 🧠 Why this prediction?"
                    )

                    left, right = st.columns(2)

                    with left:

                        st.markdown(
                            "#### Positive contributors"
                        )

                        if positive:

                            st.dataframe(
                                pd.DataFrame(
                                    positive,
                                    columns=[
                                        "Feature",
                                        "Contribution"
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.info(
                                "No strong positive feature found."
                            )

                    with right:

                        st.markdown(
                            "#### Negative contributors"
                        )

                        if negative:

                            st.dataframe(
                                pd.DataFrame(
                                    negative,
                                    columns=[
                                        "Feature",
                                        "Contribution"
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True
                            )

                        else:

                            st.info(
                                "No strong negative feature found."
                            )


# =========================================================
# WHY?
# =========================================================

elif page == "Why?":

    st.html(
        """
        <div class="section-title">
            🧠 Why? — Explainable AI
        </div>
        """
    )

    st.write(
        "These are the TF-IDF features with the strongest "
        "positive and negative influence on the Logistic "
        "Regression classifier."
    )

    if MODEL_READY:

        features = (
            vectorizer
            .get_feature_names_out()
        )

        coefficients = model.coef_[0]

        explanation = pd.DataFrame(
            {
                "Feature": features,
                "Coefficient": coefficients
            }
        )

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

        combined = pd.concat(
            [
                negative,
                positive
            ]
        ).drop_duplicates()

        fig = px.bar(
            combined.sort_values(
                "Coefficient"
            ),
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Most Influential Sentiment Features"
        )

        fig.update_layout(
            height=700,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div class="footer">

        <b>SocialPulse</b>
        • Feel the pulse of social media.

        <br><br>

        Sentiment140 • TF-IDF • Logistic Regression
        • K-Means • Explainable AI

    </div>
    """
)

