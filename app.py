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
def initialize_nltk():

    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4")
    ]

    for resource_path, package_name in resources:

        try:
            nltk.data.find(resource_path)

        except LookupError:
            nltk.download(
                package_name,
                quiet=True
            )

    try:
        stop_words = set(
            stopwords.words("english")
        )
    except LookupError:
        stop_words = set()

    return (
        stop_words,
        WordNetLemmatizer()
    )


stop_words, lemmatizer = initialize_nltk()


# =========================================================
# NEGATIONS
# =========================================================

NEGATIONS = {
    "no", "not", "nor", "never", "none", "neither",
    "don't", "doesn't", "didn't", "isn't", "aren't",
    "wasn't", "weren't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't",
    "haven't", "hasn't", "hadn't"
}

stop_words = stop_words - NEGATIONS


# =========================================================
# TEXT CLEANING
# Matches training preprocessing
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
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_main_model():

    model = joblib.load(
        MODEL_PATH
    )

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    return model, vectorizer


@st.cache_resource
def load_topic_model():

    if not os.path.exists(TOPIC_MODEL_PATH):
        return None

    return joblib.load(
        TOPIC_MODEL_PATH
    )


try:

    model, vectorizer = load_main_model()

    MODEL_READY = True
    MODEL_ERROR = ""

except Exception as exc:

    model = None
    vectorizer = None

    MODEL_READY = False
    MODEL_ERROR = str(exc)


try:

    topic_model = load_topic_model()

except Exception:

    topic_model = None


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
<style>

.stApp {

    background:
        radial-gradient(
            circle at 7% 7%,
            rgba(245,158,11,.16),
            transparent 24%
        ),
        radial-gradient(
            circle at 94% 8%,
            rgba(124,58,237,.12),
            transparent 25%
        ),
        linear-gradient(
            180deg,
            #fffaf3 0%,
            #f7f3ed 100%
        );

    color: #28232a;
}


.block-container {

    max-width: 1450px;

    padding-top: 1.5rem;

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

    color: #2b2630 !important;
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
            #ffe6dc,
            #eee5ff,
            #fff0c9
        );

    background-size: 300% 300%;

    animation:
        heroMove 12s ease infinite;

    border:
        1px solid
        rgba(124,58,237,.09);

    box-shadow:
        0 18px 55px
        rgba(70,45,20,.10);
}


.hero-orb {

    position: absolute;

    width: 260px;
    height: 260px;

    right: -90px;
    top: -120px;

    border-radius: 50%;

    background:
        rgba(245,158,11,.16);

    filter: blur(65px);

    animation:
        orbMove 7s ease-in-out infinite alternate;
}


.hero-inner {

    position: relative;

    z-index: 2;
}


.badge {

    display: inline-block;

    padding:
        7px 14px;

    border-radius:
        999px;

    background:
        rgba(255,255,255,.65);

    color:
        #77451c;

    font-size:
        .78rem;

    font-weight:
        800;
}


.hero-title {

    margin-top:
        14px;

    font-size:
        3.3rem;

    font-weight:
        900;

    color:
        #2a2429;
}


.hero-subtitle {

    max-width:
        900px;

    margin-top:
        10px;

    font-size:
        1.05rem;

    line-height:
        1.7;

    color:
        #696168;
}


/* KPI */

.kpi {

    padding:
        22px;

    min-height:
        110px;

    border-radius:
        22px;

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

}


.kpi:hover {

    transform:
        translateY(-7px);

    box-shadow:
        0 18px 38px
        rgba(55,36,15,.12);

}


.kpi-label {

    color:
        #81787e;

    font-size:
        .82rem;

    font-weight:
        700;
}


.kpi-value {

    margin-top:
        8px;

    font-size:
        1.55rem;

    font-weight:
        900;

    color:
        #28232a;
}


/* SECTIONS */

.section-title {

    margin-top:
        38px;

    margin-bottom:
        8px;

    font-size:
        1.7rem;

    font-weight:
        900;

    color:
        #28232a;
}


/* CARD */

.card {

    padding:
        22px;

    border-radius:
        22px;

    background:
        rgba(255,255,255,.90);

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


/* TEXT INPUT */

.stTextArea textarea {

    background:
        #fffdfa !important;

    color:
        #241f27 !important;

    border:
        1px solid
        #cfc5bd !important;

    border-radius:
        16px !important;

    font-size:
        1rem !important;
}


.stTextArea textarea::placeholder {

    color:
        #776e74 !important;

    opacity:
        1 !important;
}


/* BUTTONS */

.stButton > button {

    min-height:
        45px;

    border-radius:
        13px !important;

    border:
        none !important;

    background:
        linear-gradient(
            135deg,
            #f59e0b,
            #8b5cf6
        ) !important;

    color:
        white !important;

    font-weight:
        800 !important;

    box-shadow:
        0 8px 20px
        rgba(124,58,237,.13);

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

.result {

    padding:
        28px;

    margin-top:
        20px;

    border-radius:
        22px;

    text-align:
        center;

    animation:
        resultPop .45s ease both;
}


.result-positive {

    background:
        linear-gradient(
            135deg,
            #e9f8ef,
            #fbfffc
        );

    border:
        1px solid
        #a9ddc0;
}


.result-negative {

    background:
        linear-gradient(
            135deg,
            #fff0ef,
            #fffafa
        );

    border:
        1px solid
        #ecb4af;
}


.result-title {

    font-size:
        2rem;

    font-weight:
        900;
}


.result-sub {

    margin-top:
        8px;

    color:
        #665e64;
}


/* INFO */

.insight {

    padding:
        18px 20px;

    margin-top:
        18px;

    border-radius:
        17px;

    background:
        #fff8e9;

    border:
        1px solid
        #efd299;

    color:
        #624820;
}


/* FOOTER */

.footer {

    margin-top:
        60px;

    padding:
        25px;

    text-align:
        center;

    color:
        #898187;

    border-top:
        1px solid
        rgba(50,40,30,.08);
}


/* ANIMATIONS */

@keyframes heroMove {

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


@keyframes orbMove {

    from {
        transform:
            translate(0,0)
            scale(1);
    }

    to {
        transform:
            translate(-35px,25px)
            scale(1.15);
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


@media (max-width: 800px) {

    .hero {
        padding:
            30px 24px;
    }

    .hero-title {
        font-size:
            2.25rem;
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
        "Sentiment140"
    )

    st.caption(
        "1.6M Tweets"
    )

    st.caption(
        "NLP • TF-IDF • ML • K-Means"
    )


# =========================================================
# HERO
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="hero-orb"></div>

        <div class="hero-inner">

            <div class="badge">
                ☀️ AI-POWERED SOCIAL MEDIA INTELLIGENCE
            </div>

            <div class="hero-title">
                SocialPulse
            </div>

            <div class="hero-subtitle">
                Discover what people are saying,
                how sentiment moves, which themes
                dominate the conversation, and why
                the model makes its prediction.
            </div>

        </div>

    </div>
    """
)


# =========================================================
# KPI
# =========================================================

cols = st.columns(4)

kpis = [
    ("Dataset", "1.6M Tweets"),
    ("Sentiment Split", "50% / 50%"),
    ("Discovered Clusters", "6"),
    ("Best Accuracy", "80.61%")
]

for col, (label, value) in zip(
    cols,
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
# PREDICTION ENGINE
# =========================================================

def analyze_post(text):

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

    topic_id = None

    if topic_model is not None:

        try:

            topic_id = int(
                topic_model.predict(
                    vector
                )[0]
            ) + 1

        except Exception:

            topic_id = None

    return {
        "cleaned": cleaned,
        "vector": vector,
        "prediction": prediction,
        "probabilities": probabilities,
        "topic_id": topic_id
    }


# =========================================================
# CONTRIBUTIONS
# =========================================================

def get_contributions(vector):

    if not MODEL_READY:
        return [], []

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    coefficients = model.coef_[0]

    row = vector.tocoo()

    contributions = []

    for index, value in zip(
        row.col,
        row.data
    ):

        score = (
            float(value)
            *
            float(coefficients[index])
        )

        contributions.append(
            (
                feature_names[index],
                score
            )
        )

    positive = sorted(
        [
            item
            for item in contributions
            if item[1] > 0
        ],
        key=lambda item: item[1],
        reverse=True
    )[:6]

    negative = sorted(
        [
            item
            for item in contributions
            if item[1] < 0
        ],
        key=lambda item: item[1]
    )[:6]

    return positive, negative


# =========================================================
# DISPLAY RESULT
# =========================================================

def display_result(result):

    probabilities = result["probabilities"]

    positive = (
        float(probabilities[1]) * 100
    )

    negative = (
        float(probabilities[0]) * 100
    )

    prediction = result["prediction"]

    confidence = (
        positive
        if prediction == 1
        else negative
    )

    if prediction == 1:

        st.html(
            f"""
            <div class="result result-positive">

                <div class="result-title">
                    😊 POSITIVE
                </div>

                <div class="result-sub">
                    Confidence:
                    <b>{confidence:.2f}%</b>
                </div>

            </div>
            """
        )

    else:

        st.html(
            f"""
            <div class="result result-negative">

                <div class="result-title">
                    😞 NEGATIVE
                </div>

                <div class="result-sub">
                    Confidence:
                    <b>{confidence:.2f}%</b>
                </div>

            </div>
            """
        )

    a, b, c = st.columns(3)

    with a:

        st.metric(
            "Positive",
            f"{positive:.2f}%"
        )

    with b:

        st.metric(
            "Negative",
            f"{negative:.2f}%"
        )

    with c:

        if result["topic_id"] is not None:

            st.metric(
                "Detected Cluster",
                f"Cluster {result['topic_id']}"
            )

        else:

            st.metric(
                "Detected Cluster",
                "Unavailable"
            )

    chart_df = pd.DataFrame(
        {
            "Sentiment": [
                "Positive",
                "Negative"
            ],
            "Probability": [
                positive,
                negative
            ]
        }
    )

    fig = px.bar(
        chart_df,
        x="Sentiment",
        y="Probability",
        text="Probability",
        title="Prediction Confidence Breakdown"
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        height=360,
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
        use_container_width=True
    )


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
        "Explore the pulse of social media or try "
        "an instant sentiment prediction."
    )

    st.markdown(
        "### ⚡ Quick Demo"
    )

    if "demo_text" not in st.session_state:

        st.session_state.demo_text = ""

    d1, d2, d3 = st.columns(3)

    with d1:

        if st.button(
            "😊 Happy",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "I absolutely love this product! "
                "It works perfectly."
            )

    with d2:

        if st.button(
            "😡 Angry",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "This service is terrible. "
                "I am extremely disappointed."
            )

    with d3:

        if st.button(
            "🤩 Excited",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "What an amazing day! "
                "Everything is going perfectly."
            )

    home_text = st.text_area(
        "What are people saying?",
        value=st.session_state.demo_text,
        placeholder=(
            "Write a social media post here..."
        ),
        height=150
    )

    if st.button(
        "✨ Analyze Sentiment",
        use_container_width=True,
        key="home_predict"
    ):

        if not MODEL_READY:

            st.error(
                f"Model could not be loaded: {MODEL_ERROR}"
            )

        elif not home_text.strip():

            st.warning(
                "Please enter a social media post."
            )

        else:

            with st.spinner(
                "Reading the pulse..."
            ):

                time.sleep(.35)

                result = analyze_post(
                    home_text
                )

            if result is None:

                st.error(
                    "No usable text remained after preprocessing."
                )

            else:

                display_result(result)


# =========================================================
# PULSE
# =========================================================

elif page == "Pulse":

    st.html(
        '<div class="section-title">💓 Sentiment Pulse</div>'
    )

    st.write(
        "The overall Positive vs Negative balance "
        "across the 1.6M-tweet dataset."
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
        hole=.62,
        title="Overall Sentiment Balance"
    )

    fig.update_traces(
        textinfo="percent+label",
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
        use_container_width=True
    )

    st.html(
        """
        <div class="insight">

            💡 <b>Dataset balance:</b>
            800,000 Positive and 800,000 Negative tweets.

        </div>
        """
    )


# =========================================================
# TRENDS
# =========================================================

elif page == "Trends":

    st.html(
        '<div class="section-title">📈 Sentiment Trends</div>'
    )

    if os.path.exists(MONTHLY_PATH):

        monthly = pd.read_csv(
            MONTHLY_PATH
        )

        month_col = monthly.columns[0]

        long_monthly = monthly.melt(
            id_vars=[month_col],
            var_name="Sentiment",
            value_name="Percentage"
        )

        fig = px.area(
            long_monthly,
            x=month_col,
            y="Percentage",
            color="Sentiment",
            markers=True,
            title="Monthly Sentiment Movement"
        )

        fig.update_layout(
            height=520,
            yaxis_title="Sentiment Share (%)",
            xaxis_title="Month",
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Monthly trend data is unavailable."
        )

    st.markdown(
        "### 🚨 Unusual Sentiment Activity"
    )

    if os.path.exists(ANOMALY_PATH):

        anomaly = pd.read_csv(
            ANOMALY_PATH
        )

        if "Anomaly" in anomaly.columns:

            flags = (
                anomaly["Anomaly"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )

            detected = anomaly[
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
            "View anomaly analysis"
        ):

            st.dataframe(
                anomaly,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Anomaly analysis data is unavailable."
        )


# =========================================================
# TOPICS
# =========================================================

elif page == "Topics":

    st.html(
        '<div class="section-title">🧩 Topic Intelligence</div>'
    )

    st.write(
        "Six latent conversation clusters discovered "
        "using K-Means clustering on TF-IDF features."
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
            use_container_width=True
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

        topic_col = (
            topic_sentiment.columns[0]
        )

        long_topic = topic_sentiment.melt(
            id_vars=[topic_col],
            var_name="Sentiment",
            value_name="Percentage"
        )

        fig = px.bar(
            long_topic,
            x=topic_col,
            y="Percentage",
            color="Sentiment",
            barmode="group",
            title="Sentiment Mix Across Clusters"
        )

        fig.update_layout(
            height=500,
            yaxis_title="Percentage (%)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# LANGUAGE
# =========================================================

elif page == "Language":

    st.html(
        '<div class="section-title">🔤 Language Intelligence</div>'
    )

    if os.path.exists(
        TOP_WORDS_PATH
    ):

        words = pd.read_csv(
            TOP_WORDS_PATH
        )

        word_col = words.columns[0]
        freq_col = words.columns[1]

        words = words.sort_values(
            freq_col,
            ascending=True
        )

        fig = px.bar(
            words,
            x=freq_col,
            y=word_col,
            orientation="h",
            text=freq_col,
            title="Top 20 Most Frequent Words"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=650,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    a, b = st.columns(2)

    with a:

        positive_path = os.path.join(
            ASSET_DIR,
            "positive_wordcloud.png"
        )

        if os.path.exists(positive_path):

            st.image(
                positive_path,
                caption="Positive Language",
                use_container_width=True
            )

    with b:

        negative_path = os.path.join(
            ASSET_DIR,
            "negative_wordcloud.png"
        )

        if os.path.exists(negative_path):

            st.image(
                negative_path,
                caption="Negative Language",
                use_container_width=True
            )


# =========================================================
# MODELS
# =========================================================

elif page == "Models":

    st.html(
        '<div class="section-title">🤖 Model Arena</div>'
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
            "F1 Score": [
                80.92,
                78.43,
                80.70
            ]
        }
    )

    selected_name = st.selectbox(
        "Inspect model",
        performance["Model"]
    )

    selected = performance[
        performance["Model"]
        == selected_name
    ].iloc[0]

    categories = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    values = [
        selected["Accuracy"],
        selected["Precision"],
        selected["Recall"],
        selected["F1 Score"]
    ]

    radar = go.Figure()

    radar.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=selected_name
        )
    )

    radar.update_layout(
        height=520,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[70, 90]
            )
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        radar,
        use_container_width=True
    )

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.html(
        """
        <div class="insight">

            🏆 <b>Best overall model:</b>
            Logistic Regression with 80.61% test accuracy
            and 80.92% F1 score.

        </div>
        """
    )


# =========================================================
# PREDICT
# =========================================================

elif page == "Predict":

    st.html(
        '<div class="section-title">🔮 Live Prediction</div>'
    )

    st.write(
        "Enter a social media post and get sentiment, "
        "confidence, probability and cluster information."
    )

    if not MODEL_READY:

        st.error(
            f"Model could not be loaded: {MODEL_ERROR}"
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

                    result = analyze_post(
                        text
                    )

                if result is None:

                    st.error(
                        "No usable text remained after preprocessing."
                    )

                else:

                    display_result(
                        result
                    )

                    st.markdown(
                        "### 🧹 Processed Text"
                    )

                    st.code(
                        result["cleaned"]
                    )

                    positive, negative = (
                        get_contributions(
                            result["vector"]
                        )
                    )

                    st.markdown(
                        "### 🧠 Why this prediction?"
                    )

                    a, b = st.columns(2)

                    with a:

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
                                "No strong positive contributor."
                            )

                    with b:

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
                                "No strong negative contributor."
                            )


# =========================================================
# WHY / EXPLAINABILITY
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
        "These features have the strongest learned "
        "influence on the Logistic Regression classifier."
    )

    if MODEL_READY:

        features = (
            vectorizer
            .get_feature_names_out()
        )

        coefficients = (
            model.coef_[0]
        )

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
            title="Most Influential Features"
        )

        fig.update_layout(
            height=700,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.html(
            """
            <div class="insight">

                💡 Positive coefficients push the prediction
                toward Positive sentiment, while negative
                coefficients push it toward Negative sentiment.

            </div>
            """
        )

    else:

        st.error(
            "The trained model is unavailable."
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
