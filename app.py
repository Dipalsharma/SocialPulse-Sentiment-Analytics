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
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSET_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

ADVANCED_DIR = os.path.join(
    BASE_DIR,
    "advanced"
)

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

MONTHLY_PATH = os.path.join(
    ASSET_DIR,
    "monthly_percentage.csv"
)

TOP_WORDS_PATH = os.path.join(
    ASSET_DIR,
    "top_words.csv"
)

TOPIC_KEYWORDS_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_keywords.csv"
)

TOPIC_DISTRIBUTION_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_distribution.csv"
)

TOPIC_SENTIMENT_PATH = os.path.join(
    ADVANCED_DIR,
    "topic_sentiment_percentage.csv"
)

ANOMALY_PATH = os.path.join(
    ADVANCED_DIR,
    "monthly_anomalies.csv"
)


# =========================================================
# NLTK
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
        stops = set(
            stopwords.words("english")
        )
    except Exception:
        stops = set()

    return (
        stops,
        WordNetLemmatizer()
    )


try:

    STOP_WORDS, LEMMATIZER = initialize_nltk()

except Exception:

    STOP_WORDS = set()
    LEMMATIZER = None


NEGATIONS = {
    "no",
    "not",
    "nor",
    "never",
    "none",
    "neither",
    "don't",
    "doesn't",
    "didn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
    "shouldn't",
    "haven't",
    "hasn't",
    "hadn't"
}

STOP_WORDS = STOP_WORDS - NEGATIONS


# =========================================================
# TEXT PREPROCESSING
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

        if word in STOP_WORDS:
            continue

        if LEMMATIZER is not None:

            try:
                word = LEMMATIZER.lemmatize(word)
            except Exception:
                pass

        cleaned_words.append(word)

    return " ".join(cleaned_words)


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

    topic_model = None

    if os.path.exists(TOPIC_MODEL_PATH):

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

    MODEL_READY = True
    MODEL_ERROR = ""

except Exception as exc:

    model = None
    vectorizer = None
    topic_model = None

    MODEL_READY = False
    MODEL_ERROR = str(exc)


# =========================================================
# SOCIALPULSE DESIGN SYSTEM
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
            rgba(255, 186, 120, 0.28),
            transparent 24%
        ),
        radial-gradient(
            circle at 92% 12%,
            rgba(180, 160, 255, 0.24),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 100%,
            rgba(137, 208, 255, 0.16),
            transparent 30%
        ),
        linear-gradient(
            180deg,
            #fcfaf7 0%,
            #f5f2ed 100%
        );

    color: #29252b;
}

.block-container {

    max-width: 1450px;

    padding-top: 1.4rem;
    padding-bottom: 4rem;
}


/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {

    background:
        rgba(255, 253, 249, 0.96);

    border-right:
        1px solid
        rgba(50, 40, 30, 0.07);
}

section[data-testid="stSidebar"] * {

    color:
        #312b33 !important;
}


/* ======================================================
   HERO
   ====================================================== */

.hero {

    position: relative;

    overflow: hidden;

    min-height: 300px;

    padding: 46px;

    border-radius:
        32px;

    margin-bottom:
        26px;

    background:
        linear-gradient(
            120deg,
            #fff0d7,
            #ffe4da,
            #eee8ff,
            #e1f4ff,
            #fff0d7
        );

    background-size:
        350% 350%;

    animation:
        heroFlow 13s ease infinite;

    border:
        1px solid
        rgba(255,255,255,.85);

    box-shadow:
        0 24px 70px
        rgba(78, 58, 40, .12);
}


/* Floating 3D shapes */

.orb {

    position:
        absolute;

    border-radius:
        50%;

    filter:
        blur(1px);

    box-shadow:
        inset -14px -14px 22px
        rgba(0,0,0,.06),

        inset 10px 10px 18px
        rgba(255,255,255,.55),

        0 18px 35px
        rgba(50,40,30,.10);

    animation:
        float 7s ease-in-out infinite alternate;
}

.orb-one {

    width:
        105px;

    height:
        105px;

    right:
        11%;

    top:
        42px;

    background:
        linear-gradient(
            145deg,
            #ffc37e,
            #f59e60
        );
}

.orb-two {

    width:
        68px;

    height:
        68px;

    right:
        24%;

    bottom:
        36px;

    background:
        linear-gradient(
            145deg,
            #c9bcff,
            #9e8df5
        );

    animation-delay:
        1.5s;
}

.orb-three {

    width:
        48px;

    height:
        48px;

    right:
        6%;

    bottom:
        33px;

    background:
        linear-gradient(
            145deg,
            #91d8ff,
            #60bbed
        );

    animation-delay:
        .8s;
}


.hero-content {

    position:
        relative;

    z-index:
        5;

    max-width:
        780px;
}


.badge {

    display:
        inline-block;

    padding:
        8px 14px;

    border-radius:
        999px;

    background:
        rgba(255,255,255,.64);

    border:
        1px solid
        rgba(255,255,255,.85);

    color:
        #795333;

    font-size:
        .78rem;

    font-weight:
        800;

    letter-spacing:
        .35px;
}


.hero-title {

    margin-top:
        15px;

    font-size:
        3.5rem;

    line-height:
        1;

    font-weight:
        950;

    letter-spacing:
        -2px;

    color:
        #272229;
}


.hero-subtitle {

    max-width:
        760px;

    margin-top:
        15px;

    font-size:
        1.05rem;

    line-height:
        1.75;

    color:
        #675e66;
}


/* ======================================================
   FLOATING KPI CARDS
   ====================================================== */

.kpi {

    position:
        relative;

    padding:
        22px;

    min-height:
        115px;

    border-radius:
        23px;

    background:
        rgba(
            255,
            255,
            255,
            .88
        );

    border:
        1px solid
        rgba(
            68,
            55,
            40,
            .08
        );

    box-shadow:
        0 12px 34px
        rgba(
            55,
            39,
            20,
            .08
        );

    transition:
        transform .30s ease,
        box-shadow .30s ease;
}

.kpi:hover {

    transform:
        translateY(-8px)
        rotate(-.35deg);

    box-shadow:
        0 22px 42px
        rgba(
            55,
            39,
            20,
            .13
        );
}

.kpi-label {

    color:
        #857c82;

    font-size:
        .79rem;

    font-weight:
        750;
}

.kpi-value {

    margin-top:
        9px;

    color:
        #2b252b;

    font-size:
        1.55rem;

    font-weight:
        900;
}


/* ======================================================
   SECTION
   ====================================================== */

.section-title {

    margin-top:
        38px;

    margin-bottom:
        8px;

    font-size:
        1.75rem;

    font-weight:
        900;

    color:
        #29242a;
}


/* ======================================================
   GLASS CARD
   ====================================================== */

.glass-card {

    padding:
        22px;

    border-radius:
        24px;

    background:
        rgba(
            255,
            255,
            255,
            .84
        );

    border:
        1px solid
        rgba(
            255,
            255,
            255,
            .94
        );

    box-shadow:
        0 16px 40px
        rgba(
            55,
            39,
            20,
            .07
        );

    transition:
        all .28s ease;
}

.glass-card:hover {

    transform:
        translateY(-5px);

    box-shadow:
        0 22px 50px
        rgba(
            55,
            39,
            20,
            .11
        );
}


/* ======================================================
   QUICK ACTION
   ====================================================== */

.quick-card {

    padding:
        20px;

    border-radius:
        20px;

    background:
        linear-gradient(
            135deg,
            rgba(255,237,211,.92),
            rgba(239,231,255,.92)
        );

    border:
        1px solid
        rgba(255,255,255,.90);
}


/* ======================================================
   INPUT
   ====================================================== */

.stTextArea label {

    color:
        #4e464e !important;

    font-weight:
        800 !important;
}

.stTextArea textarea {

    background:
        rgba(255,255,255,.94) !important;

    color:
        #242028 !important;

    border:
        1px solid
        #d7cec6 !important;

    border-radius:
        18px !important;

    font-size:
        1rem !important;

    box-shadow:
        inset 0 2px 8px
        rgba(40,30,20,.035);
}

.stTextArea textarea::placeholder {

    color:
        #81777e !important;

    opacity:
        1 !important;
}


/* ======================================================
   BUTTON
   ====================================================== */

.stButton > button {

    min-height:
        46px;

    border:
        none !important;

    border-radius:
        14px !important;

    background:
        linear-gradient(
            135deg,
            #f59e0b,
            #f28a62,
            #8b5cf6
        ) !important;

    color:
        #ffffff !important;

    font-weight:
        850 !important;

    box-shadow:
        0 10px 24px
        rgba(124,58,237,.14);

    transition:
        all .25s ease !important;
}

.stButton > button:hover {

    transform:
        translateY(-3px)
        scale(1.01);

    box-shadow:
        0 16px 32px
        rgba(124,58,237,.20);
}


/* ======================================================
   RESULT
   ====================================================== */

.result {

    margin-top:
        20px;

    padding:
        30px;

    border-radius:
        24px;

    text-align:
        center;

    animation:
        resultPop .45s ease both;
}

.result-positive {

    background:
        linear-gradient(
            135deg,
            #e7f8ee,
            #f9fffb
        );

    border:
        1px solid
        #a7d9bc;
}

.result-negative {

    background:
        linear-gradient(
            135deg,
            #fff0ee,
            #fffafa
        );

    border:
        1px solid
        #ebb2ad;
}

.result-title {

    font-size:
        2.2rem;

    font-weight:
        950;
}

.result-sub {

    margin-top:
        8px;

    color:
        #685f66;

}


/* ======================================================
   INSIGHT
   ====================================================== */

.insight {

    padding:
        19px 21px;

    border-radius:
        18px;

    background:
        #fff7e8;

    border:
        1px solid
        #efd19a;

    color:
        #624a22;

}


/* ======================================================
   ANIMATION
   ====================================================== */

@keyframes heroFlow {

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

@keyframes float {

    from {
        transform:
            translateY(0px)
            rotate(-5deg);
    }

    to {
        transform:
            translateY(-18px)
            rotate(5deg);
    }

}

@keyframes resultPop {

    from {
        opacity: 0;
        transform:
            scale(.95)
            translateY(8px);
    }

    to {
        opacity: 1;
        transform:
            scale(1)
            translateY(0);
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
        "NLP • TF-IDF • Machine Learning"
    )


# =========================================================
# HERO
# =========================================================

st.html(
    """
    <div class="hero">

        <div class="orb orb-one"></div>
        <div class="orb orb-two"></div>
        <div class="orb orb-three"></div>

        <div class="hero-content">

            <div class="badge">
                ☀️ AI SOCIAL INTELLIGENCE
            </div>

            <div class="hero-title">
                SocialPulse
            </div>

            <div class="hero-subtitle">
                Understand what people feel,
                discover conversation patterns,
                and explore the pulse of
                1.6 million social media posts.
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
    ("Positive / Negative", "50% / 50%"),
    ("Discovered Clusters", "6"),
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
# HOME
# =========================================================

if page == "Home":

    st.html(
        """
        <div class="section-title">
            Your social media pulse
        </div>
        """
    )

    st.write(
        "Explore the analytics below or test the AI "
        "with your own social media post."
    )

    st.markdown(
        "### ⚡ Instant Demo"
    )

    if "demo_text" not in st.session_state:

        st.session_state.demo_text = ""

    a, b, c = st.columns(3)

    with a:

        if st.button(
            "😊 Happy",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "I absolutely love this product! "
                "Everything works perfectly."
            )

    with b:

        if st.button(
            "😡 Angry",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "This service is terrible. "
                "I am extremely disappointed."
            )

    with c:

        if st.button(
            "🤩 Excited",
            use_container_width=True
        ):

            st.session_state.demo_text = (
                "What an amazing experience! "
                "I am so happy with the results."
            )

    home_text = st.text_area(
        "What are people saying?",
        value=st.session_state.demo_text,
        placeholder=(
            "Write a social media post here..."
        ),
        height=145,
        key="home_text"
    )

    if st.button(
        "✨ Analyze Sentiment",
        use_container_width=True,
        key="home_analyze"
    ):

        if not MODEL_READY:

            st.error(
                f"Model unavailable: {MODEL_ERROR}"
            )

        elif not home_text.strip():

            st.warning(
                "Please enter a post first."
            )

        else:

            with st.spinner(
                "Reading the social pulse..."
            ):

                time.sleep(.35)

                cleaned = clean_text(
                    home_text
                )

                if not cleaned:

                    st.error(
                        "No usable text remained after cleaning."
                    )

                else:

                    vector = vectorizer.transform(
                        [cleaned]
                    )

                    prediction = int(
                        model.predict(vector)[0]
                    )

                    probabilities = (
                        model
                        .predict_proba(vector)[0]
                    )

                    positive = (
                        float(probabilities[1]) * 100
                    )

                    negative = (
                        float(probabilities[0]) * 100
                    )

                    if prediction == 1:

                        confidence = positive

                        st.html(
                            f"""
                            <div class="result result-positive">

                                <div class="result-title">
                                    😊 POSITIVE
                                </div>

                                <div class="result-sub">
                                    Confidence:
                                    <b>
                                        {confidence:.2f}%
                                    </b>
                                </div>

                            </div>
                            """
                        )

                    else:

                        confidence = negative

                        st.html(
                            f"""
                            <div class="result result-negative">

                                <div class="result-title">
                                    😞 NEGATIVE
                                </div>

                                <div class="result-sub">
                                    Confidence:
                                    <b>
                                        {confidence:.2f}%
                                    </b>
                                </div>

                            </div>
                            """
                        )


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

    st.write(
        "A balanced dataset with an equal number of "
        "Positive and Negative tweets."
    )

    pulse = pd.DataFrame(
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
        pulse,
        names="Sentiment",
        values="Tweets",
        hole=.68,
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

            💡 <b>Pulse:</b>
            the dataset contains
            <b>800,000 Positive</b> and
            <b>800,000 Negative</b> tweets.

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
            height=530,
            hovermode="x unified",
            yaxis_title="Sentiment Share (%)",
            xaxis_title="Month",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Monthly trend data not found."
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
                    f"{len(detected)} potential "
                    "statistical anomaly period(s) detected."
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
        "K-Means clustering discovers six latent "
        "conversation clusters from TF-IDF features."
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
            title="Discovered Conversation Clusters"
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

        if topic_sentiment.shape[1] >= 3:

            first_column = (
                topic_sentiment.columns[0]
            )

            long_topic = topic_sentiment.melt(
                id_vars=[first_column],
                var_name="Sentiment",
                value_name="Percentage"
            )

            fig = px.bar(
                long_topic,
                x=first_column,
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
                use_container_width=True
            )


# =========================================================
# LANGUAGE
# =========================================================

elif page == "Language":

    st.html(
        """
        <div class="section-title">
            🔤 Language
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
        frequency_col = words.columns[1]

        words = words.sort_values(
            frequency_col,
            ascending=True
        )

        fig = px.bar(
            words,
            x=frequency_col,
            y=word_col,
            orientation="h",
            text=frequency_col,
            title="Top 20 Most Frequent Words"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=640,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
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
        }
    )

    selected_name = st.selectbox(
        "Compare model",
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=selected_name
        )
    )

    fig.update_layout(
        height=540,
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
        fig,
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

            🏆 <b>Selected final model:</b>
            Logistic Regression achieved
            <b>80.61% test accuracy</b> and
            <b>80.92% F1 score</b>.

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
            🔮 AI Sentiment Predictor
        </div>
        """
    )

    st.write(
        "Write a social media post and discover "
        "its sentiment, confidence and model reasoning."
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
            height=175
        )

        if st.button(
            "✨ Analyze Sentiment",
            use_container_width=True,
            key="predict_main"
        ):

            if not text.strip():

                st.warning(
                    "Please enter a social media post."
                )

            else:

                with st.spinner(
                    "Analyzing..."
                ):

                    time.sleep(.35)

                    result = None

                    cleaned = clean_text(
                        text
                    )

                    if cleaned:

                        vector = vectorizer.transform(
                            [cleaned]
                        )

                        prediction = int(
                            model.predict(vector)[0]
                        )

                        probabilities = (
                            model
                            .predict_proba(vector)[0]
                        )

                        result = {
                            "cleaned": cleaned,
                            "vector": vector,
                            "prediction": prediction,
                            "probabilities": probabilities
                        }

                    if result is None:

                        st.error(
                            "No usable text remained after cleaning."
                        )

                    else:

                        probabilities = (
                            result["probabilities"]
                        )

                        positive = (
                            float(probabilities[1])
                            * 100
                        )

                        negative = (
                            float(probabilities[0])
                            * 100
                        )

                        if result["prediction"] == 1:

                            confidence = positive

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

                            confidence = negative

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

                        c1, c2 = st.columns(2)

                        with c1:

                            st.metric(
                                "Positive Probability",
                                f"{positive:.2f}%"
                            )

                            st.progress(
                                min(
                                    positive / 100,
                                    1.0
                                )
                            )

                        with c2:

                            st.metric(
                                "Negative Probability",
                                f"{negative:.2f}%"
                            )

                            st.progress(
                                min(
                                    negative / 100,
                                    1.0
                                )
                            )

                        st.markdown(
                            "### 🧹 Processed Text"
                        )

                        st.code(
                            result["cleaned"]
                        )


# =========================================================
# WHY
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
        "See which TF-IDF features have the strongest "
        "learned influence on the classifier."
    )

    if not MODEL_READY:

        st.error(
            f"Model unavailable: {MODEL_ERROR}"
        )

    else:

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
            title="Most Influential Sentiment Features"
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
