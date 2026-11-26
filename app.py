
import streamlit as st
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Product Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

# Load trained pipeline
MODEL_PATH = Path(__file__).parent / "models" / "best_sentiment_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Application interface
st.title("🛍️ Product Review Sentiment Analysis")

st.write(
    "Enter a product review below and our machine learning "
    "model will predict its sentiment."
)

review = st.text_area(
    "Your product review",
    placeholder="Example: This product is amazing. I love its quality!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]

        sentiment_labels = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

        sentiment = sentiment_labels[int(prediction)]

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"😊 {sentiment} Sentiment")
        elif sentiment == "Negative":
            st.error(f"😞 {sentiment} Sentiment")
        else:
            st.info(f"😐 {sentiment} Sentiment")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([review])[0]

            st.subheader("Prediction probabilities")

            for label, probability in zip(
                ["Negative", "Neutral", "Positive"],
                probabilities
            ):
                st.write(f"**{label}:** {probability:.2%}")
                st.progress(float(probability))

st.divider()

st.caption(
    "Developed using TF-IDF and Logistic Regression. "
    "Predictions are based on patterns learned from product reviews."
)
