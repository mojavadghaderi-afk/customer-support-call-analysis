import nltk
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_analyzer():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def classify_sentiment(text):
    analyzer = get_analyzer()
    score = analyzer.polarity_scores(str(text))["compound"]

    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"


def analyze_dataframe(df, text_column="text"):
    result = df.copy()

    analyzer = get_analyzer()

    scores = result[text_column].fillna("").astype(str).apply(
        analyzer.polarity_scores
    )

    result["compound"] = scores.apply(lambda x: x["compound"])

    result["sentiment_label"] = result["compound"].apply(
        lambda x: (
            "positive" if x >= 0.05
            else "negative" if x <= -0.05
            else "neutral"
        )
    )

    return result


def count_true_positives(
    df,
    actual_column="sentiment",
    predicted_column="sentiment_label"
):
    actual = df[actual_column].astype(str).str.lower().str.strip()
    predicted = df[predicted_column].astype(str).str.lower().str.strip()

    return int(((actual == "positive") & (predicted == "positive")).sum())
