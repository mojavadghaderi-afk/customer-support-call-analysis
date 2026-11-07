
import nltk
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
    if text_column not in df.columns:
        raise ValueError(f"Missing column: {text_column}")

    result = df.copy()
    analyzer = get_analyzer()

    scores = result[text_column].fillna("").astype(str).apply(
        analyzer.polarity_scores
    )

    result["compound"] = scores.apply(
        lambda x: x["compound"]
    )

    result["predicted_sentiment"] = result["compound"].apply(
        lambda x: (
            "positive" if x >= 0.05
            else "negative" if x <= -0.05
            else "neutral"
        )
    )

    return result


def count_true_positives(
    df,
    actual_column="sentiment_label",
    predicted_column="predicted_sentiment"
):
    actual = df[actual_column].astype(str).str.lower().str.strip()

    predicted = (
        df[predicted_column].astype(str).str.lower().str.strip()
    )

    return int(
        ((actual == "positive") & (predicted == "positive")).sum()
    )
