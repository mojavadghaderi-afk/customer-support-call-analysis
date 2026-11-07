
import pandas as pd
from pathlib import Path

from src.audio_utils import (
    transcribe_audio,
    get_audio_metadata
)

from src.sentiment import (
    analyze_dataframe,
    count_true_positives
)

from src.entities import (
    load_nlp,
    most_frequent_entity
)

from src.similarity import (
    find_most_similar_text
)


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path):
    path = Path(path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


def run_project(
    audio_file="data/sample_customer_call.wav",
    csv_file="data/customer_call_transcriptions.csv"
):

    audio_path = resolve_path(audio_file)
    csv_path = resolve_path(csv_file)

    # Check input files
    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    # 1. Audio processing
    transcribed_text = transcribe_audio(str(audio_path))

    frame_rate, number_channels = get_audio_metadata(
        str(audio_path)
    )

    # 2. Load customer calls
    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("CSV must contain a text column.")

    if "sentiment_label" in df.columns:
        actual_column = "sentiment_label"
    elif "sentiment" in df.columns:
        actual_column = "sentiment"
    else:
        raise ValueError(
            "CSV must contain sentiment_label or sentiment."
        )

    # 3. Sentiment analysis
    analyzed_df = analyze_dataframe(
        df,
        text_column="text"
    )

    true_positive = count_true_positives(
        analyzed_df,
        actual_column=actual_column,
        predicted_column="predicted_sentiment"
    )

    # 4. Named Entity Recognition
    nlp = load_nlp()

    most_freq_ent = most_frequent_entity(
        analyzed_df["text"].fillna("").tolist(),
        nlp=nlp
    )

    # 5. Similarity search
    most_similar_text = find_most_similar_text(
        "wrong package delivery",
        analyzed_df["text"].fillna("").tolist()
    )

    # 6. Save results
    output_dir = BASE_DIR / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "analyzed_customer_calls.csv"

    analyzed_df.to_csv(output_path, index=False)

    # 7. Display results
    print("\nCUSTOMER SUPPORT ANALYSIS")
    print("=" * 45)

    print("Transcribed text:", transcribed_text)
    print("Frame rate:", frame_rate)
    print("Number of channels:", number_channels)
    print("True positive:", true_positive)
    print("Most frequent entity:", most_freq_ent)
    print("Most similar text:", most_similar_text)

    print("\nAnalysis saved to:", output_path)


if __name__ == "__main__":
    run_project()
