
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


def run_project(
    audio_file="sample_customer_call.wav",
    csv_file="customer_call_transcriptions.csv"
):

    audio_path = BASE_DIR / audio_file
    csv_path = BASE_DIR / csv_file

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # 1. Audio processing
    transcribed_text = transcribe_audio(str(audio_path))

    frame_rate, number_channels = get_audio_metadata(
        str(audio_path)
    )

    # 2. Load customer calls
    df = pd.read_csv(csv_path)

    if not {"text", "sentiment"}.issubset(df.columns):
        raise ValueError(
            "CSV must contain text and sentiment columns."
        )

    # 3. Sentiment analysis
    analyzed_df = analyze_dataframe(df)

    true_positive = count_true_positives(analyzed_df)

    # 4. Named Entity Recognition
    nlp = load_nlp()

    most_freq_ent = most_frequent_entity(
        analyzed_df["text"].fillna("").tolist(),
        nlp=nlp
    )

    # 5. Text similarity
    most_similar_text = find_most_similar_text(
        "wrong package delivery",
        analyzed_df["text"].fillna("").tolist()
    )

    # Save analysis results
    output_path = BASE_DIR / "analyzed_customer_calls.csv"
    analyzed_df.to_csv(output_path, index=False)

    # 6. Display results
    print("\nCUSTOMER SUPPORT ANALYSIS")
    print("-" * 40)

    print("Transcribed text:", transcribed_text)
    print("Frame rate:", frame_rate)
    print("Number of channels:", number_channels)
    print("True positive:", true_positive)
    print("Most frequent entity:", most_freq_ent)
    print("Most similar text:", most_similar_text)

    print("\nAnalysis saved to:", output_path)


if __name__ == "__main__":
    run_project()
