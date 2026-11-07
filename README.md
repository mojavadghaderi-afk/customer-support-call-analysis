
# Customer Support Call Analysis

A Python project for analyzing customer support calls using Speech Recognition and Natural Language Processing (NLP).

## Features

- Audio transcription using SpeechRecognition.
- Audio metadata extraction using Pydub.
- Sentiment analysis using NLTK VADER.
- Named Entity Recognition using spaCy.
- Similar customer call search using TF-IDF and cosine similarity.
- Export analysis results to CSV.

## Technologies

- Python
- SpeechRecognition
- Pydub
- Pandas
- NLTK
- spaCy
- Scikit-learn

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy English model:

```bash
python -m spacy download en_core_web_sm
```

Install FFmpeg on Linux:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## Input Files

Place the following files in the configured data directory:

- sample_customer_call.wav
- customer_call_transcriptions.csv

The CSV must contain call text and sentiment labels.

## Run the Project

```bash
python main.py
```

## Output

The analysis report is saved as:

```text
reports/analyzed_customer_calls.csv
```

## Evaluation

The project uses VADER as a baseline sentiment analyzer.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Threshold experiments on the current dataset are exploratory and should not be interpreted as independent test performance.

## Limitations

VADER is a lexicon-based sentiment analyzer and may not fully understand contextual emotions in customer conversations.

The speech recognition component requires internet access.

## Project Status

Core audio processing and NLP pipeline implemented and tested.
