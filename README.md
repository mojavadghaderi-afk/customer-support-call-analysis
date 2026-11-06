# Customer Support Call Analysis

**AI-Powered Speech Recognition and NLP Project**

## Overview
This project analyzes customer support calls using Speech Recognition and Natural Language Processing (NLP).

## Objectives
- Convert audio calls into text using SpeechRecognition.
- Extract audio metadata using Pydub.
- Perform sentiment analysis using VADER.
- Identify named entities using spaCy.
- Find similar customer support calls using semantic similarity.

## Technologies
- Python
- SpeechRecognition
- Pydub
- Pandas
- NLTK (VADER)
- spaCy
- NumPy

## Project Results

| Metric | Result |
|---|---|
| Audio transcription | Successfully completed |
| Frame rate | 44,100 Hz |
| Audio channels | 1 (Mono) |
| True positive predictions | 2 |
| Most frequent entity | yesterday |
| Most similar call | wrong package delivered |

## Project Structure

```text
customer-support-call-analysis/
│
├── notebook.ipynb
├── README.md
└── .gitignore
```

## Installation

```bash
pip install SpeechRecognition pydub pandas nltk spacy numpy
python -m spacy download en_core_web_sm
```

## Execution
Open `notebook.ipynb` in Jupyter Notebook or Google Colab and execute the cells.

**Note:** The audio and CSV datasets must be available in the execution environment.

## Author
Mohammad Javad Ghaderi
