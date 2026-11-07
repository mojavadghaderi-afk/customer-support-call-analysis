from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_most_similar_text(query, texts):

    texts = [str(text) for text in texts if str(text).strip()]

    if not texts:
        return ""

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform([query] + texts)

    scores = cosine_similarity(
        matrix[0:1],
        matrix[1:]
    ).flatten()

    best_index = scores.argmax()

    return texts[best_index]


def similarity_score(query, text):

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform([query, text])

    return float(
        cosine_similarity(matrix[0:1], matrix[1:])[0][0]
    )
