import spacy
from collections import Counter


def load_nlp(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            "spaCy model is missing. Run: "
            "python -m spacy download en_core_web_sm"
        ) from exc


def extract_entities(text, nlp=None):
    if nlp is None:
        nlp = load_nlp()

    doc = nlp(str(text))

    return [
        {
            "text": ent.text,
            "label": ent.label_,
            "description": spacy.explain(ent.label_)
        }
        for ent in doc.ents
    ]


def most_frequent_entity(texts, nlp=None):
    if nlp is None:
        nlp = load_nlp()

    counter = Counter()
    original_names = {}

    for doc in nlp.pipe(
        (str(text) for text in texts),
        batch_size=64
    ):
        for ent in doc.ents:
            name = ent.text.strip()

            if name:
                key = name.casefold()
                counter[key] += 1
                original_names.setdefault(key, name)

    if not counter:
        return ""

    most_common = counter.most_common(1)[0][0]

    return original_names[most_common]
