import spacy
import os
# noinspection PyUnresolvedReferences
from spacy.lang.es import Spanish
from spacy_spanish_lemmatizer.lemmatizer import Lemmatizer  # noqa # pylint: disable=unused-import


class SpanishSpacyLematizer:
    def __init__(self):
        os.system("python -m spacy_spanish_lemmatizer download wiki")
        self.nlp = spacy.load('es_core_news_sm', disable=["ner", "attribute_ruler"])
        self.nlp.replace_pipe("lemmatizer", "spanish_lemmatizer")

    def __call__(self, text):
        results = [" ".join([t.lemma_ for t in d]) for d in self.nlp.pipe([text] if isinstance(text, str) else text,
                                                                          n_process=1, batch_size=2000)]

        return results[0] if isinstance(text, str) else results


spanish_spacy_lemitizer = SpanishSpacyLematizer()


def lemmatize_text(text):
    """Converts a text to another text with lemmitized words."""
    if isinstance(text, str):
        text = [text]
    # return spanish_stanza_lemitizer(text)
    lemmas = spanish_spacy_lemitizer(text)
    return lemmas if len(text) > 1 else lemmas[0]




