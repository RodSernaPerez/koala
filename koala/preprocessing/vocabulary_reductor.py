from typing import Iterable, List

from gensim.models import Phrases

from koala.metrics import hamming_similarity
# from callcenters.nlp.text_processing import is_spanish_word
from koala.vectorizer.wordembedder import WordEmbbedder


class ReductorFunction:
    """Function to replace similar words by a representative."""
    MAP_CLOSE_WORDS = False  # Replaces also words that are written very similarly.

    def __init__(self, dict_synons: dict, extra_data={}):
        self.dict_synons = dict_synons
        if self.MAP_CLOSE_WORDS:
            self.dict_synons = self._add_similar_words_mapping(self.dict_synons)
        self.dict_synons = {k: v for k, v in self.dict_synons.items() if k != v}
        self.extra_data = extra_data
        print("Vocab reduction function crated with {} terms in dict".format(len(self.dict_synons)))

    def _add_similar_words_mapping(self, dict_synonymous):
        """Using hamming distance for not existing words
        Tries to fix lemmatizing problems."""

        words = dict_synonymous.keys()
        for w in list(words):
            if "_" in w or "-" in w:
                continue
            if is_spanish_word(w):
                continue
            if dict_synonymous.get(w, w) != w:
                continue
            for other_word in words:
                if other_word == w:
                    continue
                try:
                    similarity = hamming_similarity(w, other_word)
                    if similarity >= 0.85:
                        dict_synonymous[w] = dict_synonymous.get(other_word, other_word)
                        break
                except KeyError:  # Word is not in we model
                    break
        return dict_synonymous

    def __call__(self, text: str):
        """Applies reductor on a text."""
        tokens = text.split()
        # First, applies bigrams replacements.
        for i in range(len(tokens) - 1):
            try:
                bigram = "_".join(tokens[i: i + 2])

                translated_bigram = self.dict_synons[bigram]
                if bigram == translated_bigram:
                    raise KeyError
                tokens_in_bigram = translated_bigram.split("_")
                if len(tokens_in_bigram) == 1:
                    tokens_in_bigram = ["", tokens_in_bigram[0]]
                tokens[i] = self.dict_synons.get(tokens_in_bigram[0], tokens_in_bigram[0])
                tokens[i + 1] = self.dict_synons.get(tokens_in_bigram[1], tokens_in_bigram[1])
            except KeyError:
                pass

        text = " ".join(tokens).replace("_", " ")
        tokens = text.split()

        # Then applies token replacement
        tokens = [self.dict_synons.get(t, t) for t in tokens]

        # The algorithm causes some words to be duplicated. They are removed.
        tokens = [tokens[i] for i in range(len(tokens)) if (i == 0) | (tokens[i] != tokens[i - 1])]
        text = " ".join(" ".join(tokens).split()).replace("_", " ")
        return text


class VocabReductionPipeline:
    """Vocabulary reduction pipeline.
    Applies several vocabulary reduction functions."""

    def __init__(self, reductors: List[ReductorFunction]):
        self.reductors = reductors

    def __call__(self, text: str, n_reductors=None):
        if n_reductors:
            reductors = self.reductors[:n_reductors]
        else:
            reductors = self.reductors
        for r in reductors:
            text = r(text)
        return text


class VocabularyReductor:
    """Object capable fo reduce the vocabulary of a function.

    This algorithm works in three steps:
    - Trains a word embedding model on the input texts.
    - Applies hierarchical clustering to word vectors to group them.
    - Replaces words in the same cluster by a representative.
    """

    def __init__(self, job_name: str = ""):
        """
        Inits the object
        :str job_name: name of the job, gives the name of the folder when middle results are kept.
        """
        self.job_name = job_name

    def get_vocab_reductor(self, texts: Iterable[str],
                           number_iterations_words: int = 1,
                           number_iterations_bigrams: int = 1,
                           id_start=0) -> VocabReductionPipeline:
        """Returns function that reduces the vocab.
        :param texts: list of texts used to train.
        :param number_iterations_words: number of times the algorithm is run on individual words.
        :param number_iterations_bigrams: number of times the algorithm is run on bigrams.
        :param id_start: middle models are kept in disks with an incrementar count. This parameter gives the first one.
        """
        pipeline = []
        for i in range(number_iterations_words + number_iterations_bigrams):
            if i != 0:
                texts = [pipeline[-1](t) for t in texts]

            if i > number_iterations_words:
                tokenizer = self._build_tokenizer(texts)
                WordEmbbedder.WE_PARAMS["window"] = 5
            else:
                WordEmbbedder.WE_PARAMS["window"] = 2

                def tokenizer(x):
                    return x.split()

            word_embedder = self._get_word_embedder(texts, tokenizer, id_model=i + id_start)
            dict_synons = word_embedder.dict_synons.copy()
            word_embedder.set_disk_mode()  # This removes the model from memory and stores it in disk to save memory.

            data_to_keep = {"tokenizer": tokenizer,
                            "word_embe": word_embedder}

            pipeline.append(ReductorFunction(dict_synons, data_to_keep))

        return VocabReductionPipeline(pipeline)

    def _build_tokenizer(self, texts):
        """Returns a tokenizer that uses bigrams."""
        print("Building tokenizer")
        phrases = Phrases([x.split() for x in texts], min_count=1, threshold=0.05, scoring="npmi",
                          max_vocab_size=1_000).freeze()

        def tokenizer(text):
            return phrases[text.split()]

        return tokenizer

    def _get_word_embedder(self, texts, tokenizer, id_model=0):
        """Computes a word embedder."""
        print("Building word embedding")
        word_embedder = WordEmbbedder(texts, tokenizer,
                                      id_model="wordreductor_{}".format(id_model),
                                      max_distance_clustering=0.2,
                                      job_name=self.job_name)
        return word_embedder
