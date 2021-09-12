import random
from typing import List, Union, Callable
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from scipy.cluster.hierarchy import fclusterdata
import matplotlib.pyplot as plt

from koala.config import MODELS_FOLDER
from koala.config import logging


class WordEmbbedder:
    _NAME_MODEL_DISK = "trained_word2vec.bin"

    @property
    def name_model_in_disk(self):
        name = "trained_word2vec_{}.bin".format(self.id_model)
        if self.job_name != "":
            name = self.job_name + "/" + name
        return name

    WE_PARAMS = {
        "number_samples": 1_000_000,
        "size_vectors": 48,  # better if multiple of 4
        "window": 3,
        "min_count": 100,
        "max_vocab_size": 15_000,
        "epochs": 18}

    def __init__(self, train_texts: List[str], tokenizer: Callable = lambda x: x.split(),
                 id_model: str = "0", max_distance_clustering: float = 0.25,
                 job_name: str = ""):
        """
        Builds the WordEmbedder.
        :param train_texts: list of text will be used to train the model.
        :param tokenizer: function that expects a string and returns a list of tokens.
        :param id_model: Name for the model, used to be kept in disk
        :param max_distance_clustering: maximum distance between words to consider them synonymous.
        :param job_name: name for the folder where intermediate data will be saved.
        """
        self.job_name = job_name
        self.tokenizer = tokenizer
        self.train_texts = [tokenizer(x) for x in set(train_texts)]
        self.train_texts = [x for x in self.train_texts if len(x) > 4]
        self.max_distance_clustering = max_distance_clustering
        self.id_model = id_model
        self._we_model = None
        self._we_model_params = {}
        self.use_disk = False
        self._dict_synons = {}

        Path(str(MODELS_FOLDER / self.name_model_in_disk)).mkdir(parents=True, exist_ok=True)

    def load_from_disk(self):
        return KeyedVectors.load_word2vec_format(str(MODELS_FOLDER / self.name_model_in_disk), binary=True)

    def set_disk_mode(self):
        """Used to save memory.

        Removes all cached data, and when the model is used it will be loaded from disk."""
        self.use_disk = True
        self.train_texts = None
        del self._we_model
        self._we_model = None

    @property
    def model(self) -> KeyedVectors:
        """Returns the word embedding model."""
        if self.use_disk:  # Loads from disk
            return self.load_from_disk()
        if self._we_model is not None and self._we_model_params == self.WE_PARAMS:  # Returns cached model
            return self._we_model.wv
        else:  # Trains the model
            self._dict_synons = {}
            logging.info("Fitting Word2Vec with params {}".format(self.WE_PARAMS))
            model = Word2Vec(
                sentences=random.choices(self.train_texts,
                                         k=self.WE_PARAMS["number_samples"]),
                vector_size=self.WE_PARAMS["size_vectors"],
                window=self.WE_PARAMS["window"],
                min_count=self.WE_PARAMS["min_count"],
                workers=4,
                max_vocab_size=self.WE_PARAMS["max_vocab_size"],
                epochs=self.WE_PARAMS["epochs"])
            self._we_model = model
            self._we_model_params = self.WE_PARAMS
            model.wv.save_word2vec_format(str(MODELS_FOLDER / self.name_model_in_disk), binary=True)
            return self._we_model.wv

    def _build_word_embedding_similarity_synons(self):
        """Uses clustering on word embeddings to find words with the same meaning."""
        logging.info("Building dictionary of synons.")
        dict_synonymous = {}
        all_words = self.get_words_in_model()

        def metric_func(x, y):
            x = all_words[int(x[0])]
            y = all_words[int(y[0])]
            return 1 - self.similarity(x, y)

        logging.debug("Performing clustering on {} words".format(len(all_words)))
        res = fclusterdata([[all_words.index(w)] for w in all_words],
                           metric=metric_func,
                           t=self.max_distance_clustering,
                           criterion="distance",
                           method="centroid")
        clusters = [[j for j in range(len(res)) if res[j] == i] for i in range(min(res), max(res) + 1)]

        for cl in clusters:
            words_in_cluster = [all_words[j] for j in cl]

            counters_cl = [self.model.get_vecattr(w, "count") for w in words_in_cluster]
            # penalize tokens in cluster with more than one word
            counters_cl = [counters_cl[i] if "_" not in words_in_cluster[i] else counters_cl[i] - 10_000_000
                           for i in range(len(words_in_cluster))]
            name_cluster = words_in_cluster[np.argmax(counters_cl)]
            for w in words_in_cluster:
                dict_synonymous[w] = name_cluster
        return dict_synonymous

    @property
    def dict_synons(self):
        """Returns a dict of word synonymous."""
        if not len(self._dict_synons):
            self._dict_synons = self._build_word_embedding_similarity_synons()
        return self._dict_synons

    def _vectorize_text(self, text: str) -> Union[np.ndarray, None]:
        """Returns the vector of a text as the mean of all its vectors"""

        def encode_word(w):
            try:
                vector = self.model[w]
            except KeyError:
                vector = None
            return vector

        result = 0.
        c = 0
        for v in filter(lambda x: x is not None, map(encode_word, text.split())):
            result += v
            c += 1

        return np.array(result / c, dtype="double") if c != 0 else None

    def get_words_in_model(self):
        """Returns the list of words present in model."""
        return list(self.model.key_to_index.keys())

    def get_close_words(self, word: str, n: int):
        """For a given word, returns n closest words in model."""
        return self.model.similar_by_word(word, n)

    def similarity(self, x, y):
        """Computes the similarity between two words (cosine distance of their vectors)."""
        return self.model.similarity(x, y)

    def encode(self, word):
        """Returns the vector of a word."""
        if isinstance(word, str):
            word = [word]
        vectors = [self.model[w] for w in word]
        return vectors[0] if len(vectors) == 1 else vectors

    def plot_similar_words(self, word: str, n: int = 50, embedding_technique="pca"):
        """Uses a trained word embedding to plot similar words to a given one.
        :type word: str Word to encode
        :type n: int Number of words to plot
        :type embedding_technique: str technique to run the 2d embedding. Can be 'pca' or 'tsne'.
        """
        close_words = [word] + [x[0] for x in self.get_close_words(word, n)]

        word_labels, arr = zip(*[(x, self.encode(x)) for x in close_words])

        colors = []
        syn_of_word = self.dict_synons.get(word, word)
        for w in word_labels:
            if w == word:
                colors.append("blue")
            elif syn_of_word == self.dict_synons.get(w, w):
                colors.append("green")
            else:
                colors.append("red")
        Y = self._encode_for_2d(arr, embedding_technique)
        self._scatter_with_labels(Y, word_labels, colors,
                                  title="Closest words to word {} (dim reduction technique: {})".
                                  format(word, embedding_technique))

    def _encode_for_2d(self, values, embedding_technique="pca"):
        """Reduces dimension of vectors to only two dimension.

        Applies a dimension reduction algorithm to get two dimensional vectors. Supports PCA and TSNE."""

        def get_encoder_class():
            dict_encoders = {"pca": PCA, "tsne": TSNE}
            try:
                return dict_encoders[embedding_technique]
            except KeyError:
                raise ValueError("Indicated embedding technique '{}' is not valid. Options are: ".
                                 format(embedding_technique, dict_encoders.keys()))

        encoder = get_encoder_class()(n_components=2, random_state=0)
        Y = encoder.fit_transform(values)
        return Y

    def _scatter_with_labels(self, data, labels, color, title):
        x_coords, y_coords = data[:, 0], data[:, 1]
        # display scatter plot
        plt.figure(figsize=(20, 10))
        plt.scatter(x_coords, y_coords, color=color)

        for label, x, y in zip(labels, x_coords, y_coords):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min() - 0.00005, x_coords.max() - 0.00005)
        plt.ylim(y_coords.min() - 0.00005, y_coords.max() - 0.00005)
        plt.title(title, color="blue")
        plt.axis('off')
        plt.show()
