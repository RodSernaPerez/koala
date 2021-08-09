import itertools
from typing import Dict, List

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.special import softmax
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import tensorflow as tf
from tensorflow import keras

from koala.tf_layers import TripletLossLayer

K = keras.backend


class Classifier:
    def fit(self, dict_vectors_per_value: Dict) -> None:
        return None

    def classify_vector(self, vector: np.array) -> Dict:
        return {}

    def apply_softmax_on_dict(self, dict_score_per_label: Dict[int, float]) -> Dict[int, float]:
        labels, distances = zip(*dict_score_per_label.items())
        distances = softmax(distances)
        results = {k: v for k, v in zip(labels, distances)}
        return results

    def apply_norm_1_normalizing_on_dict(self, dict_score_per_label: Dict[int, float]) -> Dict[int, float]:
        labels, distances = zip(*dict_score_per_label.items())
        distances = np.asarray([max(0, x) for x in distances])
        distances = distances / sum(distances)
        results = {k: v for k, v in zip(labels, distances)}
        return results


class DistanceToAverageClassifier(Classifier):

    def __init__(self, type_distance_function: str, apply_normalization_model=True):
        self.av_vectors = {}
        self.type_distance_function = type_distance_function
        self.normalize_model = None
        self.labels = []
        self.apply_normalization_model = apply_normalization_model

    @property
    def similarity_function(self):
        if self.type_distance_function == "cosine":
            return cosine_similarity
        if self.type_distance_function == "euclidean":
            return lambda x, y: 1 - euclidean_distances(x, y) / max(np.linalg.norm(x), np.linalg.norm(y))

    def compute_similarities_to_means(self, vector: np.array):
        sim_to_each_class = {}
        for label in self.labels:
            sim_to_each_class[label] = \
                self.similarity_function(vector.reshape(1, -1), self.av_vectors[label].reshape(1, -1))[0][0]
        return sim_to_each_class

    def similarities_as_features(self, dict_similarities: Dict) -> np.array:
        r = np.asarray([dict_similarities[label] for label in self.labels])
        return r

    def similarities_to_probs(self, sim_to_each_class):
        features_for_normalize = self.similarities_as_features(sim_to_each_class)
        results = self.normalize_model.predict([features_for_normalize])
        return {label: r for label, r in zip(self.labels, results[0])}

    def classify_vector(self, vector: np.array):
        sim_to_each_class = self.compute_similarities_to_means(vector)
        if self.apply_normalization_model:
            results = self.similarities_to_probs(sim_to_each_class)
            results = self.apply_norm_1_normalizing_on_dict(results)
        else:
            results = self.apply_softmax_on_dict(sim_to_each_class)
        return results

    def fit(self, dict_vectors_per_value: Dict):
        def to_one_hot(i):
            x = [0] * len(self.labels)
            x[self.labels.index(i)] = 1
            return x

        self.labels = sorted(dict_vectors_per_value.keys())
        self.av_vectors = {k: np.mean(v, axis=0) for k, v in dict_vectors_per_value.items()}
        distances_to_means = []
        vector_of_labels = []
        for label in self.labels:
            vectors = dict_vectors_per_value[label]
            for v in vectors:
                similarities = self.compute_similarities_to_means(v)
                distances_to_means.append(self.similarities_as_features(similarities))
            vector_of_labels += [to_one_hot(label)] * len(vectors)
        self.normalize_model = LinearRegression().fit(distances_to_means, vector_of_labels)


class NeighborsClassifier(Classifier):
    def __init__(self):
        self.model = None
        self.labels = []

    def fit(self, dict_vectors_per_value: Dict):
        labels = sorted(list(dict_vectors_per_value.keys()))
        self.labels = labels
        number_labels = len(labels)
        self.model = KNeighborsClassifier(1)
        extended_dataset = []
        for k in labels:
            values = dict_vectors_per_value[k]
            for v in values:
                extended_dataset.append((k, v))

        y, X = zip(*extended_dataset)
        self.model.fit(X, y)

    def classify_vector(self, vector: np.array):
        predictions = self.model.predict_proba([vector])[0]
        return {label: prob for label, prob in zip(self.model.classes_, predictions)}


class AverageOfDistancesClassifier(Classifier):
    def __init__(self, type_distance_function: str):
        self.train_vectors = {}
        self.type_distance_function = type_distance_function

    @property
    def similarity_function(self):
        if self.type_distance_function == "cosine":
            return cosine_similarity
        if self.type_distance_function == "euclidean":
            return lambda x, y: 1 - euclidean_distances(x, y) / max(np.linalg.norm(x), np.linalg.norm(y))

    def fit(self, dict_vectors_per_value: Dict):
        self.train_vectors = dict_vectors_per_value

    def classify_vector(self, vector: np.array):
        score_per_label = {}
        for label, vectors in self.train_vectors.items():
            score_per_label[label] = np.mean(
                [self.similarity_function(vector.reshape(1, -1), v.reshape(1, -1)) for v in vectors])
        return self.apply_softmax_on_dict(score_per_label)


class SimilarityClassifier(Classifier):
    def __init__(self, type_model="logistic", dim_reduction_model=None):
        self.model = None
        self.labels = []
        self.type_model = type_model
        self.support_vectors = None
        self.dim_reduction_model = dim_reduction_model

    @property
    def dim_vectors(self) -> int:
        if self.dim_reduction_model is None:
            return 768
        else:
            return self.dim_reduction_model.n_components

    def create_model(self):
        if self.type_model == "logistc":
            return self._create_similarity_model_logistic()
        elif self.type_model == "dense":
            return self._create_similarity_model_dense()
        else:
            raise ValueError("Not valid model")

    def _merge_vectors(self, vector_a: np.array, vector_b: np.asarray) -> np.array:
        diff = vector_a - vector_b
        return np.multiply(diff, diff)

    def reduce_dim(self, vector):
        if not self.dim_reduction_model:
            return vector
        else:
            return self.dim_reduction_model.transform([vector])[0].reshape(-1)

    def create_extended_dataset(self, dict_vectors_per_value: Dict):
        samples = []
        for label, vectors in dict_vectors_per_value.items():
            samples += [(self.reduce_dim(v), label) for v in vectors]
        return samples

    def create_dataset_similarity(self, list_samples: list):
        dataset = []
        for i in range(len(list_samples)):
            for j in range(i, len(list_samples)):
                vectors_merged = self._merge_vectors(list_samples[i][0], list_samples[j][0])
                target = 1 if list_samples[i][1] == list_samples[j][1] else 0
                dataset.append((vectors_merged, target))
        return dataset

    def _create_similarity_model_logistic(self):
        return LinearRegression()

    def _create_similarity_model_dense(self):
        input_ = keras.layers.Input(shape=(self.dim_vectors,))
        x = keras.layers.Dense(25)(input_)
        x = keras.layers.Dense(1)(x)
        model = keras.models.Model(input_, x)
        model.compile("adam", "mse", metrics=["accuracy"])
        return model

    def fit(self, dict_vectors_per_value: Dict):
        self.labels = list(dict_vectors_per_value.keys())
        self.model = self.create_model()
        self.support_vectors = self.create_extended_dataset(dict_vectors_per_value)
        train_data = zip(*self.create_dataset_similarity(self.support_vectors))
        x, y = train_data
        self.model.fit(np.asarray(x), np.asarray(y), epochs=1, verbose=0)

    def classify_vector(self, vector: np.array):
        vector = self.reduce_dim(vector)
        features = np.asarray([self._merge_vectors(vector, v[0]) for v in self.support_vectors])
        similarities = self.model.predict(features)
        probs_per_label = self._get_score_per_label(similarities)
        return self.apply_softmax_on_dict(probs_per_label)

    def _get_score_per_label(self, similarities):
        similarities_per_label = {l: [] for l in self.labels}
        for v, s in zip(self.support_vectors, similarities):
            similarities_per_label[v[1]] = similarities_per_label[v[1]] + [s]
        return {k: np.max(v) for k, v in similarities_per_label.items()}


class SiameseNetworkClassifier(Classifier):
    def __init__(self, dim_reduction_model=None):
        self.dim_reduction_model = dim_reduction_model
        self.siamese_model = None

    @property
    def dim_vectors(self) -> int:
        if self.dim_reduction_model is None:
            return 768
        else:
            return self.dim_reduction_model.n_components

    def build_siamese_model(self):
        input_text1 = keras.layers.Input(shape=(self.dim_vectors,))
        x = keras.layers.Dense(self.dim_vectors // 2, activation='relu', name="dense_1")(input_text1)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(self.dim_vectors // 4, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(0.001), name="dense_2")(x)
        x = keras.layers.Dropout(0.4)(x)
        dense_layer = keras.layers.Dense(self.dim_vectors // 2, name='dense_3')(x)
        norm_layer = keras.layers.Lambda(lambda x: K.l2_normalize(x, axis=1), name='norm_layer')(dense_layer)

        model = keras.models.Model(inputs=[input_text1], outputs=norm_layer, name="siamese")
        return model

    def build_model(self):
        anchor = keras.layers.Input(shape=(self.dim_vectors,), name="anchor")
        positive = keras.layers.Input(shape=(self.dim_vectors,), name="positive")
        negative = keras.layers.Input(shape=(self.dim_vectors,), name="negative")

        self.siamese_model = self.build_siamese_model()
        encoded_inputs = [self.siamese_model(x) for x in [anchor, positive, negative]]
        output = TripletLossLayer(2)(encoded_inputs)
        self.model_triplet = keras.models.Model([anchor, positive, negative], output)
        self.model_triplet.compile(loss=None, optimizer="adam")
        return encoded_inputs

    def generate_triplet_batchs(self, dict_vectors_per_value, batch_size: int = 32):
        def generate_triplet():
            anchor_class = np.random.choice(self.labels)
            negative_class = np.random.choice([l for l in self.labels if l != anchor_class])
            a, p = np.random.choice(range(len(dict_vectors_per_value[anchor_class])), 2, replace=False)

            n = np.random.choice(range(len(dict_vectors_per_value[negative_class])))
            a = dict_vectors_per_value[anchor_class][a]
            p = dict_vectors_per_value[anchor_class][p]
            n = dict_vectors_per_value[negative_class][n]
            return a, p, n

        while True:
            batch_data = [generate_triplet() for _ in range(batch_size)]
            a_batch, p_batch, n_batch = zip(*batch_data)
            yield {"anchor": np.asarray(a_batch),
                   "positive": np.asarray(p_batch),
                   "negative": np.asarray(n_batch)}, []

    def reduce_dim_of_vectors_in_dict(self, dict_vectors_per_value: Dict) -> Dict:
        if self.dim_reduction_model is None:
            return dict_vectors_per_value
        dict_reduced = {}
        for k, vectors in dict_vectors_per_value.items():
            dict_reduced[k] = self.dim_reduction_model.transform(vectors)
        return dict_reduced

    def fit(self, dict_vectors_per_value: Dict) -> None:
        self.dict_vectors_per_value = self.reduce_dim_of_vectors_in_dict(dict_vectors_per_value)
        self.labels = list(self.dict_vectors_per_value.keys())
        self.build_model()
        self.model_triplet.fit(self.generate_triplet_batchs(self.dict_vectors_per_value, 32),
                               epochs=20, steps_per_epoch=20, verbose=0)
        self.dict_vectors_per_value = {k: self.siamese_model.predict(np.asarray(v))
                                       for k, v in self.dict_vectors_per_value.items()}
        self.classification_model = DistanceToAverageClassifier("cosine")
        self.classification_model.fit(self.dict_vectors_per_value)

    def reduce_dim(self, vector):
        if not self.dim_reduction_model:
            return vector
        else:
            return self.dim_reduction_model.transform([vector])[0].reshape(-1)

    def classify_vector(self, vector: np.array) -> Dict:
        vector_reduced = self.reduce_dim(vector)
        vector_transformed = self.siamese_model.predict(np.asarray([vector_reduced]))[0]
        return self.classification_model.classify_vector(vector_transformed)


class MultiClassifier(Classifier):
    def __init__(self):
        self.classifiers = [DistanceToAverageClassifier("cosine"), NeighborsClassifier()]

    def fit(self, vectors_per_label):
        for p in self.classifiers:
            p.fit(vectors_per_label)

    def classify_vector(self, vector: np.array) -> Dict:
        results_per_classifier = [p.classify_vector(vector) for p in self.classifiers]
        return self._join_probs(results_per_classifier)

    def _join_probs(self, list_of_dicts: List[Dict[int, float]]) -> Dict[int, float]:
        dict_label_to_props = {}
        for x in list_of_dicts:
            for label, prop in x.items():
                dict_label_to_props[label] = dict_label_to_props.get(label, []) + [prop]
        dict_label_to_average_prop = {k: np.prod(v) for k, v in dict_label_to_props.items()}
        probs_per_label = self.apply_softmax_on_dict(dict_label_to_average_prop)
        return probs_per_label
