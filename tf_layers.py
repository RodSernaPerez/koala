from typing import List

from tensorflow import keras
import tensorflow as tf

K = keras.backend


class TripletLossLayer(keras.layers.Layer):
    def __init__(self, alpha: float, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs: List[tf.Tensor]):
        a, p, n = inputs
        p_dist = K.sum(K.square(a - p), axis=-1)
        n_dist = K.sum(K.square(a - n), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs: List[tf.Tensor]):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
