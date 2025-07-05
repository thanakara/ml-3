import os
import tensorflow as tf
from functools import partial


def model_factory() -> tf.keras.Model:
    Dense = partial(tf.keras.layers.Dense, activation="relu")
    pass

class ModelFactory(object):
    def __init__(self, seed=42) -> None:
        self.seed = seed
    
    def _build(self, num_hidden: int, name: str) -> tf.keras.Sequential:
        Dense = partial(tf.keras.layers.Dense, activation="relu")
        tf.keras.utils.set_random_seed(self.seed)
        model = tf.keras.Sequential([
            tf.keras.Input([28, 28]),
            tf.keras.layers.Flatten()
        ], name=name)
        units = 2 ** (num_hidden + 4)
        for _ in range(num_hidden):
            model.add(Dense(units=units))
            units //= 2
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        return model
