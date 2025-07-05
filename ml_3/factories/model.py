import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from functools import partial



def get_fc_part(layers) -> tf.keras.Model:
    input_dim, *hidden, output_layer = layers
    Dense = partial(tf.keras.layers.Dense)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_dim),
        tf.keras.layers.Flatten()
    ])
    for layer in hidden:
        units, activation = layer
        model.add(Dense(units=units, activation=activation))
    output_dim, activation = output_layer
    model.add(Dense(units=output_dim, activation=activation))
    return model

def get_cnn_part(layers):
    (x, y), *hidden = layers
    Conv = partial(tf.keras.layers.Conv2D)
    model = tf.keras.Sequential([
        tf.keras.Input([x, y]),
        tf.keras.layers.Reshape([x, y, 1])
    ])
    for layer in hidden:
        filters, kernel, strides, padding, activation = layer
        model.add(Conv(filters=filters, kernel_size=kernel, strides=strides,
                       padding=padding, activation=activation))
        model.add(tf.keras.layers.MaxPool2D())
    return model


def model_factory() -> tf.keras.Model:
    pass