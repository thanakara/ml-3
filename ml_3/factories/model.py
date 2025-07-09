import tensorflow as tf
from hydra.utils import instantiate
from ml_3.model.layers import Dense, Conv2D
from dataclasses import asdict


def keras_builder(layers: list) -> tf.keras.Model:
    model = tf.keras.Sequential()

    for layer, args in layers.items():
        if "flatten" in layer:
            model.add(tf.keras.layers.Flatten(input_shape=args.input_shape))
        else:
            layer_args = instantiate(args)
            if isinstance(layer_args, Dense):
                layer_args = asdict(layer_args)
                layer_args.pop("in_units")
                out_units = layer_args.pop("out_units")
                model.add(tf.keras.layers.Dense(
                    units=out_units,
                    **layer_args
                ))
            elif isinstance(layer_args, Conv2D):
                layer_args = asdict(layer_args)
                model.add(tf.keras.layers.Conv2D(**layer_args))

    return model