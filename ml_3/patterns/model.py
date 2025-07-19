from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from hydra.utils import call, get_class
from tensorflow import keras

from ml_3.model.layers import Dense


class Builder(ABC):
    @abstractmethod
    def build(self, layers):
        pass


class TorchBuilder(Builder):
    def __init__(self, config) -> None:
        self.config = config
        

    def build(self) -> nn.Module:
        layers = self.config.model.layers
        model = torch.nn.Sequential()

        for name, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = call(layer.args) if layer.args else {}
            module = layer_cls(**self.torch_args(args=args) if args else args)
            model.add_module(name=name, module=module)

        return model

    def torch_args(self, args: Dense) -> dict:
        return {
            "in_features": args.in_units,
            "out_features": args.out_units,
            "bias": args.use_bias,
            "activation": args.activation,
            "kernel_initializer": args.kernel_initializer,
            "bias_initializer": args.bias_initializer,
            "kernel_regularizer": args.kernel_regularizer,
            "bias_regularizer": args.bias_regularizer,
            "activity_regularizer": args.activity_regularizer,
            "kernel_constraint": args.kernel_constraint,
            "bias_constraint": args.bias_constraint,
        }


class KerasBuilder(Builder):
    def __init__(self, config) -> None:
        self.config = config

    def build(self):
        layers = self.config.model.layers
        model = keras.Sequential()

        for _, layer in layers.items():
            layer_cls = get_class(layer.cls)
            args = call(layer.args) if layer.args else {}
            module = layer_cls(**self.keras_args(args=args) if args else args)
            model.add(module)

        return model

    def keras_args(self, args: Dense) -> dict:
        return {
            "units": args.out_units,
            "use_bias": args.use_bias,
            "activation": args.activation,
            "kernel_initializer": args.kernel_initializer,
            "bias_initializer": args.bias_initializer,
            "kernel_regularizer": args.kernel_regularizer,
            "bias_regularizer": args.bias_regularizer,
            "activity_regularizer": args.activity_regularizer,
            "kernel_constraint": args.kernel_constraint,
            "bias_constraint": args.bias_constraint,
        }
    
    def load_data(self):
        dataset_factory = call(self.config.backend.data_factory)
        tf_schema = dataset_factory.load_and_preprocess_data(self.config)

        return tf_schema