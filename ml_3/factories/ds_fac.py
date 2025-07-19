from abc import ABC, abstractmethod
from dataclasses import dataclass
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class TFSchema:
    train: tf.data.Dataset
    valid: tf.data.Dataset
    test: tf.data.Dataset


class DataFactory(ABC):

    @abstractmethod
    def load_and_preprocess_data(self, config, dataset):
        pass


class KerasDataFactory(DataFactory):

    def load_and_preprocess_data(self, config) -> TFSchema:

        if config.dataset.name == "mnist":
            dataset = tf.keras.datasets.mnist
        elif config.dataset.name == "fashion_mnist":
            dataset = tf.keras.datasets.fashion_mnist
        else:
            raise TypeError

        (X_train_full, y_train_full), (X_test, y_test) = dataset.load_data()
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full,
            y_train_full,
            test_size=config.dataset.test_size,
            random_state=config.seed,
            shuffle=config.dataset.shuffle
            )
        
        if config.dataset.scale is True:
            shape = X_train.shape[1:]
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train, X_valid, X_test = [set_.ravel().reshape(-1, 1)
                                        for set_ in (X_train, X_valid, X_test)]
            X_train, X_valid, X_test = [scaler.fit_transform(set_)
                                        for set_ in (X_train, X_valid, X_test)]
            X_train, X_valid, X_test = [set_.reshape(-1, *shape)
                                        for set_ in (X_train, X_valid, X_test)]
            
        return TFSchema(
        train=tf.data.Dataset.from_tensor_slices((X_train, y_train)),
        valid=tf.data.Dataset.from_tensor_slices((X_valid, y_valid)),
        test=tf.data.Dataset.from_tensor_slices((X_test,y_test))
        )