import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class ModelFactory(object):
    def __init__(self, seed=42) -> None:
        self.seed = seed
    
    def _build(self, num_hidden: int, name: str) -> tf.keras.Sequential:
        from functools import partial
        Dense = partial(tf.keras.layers.Dense, activation="relu")
        tf.keras.utils.set_random_seed(self.seed)
        model = tf.keras.Sequential([
            tf.keras.Input([28, 28]),
            tf.keras.layers.Flatten()
        ], name=name)
        units = 2 ** (num_hidden + 4)
        for _ in range(num_hidden):
            model.add(Dense(units))
            units //= 2
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        return model
 
class MNISTFactory(object):
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset

    def _build_dataset(self, test_size: int | float,
                       scale=True, seed=42
                       ) -> tuple[tf.data.Dataset]:
        from sklearn.model_selection import train_test_split
        if self.dataset == "mnist":
            ds = tf.keras.datasets.mnist

        elif self.dataset == "fashion_mnist":
            ds = tf.keras.datasets.fashion_mnist
        
        else:
            raise TypeError
        
        (X_train_full, y_train_full), (X_test, y_test) = ds.load_data()
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=test_size,
            random_state=seed, shuffle=True)
        if scale:
            X_train = X_train / 255.0
            X_valid = X_valid / 255.0
            X_test = X_test / 255.0
        
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return train_ds, valid_ds, test_ds