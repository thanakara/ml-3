from dataclasses import dataclass
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Dataset:
    train: tf.data.Dataset
    valid: tf.data.Dataset
    test: tf.data.Dataset
    
def get_gs_dataset(
    ds,                 # TODO: hint
    test_size: int,
    seed: int,
    scale: bool=True,
    ) -> Dataset:
    
    (X_train_full, y_train_full), (X_test, y_test) = ds.load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full,
        y_train_full,
        test_size=test_size,
        random_state=seed,
        shuffle=True
        )
    
    if scale:
        shape = X_train.shape[1:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train, X_valid, X_test = [set_.ravel().reshape(-1, 1) for set_ in (X_train, X_valid, X_test)]
        X_train, X_valid, X_test = [scaler.fit_transform(set_) for set_ in (X_train, X_valid, X_test)]
        X_train, X_valid, X_test = [set_.reshape(-1, *shape) for set_ in (X_train, X_valid, X_test)]
    
    
    return Dataset(
        train=tf.data.Dataset.from_tensor_slices((X_train, y_train)),
        valid=tf.data.Dataset.from_tensor_slices((X_valid, y_valid)),
        test=tf.data.Dataset.from_tensor_slices((X_test,y_test))
        )


def dataset_factory(name: str, test_size: int | float, seed: int, scale: bool=True) -> Dataset:
    if name == "mnist":
        return get_gs_dataset(
            ds=tf.keras.datasets.mnist,
            test_size=test_size,
            seed=seed,
            scale=scale
            )
    elif name == "fashion_mnist":
        return get_gs_dataset(
            ds=tf.keras.datasets.fashion_mnist,
            test_size=test_size,
            seed=seed,
            scale=scale
            )
        
    raise NotImplementedError