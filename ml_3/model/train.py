import tensorflow as tf
import logging
from hydra.utils import instantiate
from datetime import datetime

logger = logging.getLogger(__name__)

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.num_epochs = 0

    def on_train_begin(self, logs: dict=None) -> None:
        self.start_time = datetime.now()
        
    def on_epoch_end(self, epoch: int, logs: dict=None) -> None:
        self.num_epochs += 1
        tl = logs["loss"]
        ta = logs["accuracy"]
        vl = logs["val_loss"]
        va = logs["val_accuracy"]

        msg = f"Epoch {epoch + 1} - loss: {tl:.4f} - val_loss: {vl:.4f} - accuracy: {ta:.2%} - val_accuracy: {va:.2%}"
        logger.info(msg)
        
    def on_train_end(self, logs: dict=None) -> None:
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Duration: {duration} seconds.")


def compile_and_fit(
        model: tf.keras.Sequential,
        train_ds: tf.data.Dataset,
        valid_ds: tf.data.Dataset,
        backend: dict
        ) -> tf.keras.callbacks.History:
    logger.info(f"Training on {backend['device'].upper()}")

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=backend["patience"],
        restore_best_weights=True
    )

    optimizer = instantiate(backend["optimizer"])
    model.compile(
        optimizer=optimizer,
        loss=backend["loss"],
        metrics=backend["metrics"]
    )
    batch_size = backend["batch_size"]
    with tf.device(backend["device"]):
        history = model.fit(
            train_ds.batch(batch_size),
            validation_data=valid_ds.batch(batch_size),
            epochs=backend["epochs"],
            verbose=False,
            callbacks=[CustomCallback(), early_stopping_cb]
        )
        
    return history
