import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
import logging

from ml_3.definitions.constants import CONFIG_PATH
from omegaconf import OmegaConf, DictConfig
from ml_3.model.layers import flatten, calc_channels
OmegaConf.register_new_resolver(name="flatten", resolver=flatten)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)

from ml_3.factories.dataset import dataset_factory
from ml_3.factories.model import keras_builder
from ml_3.model.train import compile_and_fit


logger = logging.getLogger(__name__)
config_path = str(CONFIG_PATH)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):
    Dataset = dataset_factory(
        name=config.name,
        test_size=config.test_size,
        seed=config.seed
    )
    model = keras_builder(layers=config.model.layers)
    history = compile_and_fit(
        model=model,
        train_ds=Dataset.train,
        valid_ds=Dataset.valid,
        backend=config.backend
    )

    return history

if __name__ == "__main__":
    main()