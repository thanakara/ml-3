import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
import logging
from ml_3.definitions.constants import CONFIG_PATH
from omegaconf import OmegaConf, DictConfig
from ml_3.model.layers import flatten, calc_channels, Dense
from ml_3.factories.model import keras_builder


logger = logging.getLogger(__name__)
config_path = str(CONFIG_PATH)
OmegaConf.register_new_resolver(name="flatten", resolver=flatten)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)
    model = keras_builder(config.model.layers)
    print(model.summary())


if __name__ == "__main__":
    main()