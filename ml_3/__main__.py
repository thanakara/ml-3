import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
import logging

from ml_3.definitions.constants import CONFIG_PATH
from omegaconf import OmegaConf, DictConfig
from ml_3.model.layers import flatten, calc_channels
from ml_3.factories.model import KerasBuilder

OmegaConf.register_new_resolver(name="flatten", resolver=flatten)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)

logger = logging.getLogger(__name__)
config_path = str(CONFIG_PATH)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    builder = hydra.utils.call(config.backend.builder)
    model = builder.build(config.model.layers)
    print(model.__class__)

if __name__ == "__main__":
    main()