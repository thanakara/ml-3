import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
import logging

from ml_3.definitions.constants import CONFIG_PATH
from omegaconf import OmegaConf, DictConfig
from ml_3.model.layers import get_out_units, calc_channels

OmegaConf.register_new_resolver(name="get_out_units", resolver=get_out_units)
OmegaConf.register_new_resolver(name="calc_channels", resolver=calc_channels)

logger = logging.getLogger(__name__)
config_path = str(CONFIG_PATH)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):
    OmegaConf.resolve(config)

    builder = hydra.utils.call(config.backend.builder, config)
    valid_ds = builder.load_data().valid
    for X, y in valid_ds.batch(config.batch_size).take(1):
        print(X.shape)
        print(y)

if __name__ == "__main__":
    main()