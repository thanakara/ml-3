import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra
import logging
from ml_3.definitions.constants import CONFIG_PATH
from ml_3.factories.dataset import dataset_factory
from omegaconf import OmegaConf, DictConfig


logger = logging.getLogger(__name__)
config_path = str(CONFIG_PATH)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):

    logger.info("Resolving Hydra Configuration")
    OmegaConf.resolve(config)
    logger.debug(config.model.layers)
    ds = dataset_factory(
        name=config.dataset.name,
        test_size=0.2,
        seed=config.seed
    )
    
    valid_ds = ds.valid
    for img, label in valid_ds.batch(config.batch_size).take(3):
        logger.info(img.shape)
        logger.debug(label.numpy())

if __name__ == "__main__":
    main()