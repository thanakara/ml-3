import hydra
import logging
from ml_3.definitions.constans import CONFIG_DIR
from conf.config import MyConfig


logger = logging.getLogger(__name__)
config_path = str(CONFIG_DIR)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: MyConfig):
    logger.info("INFO level msg.")
    logger.info(cfg.dataset)
    return cfg


if __name__ == "__main__":
    main()