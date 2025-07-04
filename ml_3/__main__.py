import hydra
import logging
from ml_3.definitions.constans import CONFIG_DIR
from conf.config import MyConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
config_path = str(CONFIG_DIR)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: MyConfig):
    logging.info("Resolving Hydra Configuration")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()