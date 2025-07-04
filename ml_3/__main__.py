import hydra
import logging
from ml_3.definitions.constans import CONFIG_DIR
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)
cfg_path = str(CONFIG_DIR)

@hydra.main(version_base=None, config_path=cfg_path, config_name="config")
def main(cfg: DictConfig):
    logging.info("Resolving Hydra Configuration")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()