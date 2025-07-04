import hydra
import logging
from ml_3.definitions.constans import CONFIG_DIR
from conf.config import MyConfig
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)
config_path = str(CONFIG_DIR)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: MyConfig):
    logger.info("INFO level msg.")
    logger.info(hydra.utils.get_original_cwd())
    logger.debug(OmegaConf.to_yaml(cfg))
    logger.debug(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

if __name__ == "__main__":
    main()