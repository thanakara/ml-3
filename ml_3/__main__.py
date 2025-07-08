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
    # print(OmegaConf.to_yaml(config))
    """
    WANT:
     + Resolve
     + Use {oc.select}
     + utils.instansiate()
     + print summary
    """

if __name__ == "__main__":
    main()