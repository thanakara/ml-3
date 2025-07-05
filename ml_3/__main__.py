import os
import hydra
import logging
from ml_3.definitions.constans import CONFIG_DIR
from omegaconf import OmegaConf, DictConfig

from ml_3.model.baseline_factories import MNISTFactory, ModelFactory
from ml_3.model.train import compile_and_fit

logger = logging.getLogger(__name__)
config_path = str(CONFIG_DIR)

@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(config: DictConfig):
    logger.info("Resolving Hydra Configuration")
    config = OmegaConf.to_container(cfg=config, resolve=True)
    mnist_factory = MNISTFactory(dataset=config["dataset"])
    train_ds, valid_ds, _ = mnist_factory._build_dataset(
        test_size=0.2
        )
    model_factory = ModelFactory()
    model = model_factory._build(
        num_hidden=config["layers"],
        name=config["model"]
        )
    history = compile_and_fit(
        model=model,
        train_ds=train_ds,
        valid_ds=valid_ds,
        backend=config["backend"]
    )
    return history

if __name__ == "__main__":
    main()