import hydra
from omegaconf import DictConfig
from hydra.experimental.callback import Callback

class MyCallback(Callback):
    def __init__(self, blob: str) -> None:
        self.blob = blob
    
    def on_job_end(self, config: DictConfig, **kwargs) -> None:
        print("Job ended...")