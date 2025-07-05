import logging
from typing import Any
from omegaconf import DictConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

import pathlib
import pandas as pd

class MyCallback(Callback):

    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def on_job_start(self,
                     config: DictConfig,
                     *,
                     task_function: TaskFunction,
                     **kwargs: Any
                     ) -> None:
        pass

    def on_job_end(self,
                   config: DictConfig,
                   job_return: JobReturn,
                   **kwargs
                   ) -> None:
        
        output_dir = pathlib.Path(config.hydra.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None

        history = job_return.return_value
        history = pd.DataFrame(history.history)
        history.index = range(1, len(history) + 1)
        history.to_json(output_dir.joinpath("history.json"), indent=4)