import logging
from typing import Any
from conf.config import MyConfig
from hydra.types import TaskFunction
from hydra.core.utils import JobReturn
from hydra.experimental.callback import Callback

import pathlib
import pandas as pd

class MyCallback(Callback):

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def on_job_start(self,
                     config: MyConfig,
                     *,
                     task_function: TaskFunction,
                     **kwargs: Any) -> None:
        pass

    def on_job_end(self,
                   config: MyConfig,
                   job_return: JobReturn,
                   **kwargs) -> None:
        
        output_dir = pathlib.Path(config.hydra.runtime.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        configuration = job_return.return_value
        configuration = pd.DataFrame(configuration)
        configuration.T.to_json(output_dir.joinpath("conf.json"))