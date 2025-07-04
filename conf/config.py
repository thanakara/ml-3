from dataclasses import dataclass

@dataclass
class Paths:
    log: str

@dataclass
class Params:
    shuffle: bool
    batch_size: int
    learning_rate: float

@dataclass
class MyConfig:
    paths: Paths
    params: Params