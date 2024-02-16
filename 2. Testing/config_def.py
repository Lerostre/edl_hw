from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import MISSING

@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    hidden_size: int = 256

@dataclass
class DiffusionConfig:
    unet: UNetConfig = MISSING
    betas: list[float] = field(default_factory=lambda: [1e-4, 0.02])
    num_timesteps: int =  1000

@dataclass
class LoaderConfig:
    batch_size: int = 128
    num_workers: int = 4

# @dataclass
# class OptimizerConfig:
#     name: str = MISSING
#     lr: float = 1e-5
#     momentum: float = 0.9
#     weight_decay: float = 0
    
#     def __init__(self,
#         lr: float = 1e-5,
#         momentum: float = 0.9,
#         weight_decay: float = 0
#     ) -> None:
#         self.lr = lr
#         self.momentum = momentum
#         self.weight_decay = weight_decay

@dataclass
class TrainConfig:
    random_seed: int = MISSING
    num_epochs: int = 100
    device: str = "cuda"
    subset_size: float | int = 1
    augmentation: Any = MISSING
    loader: LoaderConfig = MISSING
    optimizer: Any = MISSING
    # optimizer: OptimizerConfig = OptimizerConfig()

@dataclass
class DBConfig:
    log: bool = True
    project: str | None = MISSING
    name: str | None = MISSING

@dataclass
class Config:
    model: DiffusionConfig
    train: TrainConfig
    db: DBConfig

def access_configs(config_store):
    config_store.store(name="config", node=Config)
    config_store.store(name="model", node=DiffusionConfig)
    config_store.store(name="train", node=TrainConfig)
    config_store.store(name="db", node=DBConfig)