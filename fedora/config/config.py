# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, asdict
from typing import Optional
import typing as t
from functools import partial
import os


from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path, get_object
import torch
from torch import cuda
from torch.backends import mps
from pandas import json_normalize
import logging

from fedora.client.baseclient import BaseFlowerClient
from fedora.client.fedstdevclient import FedstdevClient
from fedora.client.fedgradstdclient import FedgradstdClient

from fedora.server.baseserver import BaseFlowerServer
from fedora.utils import Range, get_free_gpus, arg_check, get_free_gpu
from fedora.config.strategyconf import StrategySchema, register_strategy_configs

from fedora.config.splitconf import SplitConfig
from fedora.config.trainconf import TrainConfig
from fedora.config.clientconf import ClientConfig, register_client_configs
logger = logging.getLogger(__name__)

OPTIMIZER_MAPS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop
}

LRSCHEDULER_MAPS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cyclic": torch.optim.lr_scheduler.CyclicLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}

CLIENT_MAPS = {
    "fedstdev": FedstdevClient,
    "fedgradstd": FedgradstdClient,
    "base": BaseFlowerClient,
    "baseflower": BaseFlowerClient,
    "fedavg": BaseFlowerClient,
}

SERVER_MAPS = {
    "base": BaseFlowerServer,
    "baseflower": BaseFlowerServer,
}


########## Simulator Configurations ##########


def default_resources():
    return {"num_cpus": 1, "num_gpus": 0.0}


# TODO: Isolate result manager configs from this
# TODO: Develop a client and strategy compatibility checker
@dataclass
class SimConfig:
    seed: int
    num_clients: int
    use_tensorboard: bool
    num_rounds: int
    use_wandb: bool
    save_csv: bool
    checkpoint_every: int = field(default=10)
    out_prefix: str = field(default="")
    # plot_every: int = field(default=10)
    eval_every: int = field(default=1)
    eval_type: str = field(default="both")
    mode: str = field(default="federated")
    # flower: Optional[FlowerConfig]
    flwr_resources: dict = field(default_factory=default_resources)

    def __post_init__(self):
        assert self.mode in [
            "federated",
            "standalone",
            "centralized",
            "flower",
        ], f"Unknown simulator mode: {self.mode}"
        assert (
            self.use_tensorboard or self.use_wandb or self.save_csv
        ), f"Select any one logging method atleast to avoid losing results"


########## Training Configurations ##########


########## Client Configurations ##########


@dataclass
class ClientSchema:
    _target_: str
    _partial_: bool
    cfg: ClientConfig
    train_cfg: TrainConfig


########## Server Configurations ##########
@dataclass
class ServerConfig:
    eval_type: str = field(default="both")
    eval_every: int = field(default=1)


@dataclass
class ServerSchema:
    _target_: str
    _partial_: bool
    cfg: ServerConfig
    train_cfg: TrainConfig



########## Dataset configurataions ##########


# TODO: Add support for custom transforms
@dataclass
class TransformsConfig:
    resize: Optional[dict] = field(default_factory=dict)
    normalize: Optional[dict] = field(default_factory=dict)
    train_cfg: Optional[list] = field(default_factory=list)

    def __post_init__(self):
        train_tf = []
        for key, val in self.__dict__.items():
            if key == "normalize":
                continue
            else:
                if val:
                    train_tf.append(val)
        self.train_cfg = train_tf

    # construct



@dataclass
class DatasetConfig:
    name: str
    data_path: str
    dataset_family: str
    transforms: Optional[TransformsConfig]
    test_fraction: Optional[float]
    seed: Optional[int]
    federated: bool
    split_conf: SplitConfig
    subsample: bool = False
    subsample_fraction: float = 0.0  # subsample the dataset with the given fraction

    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        self.data_path = to_absolute_path(self.data_path)
        if self.federated == False:
            assert (
                self.split_conf.num_splits == 1
            ), "Non-federated datasets should have only one split"


########## Model Configurations ##########
@dataclass
class ModelConfig:
    _target_: str
    hidden_size: int


@dataclass
class ModelConfigGN(ModelConfig):
    num_groups: int


@dataclass
class DatasetModelSpec:
    num_classes: int
    in_channels: int


@dataclass
class ModelInitConfig:
    init_type: str
    init_gain: float


########## Master Configurations ##########
@dataclass
class Config:
    mode: str = field()
    desc: str = field()
    simulator: SimConfig = field()
    server: ServerSchema = field()
    strategy: StrategySchema = field()
    client: ClientSchema = field()
    train_cfg: TrainConfig = field()

    dataset: DatasetConfig = field()
    model: ModelConfig = field()
    model_init: ModelInitConfig = field()
    log_conf: list = field(default_factory=list)

    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):
        if self.simulator.mode == "centralized":
            self.dataset.federated = False
            logger.info("Setting federated cfg in dataset cfg to False")
        else:
            assert (
                self.dataset.split_conf.num_splits == self.simulator.num_clients
            ), "Number of clients in dataset and simulator should be equal"

        if self.train_cfg.device == "mps" or self.train_cfg.device == "cpu":
            # GPU support in flower for MPS is not available
            self.simulator.flwr_resources = default_resources()

        if self.mode == "debug":
            set_debug_mode(self)


########## Debug Configurations ##########
def set_debug_mode(cfg: Config):
    """Debug mode overrides to the configuration object"""
    logger.root.setLevel(logging.DEBUG)
    cfg.simulator.use_wandb = False
    cfg.simulator.use_tensorboard = False
    cfg.simulator.save_csv = True

    logger.debug(f"[Debug Override] Setting use_wandb to: {cfg.simulator.use_wandb}")
    cfg.simulator.num_rounds = 2
    logger.debug(f"[Debug Override] Setting rounds to: {cfg.simulator.num_rounds}")
    cfg.client.train_cfg.epochs = 1
    logger.debug(f"[Debug Override] Setting epochs to: {cfg.client.train_cfg.epochs}")

    cfg.simulator.num_clients = 3
    cfg.dataset.split_conf.num_splits = 3
    if cfg.dataset.split_conf.split_type in [
        "n_label_flipped_clients",
        "n_noisy_clients",
        "n_distinct_noisy_clients",
        "n_distinct_label_flipped_clients",
    ]:
        cfg.dataset.split_conf.num_noisy_clients = 2
    if hasattr(cfg.strategy.cfg, "num_clients"):
        cfg.strategy.cfg.num_clients = 3  # type: ignore
    logger.debug(
        f"[Debug Override] Setting num clients to: {cfg.simulator.num_clients}"
    )

    cfg.dataset.subsample = True
    cfg.dataset.subsample_fraction = 0.05


def register_configs():
    # Register a new configuration scheme to be validated against from the config file
    register_strategy_configs()
    register_client_configs()
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="client", name="client_schema", node=ClientSchema)
    cs.store(group="server", name="base_server", node=ServerSchema)
    cs.store(group="server", name="base_flower_server", node=ServerSchema)

    cs.store(group="train_cfg", name="base_train", node=TrainConfig)
   

    cs.store(group="model", name="resnet18gn", node=ModelConfigGN)
    cs.store(group="model", name="resnet34gn", node=ModelConfigGN)
    cs.store(group="model", name="resnet10gn", node=ModelConfigGN)
    # cs.store(group='model', name='resnet18gn', node=ModelConfigGN)


    # cs.store(group='server/cfg', name='base_fedavg', node=FedavgConfig)
    # cs.store(group='server/cfg', name='fedstdev_server', node=FedstdevServerConfig)

