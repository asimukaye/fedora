# Master config file to store config dataclasses and do validation
from dataclasses import dataclass, field, asdict
from functools import partial


from hydra.core.config_store import ConfigStore
import logging

from fedora.client.baseclient import BaseFlowerClient
from fedora.client.fedstdevclient import FedstdevClient
from fedora.client.fedgradstdclient import FedgradstdClient

from fedora.server.baseserver import BaseFlowerServer

from fedora.strategy.fedavg import FedAvgStrategy
from fedora.strategy.fedavgmanual import FedavgManual
from fedora.strategy.fedopt import FedOptStrategy
from fedora.strategy.fedstdev import FedstdevStrategy
from fedora.strategy.fedgradstd import FedgradstdStrategy
from fedora.strategy.ties import TiesStrategy
from fedora.strategy.cgsv import CgsvStrategy


# from fedora.utils import Range, get_free_gpus, arg_check, get_free_gpu
from fedora.config.strategyconf import StrategyConfig, register_strategy_configs

from fedora.config.commonconf import TrainConfig, ModelConfigGN, ModelInitConfig, DatasetConfig, SimConfig, default_resources, ModelConfig, ServerConfig, initialize_module, partial_initialize_module
from fedora.config.clientconf import ClientConfig, register_client_configs
from fedora.config.splitconf import SplitConfig, register_split_configs
logger = logging.getLogger(__name__)

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

STRATEGY_MAPS = {
    "fedavg": FedAvgStrategy,
    "base": FedAvgStrategy,
    "fedavgmanual": FedavgManual,
    "fedopt": FedOptStrategy,
    "cgsv": CgsvStrategy,
    "ties": TiesStrategy,
    "fedstdev": FedstdevStrategy,
    "fedgradstd": FedgradstdStrategy,
}


@dataclass
class StrategySchema:
    name: str
    cfg: StrategyConfig

########## Client Configurations ##########

@dataclass
class ClientSchema:

    name: str
    cfg: ClientConfig
    train_cfg: TrainConfig


@dataclass
class ServerSchema:
    name: str
    cfg: ServerConfig
    train_cfg: TrainConfig

def get_client_partial(client_schema: ClientSchema)->partial[BaseFlowerClient]:
    return partial_initialize_module(CLIENT_MAPS, client_schema, ignore_args=["client_id", "model", "dataset"])

def get_server_partial(server_schema: ServerSchema)->partial[BaseFlowerServer]:
    return partial_initialize_module(SERVER_MAPS, server_schema, ignore_args=["clients", "model", "strategy", "dataset", "result_manager"])

def get_strategy_partial(strategy_schema: StrategySchema)->partial[FedAvgStrategy]:
    return partial_initialize_module(STRATEGY_MAPS, strategy_schema, ignore_args=["model", "res_man"])

########## Master Configurations ######
# ####
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
    split: SplitConfig = field()
    model: ModelConfig = field()
    model_init: ModelInitConfig = field()
    log_cfg: list = field(default_factory=list)

    # metrics: MetricConfig = field(default=MetricConfig)

    def __post_init__(self):

        self.server_partial = get_server_partial(self.server)

        self.client_partial = get_client_partial(self.client)

        self.strategy_partial = get_strategy_partial(self.strategy)
   
        if self.simulator.mode == "centralized":
            self.dataset.federated = False
            logger.info("Setting federated cfg in dataset cfg to False")
        else:
            if self.dataset.federated == True:
                assert (
                    self.split.num_splits == self.simulator.num_clients
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
    cfg.split.num_splits = 3
    # if cfg.dataset.split_conf.name in [
    #     "n_label_flipped_clients",
    #     "n_noisy_clients",
    #     "n_distinct_noisy_clients",
    #     "n_distinct_label_flipped_clients",
    # ]:
    #     cfg.dataset.split_conf.num_noisy_clients = 2
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
    register_split_configs()

    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    cs.store(group="client", name="client_schema", node=ClientSchema)
    cs.store(group="server", name="base_server", node=ServerSchema)
    cs.store(group="server", name="base_flower_server", node=ServerSchema)
    cs.store(group="strategy", name="strategy_schema", node=StrategySchema)
    cs.store(group="train_cfg", name="base_train", node=TrainConfig)
   

    cs.store(group="model", name="resnet18gn", node=ModelConfigGN)
    cs.store(group="model", name="resnet34gn", node=ModelConfigGN)
    cs.store(group="model", name="resnet10gn", node=ModelConfigGN)
    # cs.store(group='model', name='resnet18gn', node=ModelConfigGN)
