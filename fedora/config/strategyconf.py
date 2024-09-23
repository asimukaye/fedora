from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from fedora.strategy.fedavg import FedAvgStrategy
from fedora.strategy.fedavgmanual import FedavgManual
from fedora.strategy.fedopt import FedOptStrategy
from fedora.strategy.fedstdev import FedstdevStrategy
from fedora.strategy.fedgradstd import FedgradstdStrategy
from fedora.strategy.ties import TiesStrategy
from fedora.strategy.cgsv import CgsvStrategy

import importlib
from fedora.utils import Range

# strategy = importlib.import_module("fedora.strategy")

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

########### Strategy Configurations ##########
@dataclass
class StrategyConfig:
    train_fraction: float
    eval_fraction: float

    def __post_init__(self):
        assert self.train_fraction == Range(
            0.0, 1.0
        ), f"Invalid value {self.train_fraction} for sampling fraction"
        assert self.eval_fraction == Range(0.0, 1.0)

@dataclass
class TiesConfig(StrategyConfig):
    K: float = 20
    lamda: float = 0.5
    merge_func: str = "dis-mean"

    def __post_init__(self):
        super().__post_init__()
        assert self.merge_func in ["dis-mean", "dis-max", "dis-sum"]


@dataclass
class FedOptConfig(StrategyConfig):
    alpha: float = 0.95
    gamma: float = 0.5
    delta_normalize: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.delta_normalize:
            assert (
                self.gamma > 0.0
            ), "Gamma should be greater than 0 for delta normalization"
        assert self.alpha == Range(0.0, 1.0), f"Invalid value {self.alpha} for alpha"


@dataclass
class FedstdevConfig(StrategyConfig):
    """Config schema for Fedstdev strategy config"""

    weighting_strategy: str
    betas: list[float]
    beta_0: float
    alpha: float
    num_clients: int


@dataclass
class CGSVConfig(StrategyConfig):
    num_clients: int
    beta: float = 1.5
    alpha: float = 0.95
    gamma: float = 0.15
    delta_normalize: bool = False
    sparsify_gradients: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.delta_normalize:
            assert (
                self.gamma > 0.0
            ), "Gamma should be greater than 0 for delta normalization"
        assert self.alpha == Range(0.0, 1.0), f"Invalid value {self.alpha} for alpha"
        assert self.beta >= 1.0, f"Invalid value {self.beta} for beta"
@dataclass
class FedavgManualConfig(StrategyConfig):
    weights: list[float]
    num_clients: int

    def __post_init__(self):
        super().__post_init__()
        assert (
            len(self.weights) == self.num_clients
        ), "Number of weights should be equal to number of clients"
        # Auto normalize the weights
        self.weights = [w / sum(self.weights) for w in self.weights]


@dataclass
class StrategySchema:
    _target_: str
    # _partial_: bool
    cfg: StrategyConfig

def register_strategy_configs():
    cs = ConfigStore.instance()
    
    cs.store(group="strategy/cfg", name="base_cgsv", node=CGSVConfig)
    cs.store(group="strategy", name="strategy_schema", node=StrategySchema)
    cs.store(group="strategy/cfg", name="base_strategy", node=StrategyConfig)
    cs.store(group="strategy/cfg", name="fedavgmanual", node=FedavgManualConfig)
    cs.store(group="strategy/cfg", name="fedopt", node=FedOptConfig)
    cs.store(group="strategy/cfg", name="cgsv", node=CGSVConfig)
    cs.store(group="strategy/cfg", name="ties", node=TiesConfig)
    cs.store(group="strategy/cfg", name="fedstdev_strategy", node=FedstdevConfig)