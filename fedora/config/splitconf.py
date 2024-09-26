from dataclasses import dataclass, field
import typing as t
from hydra.core.config_store import ConfigStore
@dataclass
class SplitConfig:
    name: str
    num_splits: int  # should be equal to num_clients
    # Train test split ratio within the client,
    # Now this is auto determined by the test set size
    test_fractions: list[float] = field(init=False, default_factory=list)

# TODO: find a way to handle lists
@dataclass
class NoisyImageSplitConfig(SplitConfig):
    num_noisy_clients: int
    noise_mu: float 
    noise_sigma: float 
@dataclass
class NoisyLabelSplitConfig(SplitConfig):
    num_noisy_clients: int
    noise_flip_percent: float 

@dataclass
class PathoSplitConfig(SplitConfig):
    num_class_per_client: int

@dataclass
class DirichletSplitConfig(SplitConfig):
    alpha: float

@dataclass
class DataImbalanceSplitConfig(SplitConfig):
    num_imbalanced_clients: int


def register_split_configs():
    cs = ConfigStore.instance()
    cs.store(group="split", name="iid", node=SplitConfig)
    cs.store(group="split", name="patho", node=PathoSplitConfig)
    cs.store(group="split", name="dirichlet", node=DirichletSplitConfig)
    cs.store(group="split", name="noisylabel", node=NoisyLabelSplitConfig)
    cs.store(group="split", name="noisyimage", node=NoisyImageSplitConfig)
    