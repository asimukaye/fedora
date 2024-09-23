from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

@dataclass
class ClientConfig:
    start_epoch: int = field(default=0)
    n_iters: int = field(
        init=False, default=-1
    )  # TO be initialized after dataset is loaded
    data_shuffle: bool = field(default=False)


def default_seed():
    return [1, 2, 3]


@dataclass
class FedstdevClientConfig(ClientConfig):
    seeds: list[int] = field(default_factory=default_seed)
    client_ids: list[str] = field(default_factory=list)


@dataclass
class FedgradstdClientConfig(ClientConfig):
    seeds: list[int] = field(default_factory=default_seed)
    client_ids: list[str] = field(default_factory=list)
    abs_before_mean: bool = field(default=False)
    # def __post_init__(self):
    #     super().__post_init__()

def register_client_configs():
    cs = ConfigStore.instance()

    cs.store(group="client/cfg", name="base_client", node=ClientConfig)
    cs.store(group="client/cfg", name="fedstdev_client", node=FedstdevClientConfig)
    cs.store(group="client/cfg", name="fedgradstd_client", node=FedgradstdClientConfig)