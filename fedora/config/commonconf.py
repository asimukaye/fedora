from dataclasses import dataclass, field
import os
from typing import Optional
from torch.backends import mps
from torch import cuda
from fedora.utils import Range, get_free_gpus, arg_check, get_free_gpu
from fedora.config.splitconf import SplitConfig
from hydra.utils import to_absolute_path
import logging
logger = logging.getLogger(__name__)

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


@dataclass
class MetricConfig:
    eval_metrics: list
    # fairness_metrics: list
    log_to_file: bool = False
    file_prefix: str = field(default="")
    cwd: Optional[str] = field(default=None)

    def __post_init__(self):
        self.cwd = os.getcwd() if self.cwd is None else self.cwd

########## Server Configurations ##########
@dataclass
class ServerConfig:
    eval_type: str = field(default="both")
    eval_every: int = field(default=1)


########## Training Configurations ##########

@dataclass
class TrainConfig:
    epochs: int = field()
    device: str = field()
    batch_size: int = field()
    eval_batch_size: int = field()
    optimizer: dict = field()
    criterion: dict = field()
    lr: float = field()  # Client LR is optional
    lr_scheduler: Optional[dict] = field()
    lr_decay: Optional[float] = field()
    metric_cfg: MetricConfig = field()

    def __post_init__(self):
        assert self.batch_size >= 1
        if self.device == "auto":
            if cuda.is_available():
                # Set visible GPUs
                # TODO: MAke the gpu configurable
                gpu_ids = get_free_gpus()
                # logger.info('Selected GPUs:')
                logger.info("Selected GPUs:" + ",".join(map(str, gpu_ids)))
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

                if cuda.device_count() > 1:
                    self.device = f"cuda:{get_free_gpu()}"
                else:
                    self.device = "cuda"

            elif mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"Auto Configured device to: {self.device}")
        if self.lr_scheduler:
            arg_check(self.lr_scheduler)
        arg_check(self.optimizer)

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
