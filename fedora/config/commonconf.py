from dataclasses import dataclass, field, asdict, is_dataclass
import os
import psutil
import inspect
from typing import Optional
from functools import partial
import torch
from torch.backends import mps
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from torch import cuda
from fedora.utils import Range, get_free_gpus, arg_check, get_free_gpu
from fedora.config.splitconf import SplitConfig
from hydra.utils import to_absolute_path
import logging
logger = logging.getLogger(__name__)

########## Simulator Configurations ##########

OPTIMIZER_MAP = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "rmsprop": torch.optim.RMSprop
}

LRSCHEDULER_MAP = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cyclic": torch.optim.lr_scheduler.CyclicLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    "cosine_warmup": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
}

LOSS_MAP = {
    "crossentropy": torch.nn.CrossEntropyLoss,
    "nll": torch.nn.NLLLoss,
    "mse": torch.nn.MSELoss,
    "l1": torch.nn.L1Loss,
    "bce": torch.nn.BCELoss,
    "bcelogits": torch.nn.BCEWithLogitsLoss,
    "ctc": torch.nn.CTCLoss,
    "kl": torch.nn.KLDivLoss,
    "margin": torch.nn.MultiMarginLoss,
    "smoothl1": torch.nn.SmoothL1Loss,
    "huber": torch.nn.SmoothL1Loss,
    "triplet": torch.nn.TripletMarginLoss,
    "hinge": torch.nn.MultiMarginLoss,
    "cosine": torch.nn.CosineEmbeddingLoss,
    "poisson": torch.nn.PoissonNLLLoss
}

def get_module_args(map: dict, obj: dict, ignore_args: list = []):
    if is_dataclass(obj):
        # ic(obj.__dict__)      
        # obj = asdict(obj)
        obj_dict = obj.__dict__.copy()
        # ic(obj)
    elif isinstance(obj, dict):
        obj_dict = obj.copy()
    else:
        raise ValueError("Unknown object type: ", type(obj))
    
    module = map[obj_dict['name']]
    obj_dict.pop("name")

    arg_names = obj_dict.keys()
    
    check_module_args(module, arg_names, ignore_args=ignore_args)
    return module, obj_dict
    

def check_module_args(module, given_args, ignore_args: list = []): 
    # Figure out usage with string functions
    # Check if the argument spec is compatible wi

    all_args = inspect.signature(module).parameters.values()
    required_args = [
        arg.name for arg in all_args if arg.default == inspect.Parameter.empty
    ]
    for ign in ignore_args:
        if ign in required_args:
            required_args.remove(ign)
    # collect eneterd arguments
    for argument in required_args:
        if argument in given_args:
            logger.debug(f"Found required argument: {argument}")
        else:
            logger.error(f"Missing required argument: {argument}")
            raise ValueError(f"Missing required argument: {argument}")


def initialize_module(map: dict, obj):
    module, args = get_module_args(map, obj)
    return module(**args)


def partial_initialize_module(map: dict, obj, ignore_args: list):
    module, args = get_module_args(map, obj, ignore_args=ignore_args)
    return partial(module, **args)

def default_resources():
    return {"num_cpus": 1, "num_gpus": 0.0}


# TODO: Isolate result manager configs from this
@dataclass
class ResultConfig:
    use_tensorboard: bool
    use_wandb: bool
    save_csv: bool
    out_prefix: Optional[str] = field(default="")
    # plot_every: int = field(default=10)

    def __post_init__(self):
        assert (
            self.use_tensorboard or self.use_wandb or self.save_csv
        ), f"Select any one logging method atleast to avoid losing results"

# TODO: Develop a client and strategy compatibility checker
@dataclass
class SimConfig:
    seed: int
    num_clients: int
    # use_tensorboard: bool
    num_rounds: int
    # use_wandb: bool
    # save_csv: bool
    device: str = field()
    checkpoint_every: int = field(default=10)
    # out_prefix: str = field(default="")
    # plot_every: int = field(default=10)
    # eval_every: int = field(default=1)
    # eval_type: str = field(default="both")
    mode: str = field(default="federated")
    
    # flower: Optional[FlowerConfig]
    # flwr_resources: dict = field(default_factory=default_resources)

    def __post_init__(self):
        assert self.mode in [
            "federated",
            "standalone",
            "centralized",
            "flower",
        ], f"Unknown simulator mode: {self.mode}"
        assert self.device in [
            "auto",
            "cuda",
            "cpu",
            "mps",
        ], f"Unknown device : {self.device}"

        gpu_ids = []
        if self.device == "auto":
            if cuda.is_available():
                # Set visible GPUs
                # TODO: MAke the gpu configurable
                gpu_ids = get_free_gpus()
                # logger.info('Selected GPUs:')
                logger.info("Selected GPUs:" + ",".join(map(str, gpu_ids)))

                # Disabling below line due to cluster policies
                # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

                if cuda.device_count() > 1:
                    self.device = f"cuda:{get_free_gpu()}"
                else:
                    self.device = "cuda"

            elif mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            logger.info(f"Auto Configured device to: {self.device}")
        else:
            self.device = self.device
        
        if self.mode == "flower":
        # if True:
            # num_cpus = psutil.cpu_count()
            num_free_cpus = 0
            cpu_usage = psutil.cpu_percent(interval=1.0, percpu=True)
            for cpu in cpu_usage:
                # If the CPU usage is less than 60%, consider it as free
                if cpu < 60:
                    num_free_cpus += 1
            cpus_per_client = max(1, int(num_free_cpus/self.num_clients))
            logger.info(f"Number of CPUs per client: {cpus_per_client}")

            if self.device == "mps" or self.device == "cpu":
                # GPU support in flower for MPS is not available
                self.flwr_resources = {"num_cpus": cpus_per_client, "num_gpus": 0}
            else:
                num_gpus = len(gpu_ids)
                gpu_per_client = num_gpus/self.num_clients
                self.flwr_resources = {"num_cpus": cpus_per_client, "num_gpus": gpu_per_client}
        # assert (
        #     self.use_tensorboard or self.use_wandb or self.save_csv
        # ), f"Select any one logging method atleast to avoid losing results"


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

# @dataclass
# class OptimizerConfig:
#     name: str
#     lr: float
#     momentum: Optional[float] = None
#     weight_decay: Optional[float] = None
########## Training Configurations ##########

@dataclass
class TrainConfig:
    epochs: int = field()
    batch_size: int = field()
    eval_batch_size: int = field()
    optimizer: dict = field()
    loss_name: str = field()
    lr_scheduler: Optional[dict] = field()
    metric_cfg: MetricConfig = field()
    device: Optional[str] = field(default=None)

    def __post_init__(self):
        self.optim_partial: partial[Optimizer] = partial_initialize_module(OPTIMIZER_MAP, self.optimizer, ignore_args=["params"])
        if self.lr_scheduler:
            self.lr_scheduler_partial: partial[LRScheduler] = partial_initialize_module(LRSCHEDULER_MAP, self.lr_scheduler, ignore_args=["optimizer"])
        self.loss_fn: Module = LOSS_MAP[self.loss_name]()

        assert self.batch_size >= 1
       

########## Dataset configurataions ##########


# TODO: Add support for custom transforms
@dataclass
class TransformsConfig:
    resize: Optional[dict] = field(default_factory=dict)
    normalize: Optional[dict] = field(default_factory=dict)
    train: Optional[list] = field(default_factory=list)

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
    # split_conf: SplitConfig
    subsample: bool = False
    subsample_fraction: float = 0.0  # subsample the dataset with the given fraction

    def __post_init__(self):
        # assert self.test_fraction == Range(0.0, 1.0), f'Invalid value {self.test_fraction} for test fraction'
        self.data_path = to_absolute_path(self.data_path)



########## Model Configurations ##########
@dataclass
class ModelConfig:
    name: str
    hidden_channels: int


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
