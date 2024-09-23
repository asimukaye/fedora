from dataclasses import dataclass, field
import os
from typing import Optional
from torch.backends import mps
from torch import cuda
from fedora.utils import Range, get_free_gpus, arg_check, get_free_gpu
import logging
logger = logging.getLogger(__name__)

@dataclass
class MetricConfig:
    eval_metrics: list
    # fairness_metrics: list
    log_to_file: bool = False
    file_prefix: str = field(default="")
    cwd: Optional[str] = field(default=None)

    def __post_init__(self):
        self.cwd = os.getcwd() if self.cwd is None else self.cwd

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
