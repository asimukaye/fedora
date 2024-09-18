import os
import glob
import typing as t
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from copy import deepcopy
import time
import logging
import random
import numpy as np
import torch
from functools import partial
import wandb
from torch.nn import Module
from torch.backends import cudnn, mps
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from hydra.utils import instantiate
import flwr as fl

# from fedora.server.baseserver import BaseServer
from fedora.server.baseserver import BaseFlowerServer
from fedora.client.abcclient import simple_evaluator, simple_trainer
from fedora.client.baseclient import BaseFlowerClient

from fedora.dataloader import load_federated_dataset
from fedora.config import Config, SimConfig
from fedora.utils import log_tqdm, log_instance, generate_client_ids, get_time

from fedora.models.model import init_model
from fedora.results.resultmanager import ResultManager
from fedora.results.metricmanager import MetricManager
import fedora.customtypes as fT

from fedora.simulator.centralized import run_centralized_simulation
from fedora.simulator.flower import run_flower_simulation
from fedora.simulator.federated import run_federated_simulation, run_standalone_simulation

logger = logging.getLogger("Simulator")

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    logger.info(f"[SEED] Simulator global seed is set to: {seed}!")


def save_checkpoints(server: BaseFlowerServer, clients: dict[str, BaseFlowerClient]):
    if server:
        server.save_checkpoint()
    if clients:
        for client in clients.values():
            client.save_checkpoint()


def set_global_n_iters(cfg: Config, client_sets: fT.ClientDatasets_t):
    """Set the value of n_iters for the client"""
    # Setting the n_iters size
    total_size = sum([len(train_set) for train_set, _ in client_sets])

    per_client_set_size = total_size // cfg.simulator.num_clients
    cfg.client.cfg.n_iters = per_client_set_size // cfg.client.train_cfg.batch_size + (
        1 if per_client_set_size % cfg.client.train_cfg.batch_size else 0
    )
    logger.debug(f"[DATA_SPLIT] N iters: `{cfg.client.cfg.n_iters}`")
    logger.debug(f"[DATA_SPLIT] batch size : `{cfg.client.train_cfg.batch_size}`")
    return cfg


def init_dataset_and_model(cfg: Config) -> tuple[fT.ClientDatasets_t, Dataset, Module]:
    """Initialize the dataset and the model here"""
    # NOTE: THIS FUNCTION MODIFIES THE RANDOM NUMBER SEEDS

    client_sets, test_set, dataset_model_spec = load_federated_dataset(cfg.dataset)

    model_args = asdict(dataset_model_spec)
    model_instance = init_model(cfg.model, cfg.model_init, model_args)

    # Required for keeping iterations constant if batch size varies
    cfg = set_global_n_iters(cfg, client_sets)

    return client_sets, test_set, model_instance


class Simulator:
    """Simulator orchestrating the whole process of federated learning."""

    def __init__(self, cfg: Config):
        self.start_time = time.time()
        self._round: int = 0
        self.cfg: Config = cfg
        self.sim_cfg: SimConfig = cfg.simulator

        logger.info(f"[SIM MODE] : {self.sim_cfg.mode}")
        logger.info(f'[SERVER] : {self.cfg.server._target_.split(".")[-1]}')
        logger.info(f'[STRATEGY] : {self.cfg.strategy._target_.split(".")[-1]}')
        logger.info(f'[CLIENT] : {self.cfg.client._target_.split(".")[-1]}')

        logger.info(f"[NUM ROUNDS] : {self.sim_cfg.num_rounds}")
        set_seed(cfg.simulator.seed)

        self.client_sets, self.test_set, self.model_instance = init_dataset_and_model(
            cfg=cfg
        )

        # NOTE: cfg object conversion to asdict breaks when init fields are not set
        if self.sim_cfg.use_wandb:
            wandb.init(
                project="fedora",
                job_type=cfg.mode,
                config=asdict(cfg),
                resume=True,
                notes=cfg.desc,
            )

        # Till here can be factorized

        # TODO: consolidate checkpointing and resuming logic systematically
        self.is_resumed = False

        # server_ckpt, client_ckpts = self.find_checkpoint()

        # # FIXME: need to fix resume logic
        # if self.is_resumed:
        #     logger.info('------------ Resuming training ------------')
        #     self.load_state(server_ckpt, client_ckpts)
        logger.debug(f"Init time: {time.time() - self.start_time} seconds")

    def run_simulation(self):
        match self.sim_cfg.mode:
            case "federated":
                run_federated_simulation(
                    self.cfg, self.client_sets, self.test_set, self.model_instance
                )
            case "standalone":
                run_standalone_simulation(
                    self.cfg, self.client_sets, self.test_set, self.model_instance
                )
            case "centralized":
                run_centralized_simulation(
                    self.cfg, self.client_sets, self.test_set, self.model_instance
                )
            case "flower":
                run_flower_simulation(
                    self.cfg, self.client_sets, self.test_set, self.model_instance
                )
            case _:
                raise AssertionError(f"Mode: {self.sim_cfg.mode} is not implemented")

    def close(self):
        total_time = time.time() - self.start_time
        logger.info(f"Total runtime: {total_time} seconds")
        logger.info(f"Per loop runtime: {total_time/self.sim_cfg.num_rounds} seconds")

        logger.info("------------ Simulation completed ------------")

        # FIXME: Post process is broken
        # post_process(self.cfg, final_result, total_time=total_time)

        logger.info("Closing Feduciary Simulator")
