from functools import partial
import logging
from copy import deepcopy
import os
import json
import torch
from torch.nn import Module
from torch.backends import mps
from torch.utils.data import Dataset
import flwr as fl
from hydra.utils import instantiate

from fedora.utils import generate_client_ids
from fedora.config.masterconf import Config
from fedora.simulator.utils import make_checkpoint_dirs
from fedora.results.resultmanager import ResultManager
from fedora.client.baseclient import BaseFlowerClient
from fedora.server.baseserver import BaseFlowerServer
import fedora.customtypes as fT


logger = logging.getLogger(__name__)

def run_flower_simulation(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Dataset,
    model: Module,
):

    # all_client_ids = generate_client_ids(cfg.simulator.num_clients)
    # make_checkpoint_dirs(has_server=True, client_ids=all_client_ids)
    clients: dict[str, BaseFlowerClient] = dict()

    # client_datasets_map = {}
    # for cid, dataset in zip(all_client_ids, client_datasets):
    #     client_datasets_map[cid] = dataset

    if torch.cuda.is_available():
        # Device has to be cuda 0 as flower creates its own device namespace while running
        cfg.client.train_cfg.device = "cuda:0"
        cfg.server.train_cfg.device = "cuda:0"
    # elif mps.is_available():
    #     # MPS support is still not mature in PyTorch and Flower
    #     cfg.client.train_cfg.device = "mps"
    #     cfg.server.train_cfg.device = "mps"
    else:
        cfg.client.train_cfg.device = "cpu"
        cfg.server.train_cfg.device = "cpu"

    result_manager = ResultManager(cfg.simulator, logger=logger)

    strategy = instantiate(cfg.strategy, model=model, res_man=result_manager)

    flwr_strategy_partial = instantiate(cfg.server)
    flwr_strategy: BaseFlowerServer = flwr_strategy_partial(
        model=model,
        dataset=server_dataset,
        clients=clients,
        strategy=strategy,
        result_manager=result_manager,
    )

    # def _client_fn(cid: str):
    #     client_partial: partial = instantiate(cfg.client)
    #     _model = deepcopy(model)
    #     _datasets = client_datasets_map[cid]

    #     return client_partial(client_id=cid, dataset=_datasets, model=_model)
    
    def _client_fn(context: fl.common.Context):
        logger.info(context.node_id)
        logger.info(context.node_config)

        partition_id = int(context.node_config["partition_id"])
        with open(f"client_{partition_id}.json", "w") as f:
            json.dump(context.__dict__, f)
            
        client_partial: partial = instantiate(cfg.client)
    
        # cid = str(context.node_id)
        partition_id = int(context.node_config["partition_id"])
        with open(f"client_{partition_id}.json", "w") as f:
            json.dump(context.__dict__, f)
        cid = str(f'{partition_id:04}')
        _model = deepcopy(model)
        # _datasets = client_datasets_map[cid]
        _datasets = client_datasets[partition_id]

        return client_partial(client_id=cid, dataset=_datasets, model=_model)

    # Flower simulation arguments
    # runtime_env = {"env_vars": {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids))}}
    runtime_env = {}
    runtime_env["working_dir"] = os.getcwd()

    fl.simulation.start_simulation(
        strategy=flwr_strategy,
        client_fn=_client_fn,
        # clients_ids=all_client_ids,
        num_clients=cfg.simulator.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.simulator.num_rounds),
        ray_init_args={"runtime_env": runtime_env},
        client_resources=cfg.simulator.flwr_resources,
    )

def run_flower_standalone_simulation(
    cfg: Config,
    client_datasets: fT.ClientDatasets_t,
    server_dataset: Dataset,
    model: Module,
):
    pass

