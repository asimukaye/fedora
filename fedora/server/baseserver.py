from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from functools import partial
import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

import flwr as fl
import flwr.server.strategy as fl_strat
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module, Parameter
from torch import Tensor


from torch.optim.lr_scheduler import LRScheduler

from fedora.client.baseclient import BaseFlowerClient, simple_evaluator
from fedora.config.commonconf import TrainConfig, ServerConfig
from fedora.results.resultmanager import ResultManager
from fedora.results.metricmanager import MetricManager

from fedora.utils import (
    log_tqdm,
    unroll_param_keys,
    roll_param_keys,
    get_model_as_ndarray,
    convert_param_dict_to_ndarray,
    convert_ndarrays_to_param_dict,
    get_time,
)

import fedora.customtypes as fT

from fedora.server.abcserver import ABCServer

# from fedora.config import *
from fedora.strategy.fedavg import FedAvgStrategy

logger = logging.getLogger(__name__)


def _nest_dict_rec(k: str, v, out: dict):
    k, *rest = k.split(".", 1)
    if rest:
        _nest_dict_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def nest_dict(flat: dict) -> dict:
    result = {}
    for k, v in flat.items():
        _nest_dict_rec(k, v, result)
    return result


def flatten_dict(nested: dict) -> dict:
    return pd.json_normalize(nested, sep=".").to_dict("records")[0]


# Flower conversion adapters functions


def flower_metrics_to_results(flwr_res: FitRes | EvaluateRes) -> fT.Result:
    # print(flwr_res.metrics.keys())
    nested = nest_dict(flwr_res.metrics)
    # print("Nested: ", nested)
    # The loss value is part of metrics and thus ignored
    return fT.Result(
        actor=nested["actor"],
        event=nested["event"],
        phase=nested["phase"],
        _round=nested["_round"],
        metadata=nested["metadata"],
        metrics=nested["metrics"],
        size=flwr_res.num_examples,
    )


def flower_eval_results_adapter(
    results: List[Tuple[ClientProxy, EvaluateRes]]
) -> dict[str, fT.Result]:
    client_results = {}
    for client, result in results:
        client_results[client.cid] = flower_metrics_to_results(result)
    return client_results


def flower_train_results_adapter(
    results: List[Tuple[ClientProxy, FitRes]]
) -> fT.ClientResults_t:
    client_results = {}
    for client, result in results:
        nd_params = fl.common.parameters_to_ndarrays(result.parameters)
        param_keys = unroll_param_keys(result.metrics)  # type: ignore
        client_params = convert_ndarrays_to_param_dict(param_keys, nd_params)
        client_result = flower_metrics_to_results(result)
        client_results[client.cid] = fT.ClientResult(
            params=client_params, result=client_result
        )

    return client_results


def client_in_to_flower_fitin(client_in: fT.ClientIns) -> FitIns:
    config = client_in.metadata
    config["_round"] = client_in._round
    config["_request"] = client_in.request

    config.update(roll_param_keys(list(client_in.params.keys())))

    nd_param = convert_param_dict_to_ndarray(client_in.params)
    # nd_param = convert_param_list_to_ndarray(client_in.params)

    flower_param = fl.common.ndarrays_to_parameters(nd_param)
    return FitIns(parameters=flower_param, config=config)


class BaseFlowerServer(ABCServer, fl_strat.Strategy):
    """Central server orchestrating the whole process of federated learning."""

    name: str = "BaseFlowerServer"

    # NOTE: It is must to redefine the init function for child classes with a call to super.__init__()
    def __init__(
        self,
        clients: dict[str, BaseFlowerClient],
        model: Module,
        cfg: ServerConfig,
        strategy: FedAvgStrategy,
        train_cfg: TrainConfig,
        dataset: Dataset,
        result_manager: ResultManager,
    ):
        self.model = model
        self.clients = clients
        self.num_clients = len(self.clients)
        self.strategy = strategy
        self.cfg = cfg
        self.train_cfg = train_cfg
        self.server_dataset = dataset
        self.metric_manager = MetricManager(train_cfg.metric_cfg, 0, "server")
        if result_manager:
            self.result_manager = result_manager

        self.param_keys = list(self.model.state_dict().keys())
        defaults = dict(lr=train_cfg.optimizer["lr"])
        self._optimizer = torch.optim.Optimizer(
            self.model.parameters(), defaults=defaults
        )

        # lrs_partial = train_cfg.lr_scheduler
        lrs_partial = train_cfg.lr_scheduler_partial
        self.lr_scheduler: LRScheduler = lrs_partial(self._optimizer)

    @classmethod
    def send_model_to_client(
        cls,
        client_ins: fT.ClientIns,
        client: BaseFlowerClient,
        request_type: fT.RequestType,
        _round=-1,
    ) -> fT.RequestOutcome:
        ''' Send the model to the client and get the outcome of the request'''
        # It calls the clients download function directly. 
        # In real world, the server should not have access to the client's download function. This should instead be a network call to the client
        client_ins.request = request_type
        client_ins._round = _round
        out = client.download(client_ins)
        return out

    def _broadcast_models(
        self,
        ids: list[str],
        clients_ins: dict[str, fT.ClientIns],
        request_type=fT.RequestType.NULL,
    ) -> fT.RequestOutcomes_t:
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """

        # HACK: does lr scheduling need to be done for select ids ??
        if self.lr_scheduler:
            current_lr = self.lr_scheduler.get_last_lr()[-1]
            [client.set_lr(current_lr) for client in self.clients.values()]

        # Uncomment this when adding GPU support to server
        # self.model.to('cpu')
        results = {}

        for idx in log_tqdm(ids, desc=f"broadcasting models: ", logger=logger):
            results[idx] = self.send_model_to_client(
                clients_ins[idx], self.clients[idx], request_type, self._round
            )

        return results

    # TODO: Support custom client wise ins
    def _collect_results(
        self, ids: list[str], request_type=fT.RequestType.NULL
    ) -> fT.ClientResults_t:
        """broadcast the global model to all the clients.
        Args:
            ids (_type_): client ids
        """

        # Uncomment this when adding GPU support to server
        # self.model.to('cpu')
        results: dict[str, fT.ClientResult] = {}
        for idx in log_tqdm(ids, desc=f"collecting results: ", logger=logger):
            results[idx] = self.clients[idx].upload(request_type)
            # self.result_manager.json_dump(results[idx].result, f'client_{idx}_res', 'post_agg', 'flowerserver', request_type.name)
            # if request_type == fT.RequestType.TRAIN:
            #     # ic(idx)
            #     # ic(results[idx].params[list(results[idx].params.keys())[0]][:5])
            #     params = self.result_manager.log_parameters(results[idx].params, f'client_{idx}_params', 'post_agg', 'flowerserver', request_type.name, verbose=True)
            #     self.result_manager.json_dump(params, f'client_{idx}_params', 'post_agg', 'flowerserver', request_type.name)
        return results

    @torch.no_grad()
    def server_eval(self):
        """Evaluate the global model located at the server."""
        # FIXME: Formalize phase argument passing
        server_loader = DataLoader(
            dataset=self.server_dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
        )
        # log result
        result = simple_evaluator(
            self.model, server_loader, self.train_cfg, self.metric_manager, self._round
        )
        self.result_manager.log_general_result(
            result, phase="post_agg", actor="server", event="central_eval"
        )
        return result

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # epoch = checkpoint['epoch']
        self._round = checkpoint["round"]
        # Find a way to avoid this result manager round bug repeatedly
        self.result_manager._round = checkpoint["round"]
        self.result_manager._step_counter = checkpoint["step_count"]

        # loss = checkpoint['loss']

    def save_checkpoint(self, root_dir = '.'):
        torch.save(
            {
                "round": self._round,
                "step_count" : self.result_manager._step_counter,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            f"{root_dir}/server_ckpts/server_ckpt_{self._round:003}.pt",
        )

    def finalize(self, root_dir='.') -> None:
        # save checkpoint
        torch.save(self.model.state_dict(), f"{root_dir}/final_model.pt")
        # return all_results

    def update(self, avlble_cids: fT.ClientIds_t) -> fT.ClientIds_t:
        # TODO: Ideally, the clients keep running their training and server should only initiate a downlink transfer request. For synced federation, the server needs to wait on clients to finish their tasks before proceeding

        """Update the global model through federated learning."""
        # randomly select clients

        # broadcast the current model at the server to selected clients
        train_ids = self.strategy.train_selection(in_ids=avlble_cids)
        clients_ins = self.strategy.send_strategy(train_ids)
        outcomes = self._broadcast_models(train_ids, clients_ins, fT.RequestType.TRAIN)

        collect_ids = [
            cid for cid, out in outcomes.items() if out == fT.RequestOutcome.COMPLETE
        ]

        train_results = self._collect_results(collect_ids, fT.RequestType.TRAIN)

        strategy_ins = self.strategy.receive_strategy(train_results)
        # self.result_manager.json_dump(strategy_ins, 'strategy_ins', 'pre_agg', 'flowerserver')
        strategy_outs = self.strategy.aggregate(strategy_ins)
        self.model.load_state_dict(strategy_outs.server_params)

        # TODO: Change the function signature of log_client_results to accept client results with optional paramater logging..
        to_log = {cid: res.result for cid, res in train_results.items()}
        self.result_manager.log_clients_result(
            to_log, phase="pre_agg", event="local_train"
        )
        return collect_ids

    def local_eval(self, avlble_cids: fT.ClientIds_t):

        eval_ids = self.strategy.eval_selection(in_ids=avlble_cids)
        eval_ins = self.strategy.send_strategy(eval_ids)
        outcomes = self._broadcast_models(eval_ids, eval_ins, fT.RequestType.EVAL)
        collect_ids = [
            cid for cid, out in outcomes.items() if out == fT.RequestOutcome.COMPLETE
        ]

        eval_results = self._collect_results(collect_ids, fT.RequestType.EVAL)

        to_log = {cid: res.result for cid, res in eval_results.items()}

        self.result_manager.log_clients_result(
            to_log, event="local_eval", phase="post_agg"
        )
        return collect_ids

    # FLOWER FUNCTION OVERLOADS


    # MAPPING FLOWER STRATEGY FUNCTIONS TO SERVER STRATEGY FUNCTIONS

    # _broadcast_models + update -> configure_fit
    # _collect_results + update -> aggregate_fit
    # _broadcast_models + local_eval -> configure_evaluate
    # _collect_results + local_eval -> aggregate_evaluate
    # server_eval -> evaluate

    # NOTE: The following functions are overloads of the flwr server strategy functions
    # The functions are overloaded to provide a more customized and flexible interface to the server
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""

        ndarrays = get_model_as_ndarray(self.model)
        return fl.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Create custom configs
        self._round = server_round - 1

        fit_configurations = []

        client_proxies = client_manager.all()

        out_ids = self.strategy.train_selection(list(client_proxies.keys()))
        clients_ins = self.strategy.send_strategy(out_ids)

        # FIXME: dynamic client configurations
        # client_config = {"lr": 0.001}

        for out_id in out_ids:
            client = client_proxies[out_id]
            cl_in = clients_ins[out_id]
            cl_in._round = self._round
            cl_in.request = fT.RequestType.TRAIN

            fit_in = client_in_to_flower_fitin(cl_in)
            # nd_param = convert_param_dict_to_ndarray(cl_in.params)
            # flower_param = fl.common.ndarrays_to_parameters(nd_param)
            # cl_config = cl_in.metadata
            # cl_config['_round'] = self._round

            fit_configurations.append((client, fit_in))
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes] | BaseException],
    ) -> Tuple[Parameters | None, Dict[str, Scalar]]:

        # with get_time():
        train_results = flower_train_results_adapter(results)

        strategy_ins = self.strategy.receive_strategy(train_results)
        strategy_outs = self.strategy.aggregate(strategy_ins)

        # TODO: There might be a need for a strategy out adapter
        to_log = {cid: res.result for cid, res in train_results.items()}
        self.result_manager.log_clients_result(
            to_log, phase="pre_agg", event="local_train"
        )
        # Validate the need for this.
        # with get_time():
        self.model.load_state_dict(strategy_outs.server_params)
        param_ndarrays = convert_param_dict_to_ndarray(strategy_outs.server_params)

        parameters_aggregated = fl.common.ndarrays_to_parameters(param_ndarrays)
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:

        # NOTE: Sample all is selected for now. Remember while simulating availability
        client_proxies = client_manager.all()

        out_ids = self.strategy.eval_selection(list(client_proxies.keys()))
        eval_ins = self.strategy.send_strategy(out_ids)

        eval_configurations = []

        for out_id in out_ids:
            client = client_proxies[out_id]
            cl_in = eval_ins[out_id]
            nd_param = convert_param_dict_to_ndarray(cl_in.params)
            flower_param = fl.common.ndarrays_to_parameters(nd_param)
            cl_config = cl_in.metadata
            cl_config["_round"] = self._round

            eval_configurations.append((client, EvaluateIns(flower_param, cl_config)))

        return eval_configurations

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float, Dict[str, Scalar]]:

        client_results = flower_eval_results_adapter(results)
        clien_result_stats = self.result_manager.log_clients_result(
            result=client_results, phase="post_agg", event="local_eval"
        )
        self.result_manager.flush_and_update_round(server_round - 1)
        # self._round = server_round-1

        loss_agg = clien_result_stats.stats["loss"].mean

        metrics_agg = {k: v.mean for k, v in clien_result_stats.stats.items()}

        return loss_agg, metrics_agg  # type: ignore

    def evaluate(self, server_round: int, parameters: Parameters) -> None:
        eval_res = self.server_eval()
        loss_agg = eval_res.metrics["loss"]
        # metrics_agg =  {k: v for k, v in eval_res.metrics.items()}
        metrics_agg = eval_res.metrics

        return loss_agg, metrics_agg  # type: ignore
