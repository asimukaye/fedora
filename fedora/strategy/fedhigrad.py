from collections import defaultdict
import typing as t
from typing import Any

# from dataclasses import dataclass, field
from functools import partial
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
from fedora.strategy.abcstrategy import *
from fedora.strategy.fedavg import (
    passthrough_communication,
    random_client_selection,
    ClientInProto,
)
import fedora.customtypes as fT
from flwr.server.strategy import FedAvg


@dataclass
class FedHiGradCfgProtocol(t.Protocol):
    """Protocol for base strategy config"""

    train_fraction: float
    eval_fraction: float
    delta_normalize: bool
    gamma: float


@dataclass
class FedHiGradIns(StrategyIns):
    data_size: int


AllIns_t = dict[str, FedHiGradIns]


@dataclass
class FedHiGradOuts(StrategyOuts):
    pass


class FedHiGradStrategy(ABCStrategy):
    def __init__(
        self,
        model: Module,
        cfg: FedHiGradCfgProtocol,
        res_man: ResultManager,
    ) -> None:
        # super().__init__(model, client_lr, cfg)
        self.cfg = cfg
        self.res_man = res_man

        self._server_params: dict[str, Parameter] = model.state_dict()
        self._server_deltas: dict[str, Tensor] = {
            param: torch.tensor(0.0) for param in self._server_params.keys()
        }

        if self.cfg.delta_normalize:
            self._update_fn = partial(
                gradient_average_with_delta_normalize, gamma=self.cfg.gamma
            )
        else:
            self._update_fn = gradient_average_update

        self._client_params: fT.ClientParams_t = defaultdict(dict)
        self._client_wts: dict[str, float] = defaultdict()

    def send_strategy(self, ids: fT.ClientIds_t) -> fT.ClientIns_t:
        return {
            cid: fT.ClientIns(params=self._server_params, metadata={}) for cid in ids
        }

    def receive_strategy(self, ins: fT.ClientResults_t) -> AllClientIns_t:
        return {
            cid: FedHiGradIns(cl_res.params, cl_res.result.size)
            for cid, cl_res in ins.items()
        }

    @classmethod
    def client_receive_strategy(cls, ins: fT.ClientIns) -> ClientInProto:
        return ClientInProto(in_params=ins.params)

    @classmethod
    def client_send_strategy(
        cls, ins: FedHiGradIns, result: fT.Result
    ) -> fT.ClientResult:
        return fT.ClientResult(ins.client_params, result)

    def train_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)

    def eval_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)

    def aggregate(self, strategy_ins: AllIns_t) -> FedHiGradOuts:
        # calculate client weights according to sample sizes
        total_size = sum([strat_in.data_size for strat_in in strategy_ins.values()])

        for cid, strat_in in strategy_ins.items():
            self._client_wts[cid] = float(strat_in.data_size / total_size)

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}

        self._server_params, self._server_deltas = self._update_fn(
            self._server_params, _client_params, self._client_wts
        )

        # for cid in client_ids:
        self.res_man.log_general_metric(
            self._client_wts,
            phase="post_agg",
            actor="server",
            metric_name="client_weights",
        )
        self.res_man.log_parameters(
            self._server_params, phase="post_agg", actor="server"
        )

        return FedHiGradOuts(server_params=self._server_params)
