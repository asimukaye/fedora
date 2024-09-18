from collections import defaultdict
import typing as t
import random
import torch

from torch.nn import Module, Parameter
from torch import Tensor

import torch.optim
import fedora.customtypes as fT

# from fedora.strategy.abcstrategy import ABCStrategy
from fedora.strategy.abcstrategy import *

# from fedora.strategy.abcstrategy import StrategyIns
from fedora.results.resultmanager import ResultManager

# Type declarations
ScalarWeights_t = dict[str, float]
TensorWeights_t = dict[str, Tensor]


def passthrough_communication(ins: t.Any) -> t.Any:
    """Simple passthrough communication logic"""
    return ins


def weighted_parameter_averaging(
    param_keys: t.Iterable, in_params: fT.ClientParams_t, weights: dict[str, float]
) -> fT.ActorParams_t:

    out_params = {}
    for key in param_keys:
        temp_parameter = torch.Tensor()

        for cid, client_param in in_params.items():
            if temp_parameter.numel() == 0:
                temp_parameter = weights[cid] * client_param[key].data
            else:
                temp_parameter.data.add_(weights[cid] * client_param[key].data)

        out_params[key] = temp_parameter
    return out_params


def random_client_selection(
    sampling_fraction: float, cids: list[str]
) -> fT.ClientIds_t:

    num_clients = len(cids)
    num_sampled_clients = max(int(sampling_fraction * num_clients), 1)
    sampled_client_ids = sorted(random.sample(cids, num_sampled_clients))

    return sampled_client_ids


def select_all_clients(cids: fT.ClientIds_t) -> fT.ClientIds_t:
    return cids


# @dataclass
class StrategyCfgProtocol(t.Protocol):
    """Protocol for FedAvg strategy config"""

    train_fraction: float
    eval_fraction: float


# @dataclass
class FedAvgInsProtocol(t.Protocol):
    '''Protocol for FedAvg strategy input
        Defined for ease of use on the client side without having to import the strategy
    '''
    client_params: fT.ActorParams_t
    data_size: int


@dataclass
class FedAvgIns(StrategyIns):
    client_params: fT.ActorParams_t
    data_size: int


AllIns_t = dict[str, FedAvgIns]


@dataclass
class FedAvgOuts(StrategyOuts):
    server_params: fT.ActorParams_t


@dataclass
class ClientInProto:
    in_params: fT.ActorParams_t


class FedAvgStrategy(ABCStrategy):
    def __init__(
        self, model: Module, cfg: StrategyCfgProtocol, res_man: ResultManager
    ) -> None:
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        self._param_keys = self._server_params.keys()

        # self._client_params: ClientParams_t = defaultdict(dict)
        self._client_weights: dict[str, float] = defaultdict()
        self._client_ins: dict[str, float] = defaultdict()

    @classmethod
    def client_receive_strategy(cls, ins: fT.ClientIns) -> ClientInProto:
        return ClientInProto(in_params=ins.params)

    @classmethod
    def client_send_strategy(
        cls, ins: FedAvgInsProtocol, res: fT.Result
    ) -> fT.ClientResult:
        result = res
        result.size = ins.data_size
        return fT.ClientResult(ins.client_params, res)

    def receive_strategy(self, results: fT.ClientResults_t) -> AllIns_t:

        strat_ins = {}
        for cid, res in results.items():
            strat_ins[cid] = FedAvgIns(
                client_params=res.params, data_size=res.result.size
            )
        # strat_ins = FedAvgIns(client_params=client_params,
        #                    data_sizes=client_data_sizes)
        return strat_ins

    def send_strategy(self, ids: fT.ClientIds_t) -> fT.ClientIns_t:
        """Simple send the same model to all clients strategy"""
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fT.ClientIns(params=self._server_params, metadata={})
        return clients_ins

    def train_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)

    def eval_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)

    def aggregate(self, strategy_ins: AllIns_t) -> FedAvgOuts:
        # calculate client weights according to sample sizes
        total_size = sum([strat_in.data_size for strat_in in strategy_ins.values()])

        for cid, strat_in in strategy_ins.items():
            self._client_weights[cid] = float(strat_in.data_size / total_size)

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}

        self._server_params = weighted_parameter_averaging(
            self._param_keys, _client_params, self._client_weights
        )

        outs = FedAvgOuts(server_params=self._server_params)

        return outs


# alias for the FedAvgStrategy
FedAvgStrategy = FedAvgStrategy
