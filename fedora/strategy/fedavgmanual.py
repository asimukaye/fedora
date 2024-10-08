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

from fedora.strategy.fedavg import passthrough_communication, random_client_selection, weighted_parameter_averaging

from fedora.utils import generate_client_ids

from fedora.results.resultmanager import ResultManager

@dataclass
class StrategyCfgProtocol(t.Protocol):
    '''Protocol for fedavg manual strategy config'''
    train_fraction: float
    eval_fraction: float
    weights: list[float]
    num_clients: int

@dataclass
class BaseInsProtocol(t.Protocol):
    client_params: fT.ActorParams_t
    data_size: int

@dataclass
class BaseIns(StrategyIns):
    client_params: fT.ActorParams_t
    data_size: int

AllIns_t = dict[str, BaseIns]

@dataclass
class BaseOuts(StrategyOuts):
    server_params: fT.ActorParams_t


class FedavgManual(ABCStrategy):
    def __init__(self,
                 model: Module,
                 cfg: StrategyCfgProtocol,
                 res_man: ResultManager) -> None:
        # super().__init__(model, cfg)
        self.cfg = cfg
        # * Server params is not required to be stored as a state fir
        self._server_params: dict[str, Parameter] = model.state_dict()
        self._param_keys = self._server_params.keys()

        # self._client_params: ClientParams_t = defaultdict(dict)
        client_ids = generate_client_ids(cfg.num_clients)
        self._client_weights: dict[str, float] = {cid: wt for cid, wt in zip(client_ids, cfg.weights)}

        self._client_ins: dict[str, float] = defaultdict()


    @classmethod
    def client_receive_strategy(cls, ins: fT.ClientIns) -> BaseOuts:
        base_outs = BaseOuts(
            server_params=ins.params,
        )
        return base_outs
    
    @classmethod
    def client_send_strategy(cls, ins: BaseInsProtocol, res: fT.Result) -> fT.ClientResult:
        result =res
        result.size = ins.data_size
        return fT.ClientResult(ins.client_params, res) 

    def receive_strategy(self, results: fT.ClientResults_t) -> AllIns_t:
        client_params = {}
        client_data_sizes = {}
        strat_ins = {}
        for cid, res in results.items():
            strat_ins[cid] = BaseIns(client_params=res.params, data_size=res.result.size)
        # strat_ins = BaseIns(client_params=client_params,
        #                    data_sizes=client_data_sizes)
        return strat_ins
    
    def send_strategy(self, ids: fT.ClientIds_t) -> fT.ClientIns_t:
        '''Simple send the same model to all clients strategy'''
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fT.ClientIns(
                params=self._server_params,
                metadata={}
            )
        return clients_ins
    
    def train_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)
    
    def eval_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)


    def aggregate(self, strategy_ins: AllIns_t) -> BaseOuts:
        # calculate client weights according to config
        # ic(self._client_weights)
 
        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        
        self._server_params = weighted_parameter_averaging(self._param_keys, _client_params, self._client_weights)

        outs = BaseOuts(server_params=self._server_params)

        return outs
    
