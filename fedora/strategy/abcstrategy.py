from abc import ABC, abstractmethod
import typing as t
from dataclasses import dataclass, field

import torch
from torch.nn import Module, ParameterDict, Parameter
from torch import Tensor
import torch.optim 

import fedora.customtypes as fT
from fedora.results.resultmanager import ResultManager

from fedora.config.strategyconf import StrategyConfig

@dataclass
class StrategyIns(ABC):
    '''Define the inputs required from each client for aggregation of your strategy'''
    client_params: fT.ActorParams_t

AllClientIns_t = dict[str, StrategyIns]

@dataclass
class StrategyOuts(ABC):
    '''Define the outputs of aggregation of your strategy'''
    server_params: fT.ActorParams_t


class ABCStrategy(ABC):
    # A strategy is a server strategy used to aggregate the server weights
    # It is based on the torch optimizer class to support direct integration with Pytorch LR schedulers.
    
    @abstractmethod
    def __init__(self,  model: Module,
                 cfg: StrategyConfig,
                 res_man:ResultManager,
                   **kwargs) -> None:
        pass
        # self.cfg = cfg
        # defaults = dict(lr=client_lr)

        # self.server_optimizer = torch.optim.Optimizer(model.parameters(), defaults)
        # assert len(self.server_optimizer.param_groups) == 1, f'Multi param group yet to be implemented'
    
    @classmethod
    @abstractmethod
    def receive_strategy(cls, ins: fT.ClientResults_t) -> StrategyIns:
        '''Describe how to process incoming data from client such as decompression/ decryption logic'''
        pass

    @classmethod
    @abstractmethod
    def send_strategy(cls, ids: fT.ClientIds_t) -> fT.ClientIns_t:
        '''Describe how to process the outgoing data to the client such as compression/ encryption logic'''
        pass

    @classmethod
    @abstractmethod
    def client_receive_strategy(cls, ins: fT.ClientIns) -> StrategyOuts:
        '''Describe how the client should unpack/ decompress/decrypt the incoming data'''
        pass

    @classmethod
    @abstractmethod
    def client_send_strategy(cls, ins: StrategyIns) -> fT.ClientResult:
        '''Describe how the client should pack its data for the server'''
        pass

    @abstractmethod
    def train_selection(self, in_ids: fT.ClientIds_t, **kwargs) -> fT.ClientIds_t:
        ''' Client selection criterion for training'''
        pass

    @abstractmethod
    def eval_selection(self, in_ids: fT.ClientIds_t, **kwargs) -> fT.ClientIds_t:
        ''' Client selection criterion for evaluation'''
        pass

    @abstractmethod
    def aggregate(self, *args, **kwargs) -> StrategyOuts:
        '''Client aggregation and weighting strategy'''
        pass
    
    