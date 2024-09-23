from collections import OrderedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import functools
from torch.utils.data import DataLoader
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from copy import deepcopy
import torch
from torch import Tensor
import logging
from fedora.results.metricmanager import MetricManager
from fedora.utils import log_tqdm
from tqdm import tqdm
from fedora.config.trainconf import TrainConfig
from fedora.config.clientconf import ClientConfig

import fedora.customtypes as fT
logger = logging.getLogger(__name__)

def simple_evaluator(model: Module,
                      dataloader: DataLoader,
                      cfg: TrainConfig,
                      mm: MetricManager,
                      round: int) -> fT.Result:

    mm._round = round
    model.eval()
    model.to(cfg.device)
    criterion = cfg.criterion

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss: Tensor = criterion(outputs, targets) #type: ignore
        mm.track(loss.item(), outputs, targets)
    else:
        result = mm.aggregate(len(dataloader.dataset), -1) # type: ignore
        mm.flush()
    return result


def simple_trainer(model: Module,
                    dataloader: DataLoader,
                    cfg: TrainConfig,
                    mm: MetricManager,
                    round: int) -> fT.Result:

    mm._round = round
    model.train()
    # model.float()
    model.to(cfg.device)
    criterion  = cfg.criterion
    optim_partial: functools.partial = cfg.optimizer #type: ignore
    optimizer: Optimizer = optim_partial(model.parameters(), lr=cfg.lr)

    for inputs, targets in tqdm(dataloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        outputs = model(inputs)
        loss: Tensor = criterion(outputs, targets) #type: ignore
        loss.backward()
        optimizer.step()
        mm.track(loss.item(), outputs, targets)
    else:
        result = mm.aggregate(len(dataloader.dataset), -1) # type: ignore
        mm.flush()
    return result

class ABCClient(ABC):
    """Class for client object having its own (private) data and resources to train a model.
    """

    def __init__(self,
                 cfg: ClientConfig,
                 train_cfg: TrainConfig,
                 client_id: str,
                 dataset: tuple,
                 model: Module): 
        
        self._cid = client_id 
        self._model = model


        # #NOTE: IMPORTANT: Make sure to deepcopy the config in every child class
        self.cfg = deepcopy(cfg)
        self.train_cfg = deepcopy(train_cfg)

        self.training_set = dataset[0]
        self.test_set = dataset[1]

    
    @abstractmethod
    def _create_dataloader(self, dataset, shuffle:bool)->DataLoader:
        pass
    
    @dataclass
    class ClientInProtocol:
        in_params: fT.ActorParams_t

    @abstractmethod
    def reset_model(self) -> None:
        '''Define how to reset the model'''
        

    @abstractmethod
    def download(self, round:int, model_dict: dict[str, Parameter]):
        '''Download the model from the server'''
        pass

    @abstractmethod
    def upload(self) -> OrderedDict:
        '''Upload the model back to the server'''
        pass
 
    @abstractmethod
    def unpack_train_input(self, client_ins: fT.ClientIns) -> ClientInProtocol:
        pass
    
    @abstractmethod
    def pack_train_result(self, result: fT.Result) -> fT.ClientResult:
        pass
    @abstractmethod
    def train(self, return_model=False):
        '''How the model should train'''

    @abstractmethod
    def evaluate(self):
        '''How to evaluate the model'''
        pass

    @abstractmethod
    def save_checkpoint(self, epoch=0):
        '''Define how to save a checkpoint'''
        pass
        
    @abstractmethod
    def load_checkpoint(self, ckpt_path:str):
        '''Define how to load a checkpoint'''


    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self._cid:03} >'
    

