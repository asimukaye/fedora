import logging
import math
import typing as t
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import torch
from torch.nn import Module
import numpy as np

from torch.linalg import norm

# from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CosineSimilarity, Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch.nn.functional import cosine_similarity, tanh

# from fedora.config.clientconf import ClientConfig
from fedora.config.strategyconf import RFFLConfig
from fedora.results.resultmanager import ResultManager

# from fedora.strategy import *
from fedora.strategy.abcstrategy import (
    ABCStrategy,
    StrategyIns,
    StrategyOuts,
    AllClientIns_t,
)
from fedora.strategy.fedavg import random_client_selection, ClientInProto
import fedora.customtypes as fT
from fedora.utils import generate_client_ids

# from fedora.strategy.fedopt import (
#     compute_server_delta_w_normalize,
#     compute_server_delta,
#     add_param_deltas,
# )

# from fedora.strategy.fedstdevstrategy import normalize_coefficients

logger = logging.getLogger(__name__)

# NOTE: THIS ALGORITHM HAS STABILITY ISSUES IN CASE THE LEARNING DOES NOT CONVERGE. If the RFFL values stay consistently negative, it would lead to the weights turning negative rendering a lot of subsequent steps invalid. This is a known issue with the algorithm and is not addressed in the paper.


@dataclass
class RFFLCfgProtocol(t.Protocol):
    train_fraction: float
    eval_fraction: float
    num_clients: int
    lr: float
    gamma: float
    alpha: float
    beta: float
    delta_normalize: bool
    sparsify_gradients: bool
    shard_sizes: list[int]


@dataclass
class RFFLIns(StrategyIns):
    pass


AllIns_t = dict[str, RFFLIns]


@dataclass
class RFFLOuts(StrategyOuts):
    client_params: fT.ClientParams_t


def add_update_to_params(
    params: fT.ActorParams_t, deltas: list[torch.Tensor]
) -> fT.ActorParams_t:
    """Add deltas to the server parameters"""
    for param, delta in zip(params.values(), deltas):
        param.data.add_(delta)
    return params


def flatten(grad_update: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([update.data.view(-1) for update in grad_update])


def unflatten(
    flattened: torch.Tensor, normal_shape: list[torch.Tensor]
) -> list[torch.Tensor]:
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size()))
        flattened = flattened[n_params:]
    return grad_update


def add_gradient_updates(
    grad_update_1: list[torch.Tensor],
    grad_update_2: list[torch.Tensor],
    weight: torch.Tensor,
):
    assert len(grad_update_1) == len(
        grad_update_2
    ), "Lengths of the two grad_updates not equal"

    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight


def mask_grad_update_by_magnitude(grad_update: list[torch.Tensor], mask_constant):
    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update


def mask_grad_update_by_order(
    grad_update: list[torch.Tensor], mask_order=None, mask_percentile=None, mode="all"
):

    if mode == "all":
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat(
            [update.data.view(-1).abs() for update in grad_update]
        )
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)

        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float("inf"))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)  # type: ignore
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == "layer":  # layer wise largest-values criterion
        grad_update = deepcopy(grad_update)

        mask_percentile = max(0, mask_percentile)  # type: ignore
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(
                    layer_mod, min(mask_order, len(layer_mod) - 1) # type: ignore
                )  
                grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update
    else:
        raise ValueError("Invalid mode for masking gradient updates")


class RFFLStrategy(ABCStrategy):
    name: str = "RFFLStrategy"

    def __init__(self, model: Module, cfg: RFFLConfig, res_man: ResultManager):

        self.cfg = cfg
        self.res_man = res_man
        # FIXME: client_ids should be passed as an argument
        client_ids = generate_client_ids(cfg.num_clients)

        self._server_params: dict[str, Parameter] = model.state_dict()

        self.local_grad_norm = None
        self.server_grad_norm = None

        self.rs = torch.zeros(cfg.num_clients)
        self.past_phis = []
        self.rs_dict = []
        self.r_threshold = []
        self.qs_dict = []
        self.shard_sizes = torch.tensor(cfg.shard_sizes).float()

        # self._cos_sim = CosineSimilarity(dim=0)
        # NOTE: Differing in initialization here from paper as it leads to permanently zero gradients
        # self._client_wts = {cid: 1.0 / len(client_ids) for cid in client_ids}
        # self._omegas = {cid: 1.0 / len(client_ids) for cid in client_ids}

        self._clients_params = {cid: model.state_dict() for cid in client_ids}

    def receive_strategy(self, ins: fT.ClientResults_t) -> AllIns_t:
        return {cid: RFFLIns(cl_res.params) for cid, cl_res in ins.items()}

    def send_strategy(self, ids: fT.ClientIds_t) -> fT.ClientIns_t:
        """Send custom models to respective clients"""
        clients_ins = {}
        for cid in ids:
            clients_ins[cid] = fT.ClientIns(
                params=self._clients_params[cid], metadata={}
            )
        return clients_ins

    @classmethod
    def client_receive_strategy(cls, ins: fT.ClientIns) -> ClientInProto:
        return ClientInProto(in_params=ins.params)

    @classmethod
    def client_send_strategy(cls, ins: RFFLIns, result: fT.Result) -> fT.ClientResult:
        return fT.ClientResult(ins.client_params, result)

    def train_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.train_fraction, in_ids)

    def eval_selection(self, in_ids: fT.ClientIds_t) -> fT.ClientIds_t:
        return random_client_selection(self.cfg.eval_fraction, in_ids)

    def aggregate(self, strategy_ins: AllClientIns_t) -> RFFLOuts:

        client_ids = list(strategy_ins.keys())
        _clients_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}
        ##########################

        gradients = []
        for cid, client_params in _clients_params.items():
            gradient = [
                (cparam.data - sparam.data)
                for cparam, sparam in zip(
                    client_params.values(), self._server_params.values()
                )
            ]

            flattened = flatten(gradient)
            norm_value = norm(flattened) + 1e-7  # to prevent division by zero

            if norm_value > self.cfg.gamma:
                gradient = unflatten(
                    torch.multiply(
                        torch.tensor(self.cfg.gamma), torch.div(flattened, norm_value)
                    ),
                    gradient,
                )

                # model.load_state_dict(backup.state_dict())
                self._clients_params[cid] = add_update_to_params(
                    client_params, gradient
                )

            gradients.append(gradient)

        ## Reputation Calculation
        aggregated_gradient = [
            torch.zeros(param.shape) for param in self._server_params.values()
        ]

        if self.res_man._round == 0:
            weights = torch.div(self.shard_sizes, torch.sum(self.shard_sizes))
        else:
            weights = self.rs

        for gradient, weight in zip(gradients, weights):
            add_gradient_updates(aggregated_gradient, gradient, weight=weight)

        flat_aggre_grad = flatten(aggregated_gradient)

        phis = torch.zeros(self.cfg.num_clients)
        for i, gradient in enumerate(gradients):
            phis[i] = cosine_similarity(flatten(gradient), flat_aggre_grad, 0, 1e-10)

        self.past_phis.append(phis)

        self.rs = self.cfg.alpha * self.rs + (1 - self.cfg.alpha) * phis

        # for i in range(N + A):
        #     if i not in R_set:
        #         rs[i] = 0

        self.rs = torch.div(self.rs, self.rs.sum())

        # --- reputation threshold
        # start removing participants only after 10 rounds
        # if _round >= 10:
        #     R_set_copy = dcopy(R_set)
        #     curr_threshold = threshold * (1.0/ len(R_set_copy))

        # for i in range(N + A):
        #     # only operation is to remove a reputable participant, if necessary. All others left untouched.
        #     if i in R_set_copy and rs[i] < curr_threshold:
        #         rs[i] = 0x
        #         R_set.remove(i)
        #         print("---- in round {} removing {}. ".format(_round, i))

        # self.r_threshold.append( threshold * (1.0 / len(R_set)) )
        q_ratios = torch.div(self.rs, torch.max(self.rs))

        self.rs_dict.append(self.rs)
        self.qs_dict.append(q_ratios)

        for i, cid in enumerate(client_ids):
            q_ratio = q_ratios[i]
            reward_gradient = mask_grad_update_by_order(
                aggregated_gradient, mask_percentile=q_ratio, mode="layer"
            )

            self._clients_params[cid] = add_update_to_params(
                self._clients_params[cid], reward_gradient
            )
        #######################
        self._server_params = add_update_to_params(
            self._server_params, aggregated_gradient
        )
        self.res_man.log_parameters(
            self._server_params, phase="post_agg", actor="server"
        )

        return RFFLOuts(
            server_params=self._server_params, client_params=self._clients_params
        )
