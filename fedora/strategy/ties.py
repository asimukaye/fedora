from collections import defaultdict, OrderedDict
import typing as t
import copy

import torch
from torch.nn import Module, Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from torch import Tensor, tensor
import torch.optim
import fedora.customtypes as fT

# from fedora.strategy.abcstrategy import ABCStrategy
from fedora.strategy.abcstrategy import *

from fedora.strategy.fedavg import (
    ClientInProto,
    random_client_selection,
    weighted_parameter_averaging,
)
from fedora.results.resultmanager import ResultManager

# Type declarations

class TiesCfgProtocol(t.Protocol):
    """Protocol for base strategy config"""

    train_fraction: float
    eval_fraction: float
    K: float = 1
    lamda: float = 1
    merge_func: str = "dis-mean"


class TiesInsProtocol(t.Protocol):
    client_params: fT.ActorParams_t
    data_size: int


@dataclass
class TiesIns(StrategyIns):
    client_params: fT.ActorParams_t
    data_size: int


AllIns_t = dict[str, TiesIns]


@dataclass
class TiesOuts(StrategyOuts):
    server_params: fT.ActorParams_t


# Ties conversion utils
def state_dict_to_vector(state_dict: dict, remove_keys=[]):
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    # sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return parameters_to_vector(
        [value.reshape(-1) for value in shared_state_dict.values()]
    )


def vector_to_state_dict(vector: Tensor, state_dict: dict, remove_keys=[]):
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    # sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    vector_to_parameters(vector, reference_dict.values())

    return reference_dict


def add_ptm_to_tv(tv_dict, ptm_dict):
    assert set(tv_dict.keys()) == set(
        ptm_dict.keys()
    ), "Differing parameter names in models."
    final_dict = copy.deepcopy(tv_dict)
    for k, v in ptm_dict.items():
        final_dict[k] = tv_dict[k] + v
    return final_dict


def check_parameterNamesMatch(checkpoints):
    parameter_names = set(checkpoints[0].keys())

    if len(checkpoints) >= 2:
        # raise ValueError("Number of models is less than 2.")
        for checkpoint in checkpoints[1:]:
            current_parameterNames = set(checkpoint.keys())
            if current_parameterNames != parameter_names:
                raise ValueError(
                    "Differing parameter names in models. "
                    f"The different parameters are {parameter_names.symmetric_difference(current_parameterNames)}"
                )


def check_state_dicts_equal(state_dict1, state_dict2):
    if set(state_dict1.keys()) != set(state_dict2.keys()):
        return False

    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            return False

    return True


## TIES MERGING UTILS


def topk_values_mask(M: Tensor, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = tensor(M.abs() >= kth_values)
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

# This method seems a little arbitrary
def resolve_zero_signs(sign_to_mult: Tensor, method="majority") -> Tensor:
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(input_tensor: Tensor) -> Tensor:
    sign_to_mult = torch.sign(input_tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(params: Tensor, merge_func: str, sign_to_mult: Tensor) -> Tensor:

    merge_func = merge_func.split("-")[-1]
    # ic(sign_to_mult.shape)
    # ic(sign_to_mult.unsqueeze(0).shape)
    # ic(params.shape)

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0, params > 0, params < 0
        )
        # ic(rows_to_keep.shape)
        # selected_entries = params * rows_to_keep
        # ic(selected_entries.shape)
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = params != 0
        selected_entries = params * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        # ic(non_zero_counts.shape)
        # ic(non_zero_counts.min())
        # ic(non_zero_counts.max())
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
        # ic(disjoint_aggs.shape)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks: Tensor,
    reset_thresh=0.2,
    merge_func="",
) -> Tensor:
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


class TiesStrategy(ABCStrategy):
    def __init__(
        self, model: Module, cfg: TiesCfgProtocol, res_man: ResultManager
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
        cls, ins: TiesInsProtocol, res: fT.Result
    ) -> fT.ClientResult:
        result = res
        result.size = ins.data_size
        return fT.ClientResult(ins.client_params, res)

    def receive_strategy(self, results: fT.ClientResults_t) -> AllIns_t:

        strat_ins = {}
        for cid, res in results.items():
            strat_ins[cid] = TiesIns(
                client_params=res.params, data_size=res.result.size
            )
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

    def aggregate(self, strategy_ins: AllIns_t) -> TiesOuts:
        # calculate client weights according to sample sizes

        _client_params = {cid: inp.client_params for cid, inp in strategy_ins.items()}

        flat_client_params = torch.vstack(
            [state_dict_to_vector(c_params) for c_params in _client_params.values()]
        )
        # ic(self._server_params.keys())

        flat_global_params = state_dict_to_vector(self._server_params)

        delta_params = flat_client_params - flat_global_params
        # ic(flat_client_params.shape)
        # ic(delta_params.shape)

        merged_delta_params = ties_merging(
            delta_params, reset_thresh=self.cfg.K, merge_func=self.cfg.merge_func
        )

        updated_global_params = (
            flat_global_params + self.cfg.lamda * merged_delta_params
        )
        # ic(updated_global_params.shape)
        self._server_params = vector_to_state_dict(
            updated_global_params, self._server_params
        )
        # ic(self._server_params.keys())

        outs = TiesOuts(server_params=self._server_params)

        return outs
