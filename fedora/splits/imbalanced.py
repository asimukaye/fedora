import torch
from torch.utils.data import Subset, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_imbalanced_split(dataset: Subset, num_splits: int) -> dict[int, np.ndarray]:
     # shuffle sample indices
    shuffled_indices = np.random.permutation(len(dataset))
    
    # split indices by number of clients
    split_indices = np.array_split(shuffled_indices, num_splits)
        
    # randomly remove some proportion (1% ~ 5%) of data
    keep_ratio = np.random.uniform(low=0.95, high=0.99, size=len(split_indices))
        
    # get adjusted indices
    split_indices = [indices[:int(len(indices) * ratio)] for indices, ratio in zip(split_indices, keep_ratio)]
    
    # construct a hashmap
    split_map = {k: split_indices[k] for k in range(num_splits)}
    return split_map


def get_one_imbalanced_client_split(dataset: Subset, num_splits: int) -> dict[int, np.ndarray]:
    total_size = len(dataset)
    shuffled_indices = np.random.permutation(total_size)
    # client 1 gets half the size of data compared to rest
    c1_count = int(total_size/(2*num_splits - 1))
    c1_share = shuffled_indices[:c1_count]

    rest_share = np.array_split(shuffled_indices[c1_count:], num_splits-1)

    # assert len(c1_share) + len(share) for share in rest_share == total_size
    split_map = {}
    size_check = [c1_count]
    split_map[0] = c1_share
    for k, others_share in enumerate(rest_share):
        split_map[k+1] = others_share
        size_check.append(len(others_share))
    # logger.info(f'Size total after split : {size_check}, total size: {total_size}')
    logger.info(f'Split map sizes: {size_check}')
    return split_map

