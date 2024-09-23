import logging
import numpy as np
from typing import Sequence, Protocol
import random
from torch.utils.data import Subset, Dataset
import torch
from fedora.utils  import log_tqdm
from fedora.config.splitconf import SplitConfig
import fedora.customtypes as fT
from fedora.splits.pathological import pathological_non_iid_split
from fedora.splits.dirichlet import dirichlet_noniid_split
from fedora.splits.noisy import NoisySubset, LabelNoiseSubset
from fedora.splits.imbalanced import get_imbalanced_split, get_one_imbalanced_client_split

logger = logging.getLogger(__name__)


def get_iid_split(dataset: Subset, num_splits: int, seed: int = 42) -> dict[int, np.ndarray]:
    shuffled_indices = np.random.permutation(len(dataset))
        
    # get adjusted indices
    split_indices = np.array_split(shuffled_indices, num_splits)
    
    # construct a hashmap
    split_map = {k: split_indices[k] for k in range(num_splits)}
    return split_map



def get_split_map(cfg: SplitConfig, dataset: Subset) -> dict[int, np.ndarray]:
    """Split data indices using labels.
    Args:
        cfg (DatasetConfig): Master dataset configuration class
        dataset (dataset): raw dataset instance to be split 
        
    Returns:
        split_map (dict): dictionary with key is a client index and a corresponding value is a list of indices
    """
    match cfg.split_type:
        case 'iid' | 'one_noisy_client' | 'one_label_flipped_client'|'n_label_flipped_clients'| 'n_noisy_clients' | 'n_distinct_noisy_clients' | 'n_distinct_label_flipped_clients':
            split_map = get_iid_split(dataset, cfg.num_splits)
            return split_map

        case 'imbalanced':
            split_map = get_imbalanced_split(dataset, cfg.num_splits)
            return split_map

        case 'one_imbalanced_client':
            split_map = get_one_imbalanced_client_split(dataset, cfg.num_splits)
            return split_map

        case 'patho':
            split_map = pathological_non_iid_split(dataset, cfg.num_splits, cfg.num_class_per_client)  
            return split_map
        #     raise NotImplementedError
        case 'dirichlet':
            split_map = dirichlet_noniid_split(dataset, cfg.num_splits, cfg.dirichlet_alpha)
            return split_map
        case 'leaf' |'fedvis'|'flamby':
            logger.info('[DATA_SPLIT] Using pre-defined split.')
            raise NotImplementedError
        case _ :
            logger.error('[DATA_SPLIT] Unknown datasplit type')
            raise NotImplementedError


def _construct_client_dataset(raw_train: Dataset, raw_test: Dataset, train_indices, test_indices) ->tuple[Subset, Subset]:
    train_set = Subset(raw_train, train_indices)
    test_set = Subset(raw_test, test_indices)
    return (train_set, test_set)


def get_client_datasets(cfg: SplitConfig, train_dataset: Dataset, test_dataset, match_train_distribution=False) -> list[fT.DatasetPair_t] :
    # logger.info(f'[DATA_SPLIT] dataset split: `{cfg.split_type.upper()}`')   
    split_map = get_split_map(cfg, train_dataset)
    if match_train_distribution:
        test_split_map = get_split_map(cfg, test_dataset)
    else:
        test_split_map = get_iid_split(test_dataset, cfg.num_splits)

    assert len(split_map) == len(test_split_map), 'Train and test split maps should be of same length'
    logger.info(f'[DATA_SPLIT] Simulated dataset split : `{cfg.split_type.upper()}`')
    
    # construct client datasets if None
    cfg.test_fractions = []
    client_datasets = []
    for idx, train_indices in enumerate(split_map.values()):
        train_set, test_set = _construct_client_dataset(train_dataset, test_dataset, train_indices, test_indices=test_split_map[idx])
        cfg.test_fractions.append(len(test_set)/len(train_set))
        client_datasets.append((train_set, test_set))
    
    match cfg.split_type:
        case 'one_noisy_client':
            train, test = client_datasets[0]
            patho_train = NoisySubset(train, cfg.noise.mu, cfg.noise.sigma)
            if match_train_distribution:
                test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
            client_datasets[0] = patho_train, test
        case 'one_label_flipped_client':
            train, test = client_datasets[0]
            patho_train = LabelNoiseSubset(train, cfg.noise.flip_percent)
            if match_train_distribution:
                test = LabelNoiseSubset(test, cfg.noise.flip_percent)
            client_datasets[0] = patho_train, test
        case 'n_label_flipped_clients':
            for idx in range(cfg.num_noisy_clients):
                train, test = client_datasets[idx]
                patho_train = LabelNoiseSubset(train, cfg.noise.flip_percent)
                if match_train_distribution:
                    test = LabelNoiseSubset(test, cfg.noise.flip_percent)
                client_datasets[idx] = patho_train, test
        case 'n_noisy_clients':
            for idx in range(cfg.num_noisy_clients):
                train, test = client_datasets[idx]
                patho_train = NoisySubset(train, cfg.noise.mu, cfg.noise.sigma)
                if match_train_distribution:
                    test = NoisySubset(test, cfg.noise.mu, cfg.noise.sigma)
                client_datasets[idx] = patho_train, test
        case 'n_distinct_noisy_clients':
            assert len(cfg.noise.mu) >= cfg.num_noisy_clients, 'Number of noise means should match number of patho clients'
            assert len(cfg.noise.sigma) >= cfg.num_noisy_clients, 'Number of noise sigmas should match number of patho clients'
            for idx in range(cfg.num_noisy_clients):
                train, test = client_datasets[idx]
                patho_train = NoisySubset(train, cfg.noise.mu[idx], cfg.noise.sigma[idx])
                if match_train_distribution:
                    test = NoisySubset(test, cfg.noise.mu[idx], cfg.noise.sigma[idx])
                client_datasets[idx] = patho_train, test
        case 'n_distinct_label_flipped_clients':
            assert len(cfg.noise.flip_percent) >= cfg.num_noisy_clients, 'Number of noise flip percent should match number of patho clients'
            for idx in range(cfg.num_noisy_clients):
                train, test = client_datasets[idx]
                patho_train = LabelFlippedSubset(train, cfg.noise.flip_percent[idx])
                if match_train_distribution:
                    test = LabelFlippedSubset(test, cfg.noise.flip_percent[idx])
                client_datasets[idx] = patho_train, test
        case _:
            pass
    logger.debug(f'[DATA_SPLIT] Created client datasets!')
    logger.debug(f'[DATA_SPLIT] Split fractions: {cfg.test_fractions}')
    return client_datasets

