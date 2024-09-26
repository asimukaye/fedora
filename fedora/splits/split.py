import logging
import numpy as np
from typing import Sequence, Protocol
import random
from torch.utils.data import Subset, Dataset
import torch
from fedora.utils import log_tqdm
from fedora.config.splitconf import (
    SplitConfig,
    NoisyImageSplitConfig,
    NoisyLabelSplitConfig,
    PathoSplitConfig,
    DirichletSplitConfig,
    DataImbalanceSplitConfig,
)
import fedora.customtypes as fT
from fedora.splits.pathological import pathological_non_iid_split
from fedora.splits.dirichlet import dirichlet_noniid_split
from fedora.splits.noisy import NoisySubset, LabelNoiseSubset
from fedora.splits.imbalanced import (
    get_imbalanced_split,
    get_one_imbalanced_client_split,
)

logger = logging.getLogger(__name__)


def get_iid_split(
    dataset: Subset, num_splits: int, seed: int = 42
) -> dict[int, np.ndarray]:
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
    match cfg.name:
        case "iid" | "noisyimage" | "noisylabel":
            split_map = get_iid_split(dataset, cfg.num_splits)
            return split_map

        case "dataimbalance":
            assert isinstance(cfg, DataImbalanceSplitConfig), "Invalid config type"
            if cfg.num_imbalanced_clients == 1:
                split_map = get_one_imbalanced_client_split(dataset, cfg.num_splits)
            else:
                split_map = get_imbalanced_split(dataset, cfg.num_splits)
            return split_map

        case "patho":
            assert isinstance(cfg, PathoSplitConfig), "Invalid config type"
            split_map = pathological_non_iid_split(
                dataset, cfg.num_splits, cfg.num_class_per_client
            )
            return split_map
        #     raise NotImplementedError
        case "dirichlet":
            assert isinstance(cfg, DirichletSplitConfig), "Invalid config type"
            split_map = dirichlet_noniid_split(dataset, cfg.num_splits, cfg.alpha)
            return split_map
        case "natural":
            logger.info("[DATA_SPLIT] Using pre-defined split.")
            raise NotImplementedError
        case _:
            logger.error("[DATA_SPLIT] Unknown datasplit type")
            raise NotImplementedError


def _construct_client_dataset(
    raw_train: Dataset, raw_test: Dataset, train_indices, test_indices
) -> tuple[Subset, Subset]:
    train_set = Subset(raw_train, train_indices)
    test_set = Subset(raw_test, test_indices)
    return (train_set, test_set)


def add_image_noise_to_datasets(
    client_datasets: list[fT.DatasetPair_t],
    cfg: NoisyImageSplitConfig,
    match_train_distribution=False,
) -> list[fT.DatasetPair_t]:
    if isinstance(cfg.noise_mu, list):
        assert (
            len(cfg.noise_mu) >= cfg.num_noisy_clients
        ), "Number of noise means should match number of patho clients"
        noise_mu_list = cfg.noise_mu
    else:
        noise_sigma_list = [cfg.noise_mu for _ in range(cfg.num_splits)]

    if isinstance(cfg.noise_sigma, list):
        assert (
            len(cfg.noise_sigma) >= cfg.num_noisy_clients
        ), "Number of noise sigmas should match number of patho clients"
        noise_sigma_list = cfg.noise_sigma
    else:
        noise_sigma_list = [cfg.noise_sigma for _ in range(cfg.num_splits)]

    for idx in range(cfg.num_noisy_clients):
        train, test = client_datasets[idx]
        patho_train = NoisySubset(train, noise_mu_list[idx], noise_sigma_list[idx])
        if match_train_distribution:
            test = NoisySubset(test, noise_mu_list[idx], noise_sigma_list[idx])
        client_datasets[idx] = patho_train, test
    return client_datasets


def add_label_noise_to_datasets(
    client_datasets: list[fT.DatasetPair_t],
    cfg: NoisyLabelSplitConfig,
    match_train_distribution=False,
) -> list[fT.DatasetPair_t]:
    if isinstance(cfg.noise_flip_percent, list):
        assert (
            len(cfg.noise_flip_percent) >= cfg.num_noisy_clients
        ), "Number of noise flip percent should match number of patho clients"
        noise_list = cfg.noise_flip_percent
    else:
        noise_list = [cfg.noise_flip_percent for _ in range(cfg.num_splits)]
    for idx in range(cfg.num_noisy_clients):
        train, test = client_datasets[idx]
        patho_train = LabelNoiseSubset(train, noise_list[idx])
        if match_train_distribution:
            test = LabelNoiseSubset(test, noise_list[idx])
        client_datasets[idx] = patho_train, test

    return client_datasets


def get_client_datasets(
    cfg: SplitConfig,
    train_dataset: Dataset,
    test_dataset,
    match_train_distribution=False,
) -> list[fT.DatasetPair_t]:
    # logger.info(f'[DATA_SPLIT] dataset split: `{cfg.split_type.upper()}`')
    split_map = get_split_map(cfg, train_dataset)  # type: ignore
    if match_train_distribution:
        test_split_map = get_split_map(cfg, test_dataset)
    else:
        test_split_map = get_iid_split(test_dataset, cfg.num_splits)

    assert len(split_map) == len(
        test_split_map
    ), "Train and test split maps should be of same length"
    logger.info(f"[DATA_SPLIT] Simulated dataset split : `{cfg.name.upper()}`")

    # construct client datasets if None
    cfg.test_fractions = []
    client_datasets = []
    for idx, train_indices in enumerate(split_map.values()):
        train_set, test_set = _construct_client_dataset(
            train_dataset, test_dataset, train_indices, test_indices=test_split_map[idx]
        )
        cfg.test_fractions.append(len(test_set) / len(train_set))
        client_datasets.append((train_set, test_set))

    match cfg.name:
        case "noisyimage":
            assert isinstance(cfg, NoisyImageSplitConfig), "Invalid config type"
            client_datasets = add_image_noise_to_datasets(
                client_datasets, cfg, match_train_distribution
            )
        case "noisylabel":
            assert isinstance(cfg, NoisyLabelSplitConfig), "Invalid config type"
            client_datasets = add_label_noise_to_datasets(
                client_datasets, cfg, match_train_distribution
            )
        case _:
            pass
    logger.debug(f"[DATA_SPLIT] Created client datasets!")
    logger.debug(f"[DATA_SPLIT] Split fractions: {cfg.test_fractions}")
    return client_datasets
