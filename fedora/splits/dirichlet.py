from fedora.utils  import log_tqdm
import numpy as np
from torch.utils.data import Subset, Dataset
import logging

logger = logging.getLogger(__name__)

def sample_with_mask(mask, ideal_samples_counts, concentration, num_classes, need_adjustment=False):
    num_remaining_classes = int(mask.sum())
    
    # sample class selection probabilities based on Dirichlet distribution with concentration parameter (`diri_alpha`)
    selection_prob_raw = np.random.dirichlet(alpha=np.ones(num_remaining_classes) * concentration, size=1).squeeze()
    selection_prob = mask.copy()
    selection_prob[selection_prob == 1.] = selection_prob_raw
    selection_prob /= selection_prob.sum()

    # calculate per-class sample counts based on selection probabilities
    if need_adjustment: # if remaining samples are not enough, force adjusting sample sizes...
        selected_counts = (selection_prob * ideal_samples_counts * np.random.uniform(low=0.0, high=1.0, size=len(selection_prob))).astype(int)
    else:
        selected_counts = (selection_prob * ideal_samples_counts).astype(int)
    return selected_counts


## CURRENT IMPLEMENTATION
def dirichlet_noniid_split(dataset: Subset, n_clients: int, alpha: float,) -> dict[int, np.ndarray]:
    """Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha.
    Args:
        train_labels: ndarray of train_labels.
        alpha: the parameter of Dirichlet distribution.
        n_clients: number of clients.
    Returns:
        client_idcs: a list containing sample idcs of clients.
    """
    target_set = Subset(dataset.dataset.targets, dataset.indices)
    train_labels = np.array(target_set)
    # train_labels = np.array(train_labels)
    n_classes = train_labels.max()+1
    # (n_classes, n_clients), label distribution matrix, indicating the
    # proportion of each label's data divided into each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    # (n_classes, ...), indicating the sample indices for each label
    class_idcs = [np.argwhere(train_labels == y).flatten()
                    for y in range(n_classes)]

    # Indicates the sample indices of each client
    client_idcs = [[] for _ in range(n_clients)]
    for c_idcs, fracs in zip(class_idcs, label_distribution):
        # `np.split` divides the sample indices of each class, i.e.`c_idcs`
        # into `n_clients` subsets according to the proportion `fracs`.
        # `i` indicates the i-th client, `idcs` indicates its sample indices
        for i, idcs in enumerate(np.split(c_idcs, (
                np.cumsum(fracs)[:-1] * len(c_idcs))
                .astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = {k: np.concatenate(idcs) for k, idcs in zip(range(n_clients),client_idcs)}

    return client_idcs



## ALTERNATIVE IMPLEMENTATIONS

# TODO: understand this split from paper
def get_dirichlet_split_2(dataset: Dataset, num_splits, num_classes, cncntrtn)-> dict[int, np.ndarray]:
           
    # get indices by class labels
    _, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
    class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
    
    # make hashmap to track remaining samples per class
    class_samples_counts = dict(zip([i for i in range(
        num_classes)], [len(class_idx) for class_idx in class_indices]))
    
    # calculate ideal samples counts per client
    ideal_samples_counts = len(dataset.targets) // num_classes
    if ideal_samples_counts < 1:
        err = f'[DATA_SPLIT] Decrease the number of participating clients (`args.K` < {num_splits})!'
        logger.exception(err)
        raise Exception(err)

    # assign divided shards to clients
    assigned_indices = []
    for k in log_tqdm(
        range(num_splits), 
        logger=logger,
        desc='[DATA_SPLIT] assigning to clients '
        ):
        # update mask according to the count of reamining samples per class
        # i.e., do NOT sample from class having no remaining samples
        remaining_mask = np.where(np.array(list(class_samples_counts.values())) > 0, 1., 0.)
        selected_counts = sample_with_mask(remaining_mask, ideal_samples_counts, cncntrtn, num_classes)

        # check if enough samples exist per selected class
        expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)
        valid_mask = np.where(expected_counts < 0, 1., 0.)
        
        # if not, resample until enough samples are secured
        while sum(valid_mask) > 0:
            # resample from other classes instead of currently selected ones
            adjusted_mask = (remaining_mask.astype(bool) & (~valid_mask.astype(bool))).astype(float)
            
            # calculate again if enoush samples exist or not
            selected_counts = sample_with_mask(adjusted_mask, ideal_samples_counts, cncntrtn, num_classes, need_adjustment=True)    
            expected_counts = np.subtract(np.array(list(class_samples_counts.values())), selected_counts)

            # update mask for checking a termniation condition
            valid_mask = np.where(expected_counts < 0, 1., 0.)
            
        # assign shards in randomly selected classes to current client
        indices = []
        for it, counts in enumerate(selected_counts):
            # get indices from the selected class
            selected_indices = class_indices[it][:counts]
            indices.extend(selected_indices)
            
            # update indices and statistics
            class_indices[it] = class_indices[it][counts:]
            class_samples_counts[it] -= counts
        else:
            assigned_indices.append(indices)

    # construct a hashmap
    split_map = {k: assigned_indices[k] for k in range(num_splits)}
    return split_map

def dirichlet_noniid_split_fixed(dataset: Subset, n_clients: int, alpha: float,) -> dict[int, np.ndarray]:
    target_set = Subset(dataset.dataset.targets, dataset.indices)
    train_labels = np.array(target_set)

    # train_labels = np.array(dataset.targets)
    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                    for y in range(n_classes)]
    # Indicates the sample indices of each client
    client_idcs = [np.array([], dtype=int) for _ in range(n_clients)]
    for c_idcs, fracs in zip(class_idcs, label_distribution):
        client_order = np.argsort([len(cid) for cid in client_idcs])
        idcs = np.split(c_idcs, (np.cumsum(fracs)[:-1] * len(c_idcs)).astype(int))
        idx_order = np.argsort([len(idx) for idx in idcs])[::-1]

        for neediest_client, idx in zip(client_order, idx_order):
            client_idcs[neediest_client] = np.append(client_idcs[neediest_client], idcs[idx])
    client_idcs = {k: idcs for k, idcs in zip(range(n_clients), client_idcs)}

    return client_idcs
