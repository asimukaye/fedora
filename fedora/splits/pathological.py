
from torch.utils.data import Subset, Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_one_patho_client_split(dataset: Subset, num_splits) -> dict[int, np.ndarray]:
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


def pathological_non_iid_split(dataset: Subset, n_clients: int,  n_classes_per_client: int) -> dict[int, np.ndarray]:
    target_set = Subset(dataset.dataset.targets, dataset.indices)
    train_labels = np.array(target_set)
    # ic(train_labels.shape)
    # ic(train_labels[:10])
    n_classes = train_labels.max()+1
    data_idcs = list(range(len(train_labels)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        label = train_labels[idx]
        label2index[label].append(idx)

    # ic(label2index)
    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(lst, g):
        """Divides the list `l` into `g` i.i.d. groups, i.e.direct splitting.
        Each group has `int(len(l)/g)` or `int(len(l)/g)+1` elements.
        Returns a list of different groups.
        """
        num_elems = len(lst)
        group_size = int(len(lst) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(lst[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(lst[bi + group_size * i:bi + group_size * (i + 1)])
        return glist

    n_shards = n_clients * n_classes_per_client
    # Divide the sample indices into `n_shards` i.i.d. shards
    shards = iid_divide(sorted_idcs, n_shards)

    np.random.shuffle(shards)
    # Then split the shards into `n_client` parts
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # Here `shard` is the sample indices of a shard (a list)
            # `+= shard` is to merge the list `shard` into the list
            # `clients_idcs[client_id]`
            clients_idcs[client_id] += shard

    clients_idcs = {k: np.array(idcs) for k, idcs in zip(range(n_clients),clients_idcs)}
    return clients_idcs

# FedAvg patho split

# TODO: Understand this function from FedAvg Paper
def get_patho_split(dataset: Subset, num_splits: int, num_classes: int, mincls: int) -> dict[int, np.ndarray]:
    try:
        assert mincls >= 2
    except AssertionError as e:
        logger.exception("[DATA_SPLIT] Each client should have samples from at least 2 distinct classes!")
        raise e
    
    # get unique class labels and their count
    inferred_classes, unique_inverse, unique_count = np.unique(dataset.targets, return_inverse=True, return_counts=True)
    # split the indices by class labels
    class_indices = np.split(np.argsort(unique_inverse), np.cumsum(unique_count[:-1]))
    
    assert len(inferred_classes) == num_classes, 'Inferred classes do not match the expected number of classes'
    
    # divide shards
    num_shards_per_class = num_splits* mincls // num_classes
    if num_shards_per_class < 1:
        err = f'[DATA_SPLIT] Increase the number of minimum class (`args.mincls` > {mincls}) or the number of participating clients (`args.K` > {num_splits})!'
        logger.exception(err)
        raise Exception(err)
    
    # split class indices again into groups, each having the designated number of shards
    split_indices = [np.array_split(np.random.permutation(indices), num_shards_per_class) for indices in class_indices]
    
    # make hashmap to track remaining shards to be assigned per client
    class_shards_counts = dict(zip([i for i in range(num_classes)], [len(split_idx) for split_idx in split_indices]))

    # assign divided shards to clients
    assigned_shards = []
    for _ in range(num_classes):
        # update selection proability according to the count of reamining shards
        # i.e., do NOT sample from class having no remaining shards
        selection_prob = np.where(np.array(list(class_shards_counts.values())) > 0, 1., 0.)
        selection_prob /= sum(selection_prob)
        
        # select classes to be considered
        try:
            selected_classes = np.random.choice(num_classes, mincls, replace=False, p=selection_prob)
        except: # if shard size is not fit enough, some clients may inevitably have samples from classes less than the number of `mincls`
            selected_classes = np.random.choice(num_classes, mincls, replace=True, p=selection_prob)
        
        # assign shards in randomly selected classes to current client
        for it, class_idx in enumerate(selected_classes):
            selected_shard_indices = np.random.choice(len(split_indices[class_idx]), 1)[0]
            selected_shards = split_indices[class_idx].pop(selected_shard_indices)
            if it == 0:
                assigned_shards.append([selected_shards])
            else:
                assigned_shards[-1].append(selected_shards)
            class_shards_counts[class_idx] -= 1
        else:
            assigned_shards[-1] = np.concatenate(assigned_shards[-1])

    # construct a hashmap
    split_map = {k: assigned_shards[k] for k in range(num_splits)}
    return split_map
