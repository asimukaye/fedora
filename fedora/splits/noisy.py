import torch
from torch.utils.data import Subset, Dataset
import numpy as np
from typing import Sequence, Protocol

class MappedDataset(Protocol):
    '''Dataset with the indices mapped to the original dataset'''
    @property
    def targets(self)->list:
        ...
    @property
    def class_to_idx(self)->dict:
        ...
    @property
    def indices(self)->Sequence[int]:
        ...

    def __len__(self)->int:
        ...

def check_for_mapping(dataset: Dataset) -> MappedDataset:
    if not hasattr(dataset, 'class_to_idx'):
        raise TypeError(f'Dataset {dataset} does not have class_to_idx')
    if not hasattr(dataset, 'targets'):
        raise TypeError(f'Dataset {dataset} does not have targets')
    if not hasattr(dataset, 'indices'):
        raise TypeError(f'Dataset {dataset} does not have indices')
    return dataset # type: ignore

def extract_root_dataset(subset: Subset) -> Dataset:
    if isinstance(subset.dataset, Subset):
        return extract_root_dataset(subset.dataset)
    else:
        assert isinstance(subset.dataset, Dataset), 'Unknown subset nesting' 
        return subset.dataset



def extract_root_dataset_and_indices(subset: Subset, indices = None) -> tuple[Dataset, np.ndarray] :
    # ic(type(subset.indices))
    if indices is None:
        indices = subset.indices
    np_indices = np.array(indices)
    if isinstance(subset.dataset, Subset):
        # ic(type(subset.dataset))
        mapped_indices = np.array(subset.dataset.indices)[np_indices]
        # ic(mapped_indices)
        return extract_root_dataset_and_indices(subset.dataset, mapped_indices)
    else:
        assert isinstance(subset.dataset, Dataset), 'Unknown subset nesting' 
        # ic(type(subset.dataset))
        # mapped_indices = np.array(subset.indices)[in_indices]
        # ic(np_indices)
        return subset.dataset, np_indices
    


class LabelNoiseSubset(Subset):
    """Wrapper of `torch.utils.Subset` module for label flipping.
    """
    def __init__(self, dataset: Dataset,  flip_pct: float):
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices
            root_dataset, mapped_ids = extract_root_dataset_and_indices(dataset)
            checked_dataset = check_for_mapping(root_dataset)
        else:
            self.dataset = dataset
            checked_dataset = check_for_mapping(dataset)
            self.indices = checked_dataset.indices
            mapped_ids = np.array(checked_dataset.indices)
        
        # ic(len(dataset), len(checked_dataset), len(mapped_ids))
        self.subset = self._flip_set(dataset, checked_dataset, mapped_ids, flip_pct) 

    def _flip_set(self, subset:Subset, dataset: MappedDataset, mapped_ids:np.ndarray, flip_pct: float) -> Subset:
        total_size = len(subset)
            # dataset, mapped_ids = extract_root_dataset_and_indices(subset)
        # dataset = check_for_mapping(dataset)


        samples = np.random.choice(total_size, size=int(flip_pct*total_size), replace=False)
        
        selected_indices = mapped_ids[samples]
        # ic(samples, selected_indices)
        class_ids = list(dataset.class_to_idx.values())
        for idx, dataset_idx in zip(samples, selected_indices):
            _, lbl = subset[idx]
            assert lbl == dataset.targets[dataset_idx]
            # ic(lbl, )
            excluded_labels = [cid for cid in class_ids if cid != lbl]
            # changed_label = np.random.choice(excluded_labels)
            # ic(changed_label)
            dataset.targets[dataset_idx] = np.random.choice(excluded_labels)
            # print('\n')
        return subset
    
    def __getitem__(self, index):
        inputs, targets = self.subset[index]
        return inputs, targets

    def __len__(self):
        return len(self.subset)
    
    def __repr__(self):
        return f'{repr(self.subset.dataset)}_LabelFlipped'

    

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
class NoisySubset(Subset):
    """Wrapper of `torch.utils.Subset` module for applying individual transform.
    """
    def __init__(self, subset: Subset,  mean:float, std: float):
        self.dataset = subset.dataset
        self.indices = subset.indices
        self._subset = subset
        self.noise = AddGaussianNoise(mean, std)

    def __getitem__(self, idx):
        inputs, targets = self._subset[idx]
        return self.noise(inputs), targets

    def __len__(self):
        return len(self.indices)

    
    def __repr__(self):
        return f'{repr(self.dataset)}_GaussianNoise'
