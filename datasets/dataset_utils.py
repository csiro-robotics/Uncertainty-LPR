# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset, TrainTransform, TrainSetTransform
from datasets.samplers import BatchSampler

from torchpack.utils.config import configs 


def make_datasets(debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(configs.data.aug_mode)
    train_set_transform = TrainSetTransform(configs.data.aug_mode)

    datasets['train'] = OxfordDataset(configs.data.dataset_folder, configs.data.train_file, train_transform,
                                      set_transform=train_set_transform)
    val_transform = None
    if configs.data.val_file is not None:
        datasets['val'] = OxfordDataset(configs.data.dataset_folder, configs.data.val_file, val_transform)
    return datasets

def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    if configs.data.load_mode == 1:
        def collate_fn(data_list):
            # Constructs a batch object
            clouds = [e[0] for e in data_list]
            labels = [e[1] for e in data_list]
            batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
            if dataset.set_transform is not None:
                # Apply the same transformation on all dataset elements
                batch = dataset.set_transform(batch)

            if mink_quantization_size is None:
                # Not a MinkowskiEngine based model
                batch = {'cloud': batch}
            else:
                coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                        for e in batch]
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                batch = {'coords': coords, 'features': feats, 'cloud': batch}

            # Compute positives and negatives mask
            # Compute positives and negatives mask
            positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
            negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
            positives_mask = torch.tensor(positives_mask)
            negatives_mask = torch.tensor(negatives_mask)

            # Returns (batch_size, n_points, 3) tensor and positives_mask and
            # negatives_mask which are batch_size x batch_size boolean tensors
            return batch, positives_mask, negatives_mask

    elif configs.data.load_mode == 2:
        def collate_fn(data_list):
            # Constructs a batch object
            clouds = [e[0] for e in data_list]
            labels = [e[1] for e in data_list]
            batch = []

            if dataset.set_transform is not None:
                for c in clouds:
                    batch.append(dataset.set_transform(c.unsqueeze(0)).squeeze())

            before = sum([b.shape[0] for b in batch])
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=configs.model.mink_quantization_size)
                        for e in batch]
            coords = ME.utils.batched_coordinates(coords)

            # print(before, coords.shape[0])
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats}

            # Compute positives and negatives mask
            # Compute positives and negatives mask
            positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
            negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
            positives_mask = torch.tensor(positives_mask)
            negatives_mask = torch.tensor(negatives_mask)

            # Returns (batch_size, n_points, 3) tensor and positives_mask and
            # negatives_mask which are batch_size x batch_size boolean tensors
            return batch, positives_mask, negatives_mask

    else:
        raise ValueError(f'Error: load mode {configs.data.load_mode} not valid')


    return collate_fn

def make_dataloaders(debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements

    :return:
    """
    datasets = make_datasets(debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=configs.train.batch_size,
                                 batch_size_limit=configs.train.batch_size_limit,
                                 batch_expansion_rate=configs.train.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'],  configs.model.mink_quantization_size)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=configs.train.num_workers, pin_memory=False)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=configs.train.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], configs.model.mink_quantization_size)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=configs.train.num_workers, pin_memory=False)

    return dataloders


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
