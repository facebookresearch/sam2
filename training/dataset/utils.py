# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some wrapping utilities extended from pytorch's to support repeat factor sampling in particular"""

from typing import Iterable

import torch
from torch.utils.data import (
    ConcatDataset as TorchConcatDataset,
    Dataset,
    Subset as TorchSubset,
)


class ConcatDataset(TorchConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__(datasets)

        self.repeat_factors = torch.cat([d.repeat_factors for d in datasets])

    def set_epoch(self, epoch: int):
        for dataset in self.datasets:
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


class Subset(TorchSubset):
    def __init__(self, dataset, indices) -> None:
        super(Subset, self).__init__(dataset, indices)

        self.repeat_factors = dataset.repeat_factors[indices]
        assert len(indices) == len(self.repeat_factors)


# Adapted from Detectron2
class RepeatFactorWrapper(Dataset):
    """
    Thin wrapper around a dataset to implement repeat factor sampling.
    The underlying dataset must have a repeat_factors member to indicate the per-image factor.
    Set it to uniformly ones to disable repeat factor sampling
    """

    def __init__(self, dataset, seed: int = 0):
        self.dataset = dataset
        self.epoch_ids = None
        self._seed = seed

        # Split into whole number (_int_part) and fractional (_frac_part) parts.
        self._int_part = torch.trunc(dataset.repeat_factors)
        self._frac_part = dataset.repeat_factors - self._int_part

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __len__(self):
        if self.epoch_ids is None:
            # Here we raise an error instead of returning original len(self.dataset) avoid
            # accidentally using unwrapped length. Otherwise it's error-prone since the
            # length changes to `len(self.epoch_ids)`changes after set_epoch is called.
            raise RuntimeError("please call set_epoch first to get wrapped length")
            # return len(self.dataset)

        return len(self.epoch_ids)

    def set_epoch(self, epoch: int):
        g = torch.Generator()
        g.manual_seed(self._seed + epoch)
        self.epoch_ids = self._get_epoch_indices(g)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        if self.epoch_ids is None:
            raise RuntimeError(
                "Repeat ids haven't been computed. Did you forget to call set_epoch?"
            )

        return self.dataset[self.epoch_ids[idx]]
