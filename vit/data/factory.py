# MIT License

# Copyright (c) 2021 Robin Strudel
# Copyright (c) INRIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Utility functions for dataset wrapper classes."""

import vit.utils.torch as ptu
from vit.data.cifar100 import CIFAR100Dataset
from vit.data.imagenet import ImagenetDataset
from vit.data.oxford_flowers import OxfordFlowersDataset
from vit.data.oxford_pets import OxfordPetsDataset
from vit.data.loader import Loader

DATASETS = {
    "imagenet": ImagenetDataset,
    "cifar100": CIFAR100Dataset,
    "oxford_pets": OxfordPetsDataset,
    "oxford_flowers": OxfordFlowersDataset,
}


def create_dataset(dataset_kwargs):
    """Create a dataset from configuration dictionary."""
    dataset_kwargs = dataset_kwargs.copy()
    dataset_name = dataset_kwargs.pop("dataset")
    batch_size = dataset_kwargs.pop("batch_size")
    num_workers = dataset_kwargs.pop("num_workers")
    split = dataset_kwargs.pop("split")

    dataset_cls = DATASETS.get(dataset_name, None)
    if dataset_cls is None:
        raise ValueError(f"Dataset {dataset_name} is unknown.")
    dataset = dataset_cls(split=split, **dataset_kwargs)

    dataset = Loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        distributed=ptu.distributed,
        split=split,
    )
    return dataset
