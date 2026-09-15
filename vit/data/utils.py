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
"""Utility functions for datasets."""

import torch
import torchvision.transforms.functional as F

IGNORE_LABEL = 255
STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def default_class_names(num_classes, prefix="class"):
    """Create stable fallback class names."""
    width = max(3, len(str(max(num_classes - 1, 0))))
    return [f"{prefix}_{idx:0{width}d}" for idx in range(num_classes)]


def rgb_normalize(x, stats):
    """
    x : C x *
    x in [0, 1]
    """
    return F.normalize(x, stats["mean"], stats["std"])


def rgb_denormalize(x, stats):
    """
    x : N x C x *
    x in [-1, 1]
    """
    mean = torch.tensor(stats["mean"])  # pylint: disable=E1101
    std = torch.tensor(stats["std"])  # pylint: disable=E1101
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x
