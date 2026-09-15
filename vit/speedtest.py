# Copyright © 2025 Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications based on code from Daniel Bolya et al. (ToMe)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
"""
Script to compute the throughput score of a model.

Example: see README.md
"""

import time
import warnings

import click
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from vit.data import create_dataset
from vit.model.factory import load_model

import g2tm

warnings.filterwarnings("ignore")


@torch.no_grad()
def compute_throughput(
    model: nn.Module, validation_loader: DataLoader, batch_size: int, device: str
) -> float:
    """Average throughput score of the model for one forward-pass.

    This function computes the average throughput score (frame-per-second,
    FPS) of the model for a batch inference.

    NOTE: In the paper, a batch size of 1 is used here to better reflect
    real world conditions.

    Args:
        model (nn.Module): PyTorch model.
        validation_loader (DataLoader): PyTorch dataloader.
        batch_size (int): Number of images per batch.
        device (str): Device to run the inference.

    Returns:
        mean_timing (float): Mean FPS score for one inference.
    """
    torch.cuda.empty_cache()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    warmup_iters = 50
    warmup = True
    repeat = 32

    timing = []

    torch.cuda.synchronize()
    with torch.no_grad():

        for image in tqdm(validation_loader, position=0, leave=False):
            image = image["im"].to(device)

            if warmup:
                for _ in range(warmup_iters):
                    model(image.to(device))
                warmup = False
                continue

            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeat):
                model(image.to(device))
            torch.cuda.synchronize()
            timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)  # pylint: disable=E1101

    return round((batch_size * repeat / timing.median()).item(), 2)


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--batch-size", type=int, default=1)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=2, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
@click.option("--fast-sv/--bfs", default=False, is_flag=True)
@click.option("--num-iters", default=None, type=int)
def main(
    model_path,
    dataset_name,
    batch_size,
    patch_type,
    selected_layer,
    threshold,
    prop_attn,
    iprop_attn,
    fast_sv,
    num_iters,
):
    """Compute the throughput score of the model.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        batch_size (str): Number of images per batch.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
        fast_sv (bool): Whether to use FastSV version of G2TM, compatible
            with the ONNX export.
        num_iters (int): Number of FastSV iterations.
    """

    device = "cuda:0"

    model, variant = load_model(model_path)
    dataset_kwargs = variant["dataset_kwargs"]
    dataset_kwargs["batch_size"] = batch_size
    dataset_kwargs["split"] = "test"

    if dataset_name != dataset_kwargs["dataset"]:
        print(
            "Warning: The model has been trained on "
            f"{dataset_kwargs['dataset']}, but is tested on {dataset_name}"
        )

    if patch_type == "graph":
        g2tm.graph_vit_patch(
            model, selected_layer, threshold, prop_attn, iprop_attn, fast_sv, num_iters
        )

    model.eval()
    model.to(device)

    validation_loader = create_dataset(dataset_kwargs)
    fps = compute_throughput(model, validation_loader, batch_size, device)

    print("FPS:", fps, flush=True)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
