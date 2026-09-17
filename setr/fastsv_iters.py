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
"""
Script to calibrate the number of FastSV iterations of the ONNX variant of G2TM,
reporting how often the FastSV grouping is identical to the NetworkX one.

An insufficient number of iterations can split a component in several groups (never
merge across components), so the exported model can merge *fewer* tokens than the
PyTorch one, leading to both slightly more accurate and slightly slower model than the
one it is supposed to reproduce (to confirm on hardware).

Example: see README.md
"""

import math
import os
import importlib
from contextlib import contextmanager
from itertools import islice
from typing import Tuple, List

import click
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import g2tm
from g2tm.nx_merge import get_mergeable_idxs
from g2tm.utils import cosine_similarity_masks
from g2tm.connected_components import connected_components_labels

from setr.data.utils import STATS
from setr.inference_utils import get_dataset_inference_path, dataset_prepare
from setr.model.factory import load_model
import setr.utils.torch as ptu


class _FeaturesCaptured(Exception):
    """Raised to unwind the forward pass once G2TM has received its input."""


@contextmanager
def record_merge_inputs(records: list):
    """Record the token sequences handed to G2TM, without altering the model.

    The merging function is looked up in the globals of the patch module at every call,
    so temporarily replacing it there captures its inputs. Only the features matter
    here, both groupings being recomputed from them below, so the recorder unwinds the
    forward pass instead of merging: the blocks after the G2TM one, the classification
    head and the merge itself are never run.

    Args:
        records (list): List filled with one (feat, is_encoder, distill_token) tuple
            per call to the merging function.
    """
    patch_module = importlib.import_module("g2tm.patch.graph_setr_patch")
    original = patch_module.bfs_merge

    def recorder(
        feat, threshold, is_encoder=True, distill_token=False  # pylint: disable=W0613
    ):
        records.append((feat.detach(), is_encoder, distill_token))
        raise _FeaturesCaptured

    patch_module.bfs_merge = recorder
    try:
        yield
    finally:
        patch_module.bfs_merge = original


def fast_sv_to_canonical(labels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rename groups by the smallest index they contain.

    Two groupings of the same tokens are identical if and only if their canonical
    labels are equal, whatever representative each implementation chose for a group.

    Labels are token indices, so they can be used directly as scatter bins: no
    `unique` and no read of a tensor value, hence no synchronization with the host.

    Args:
        labels (torch.Tensor): A (n, ) tensor giving the group of each token.
        positions (torch.Tensor): The (n, ) tensor of token indices.

    Returns:
        torch.Tensor: A (n, ) tensor giving the smallest token index of each token's
            group.
    """
    n = positions.numel()
    first = torch.full_like(positions, n)  # pylint: disable=E1101
    first = first.scatter_reduce(0, labels, positions, reduce="amin")
    return first[labels]


def nx_to_canonical(connected_components: List[List[int]], n: int) -> torch.Tensor:
    """Rebuild a per-token grouping from the NetworkX output.

    Args:
        connected_components (List[List[int]]): The output connected component list of
            `nx.connected_components`.
        n (int): Number of tokens.

    Returns:
        torch.Tensor: A (n, ) tensor giving the smallest token index of each token's
            group.
    """
    labels = torch.arange(n, device=ptu.device)  # pylint: disable=E1101
    for idxs in connected_components:
        smallest = min(idxs)
        labels[idxs] = smallest
    return labels


@torch.no_grad()
def compare_variants(
    model: nn.Module,
    validation_loader: DataLoader,
    threshold: float,
    max_iters: int,
    n_images: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Compare the FastSV grouping with the Breadth-First Search one.

    Both variants are fed with the very same token features, so that the only
    difference measured is the connected components algorithm itself.

    Args:
        model (nn.Module): PyTorch model patched with the G2TM module.
        validation_loader (DataLoader): PyTorch dataloader.
        threshold (float): Threshold parameter for G2TM.
        max_iters (int): Highest number of FastSV iterations to try.
        n_images (int): Number of images to process (0 for the whole dataset).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]: Per candidate
            budget, the number of exactly reproduced images and the total
            number of tokens kept; then the per-image minimal budget, the
            number of tokens kept by the Breadth-First Search and the number
            of tokens before merging.
    """
    matches = np.zeros(max_iters + 1, dtype=np.int64)
    kept_fastsv = np.zeros(max_iters + 1, dtype=np.int64)
    minimal_iters = []
    kept_bfs = 0
    n_tokens = None

    # `islice` stops pulling from the loader once enough images have been seen. A
    # filtering generator would keep decoding the whole validation set to discard it.
    loader = islice(validation_loader, n_images) if n_images else validation_loader
    total = n_images if n_images else len(validation_loader)

    for image in tqdm(loader, total=total, position=0, leave=False):
        records = []
        with record_merge_inputs(records):
            try:
                model(image.to(ptu.device))
            except _FeaturesCaptured:
                pass
        if not records:
            raise RuntimeError(
                "G2TM was never called: check --selected-layer and --patch-type."
            )

        feat, is_encoder, distill_token = records[0]
        protected = int(is_encoder) + int(distill_token)
        x_feat = feat[:, protected:, :]
        b, n, _ = x_feat.size()
        grid = int(math.sqrt(n))
        n_tokens = n
        positions = torch.arange(  # pylint: disable=E1101
            n, device=ptu.device, dtype=torch.int64
        )

        reference = nx_to_canonical(
            get_mergeable_idxs(x_feat, threshold, grid, grid, b, n, ptu.device)[0],
            n,
        )
        kept_bfs += int(reference.unique().numel())

        # The edges only depend on the features: thresholding the cosine similarities
        # once spares `max_iters` recomputations of the very same masks.
        right_mask, bottom_mask = cosine_similarity_masks(
            x_feat, threshold, grid, grid, b, n, ptu.device
        )

        smallest = None
        previous = None
        for iters in range(1, max_iters + 1):
            labels = fast_sv_to_canonical(
                connected_components_labels(right_mask, bottom_mask, grid, iters)[0],
                positions,
            )
            kept = int((labels == positions).sum())
            matched = bool(torch.equal(labels, reference))  # pylint: disable=E1101
            kept_fastsv[iters] += kept
            if matched:
                matches[iters] += 1
                if smallest is None:
                    smallest = iters

            # Pointers only decrease and the component minimum is a fixed point, so
            # an iteration that changes nothing means every larger budget gives this
            # very grouping: fill the remaining ones instead of recomputing them.
            if previous is not None and torch.equal(  # pylint: disable=E1101
                labels, previous
            ):
                kept_fastsv[iters + 1 :] += kept
                if matched:
                    matches[iters + 1 :] += 1
                break
            previous = labels
        minimal_iters.append(smallest if smallest else max_iters + 1)

    return matches, kept_fastsv, np.array(minimal_iters), kept_bfs, n_tokens


def print_report(
    matches: np.ndarray,
    kept_fastsv: np.ndarray,
    minimal_iters: np.ndarray,
    kept_bfs: int,
    n_tokens: int,
    max_iters: int,
):
    """Print the calibration table and the recommended budget.

    Args:
        matches (np.ndarray): Number of exactly reproduced images per budget.
        kept_fastsv (np.ndarray): Number of tokens kept per budget.
        minimal_iters (np.ndarray): Per-image minimal budget.
        kept_bfs (int): Number of tokens kept by the Breadth-First Search.
        n_tokens (int): Number of tokens before merging.
        max_iters (int): Highest number of FastSV iterations tried.
    """
    n_images = len(minimal_iters)
    print(f"{n_images} images loaded with {n_tokens} tokens before merging.")
    print(
        f"Reference (NetworkX connected components): {kept_bfs / n_images:.1f} "
        "tokens kept per image on average\n"
    )
    print(f"{'iters':>6}  {'images reproduced':>18}  {'extra tokens':>13}")
    for iters in range(1, max_iters + 1):
        extra = 100 * (kept_fastsv[iters] - kept_bfs) / kept_bfs
        print(
            f"{iters:>6}  {100 * matches[iters] / n_images:>17.1f}%  "
            f"{extra:>+12.3f}%"
        )

    converged = minimal_iters[minimal_iters <= max_iters]
    if len(converged) < n_images:
        print(
            f"\n{n_images - len(converged)} image(s) still differ at "
            f"--max-iters {max_iters}: raise it and run again."
        )
        return

    recommended = int(minimal_iters.max())
    print(
        f"Per-image minimum: mean {minimal_iters.mean():.1f} | "
        f"p50 {np.percentile(minimal_iters, 50):.0f} | "
        f"p95 {np.percentile(minimal_iters, 95):.0f} | "
        f"max {recommended}"
    )
    print(
        f"Library default: ceil(log_1.5(n)) + 2 = "
        f"{math.ceil(math.log(max(n_tokens, 2)) / math.log(1.5)) + 2}"
    )
    print(f"\nSuggested option: --num-iters {recommended}")


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--patch-type", default="graph", type=str)
@click.option("--selected-layer", default=2, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
@click.option("--max-iters", default=20, type=int)
@click.option("--n-images", default=0, type=int)
def main(
    model_path,
    dataset_name,
    patch_type,
    selected_layer,
    threshold,
    prop_attn,
    iprop_attn,
    max_iters,
    n_images,
):
    """Find the cheapest FastSV budget reproducing the PyTorch grouping.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
        max_iters (int): Highest number of FastSV iterations to try.
        n_images (int): Number of images to process (0 for the whole dataset).
    """
    ptu.set_gpu_mode(True)

    root_dir = os.getenv("DATASET")
    dataset_path, dataset_txt_path = get_dataset_inference_path(dataset_name, root_dir)

    model, variant = load_model(model_path)
    input_size = variant["dataset_kwargs"]["crop_size"]
    stats = STATS[variant["dataset_kwargs"]["normalization"]]

    if patch_type != "graph":
        raise ValueError("This script accept only models with G2TM applied.")

    # The Breadth-First Search variant is the reference, the FastSV one is
    # called directly on the token features it would have received.
    g2tm.graph_setr_patch(model, selected_layer, threshold, prop_attn, iprop_attn)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    validation_loader = dataset_prepare(
        dataset_path, dataset_txt_path, stats, 1, input_size, shuffle=False
    )
    print_report(
        *compare_variants(model, validation_loader, threshold, max_iters, n_images),
        max_iters,
    )


if __name__ == "__main__":
    main()  # pylint: disable=E1120
