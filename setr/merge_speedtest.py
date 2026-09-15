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
Script to compare the execution time of the three G2TM merge implementations.

The token features handed to G2TM are captured on real images, then replayed
through `nx_merge`, `bfs_merge` and `fast_sv_merge`, so that the three
implementations are timed on exactly the same inputs. Only the merging step
is measured, not the surrounding encoder.

The merging functions write into their input sequence, so a fresh copy is
made before every call, outside of the measured window.

Example: see README.md
"""

import importlib
import os
import time
from contextlib import contextmanager
from typing import Callable, Dict, List

import click
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import g2tm
from g2tm.nx_merge import nx_merge
from g2tm.bfs_merge import bfs_merge
from g2tm.fast_sv_merge import fast_sv_merge

from setr.data.utils import STATS
from setr.inference_utils import get_dataset_inference_path, dataset_prepare
from setr.model.factory import load_model
import setr.utils.torch as ptu


@contextmanager
def record_merge_inputs(records: list):
    """Record the token sequences handed to G2TM, without altering the model.

    The merging function is looked up in the globals of the patch module at
    every call, so temporarily replacing it there captures its inputs while
    still running the original implementation.

    Args:
        records (list): List filled with one (feat, is_encoder, distill_token)
            tuple per call to the merging function.
    """
    patch_module = importlib.import_module("g2tm.patch.graph_setr_patch")
    original = patch_module.bfs_merge

    def recorder(feat, threshold, is_encoder=True, distill_token=False):
        records.append((feat.detach().clone(), is_encoder, distill_token))
        return original(feat, threshold, is_encoder, distill_token)

    patch_module.bfs_merge = recorder
    try:
        yield
    finally:
        patch_module.bfs_merge = original


@torch.no_grad()
def capture_features(
    model: nn.Module, validation_loader: DataLoader, n_images: int
) -> List[tuple]:
    """Collect the token sequences G2TM receives on real images.

    Args:
        model (nn.Module): PyTorch model patched with the G2TM module.
        validation_loader (DataLoader): PyTorch dataloader.
        n_images (int): Number of batches to capture.

    Returns:
        List[tuple]: One (feat, is_encoder, distill_token) tuple per batch.
    """
    captured = []
    if n_images:
        loader = (im for i, im in enumerate(validation_loader) if i < n_images)
    else:
        loader = validation_loader
    for image in tqdm(loader, total=n_images, desc="Capturing", leave=False):
        records = []
        with record_merge_inputs(records):
            model(image.to(ptu.device))
        if not records:
            raise RuntimeError(
                "G2TM was never called: check --selected-layer and --patch-type."
            )
        captured.append(records[0])
    return captured


@torch.no_grad()
def time_implementation(
    merge_fn: Callable, captured: List[tuple], threshold: float, repeats: int
) -> Dict[str, float]:
    """Measure the execution time of one merge implementation.

    Args:
        merge_fn (Callable): Merging function to measure.
        captured (List[tuple]): Captured token sequences.
        threshold (float): Threshold parameter for G2TM.
        repeats (int): Number of measured calls per captured sequence.

    Returns:
        Dict[str, float]: Median, mean and 95th percentile latency in
            milliseconds, and the mean number of tokens kept.
    """
    latencies = []
    kept = []

    # Warm-up: first calls pay for the CUDA context and the memory allocator.
    for feat, is_encoder, distill_token in captured[: min(3, len(captured))]:
        merge_fn(feat.clone(), threshold, is_encoder, distill_token)
    if ptu.use_gpu:
        torch.cuda.synchronize()

    for feat, is_encoder, distill_token in tqdm(
        captured, total=len(captured), desc="Processing", leave=False
    ):
        for _ in range(repeats):
            # Cloned outside the measured window: the merge writes in place.
            work = feat.clone()
            if ptu.use_gpu:
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = merge_fn(work, threshold, is_encoder, distill_token)
            if ptu.use_gpu:
                torch.cuda.synchronize()
            latencies.append(1e3 * (time.perf_counter() - start))
        kept.append(out[0].size(1))

    return {
        "median": float(np.median(latencies)),
        "mean": float(np.mean(latencies)),
        "p95": float(np.percentile(latencies, 95)),
        "tokens": float(np.mean(kept)),
    }


def print_report(results: Dict[str, Dict[str, float]], n_calls: int, batch_size: int):
    """Print the timing table, slowest implementation first.

    Args:
        results (Dict[str, Dict[str, float]]): Measurements per implementation.
        n_calls (int): Number of measured calls per implementation.
    """
    order = sorted(results, key=lambda name: results[name]["median"], reverse=True)
    slowest = results[order[0]]["median"]
    fastest = results[order[-1]]["median"]

    print(f"\n{n_calls} measured calls per implementation\n")
    print(
        f"{'implementation':<16}{'median':>10}{'mean':>10}{'p95':>10}{'speed-up':>10}"
        + (f"{'tokens kept':>13}" if batch_size == 1 else "")
    )
    for name in order:
        res = results[name]
        print(
            f"{name:<16}{res['median']:>9.2f}ms{res['mean']:>9.2f}ms"
            f"{res['p95']:>9.2f}ms{slowest / res['median']:>9.1f}x"
            + (f"{res['tokens']:>13.1f}" if batch_size == 1 else "")
        )
    print(
        f"\nFastest: {order[-1]} ({fastest:.2f} ms, "
        f"{slowest / fastest:.1f}x faster than {order[0]})"
    )

    tokens = {res["tokens"] for res in results.values()}
    if max(tokens) - min(tokens) > 0.05:
        print(
            "NOTE: the implementations do not keep the same number of tokens,"
            " so they are not merging identically (see --num-iters)."
        )


@click.command()
@click.argument("model_path", type=str)
@click.argument("dataset_name", type=str)
@click.option("--selected-layer", default=2, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--num-iters", default=None, type=int)
@click.option("--batch-size", default=1, type=int)
@click.option("--n-images", default=None, type=int)
@click.option("--repeats", default=5, type=int)
def main(
    model_path,
    dataset_name,
    selected_layer,
    threshold,
    num_iters,
    batch_size,
    n_images,
    repeats,
):
    """Compare the execution time of the three G2TM merge implementations.

    Args:
        model_path (str): Path to PyTorch model.
        dataset_name (str): Name of the dataset to use.
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        num_iters (int): Number of FastSV iterations.
        batch_size (int): Number of images per batch (1 pops the merged
            tokens, more than 1 keeps and masks them).
        n_images (int): Number of batches to capture.
        repeats (int): Number of measured calls per captured sequence.
    """
    ptu.set_gpu_mode(True)

    root_dir = os.getenv("DATASET")
    dataset_path, dataset_txt_path = get_dataset_inference_path(dataset_name, root_dir)

    model, variant = load_model(model_path)
    input_size = variant["dataset_kwargs"]["crop_size"]
    stats = STATS[variant["dataset_kwargs"]["normalization"]]

    g2tm.graph_setr_patch(model, selected_layer, threshold)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    validation_loader = dataset_prepare(
        dataset_path, dataset_txt_path, stats, batch_size, input_size, shuffle=False
    )
    captured = capture_features(model, validation_loader, n_images)
    feat = captured[0][0]
    print(f"Device: {torch.cuda.get_device_name(ptu.device) if ptu.use_gpu else 'cpu'}")
    print(f"Captured {len(captured)} sequence(s) of shape {tuple(feat.shape)}")

    implementations = {
        "nx_merge": nx_merge,
        "bfs_merge": bfs_merge,
        # The FastSV budget is an argument of the implementation, not of the
        # comparison: it is bound here to keep one common calling convention.
        "fast_sv_merge": lambda feat, thr, enc, dist: fast_sv_merge(
            feat, thr, num_iters, enc, dist
        ),
    }
    results = {}
    for name, merge_fn in implementations.items():
        results[name] = time_implementation(merge_fn, captured, threshold, repeats)
    print_report(results, len(captured) * repeats, batch_size)


if __name__ == "__main__":
    main()  # pylint: disable=E1120
