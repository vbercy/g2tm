"""Script to calibrate the number of FastSV iterations of the ONNX variant of G2TM,
reporting how often the FastSV grouping is identical to the reference one.

An insufficient number of iterations can split a component in several groups (never
merge across components), so the exported model can merge *fewer* tokens than the
PyTorch one, leading to both slightly more accurate and slightly slower model than the
one it is supposed to reproduce (to confirm on hardware).

The reference is the custom BFS implementation, which is the default of this
repository and labels every token with the smallest index of its component. The
NetworkX one produces the very same grouping, so either can be used as reference.

Usage (additional LightningCLI overrides can be appended):
  CUDA_VISIBLE_DEVICES=0 python ./tools/fastsv_iters.py
  -c ./configs/augreg/ade20k/semantic/eomt_large_512.yaml
  --data_path $DATASET
  --ckpt_path ./checkpoints/semantic/ade20k/augreg/vitL/eomt_augreg_B16/model.ckpt
  --max_iters 20 --max_images 100
  --model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
"""

from __future__ import annotations
import argparse
import importlib
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=C0413

from main import LightningDataModule, LightningModule
from tools.utils import (
    has_cli_override,
    build_from_yaml,
    unwrap_network,
    iter_images,
    make_input,
)
from g2tm.bfs_merge import get_labels as bfs_get_labels
from g2tm.utils import cosine_similarity_masks
from g2tm.connected_components import connected_components_labels


class _FeaturesCaptured(Exception):
    """Raised to unwind the forward pass once G2TM has received its input."""


@contextmanager
def record_merge_inputs(records: list):
    """Record the token sequences handed to G2TM, without altering the model.

    The merging functions are looked up in the globals of the patch module at every
    call, so temporarily replacing them there captures their inputs. Both `bfs_merge`
    and `fast_sv_merge` are replaced, so the recording does not depend on the
    `--model.method` the model was patched with. Only the features matter here, both
    groupings being recomputed from them below, so the recorder unwinds the forward
    pass instead of merging: the blocks after the G2TM one, the Mask Module and the
    merge itself are never run.

    Args:
        records (list): List filled with one (feat, n_protected, grid_size) tuple
            per call to a merging function.
    """
    patch_module = importlib.import_module("g2tm.patch.graph_eomt_patch")
    originals = {
        name: getattr(patch_module, name)
        for name in ("bfs_merge", "fast_sv_merge")
    }

    def recorder(feat, threshold, n_protected, grid_size, *args, **kwargs):  # pylint: disable=W0613
        records.append((feat.detach(), n_protected, grid_size))
        raise _FeaturesCaptured

    for name in originals:
        setattr(patch_module, name, recorder)
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(patch_module, name, original)


def to_canonical(labels: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Rename groups by the smallest index they contain.

    Two groupings of the same tokens are identical if and only if their canonical
    labels are equal, whatever representative each implementation chose for a group.
    FastSV pointers are not guaranteed to have reached the component minimum when the
    iteration budget is too small, hence the renaming.

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


@torch.no_grad()
def compare_variants(
    model: LightningModule,
    datamodule: LightningDataModule,
    device: torch.device,
    threshold: float,
    max_iters: int,
    max_images: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Compare the FastSV grouping with the Breadth-First Search one.

    Both variants are fed with the very same token features, so that the only
    difference measured is the connected components algorithm itself.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        datamodule (LightningDataModule): The data loader containing several splits.
        device (torch.device): The device to use for computation.
        threshold (float): Threshold parameter for G2TM.
        max_iters (int): Highest number of FastSV iterations to try.
        max_images (int | None): Number of images to process (None for the whole
            validation split).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]: Per candidate budget, the
            number of exactly reproduced crops and the total number of tokens kept;
            then the per-crop minimal budget, the number of tokens kept by the
            Breadth-First Search and the number of tokens before merging.
    """
    matches = np.zeros(max_iters + 1, dtype=np.int64)
    kept_fastsv = np.zeros(max_iters + 1, dtype=np.int64)
    minimal_iters = []
    kept_bfs = 0
    n_tokens = None
    num_images = 0

    for img in tqdm(iter_images(datamodule), position=0, leave=False):
        for crop in make_input(model, img, device):
            records = []
            with record_merge_inputs(records):
                try:
                    model(crop)
                except _FeaturesCaptured:
                    pass
            if not records:
                raise RuntimeError(
                    "G2TM was never called: check --model.patch_type and "
                    "--model.selected_layer."
                )

            feat, n_protected, grid_size = records[0]
            x_feat = feat[:, n_protected:, :]
            b, n, _ = x_feat.size()
            grid_h, grid_w = grid_size
            n_tokens = n
            positions = torch.arange(  # pylint: disable=E1101
                n, device=device, dtype=torch.int64
            )

            reference = to_canonical(
                bfs_get_labels(x_feat, threshold, grid_h, grid_w, b, n, device)[0],
                positions,
            )
            kept_bfs += int((reference == positions).sum())

            # The edges only depend on the features: thresholding the cosine
            # similarities once spares `max_iters` recomputations of the very same
            # masks.
            right_mask, bottom_mask = cosine_similarity_masks(
                x_feat, threshold, grid_h, grid_w, b, n, device
            )

            smallest = None
            previous = None
            for iters in range(1, max_iters + 1):
                labels = to_canonical(
                    connected_components_labels(
                        right_mask, bottom_mask, grid_w, iters
                    )[0],
                    positions,
                )
                kept = int((labels == positions).sum())
                matched = bool(torch.equal(labels, reference))  # pylint: disable=E1101
                kept_fastsv[iters] += kept
                if matched:
                    matches[iters] += 1
                    if smallest is None:
                        smallest = iters

                # Pointers only decrease and the component minimum is a fixed point,
                # so an iteration that changes nothing means every larger budget gives
                # this very grouping: fill the remaining ones instead of recomputing
                # them.
                if previous is not None and torch.equal(  # pylint: disable=E1101
                    labels, previous
                ):
                    kept_fastsv[iters + 1 :] += kept
                    if matched:
                        matches[iters + 1 :] += 1
                    break
                previous = labels
            minimal_iters.append(smallest if smallest else max_iters + 1)

        num_images += 1
        if max_images is not None and num_images >= max_images:
            break

    if not minimal_iters:
        raise RuntimeError("The validation dataloader did not yield any image.")

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
        matches (np.ndarray): Number of exactly reproduced crops per budget.
        kept_fastsv (np.ndarray): Number of tokens kept per budget.
        minimal_iters (np.ndarray): Per-crop minimal budget.
        kept_bfs (int): Number of tokens kept by the Breadth-First Search.
        n_tokens (int): Number of tokens before merging.
        max_iters (int): Highest number of FastSV iterations tried.
    """
    n_crops = len(minimal_iters)
    print(f"{n_crops} crops processed with {n_tokens} tokens before merging.")
    print(
        f"Reference (custom BFS connected components): {kept_bfs / n_crops:.1f} "
        "tokens kept per crop on average\n"
    )
    print(f"{'iters':>6}  {'crops reproduced':>17}  {'extra tokens':>13}")
    for iters in range(1, max_iters + 1):
        extra = 100 * (kept_fastsv[iters] - kept_bfs) / kept_bfs
        print(
            f"{iters:>6}  {100 * matches[iters] / n_crops:>16.1f}%  {extra:>+12.3f}%"
        )

    converged = minimal_iters[minimal_iters <= max_iters]
    if len(converged) < n_crops:
        print(
            f"\n{n_crops - len(converged)} crop(s) still differ at "
            f"--max_iters {max_iters}: raise it and run again."
        )
        return

    recommended = int(minimal_iters.max())
    print(
        f"\nPer-crop minimum: mean {minimal_iters.mean():.1f} | "
        f"p50 {np.percentile(minimal_iters, 50):.0f} | "
        f"p95 {np.percentile(minimal_iters, 95):.0f} | "
        f"max {recommended}"
    )
    print(
        f"Library default: ceil(log_1.5(n)) + 2 = "
        f"{math.ceil(math.log(max(n_tokens, 2)) / math.log(1.5)) + 2}"
    )
    print(f"\nSuggested option: --model.num_iters {recommended}")


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Find the cheapest FastSV budget reproducing the PyTorch grouping."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config used by the LightningCLI.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Dataset root directory. Equivalent to the LightningCLI "
        "--data.path override.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint. Equivalent to the LightningCLI "
        "--model.ckpt_path override.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="Highest number of FastSV iterations to try.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally process only the first N validation images. "
        "Defaults to all images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to place the model and input on. Default: cuda",
    )
    args, cli_args = parser.parse_known_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be greater than 0.")

    if args.max_iters <= 0:
        parser.error("--max_iters must be greater than 0.")

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    if args.ckpt_path is not None and not has_cli_override(
        cli_args, "--model.ckpt_path"
    ):
        cli_args.extend(["--model.ckpt_path", args.ckpt_path])

    return args, cli_args


def main() -> None:
    """Calibrate the number of FastSV iterations against the BFS grouping."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    model.eval().to(device)
    net = unwrap_network(model)

    if not hasattr(net, "info"):
        raise ValueError(
            "This script only accepts models with G2TM applied. "
            "Add '--model.patch_type graph' to the command line."
        )

    # Both groupings are recomputed from the very same features, so the Mask Module
    # never runs: turning masked attention off only spares its evaluations.
    net.masked_attn_enabled = False

    for param in model.parameters():
        param.requires_grad = False

    print(f"Model           : EoMT + G2TM on {device}")
    print(f"Merging         : layer {model.selected_layer}, "
          f"threshold {model.threshold:.2f}")
    print(f"Budgets tried   : 1 to {args.max_iters}\n")

    print_report(
        *compare_variants(
            model,
            datamodule,
            device,
            model.threshold,
            args.max_iters,
            args.max_images,
        ),
        args.max_iters,
    )


if __name__ == "__main__":
    main()
