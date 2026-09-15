"""
Compute the token statistics for an EOMT model.

Uses EoMT's LightningCLI to instantiate the model and validation datamodule
without running training.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List
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


@dataclass
class TokenStats:
    """Dataclass containing token statistics of a model."""

    mean: int
    median: int
    min: int
    max: int
    q05: int
    q95: int
    num_images: int
    num_crops: int


def fill_stats(li: list, q: float = 0.9) -> dict:
    """Compute statistics from list of values.
    Args:
        li (list): List of values.
        q (float): Quantile.

    Returns:
        dict: Dictionnary containing the statistics.
    """
    return {
        "mean": np.mean(li),
        "median": np.median(li),
        f"q{int(100-q*100)}": np.quantile(li, 1 - q),
        f"q{int(q*100)}": np.quantile(li, q),
    }


def token_statistics(
    model: LightningModule,
    datamodule: LightningDataModule,
    layer_id: int,
    device: torch.device,
    max_images: int | None,
) -> TokenStats:
    """Compute the token statistics of a model.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        datamodule (LightingDataModule): The data loader containing several splits.
        layer_id (int): The ID of the layer to analyze (0-based).
        device (torch.device): The device to use for computation.
        max_images (int | None): Optionally process only the first N images. Defaults
            to all images.

    Returns:
        TokenStats: Dataclass containing the mean, median, min, max and quantiles of
            the number of tokens in each crops (patch tokens only), as well as the
            number of crops processed.
    """
    net = unwrap_network(model)
    model.eval().to(device)

    token_counts = []
    token_sizes = []
    num_images, num_crops = 0, 0
    print_warning = False

    for img in tqdm(iter_images(datamodule), position=0, leave=False):
        crops = make_input(model, img, device)

        for crop in crops:
            # The network expects inputs in [0, 1] (see LightningModule.forward)
            x, _, _ = net.get_feature_map(crop / 255.0, layer_id)
            s = net.info["size"] if hasattr(net, "info") else None

            if s is None:
                print_warning = True
                head_start = len(net.encoder.backbone.blocks) - net.num_blocks
                num_prefix_tokens = net.encoder.backbone.num_prefix_tokens + (
                    net.num_q if layer_id >= head_start else 0
                )
                token_counts.append(x.size(1) - num_prefix_tokens)
            else:
                token_counts.append(s.size(1))
                # Only keep true merged-group sizes (> 1). Size-0 entries exist
                # in the padded (batch > 1) path for merged-away tokens and
                # would pin the mean to 1; with b == 1 crops the sparse path is
                # taken and sizes are honest, but filter defensively anyway.
                token_sizes += [v for v in s[0].tolist() if v > 1]

            num_crops += 1

        num_images += 1

        if max_images is not None and num_images >= max_images:
            break

    if print_warning:
        print(
            "WARNING: token size tensor is None, either the module has not"
            " find any fusion in this image, or no fusion module has been"
            " trigered. In this case, consider increasing the layer ID."
        )

    if num_images == 0:
        raise RuntimeError("The validation dataloader did not yield any images.")
    if len(token_sizes) == 0:
        token_sizes.append(np.nan)

    return (
        TokenStats(
            np.mean(token_counts),
            np.median(token_counts),
            np.min(token_counts),
            np.max(token_counts),
            np.quantile(token_counts, 0.05),
            np.quantile(token_counts, 0.95),
            num_images,
            num_crops,
        ),
        TokenStats(
            np.mean(token_sizes),
            np.median(token_sizes),
            np.min(token_sizes),
            np.max(token_sizes),
            np.quantile(token_sizes, 0.05),
            np.quantile(token_sizes, 0.95),
            num_images,
            num_crops,
        ),
    )


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Compute token count and merged-token size statistics."
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
        "--layer_id",
        type=int,
        help="Layer to compute token statistics (1-based).",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally profile only the first N validation images. "
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

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    return args, cli_args


def main() -> None:
    """Main function to compute the token statistics of an EOMT model."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    net = unwrap_network(model)

    if not "--model.patch_type" in cli_args:
        raise ValueError(
            "No token reduction applied. This script has no "
            "interest for vanilla models with constant token "
            f"sequence length {net.n_patch_tokens}."
        )

    num_stats, size_stats = token_statistics(
        model, datamodule, args.layer_id, device, args.max_images
    )

    print("Model           : EoMT", "+ G2TM" if hasattr(net, "info") else "")
    print(f"Input           : validation dataloader images on {device}")
    print(f"Number of images: {num_stats.num_images}")
    print(
        f"Number of crops : {num_stats.num_crops} "
        f"({num_stats.num_crops / num_stats.num_images:.3f} per image)"
    )
    print("Statistics on number of tokens:")
    print(f"\tMean  : {num_stats.mean:.3f}")
    print(f"\tMedian: {num_stats.median:.3f}")
    print(f"\tMin   : {num_stats.min:.3f}")
    print(f"\tMax   : {num_stats.max:.3f}")
    print(f"\tQ05   : {num_stats.q05:.3f}")
    print(f"\tQ95   : {num_stats.q95:.3f}")
    print("Statistics on size of tokens:")
    print(f"\tMean  : {size_stats.mean:.3f}")
    print(f"\tMedian: {size_stats.median:.3f}")
    print(f"\tMin   : {size_stats.min:.3f}")
    print(f"\tMax   : {size_stats.max:.3f}")
    print(f"\tQ05   : {size_stats.q05:.3f}")
    print(f"\tQ95   : {size_stats.q95:.3f}")


if __name__ == "__main__":
    main()
