# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------
"""Script to compute the throughput of an EoMT model."""

from os import environ
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List
from tqdm import tqdm

import numpy as np
import torch
from torch.amp import autocast

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

environ.setdefault("TORCH_LOGS", "-dynamo")


@dataclass
class FPSStats:
    """Dataclass containing FPS statistics of a model."""

    mean_fps: float
    median_fps: float
    min_fps: float
    max_fps: float
    num_images: int
    num_crops: int


@torch.inference_mode()
def model_warmup(
    model: LightningModule,
    inputs: list[torch.Tensor],
    device: torch.device,
    warmup: int = 50,
    use_amp: bool = False,
) -> None:
    """Prime CUDA kernels and torch.compile caches before benchmarking.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        inputs (list[torch.Tensor]): List of crops as inputs of the model.
        device (torch.device): The device to use for computation.
        repeat (int): Number of times the measure is repeated for the same crop.
        use_amp (bool): Whether to use PyTorch AMP.
    """

    def _forward() -> None:
        for crop in inputs:
            if use_amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    model(crop)
            else:
                model(crop)

    if "cuda" in device.type:
        torch.cuda.synchronize()

    for _ in range(warmup):
        _forward()

    if "cuda" in device.type:
        torch.cuda.synchronize()


@torch.inference_mode()
def measure_fps(
    model: LightningModule,
    inputs: list[torch.Tensor],
    device: torch.device,
    repeat: int = 32,
    use_amp: bool = False,
) -> List:
    """Measures forward-pass throughput for the crops of a single image, processed one
    at a time (batch size 1), assuming warmup was already done.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        inputs (list[torch.Tensor]):
        device (torch.device): The device to use for computation.
        repeat (int): Number of times the measure is repeated for the same crop.
        use_amp (bool): Whether to use PyTorch AMP.

    Returns:
        List: Throughput scores of the model for each crop in crop per second (FPS).
    """

    def _forward(img: torch.Tensor) -> None:
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                model(img)
        else:
            model(img)

    times_s = []
    if "cuda" in device.type:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for crop in inputs:
            for _ in range(repeat):
                torch.cuda.synchronize()
                starter.record()
                _forward(crop)
                ender.record()
                torch.cuda.synchronize()
                times_s.append(starter.elapsed_time(ender) / 1000.0)
    else:
        for crop in inputs:
            for _ in range(repeat):
                t0 = time.perf_counter()
                _forward(crop)
                t1 = time.perf_counter()
                times_s.append(t1 - t0)

    # Each entry of times_s is the latency in SECONDS of a SINGLE crop, so
    # its reciprocal is directly the throughput in crops per second.

    return 1.0 / np.asarray(times_s, dtype=np.float64)


def compute_fps_stats(
    model: LightningModule,
    datamodule: LightningDataModule,
    device: torch.device,
    warmup: int = 100,
    repeat: int = 32,
    amp: bool = False,
    max_images: int = None,
) -> FPSStats:
    """Compute the throughput statistics of a model.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        datamodule (LightningDataModule): The data loader containing several splits.
        device (torch.device): The device to use for computation.
        layer_id (int): The ID of the layer to analyze (0-based).
        repeat (int): Number of times the measure is repeated for the same crop.
        use_amp (bool): Whether to use PyTorch AMP.
        max_images (int | None): Optionally profile only the first N images. Defaults
            to all images.

    Returns:
        FPSStats: Dataclass containing the mean, median, min and max FPS, as well as
            the number of images and crops processed."""
    model.eval().to(device)
    use_amp = amp and "cuda" in device.type
    val_iter = iter_images(datamodule)

    # Warmup once on the first image
    first_img = next(val_iter)
    first_input = make_input(model, first_img, device)
    model_warmup(model, first_input, device, warmup, use_amp)

    fps = np.asarray([], dtype=np.float64)
    num_images, num_crops = 0, 0

    for img in tqdm(val_iter, position=0, leave=False):
        x = make_input(model, img, device)
        new_fps = measure_fps(model, x, device, repeat, use_amp)
        fps = np.concatenate((fps, new_fps))

        num_images += 1
        num_crops += len(x)

        if max_images is not None and num_images >= max_images:
            break

    return FPSStats(
        fps.mean().item(),
        np.median(fps).item(),
        fps.min().item(),
        fps.max().item(),
        num_images,
        num_crops,
    )


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(description="EoMT's throughput measurement.")
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
        default="",
        help="Dataset root directory. Equivalent to the LightningCLI "
        "--data.path override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to place the model and input on. Default: cuda",
    )
    parser.add_argument("--amp", action="store_true", help="AMP FP16 (CUDA)")
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Number of warmup steps before measuring throughput.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="Number of times the inference is repeated for a single image.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally profile only the first N validation images. "
        "Defaults to all images.",
    )
    args, cli_args = parser.parse_known_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be greater than 0.")

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    return args, cli_args


def main() -> None:
    """Main function to compute the throughput of an EOMT model."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    net = unwrap_network(model)

    stats = compute_fps_stats(
        model, datamodule, device, args.warmup, args.repeat, args.amp, args.max_images
    )

    print("Model: EoMT", "+ G2TM" if hasattr(net, "info") else "")
    print(f"Input           : validation dataloader images on {device}")
    print(f"Number of images: {stats.num_images}")
    print(
        f"Number of crops : {stats.num_crops} "
        f"({stats.num_crops / stats.num_images:.3f} per image)"
    )
    print("Batch size      : 1 (crops processed one at a time)")
    print(f"Warmup steps    : {args.warmup}")
    print(f"Repeats         : {args.repeat}")
    print(f"AMP FP16        : {args.amp and 'cuda' in device.type}")
    print(f"Avg FPS (crops/s): {stats.mean_fps:.2f}")
    print(f"Min FPS (crops/s): {stats.min_fps:.2f}")
    print(f"Max FPS (crops/s): {stats.max_fps:.2f}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
