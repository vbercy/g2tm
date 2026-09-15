"""
Script to compute FLOPs and parameter counts for an EOMT model.

Usage (additional LightningCLI overrides can be appended):
  CUDA_VISIBLE_DEVICES=0 python ./tools/compute_flops.py
  -c ./configs/dinov2/ade20k/semantic/eomt_large_512.yaml
  --data_path $DATASET
  --model.ckpt_path ./checkpoints/semantic/ade20k/vitL/g2tm_eomt_dinov2L_L1_T0.99_B16/model.ckpt
  --model.network.masked_attn_enabled False
  --trainer.devices 1 --data.batch_size 1
  --model.patch_type graph --model.selected_layer 1 --model.threshold 0.99

Uses EoMT's LightningCLI to instantiate the model and validation datamodule
without running training.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import nn
from fvcore.nn import FlopCountAnalysis, parameter_count, parameter_count_table
from fvcore.nn.jit_handles import get_shape

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
    EoMTHead,
)


@dataclass
class FlopStats:
    """Dataclass containing FLOPs statistics of a model."""

    mean_flops: float
    num_images: int
    num_crops: int
    min_flops: float
    max_flops: float


def sdpa_flop_jit(inputs: list, outputs: list) -> float:  # pylint: disable=W0613
    """FLOP (MAC) count for aten::scaled_dot_product_attention.

    fvcore ships no handler for fused attention, so timm's default `fused_attn=True`
    path would contribute zero FLOPs. With q of shape (..., N, D) and k/v of shape
    (..., M, D), the two matmuls (QK^T and attn@V) each cost N*M*D MACs per batch-head.

    Args:
        inputs (list): Inputs of the fused attention.

    Returns:
        float: Number of flops needed for the operation with respect to the input
            provided.
    """
    q_shape = get_shape(inputs[0])
    k_shape = get_shape(inputs[1])

    n, d = q_shape[-2], q_shape[-1]
    m = k_shape[-2]

    batch = 1
    for size in q_shape[:-2]:
        batch *= size

    return batch * 2 * n * m * d


# Operators fvcore does not know about but that carry real FLOPs
EXTRA_FLOP_HANDLES = ("aten::scaled_dot_product_attention", sdpa_flop_jit)


@torch.inference_mode()
def compute_flops(module: nn.Module, x: torch.Tensor) -> float:
    """Compute the FLOPs of a module over one crop with fvcore.

    Args:
        module (nn.Module): PyTorch Module object.
        x (torch.Tensor): Inputs of the module.

    Returns:
        float: Number of flops needed for the module with respect to the input
            provided.
    """
    fca = FlopCountAnalysis(module, x)
    fca.set_op_handle(*EXTRA_FLOP_HANDLES)
    (
        fca.tracer_warnings("none")
        .unsupported_ops_warnings(False)
        .uncalled_modules_warnings(False)
    )
    return float(fca.total())


@torch.inference_mode()
def compute_model_flops(
    model: nn.Module, crops: list[torch.Tensor], mode: str
) -> list[float]:
    """Return the FLOPs of every sliding-window crop of one image, each
    processed at batch size 1.

    Args:
        model (nn.Module): The EoMT model.
        crops (list[torch.Tensor]): List of sliding-window crops of one image.
        mode (str): Whether to compute flops over images or crops.

    Returns:
        list[float]: List of FLOPs for each crop.
    """
    if mode == "crop":
        return [compute_flops(model, crop) for crop in crops]
    return sum(compute_flops(model, crop) for crop in crops)


@torch.inference_mode()
def compute_head_flops(
    head: nn.Module,
    net: nn.Module,
    crops: list[torch.Tensor],
    head_start: int,
    mode: str,
) -> list[float]:
    """Return the FLOPs of the last `model.network.num_blocks` transformer
    blocks plus the Mask Module, for every crop of one image (batch size 1).

    The embedding and earlier backbone blocks are executed to produce the same
    token sequence seen by the head, but those prefix operations are not counted.

    NOTE: this is only additive with the full-model count when
    `masked_attn_enabled` is False, i.e. EoMT's inference behaviour. With
    masked attention on, the network evaluates the Mask Module once more
    before each of the last blocks, which the head module does not replicate.

    Args:
        head (nn.Module): The EoMTHead module.
        net (nn.Module): The full EoMT network.
        crops (list[torch.Tensor]): List of sliding-window crops of one image.
        head_start (int): Index of the first block of the head.
        mode (str): Whether to compute flops over images or crops.

    Returns:
        list[float]: List of FLOPs for each crop.
    """
    if mode == "crop":
        flops = []
    else:
        total_flops = 0.0

    for crop in crops:
        # The network expects inputs in [0, 1] (see LightningModule.forward)
        x, attn_mask, rope = net.get_feature_map(crop / 255.0, head_start)
        if hasattr(net, "info"):
            x, attn_mask = net.add_query_tokens(x)
        else:
            x = torch.cat(
                (net.q.weight[None, :, :].expand(x.size(0), -1, -1), x), dim=1
            )

        if mode == "crop":
            flops.append(compute_flops(head, (x, attn_mask, rope)))
        else:
            total_flops += compute_flops(head, (x, attn_mask, rope))

    if mode == "crop":
        return flops
    else:
        return total_flops


def compute_average_flops(
    model: LightningModule,
    datamodule: LightningDataModule,
    device: torch.device,
    max_images: int | None,
    mode: str,
    only_head: bool,
) -> FlopStats:
    """Compute the average FLOPs per crop or per image over the validation set.

    Args:
        model (LightningModule): The EoMT model.
        datamodule (LightningDataModule): The data loader containing several splits.
        device (torch.device): Device to place the model and input on.
        max_images (int | None): Optionally process only the first N images. Defaults
            to all images.
        mode (str): Whether to compute flops over images or crops.
        only_head (bool): Whether to compute flops on head or on whole model.

    Returns:
        FlopStats: Dataclass containing the mean, min and max FLOPs, as well as the
            number of images and crops processed.
    """
    net = unwrap_network(model)
    model.eval().to(device)
    print(parameter_count_table(net, 2))

    if net.masked_attn_enabled:
        print(
            "NOTE: forcing network.masked_attn_enabled = False "
            "(EoMT's inference behaviour)."
        )
        net.masked_attn_enabled = False

    if only_head:
        blocks = net.encoder.backbone.blocks
        head_start = len(blocks) - net.num_blocks

        if head_start < 0:
            raise ValueError(
                "network.num_blocks is larger than the number of transformer blocks."
            )

        class_head = deepcopy(net.class_head)
        mask_head = deepcopy(net.mask_head)
        upscale = deepcopy(net.upscale)
        norm = deepcopy(net.encoder.backbone.norm)
        info = (
            net.info
            if hasattr(net, "info")
            else {"prop_attn": False, "iprop_attn": False, "size": None}
        )
        head = EoMTHead(
            blocks[head_start:],
            class_head,
            mask_head,
            upscale,
            norm,
            info,
            net.num_q,
            net.encoder.backbone.num_prefix_tokens,
            net.encoder.backbone.patch_embed.grid_size,
        )
        head.eval().to(device)

        params = int(parameter_count(head)[""])
    else:
        params = int(parameter_count(net)[""])

    total_flops = 0.0
    num_images = 0
    num_crops = 0
    min_flops = float("inf")
    max_flops = 0.0

    for img in tqdm(iter_images(datamodule), position=0, leave=False):
        crops = make_input(model, img, device)
        if only_head:
            crop_flops = compute_head_flops(head, net, crops, head_start, mode)
        else:
            crop_flops = compute_model_flops(model, crops, mode)

        # Statistics are per crop: an image yields a variable number of
        # sliding-window crops, and each crop is one forward pass.
        total_flops += sum(crop_flops)
        num_crops += len(crop_flops)
        num_images += 1
        min_flops = min(min_flops, min(crop_flops))  # pylint: disable=W3301
        max_flops = max(max_flops, max(crop_flops))  # pylint: disable=W3301

        if max_images is not None and num_images >= max_images:
            break

    if num_images == 0:
        raise RuntimeError("The validation dataloader did not yield any images.")

    return (
        params,
        FlopStats(total_flops / num_crops, num_images, num_crops, min_flops, max_flops),
    )


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Compute FLOPs with fvcore for an EOMT model."
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
        "-m",
        "--mode",
        type=str,
        default="crop",
        choices=("crop", "image"),
        help="Whether to compute flops over images or crops.",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Whether to compute flops on head or on whole model.",
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
        help="Device to place the model and input on.",
    )
    args, cli_args = parser.parse_known_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be greater than 0.")

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    return args, cli_args


def main() -> None:
    """Main function to compute FLOPs and parameter counts for an EOMT model."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    mode = args.mode
    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    net = unwrap_network(model)

    total_params, stats = compute_average_flops(
        model=model,
        datamodule=datamodule,
        device=device,
        max_images=args.max_images,
        mode=mode,
        only_head=args.head,
    )

    print(
        "Model           : EoMT" + ("'s head" if args.head else ""),
        "+ G2TM" if hasattr(net, "info") else "",
    )
    print(f"Input           : validation dataloader images on {device}")
    print(f"Number of images: {stats.num_images}")
    print(
        f"Number of crops : {stats.num_crops} "
        f"({stats.num_crops / stats.num_images:.3f} per image)"
    )
    print("Batch size      : 1 (crops processed one at a time)")
    print(f"Params          : {total_params / 1e6:.3f} M")
    print(f"Avg FLOPs/{mode}  : {stats.mean_flops / 1e9:.3f} G")
    print(f"Min FLOPs/{mode}  : {stats.min_flops / 1e9:.3f} G")
    print(f"Max FLOPs/{mode}  : {stats.max_flops / 1e9:.3f} G")


if __name__ == "__main__":
    main()
