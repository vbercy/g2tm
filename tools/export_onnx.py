"""Script to export an EoMT model (with or without G2TM) to the ONNX format.

The exported graph takes a single float32 0-255 crop (1, 3, H, W) and returns the
per-pixel class logits (1, num_classes, H, W), i.e. what EoMT's semantic
evaluation feeds to `revert_window_logits_semantic`. Wrapping the network this
way keeps a single ONNX output instead of the two per-block Python lists the
LightningModule returns.

Notes:
  - G2TM must run its FastSV implementation (`--model.method fastsv`): the BFS
    and NetworkX ones label the connected components in pure Python and are
    decorated with `@torch.compiler.disable`, so they cannot be traced.
  - `masked_attn_enabled` is forced to False, which is EoMT's inference
    behaviour (mask annealing has driven attn_mask_probs to 0 by the end of
    training) and avoids exporting the four extra Mask Module evaluations.

Example: see README.md
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

import onnx
import onnxscript.optimizer
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=C0413

from main import LightningDataModule, LightningModule
from tools.utils import has_cli_override, build_from_yaml, unwrap_network


class ONNXEoMT(nn.Module):
    """Wrap an EoMT LightningModule so that it exports to a single-output graph.

    The LightningModule returns one list of mask logits and one list of class
    logits (one entry per Mask Module evaluation). ONNX has no use for that
    structure at inference: only the last block matters. This module keeps the
    0-255 scaling inside the graph, so the exported model consumes the very same
    crops as `window_imgs_semantic` produces.
    """

    def __init__(self, model: LightningModule, img_size: tuple):
        super().__init__()
        self.model = model
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the per-pixel class logits of one crop.

        Args:
            x (torch.Tensor): float32 crop (1, 3, H, W) with values in [0, 255].

        Returns:
            torch.Tensor: Per-pixel class logits (1, num_classes, H, W).
        """
        mask_logits_per_layer, class_logits_per_layer = self.model(x)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], self.img_size, mode="bilinear"
        )

        return LightningModule.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )


def export_model(
    model: LightningModule,
    onnx_file: Path,
    img_size: tuple,
    device: torch.device,
    opset: int = 18,
    verbose: bool = False,
) -> None:
    """Export an EoMT model to ONNX and constant-fold the resulting graph.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        onnx_file (Path): Path of the ONNX file to write.
        img_size (tuple): Crop height and width expected by the model.
        device (torch.device): Device to place the model and dummy input on.
        opset (int): ONNX opset version.
        verbose (bool): Whether to print the exported graph.
    """

    wrapper = ONNXEoMT(model, img_size).eval().to(device)
    dummy_input = torch.randn(1, 3, *img_size, device=device)  # pylint: disable=E1101

    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Opset >= 18 needed for the min-reduction ScatterElements used by the G2TM
        # connected components (FastSV hooking step).
        # Batch axis deliberately left static (batch size 1, inference anyway), as a
        # symbolic batch forces the token count to be read from the tensor shapes at
        # run time, and that CPU-side scalar feeds GPU ops in the G2TM block, which
        # insert a MemcpyFromHost. Pinning it folds those away entirely.
        # dynamo=False keeps the TorchScript exporter: torch>=2.9 defaults it to
        # True, and the dynamo path rejects the data-dependent token count that
        # G2TM's sparsify produces.
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_file),
            dynamo=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=opset,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],
        )

    for _pkg in ("onnx_ir", "onnxscript"):
        logging.getLogger(_pkg).setLevel(logging.WARNING)

    # Constant folding: removes the dead `If` wrappers PyTorch emits around
    # scatter ops (their "empty src" branch is statically false here).
    onnx_model = onnxscript.optimizer.optimize(onnx.load(str(onnx_file)))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, str(onnx_file))
    if verbose:
        print(onnx.printer.to_text(onnx_model.graph))


def onnx_inference(ort_session, model: LightningModule, img: torch.Tensor):
    """Sliding-window inference of one image with an ONNX model.

    Mirrors `MaskClassificationSemantic.eval_step`, but every crop is run
    through ONNX Runtime instead of the PyTorch network.

    Args:
        ort_session (onnxruntime.InferenceSession): The exported ONNX model.
        model (LightningModule): The EoMT model, used for its windowing helpers.
        img (torch.Tensor): Image tensor (3, H, W) with values in [0, 255].

    Returns:
        torch.Tensor: Per-pixel class logits (num_classes, H, W).
    """
    crops, origins = model.window_imgs_semantic((img,))
    input_name = ort_session.get_inputs()[0].name

    crop_logits = []
    for crop in crops.split(1):
        # window_imgs_semantic yields uint8 crops (PyTorch promotes them in
        # LightningModule.forward); the exported graph is typed float32.
        outputs = ort_session.run(None, {input_name: crop.float().cpu().numpy()})
        crop_logits.append(torch.from_numpy(outputs[0]))  # pylint: disable=E1101
    crop_logits = torch.cat(crop_logits, dim=0).to(img.device)  # pylint: disable=E1101

    return model.revert_window_logits_semantic(crop_logits, origins, [img.shape[-2:]])[
        0
    ]


def eval_onnx_model(
    onnx_file: Path,
    model: LightningModule,
    datamodule: LightningDataModule,
    device: torch.device,
    max_images: int = None,
) -> float:
    """Evaluate an exported ONNX model with the mIoU, to validate the export.

    Args:
        onnx_file (Path): Path to the ONNX model file.
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        datamodule (LightningDataModule): The data loader containing several splits.
        device (torch.device): The device to use for computation.
        max_images (int | None): Optionally evaluate only the first N images.
            Defaults to all images.

    Returns:
        float: The mIoU score of the ONNX model over the validation split.
    """
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    ort_session = ort.InferenceSession(str(onnx_file), providers=providers)
    print("Available providers:", ort.get_available_providers())
    print("Active providers for the session:", ort_session.get_providers())

    metric = MulticlassJaccardIndex(
        num_classes=model.num_classes,
        validate_args=False,
        ignore_index=model.ignore_idx,
        average=None,
    ).to(device)

    datamodule.setup("validate")
    num_images = 0

    for imgs, targets in tqdm(datamodule.val_dataloader(), position=0, leave=False):
        per_pixel_targets = model.to_per_pixel_targets_semantic(
            targets, model.ignore_idx
        )
        for img, target in zip(imgs, per_pixel_targets):
            logits = onnx_inference(ort_session, model, img.to(device))
            metric.update(logits[None, ...], target.to(device)[None, ...])

            num_images += 1
            if max_images is not None and num_images >= max_images:
                break
        if max_images is not None and num_images >= max_images:
            break

    return float(metric.compute().mean())


def default_onnx_name(model: LightningModule, net: nn.Module, config: str) -> str:
    """Build a default file name describing the exported model.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        net (nn.Module): The unwrapped EoMT network.
        config (str): Path to the YAML config used by the LightningCLI.

    Returns:
        str: Name of the ONNX file, without its extension.
    """
    # the config stem already reads like "eomt_large_512"
    name = Path(config).stem
    if hasattr(net, "info"):
        name = (
            f"g2tm_{name}_L{model.selected_layer}_T{model.threshold:.2f}"
            f"_{net.info['num_iters']}it"
        )

    return name


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Export an EoMT model (with or without G2TM) to ONNX."
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
        "--onnx_name",
        type=str,
        default=None,
        help="Name to give to the ONNX model, without extension. Defaults to a "
        "name describing the config and the G2TM parameters.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version. Must be >= 18 for the G2TM FastSV graph.",
    )
    parser.add_argument(
        "--eval_onnx",
        action="store_true",
        help="Evaluate the exported model with ONNX Runtime to validate it.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally evaluate only the first N validation images. "
        "Defaults to all images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to place the model and input on. Default: cuda",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the exported ONNX graph.",
    )
    args, cli_args = parser.parse_known_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be greater than 0.")

    if args.opset < 18:
        parser.error(
            "--opset must be >= 18: the G2TM FastSV connected components rely on "
            "the min-reduction ScatterElements introduced in opset 18."
        )

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    if args.ckpt_path is not None and not has_cli_override(
        cli_args, "--model.ckpt_path"
    ):
        cli_args.extend(["--model.ckpt_path", args.ckpt_path])

    return args, cli_args


def main() -> None:
    """Export an EoMT model to ONNX and optionally validate it with ONNX Runtime."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    model.eval().to(device)
    net = unwrap_network(model)

    if not hasattr(model, "ignore_idx"):
        raise ValueError(
            "This script only supports semantic segmentation models "
            "(MaskClassificationSemantic)."
        )

    # Only the FastSV implementation is tensorized; the BFS and NetworkX ones
    # label the components in Python and are hidden from the compiler, so the
    # tracer would silently bake in the merge of the dummy input.
    if hasattr(net, "info") and not net.info["fast_sv"]:
        raise ValueError(
            "G2TM must use the FastSV implementation to be exported to ONNX. "
            "Add '--model.method fastsv' to the command line."
        )

    # EoMT runs inference without masked attention, and exporting it would bake
    # in the four extra Mask Module evaluations.
    if net.masked_attn_enabled:
        print(
            "NOTE: forcing network.masked_attn_enabled = False "
            "(EoMT's inference behaviour)."
        )
        net.masked_attn_enabled = False

    for param in model.parameters():
        param.requires_grad = False

    onnx_name = args.onnx_name or default_onnx_name(model, net, args.config)
    if args.ckpt_path is not None:
        output_dir = Path(args.ckpt_path).parent
    else:
        output_dir = Path.cwd()
    onnx_file = output_dir / f"{onnx_name}.onnx"

    print("Model           : EoMT", "+ G2TM" if hasattr(net, "info") else "")
    print(
        f"Input           : (1, 3, {model.img_size[0]}, {model.img_size[1]}) in [0, 255]"
    )
    print(f"Opset           : {args.opset}")
    print(f"Output file     : {onnx_file}")

    export_model(model, onnx_file, model.img_size, device, args.opset, args.verbose)
    size_mb = onnx_file.stat().st_size / 1e6
    print(f"Model exported to ONNX ! ({size_mb:.1f} MB)")

    if args.eval_onnx:
        print("Testing the model using ONNXRuntime...")
        miou = eval_onnx_model(onnx_file, model, datamodule, device, args.max_images)
        print(f"ONNX mIoU       : {miou * 100:.1f}")


if __name__ == "__main__":
    main()
