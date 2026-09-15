"""
Predict and save colorized semantic segmentation maps for a folder of images.

Runs an EoMT model (vanilla or G2TM-patched) on every image found in an input
folder (defaults to ./vis/ade20k/images) and saves one PNG per image where
each pixel is colored according to the ADE20K class colormap
(./vis/ade20k/cmap.yml).

Every sliding-window crop is processed at batch size 1, so a G2TM-patched
model runs on the sparse path (merged tokens physically removed from the
sequence), exactly like the other measurement tools.

Usage (additional LightningCLI overrides can be appended):
  CUDA_VISIBLE_DEVICES=0 python ./tools/predict_seg_maps.py
  -c ./configs/dinov2/ade20k/semantic/eomt_large_512.yaml
  --data_path $DATASET
  --ckpt_path ./checkpoints/semantic/ade20k/vitL/eomt_dinov2L_B16/epoch=30-step=39122.ckpt
  --output_dir ./vis/ade20k/preds
  --model.network.masked_attn_enabled False
  [--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88]

Notes:
  - --data_path is only needed to instantiate the datamodule from the config;
    no dataset image is read by this script.
  - --model.network.masked_attn_enabled False matches EoMT's inference mode
    (mask annealing removes masked attention) and is faster.
"""

from __future__ import annotations
import argparse
import sys
from typing import List
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=C0413

from main import LightningModule
from tools.utils import has_cli_override, build_from_yaml, unwrap_network

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")


def load_cmap(cmap_path: str, num_classes: int) -> np.ndarray:
    """Load the ADE20K class colormap as a (num_classes, 3) uint8 lookup table.

    Args:
        cmap_path (str): Path to the YAML colormap (list of entries with
            'id' and 'color' [R, G, B] fields).
        num_classes (int): Number of classes predicted by the model.

    Returns:
        np.ndarray: Lookup table mapping class id -> RGB color.
    """
    with open(cmap_path, encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    lut = np.zeros((num_classes, 3), dtype=np.uint8)
    for entry in entries:
        if entry["id"] < num_classes:
            lut[entry["id"]] = entry["color"]

    return lut


def list_images(images_dir: str) -> List[Path]:
    """List all the images in a directory.

    Args:
        images_dir (str): Path to the directory.

    Returns:
        list[Path]: List of paths to all the images.
    """
    paths = sorted(
        p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(
            f"No image found in {images_dir} "
            f"(expected extensions: {IMG_EXTENSIONS})."
        )
    return paths


@torch.inference_mode()
def predict_seg_map(model: LightningModule, img: torch.Tensor) -> torch.Tensor:
    """Predict the per-pixel class map for a single image.

    Follows the semantic eval pipeline (sliding-window crops + logit
    reassembly), but forwards the crops one at a time (batch size 1).

    Args:
        model (LightningModule): EoMT Lightning module (vanilla or G2TM).
        img (torch.Tensor): Image tensor (3, H, W), uint8, on the model device.

    Returns:
        torch.Tensor: Predicted class ids with shape (H, W).
    """
    crops, origins = model.window_imgs_semantic((img,))

    crop_logits = []
    for crop in crops.split(1):
        mask_logits_per_layer, class_logits_per_layer = model(crop)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear"
        )
        crop_logits.append(
            model.to_per_pixel_logits_semantic(mask_logits, class_logits_per_layer[-1])
        )
    crop_logits = torch.cat(crop_logits, dim=0)

    logits = model.revert_window_logits_semantic(
        crop_logits, origins, [img.shape[-2:]]
    )[0]

    return logits.argmax(dim=0)


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Predict colorized segmentation maps with an EoMT model."
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
        "--data.path override. Only needed to build the datamodule "
        "from the config; no dataset image is read.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the model checkpoint. Equivalent to the LightningCLI "
        "--model.ckpt_path override.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=str(ROOT / "vis" / "ade20k" / "images"),
        help="Folder of input images. Default: ./vis/ade20k/images",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=str(ROOT / "vis" / "ade20k" / "cmap.yml"),
        help="Path to the YAML class colormap. Default: ./vis/ade20k/cmap.yml",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Folder where the colorized segmentation maps will be saved.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Also save a blend of the image and its segmentation map.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally predict only the first N images of the folder. "
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

    if args.ckpt_path is not None and not has_cli_override(
        cli_args, "--model.ckpt_path"
    ):
        cli_args.extend(["--model.ckpt_path", args.ckpt_path])

    return args, cli_args


def main() -> None:
    """Main function to predict and save semantic segmentation maps."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, _ = build_from_yaml(args.config, cli_args)
    model.eval().to(device)
    net = unwrap_network(model)

    if not hasattr(model, "ignore_idx"):
        raise ValueError(
            "This script only supports semantic segmentation "
            "models (MaskClassificationSemantic)."
        )

    lut = load_cmap(args.cmap, model.num_classes)
    image_paths = list_images(args.images_dir)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Model           : EoMT", "+ G2TM" if hasattr(net, "info") else "")
    print(
        f"Input           : {len(image_paths)} images from {args.images_dir} on {device}"
    )
    print(f"Colormap        : {args.cmap}")
    print(f"Output directory: {output_dir}")

    for path in tqdm(image_paths, position=0, leave=False):
        pil_img = Image.open(path).convert("RGB")
        img = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).to(device)

        preds = predict_seg_map(model, img)

        seg_rgb = lut[preds.cpu().numpy()]
        Image.fromarray(seg_rgb).save(output_dir / f"{path.stem}_seg.png")

        if args.overlay:
            overlay = (0.5 * np.array(pil_img) + 0.5 * seg_rgb).astype(np.uint8)
            Image.fromarray(overlay).save(output_dir / f"{path.stem}_overlay.png")

    print(f"Saved {len(image_paths)} segmentation map(s) to {output_dir}")


if __name__ == "__main__":
    main()
