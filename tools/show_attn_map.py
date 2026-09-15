"""Attention and token visualizations of an EoMT model (with or without G2TM).

For a single image, this script saves, at one Transformer block:
  1 - Attention map visualizations, one per attention head
      a - Raw attention map (gaussian-smoothed).
      b - Attention map overlaying the original image.
  2 - Token visualization
      a - Token grid overlaying the original image with the merged patches
          detoured, when G2TM has already merged at the chosen block.
      b - Original ViT patch grid otherwise.
  3 - Input image with the selected patch highlighted, when the attention is
      read from a patch token.

Segmentation maps are not produced here: use `tools/predict_seg_maps.py`.

EoMT is encoder-only, so the encoder/decoder split of the original Segmenter
script does not apply. What plays the role of the decoder is the last
`num_blocks` blocks, the only ones where the `num_q` query tokens are part of
the sequence: `--token query` reads the attention of those query tokens and is
therefore rejected on earlier blocks.

Notes:
  - --data_path is only needed to instantiate the datamodule from the config;
    no dataset image is read by this script.
  - The image is resized to the crop size of the config and forwarded as a
    single crop (batch size 1), so G2TM runs on its sparse path and the token
    visualization shows the merges the model really performs.
  - `masked_attn_enabled` is forced to False, which is EoMT's inference
    behaviour (mask annealing has driven attn_mask_probs to 0 by the end of
    training).

Example: see README.md
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=C0413

from tools.utils import has_cli_override, build_from_yaml, unwrap_network

from g2tm.utils import rebuild_source
from g2tm.vis import add_grid, make_visualization


def save_attention_map(
    attention_map: np.ndarray,
    img_vis: Image.Image,
    dir_path: Path,
    file_name: str,
    cmap: str,
    sigma: float = 10.0,
):
    """Save an overlay of attention map on the image as PNG file.

    This function saves a PNG file containing an attention map (per patch)
    overlaying the original image. The attention scores are smoothed with a
    gaussian filter for prettier visualization.

    Args:
        attention_map (np.ndarray): Attention map.
        img_vis (PIL.Image.Image): Original image.
        dir_path (Path): Path to directory where PNG file is saved.
        file_name (str): Name of the saved file.
        cmap (str): Color map argument.
        sigma (float): Standard deviation of the gaussian smoothing, in pixels.
    """
    file_path = dir_path / f"{file_name}.png"
    file_path_atten_overly = dir_path / f"{file_name}_overlay.png"
    attention_map_ = gaussian_filter(attention_map, sigma=sigma)

    plt.imsave(fname=str(file_path), arr=attention_map_, format="png", cmap=cmap)

    # overlay image
    span = np.max(attention_map) - np.min(attention_map)
    attention_weights_normalized = (attention_map - np.min(attention_map)) / (
        span if span > 0 else 1.0
    )
    attention_weights_normalized = gaussian_filter(
        attention_weights_normalized, sigma=sigma
    )

    attention_map = (
        np.array(img_vis) * 0.6
        + plt.get_cmap("jet")(attention_weights_normalized)[:, :, :3] * 255 * 0.4
    ).astype(np.uint8)
    plt.imsave(
        fname=str(file_path_atten_overly), arr=attention_map, format="png", cmap=cmap
    )
    print(f"{file_path} saved.")


def load_class_names(cmap_file: str) -> dict:
    """Load the class id to class name mapping of the dataset colormap.

    Args:
        cmap_file (str): Path to the YAML colormap (list of entries with an
            'id' and a 'name' field).

    Returns:
        dict: Mapping from class id to class name.
    """
    with open(cmap_file, "r", encoding="utf-8") as f:
        entries = yaml.full_load(f)

    return {int(entry["id"]): str(entry["name"]) for entry in entries}


def attention_weights(
    net: nn.Module,
    module: nn.Module,
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    rope: Optional[torch.Tensor],
) -> torch.Tensor:
    """Compute the attention probabilities of one Multi-Head Self-Attention layer.

    Mirrors `EoMT._attn` (and its `G2TMEoMT` override) but returns the softmax
    map instead of the attended features: the fused kernel EoMT runs by default
    never materializes it.

    Args:
        net (nn.Module): The EoMT network, patched with G2TM or not.
        module (nn.Module): The MHSA layer of the block to visualize.
        x (torch.Tensor): Normalized token features (b, n, c) fed to the layer.
        mask (torch.Tensor | None): 2D boolean attention mask (b, n, n).
        rope (torch.Tensor | None): RoPE embeddings, for the backbones that use
            them (DINOv3).

    Returns:
        torch.Tensor: Attention probabilities (b, num_heads, n, n).
    """
    if mask is not None:
        mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

    if rope is not None:
        # `transformers` backbone: the attention module returns its
        # probabilities as a second output, but only its eager implementation
        # computes them (SDPA and FlashAttention return None).
        impl = module.config._attn_implementation  # pylint: disable=W0212
        module.config._attn_implementation = "eager"  # pylint: disable=W0212
        try:
            attn = module(x, mask, rope)[1]
        finally:
            module.config._attn_implementation = impl  # pylint: disable=W0212

        if attn is None:
            raise RuntimeError(
                "The backbone attention layer did not return its attention "
                "probabilities, so no attention map can be extracted from it."
            )
        return attn

    b, n, _ = x.shape

    qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, module.head_dim)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = module.q_norm(q), module.k_norm(k)

    attn = (q @ k.transpose(-2, -1)) * module.scale

    # Same (inverse) proportional re-weighting as G2TMEoMT._attn, so that the
    # map is the one the patched network actually uses.
    size = net.info["size"] if hasattr(net, "info") else None
    if size is not None and net.info["prop_attn"]:
        attn = attn + size.log()[:, None, None, :]
    elif size is not None and net.info["iprop_attn"]:
        log_size = size.log().masked_fill(size == 0, float("inf"))
        attn = attn - log_size[:, None, None, :]

    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))

    return attn.softmax(dim=-1)


@torch.inference_mode()
def block_attention(net: nn.Module, img: torch.Tensor, layer_id: int):
    """Run the network up to one block and return the attention it computes.

    Args:
        net (nn.Module): The EoMT network, patched with G2TM or not.
        img (torch.Tensor): Input crop (1, 3, H, W) with values in [0, 1].
        layer_id (int): Index of the block to visualize (0-based).

    Returns:
        attn (torch.Tensor): Attention probabilities (1, num_heads, n, n).
        x (torch.Tensor): Token sequence (1, n, c) entering the block.
    """
    blocks = net.encoder.backbone.blocks
    head_start = len(blocks) - net.num_blocks

    # Everything up to, and excluding, the attention of block `layer_id`
    x, attn_mask, rope = net.get_feature_map(img, layer_id)

    # `get_feature_map` stops right after block `layer_id - 1`, so the prologue
    # of the forward loop for block `layer_id` still has to be replayed: query
    # tokens are prepended to the sequence, and the Mask Module gates them.
    if layer_id == head_start:
        if hasattr(net, "info"):
            x, attn_mask = net.add_query_tokens(x)
        else:
            x = torch.cat(  # pylint: disable=E1101
                (net.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
            )

    if net.masked_attn_enabled and layer_id >= head_start:
        mask_logits, _ = net._predict(  # pylint: disable=W0212
            net.encoder.backbone.norm(x)
        )
        attn_mask = net._attn_mask(x, mask_logits, layer_id)  # pylint: disable=W0212

    block = blocks[layer_id]
    module = block.attn if hasattr(block, "attn") else block.attention
    attn = attention_weights(net, module, block.norm1(x), attn_mask, rope)

    return attn, x


@torch.inference_mode()
def query_predictions(
    net: nn.Module, x: torch.Tensor, class_names: dict
) -> List[tuple]:
    """Name every query token after the class its Mask Module prediction picks.

    EoMT's queries are not class-aligned, so the counterpart of the Segmenter
    script restricting the decoder maps to the classes of the groundtruth is to
    keep the queries whose prediction is not the "no object" class, which is the
    extra logit the class head appends to the dataset ones.

    Args:
        net (nn.Module): The EoMT network, patched with G2TM or not.
        x (torch.Tensor): Token sequence (1, n, c) entering the block.
        class_names (dict): Mapping from class id to class name.

    Returns:
        list[tuple[int, str | None]]: Index and predicted class name of every
            query, the name being None for the queries predicting no object.
    """
    _, class_logits = net._predict(  # pylint: disable=W0212
        net.encoder.backbone.norm(x)
    )
    preds = class_logits[0].argmax(dim=-1).tolist()

    queries = []
    for i, c in enumerate(preds):
        if int(c) == net.n_cls:
            queries.append((i, None))
        else:
            # Keep the first synonym only: ADE20K names read "building, edifice"
            name = class_names.get(int(c), f"class{int(c)}").split(",")[0]
            queries.append((i, name.strip().replace(" ", "_")))

    return queries


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(
        description="Attention and token visualizations of an EoMT model."
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
        "--image_path",
        type=str,
        required=True,
        help="Path to the image to visualize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "vis" / "ade20k" / "attn"),
        help="Folder where the visualizations are saved, in a subfolder named "
        "after the image. Default: ./vis/ade20k/attn",
    )
    parser.add_argument(
        "--cmap_file",
        type=str,
        default=str(ROOT / "vis" / "ade20k" / "cmap.yml"),
        help="Path to the YAML class colormap, read for the class names of the "
        "query tokens. Default: ./vis/ade20k/cmap.yml",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=0,
        help="Transformer block to visualize (0-based). Default: 0",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="patch",
        choices=("patch", "cls", "query"),
        help="Token whose attention is visualized: a patch token at "
        "(--x_patch, --y_patch), the [CLS] token, or the query tokens of the "
        "Mask Module. Default: patch",
    )
    parser.add_argument(
        "--x_patch",
        type=int,
        default=0,
        help="X-coordinate, in patches, of the selected patch. Default: 0",
    )
    parser.add_argument(
        "--y_patch",
        type=int,
        default=0,
        help="Y-coordinate, in patches, of the selected patch. Default: 0",
    )
    parser.add_argument(
        "--query_id",
        type=int,
        default=None,
        help="With --token query, index of the single query token to "
        "visualize. Defaults to every query predicting an actual class.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap used for the attention visualizations. Default: viridis",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Standard deviation, in pixels, of the gaussian smoothing applied "
        "to the attention maps. Default: 10",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=("cuda", "cpu"),
        help="Device to place the model and input on. Default: cuda",
    )
    args, cli_args = parser.parse_known_args()

    if args.layer_id < 0:
        parser.error("--layer_id must be positive.")

    if args.query_id is not None and args.token != "query":
        parser.error("--query_id is only meaningful with --token query.")

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    if args.ckpt_path is not None and not has_cli_override(
        cli_args, "--model.ckpt_path"
    ):
        cli_args.extend(["--model.ckpt_path", args.ckpt_path])

    return args, cli_args


def main() -> None:
    """Save the attention and token visualizations of an EoMT model."""
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    device = torch.device(args.device)
    model, _ = build_from_yaml(args.config, cli_args)
    model.eval().to(device)
    net = unwrap_network(model)

    for param in model.parameters():
        param.requires_grad = False

    # EoMT runs inference without masked attention: mask annealing has driven
    # attn_mask_probs to 0 by the end of training.
    if net.masked_attn_enabled:
        print(
            "NOTE: forcing network.masked_attn_enabled = False "
            "(EoMT's inference behaviour)."
        )
        net.masked_attn_enabled = False

    blocks = net.encoder.backbone.blocks
    head_start = len(blocks) - net.num_blocks
    num_q = net.num_q
    num_prefix = net.encoder.backbone.num_prefix_tokens
    grid_h, grid_w = net.encoder.backbone.patch_embed.grid_size
    patch_h, patch_w = net.encoder.backbone.patch_embed.patch_size

    # Sanity checks
    if args.layer_id >= len(blocks):
        raise ValueError(
            f"Provided layer_id: {args.layer_id} is not valid for a backbone "
            f"with {len(blocks)} blocks."
        )

    if args.token == "query" and args.layer_id < head_start:
        raise ValueError(
            f"Query tokens only join the sequence at block {head_start}, so "
            f"--token query is not available at block {args.layer_id}. Use "
            f"--layer_id >= {head_start}, or --token cls / --token patch."
        )

    if args.token == "patch" and (args.x_patch >= grid_w or args.y_patch >= grid_h):
        raise ValueError(
            f"Provided patch x: {args.x_patch} y: {args.y_patch} is not valid. "
            f"Patch should be in the range x: [0, {grid_w}), y: [0, {grid_h})"
        )

    if args.query_id is not None and not 0 <= args.query_id < num_q:
        raise ValueError(
            f"Provided query_id: {args.query_id} is not valid for a Mask "
            f"Module with {num_q} queries."
        )

    if patch_h != patch_w:
        raise ValueError(
            "The token visualizations assume square patches, but the backbone "
            f"uses {patch_h}x{patch_w} ones."
        )
    patch_size = patch_h

    # Open the image, resized to the crop size for the network and to the patch
    # grid for the visualizations (the two only differ when the crop size is not
    # a multiple of the patch size, which none of the shipped configs is).
    vis_size = (grid_h * patch_size, grid_w * patch_size)
    try:
        with open(args.image_path, "rb") as f:
            img = Image.open(f).convert("RGB")
            img_vis = img.resize(
                (vis_size[1], vis_size[0]), Image.BILINEAR  # pylint: disable=E1101
            )
            img_in = img.resize(
                (model.img_size[1], model.img_size[0]),
                Image.BILINEAR,  # pylint: disable=E1101
            )
    except Exception as e:
        raise ValueError(
            f"Provided image path {args.image_path} is not a valid image file."
        ) from e

    # The network expects inputs in [0, 1] (see LightningModule.forward)
    img_t = (
        torch.from_numpy(np.array(img_in))  # pylint: disable=E1101
        .permute(2, 0, 1)[None, ...]
        .to(device)
        .float()
        / 255.0
    )

    image_name = Path(args.image_path).stem
    output_dir = Path(args.output_dir) / image_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Model           : EoMT", "+ G2TM" if hasattr(net, "info") else "")
    print(f"Input           : {args.image_path} resized to {vis_size} on {device}")
    print(f"Block           : {args.layer_id} / {len(blocks) - 1}")
    print(f"Attention from  : {args.token} token(s)")
    print(f"Output directory: {output_dir}")

    print(f"Generating attention mapping for block {args.layer_id}")
    attn, x = block_attention(net, img_t, args.layer_id)

    # Sequence layout: [queries] + [prefix tokens] + [patch tokens], the queries
    # being present only from block `head_start` on.
    n_extra = num_prefix + (num_q if args.layer_id >= head_start else 0)
    n_ori = grid_h * grid_w
    n_tok = attn.size(-1) - n_extra
    n_heads = attn.size(1)

    print("Attention map shape: ", tuple(attn.shape))

    # Map each original patch to the (possibly merged) token it belongs to.
    # `unmerge_idx` has shape (1, n_ori) and only covers the patch tokens, so
    # its values are in the same space as the attention rows and columns from
    # which the query and prefix tokens have been removed.
    if n_tok != n_ori:
        idxs = net.info["unmerge_idx"][0]
    else:
        idxs = torch.arange(n_ori, device=attn.device)  # pylint: disable=E1101

    # Select the query row(s) the visualization is made of
    if args.token == "query":
        rows = attn[0, :, :num_q, n_extra:]
    elif args.token == "cls":
        # The [CLS] token is the first of the prefix tokens
        rows = attn[0, :, n_extra - num_prefix, n_extra:]
    else:
        num_patch = grid_w * args.y_patch + args.x_patch
        rows = attn[0, :, n_extra + idxs[num_patch], n_extra:]

    # Copying the attention value of the merged tokens to all tokens that have
    # been merged (keys dimension)
    maps = rows[..., idxs]
    print("Attention map shape after unmerging: ", tuple(maps.shape))

    # Reshape into image shape, then resize to match the input size
    maps = maps.reshape(n_heads, -1, grid_h, grid_w)
    maps = F.interpolate(maps.float(), size=vis_size, mode="nearest").cpu().numpy()

    # Query tokens to visualize, named after the class they predict
    if args.token == "query":
        all_queries = query_predictions(net, x, load_class_names(args.cmap_file))
        if args.query_id is not None:
            queries = [all_queries[args.query_id]]
        else:
            queries = [(i, name) for i, name in all_queries if name is not None]
            print(f"Active queries  : {len(queries)} / {num_q}")
        if not queries:
            print(
                "No query predicts an actual class at this block; nothing to "
                "visualize. Try a later --layer_id.",
                file=sys.stderr,
            )

    # Save the attention map of each head
    layer_dir = output_dir / f"layer{args.layer_id}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_heads):
        head_name = f"layer{args.layer_id}_attn-head{i}"

        if args.token == "query":
            for j, cls_name in queries:
                suffix = f"_{cls_name}" if cls_name is not None else "_no-object"
                dir_path = layer_dir / f"query{j}{suffix}"
                dir_path.mkdir(parents=True, exist_ok=True)

                save_attention_map(
                    maps[i, j],
                    img_vis,
                    dir_path,
                    f"{head_name}_query{j}",
                    args.cmap,
                    args.sigma,
                )
        else:
            if args.token == "cls":
                file_name = head_name + "_cls"
                dir_path = layer_dir / "cls"
            else:
                file_name = head_name + f"_patch_{args.x_patch}_{args.y_patch}"
                dir_path = layer_dir / f"patch_{args.x_patch}_{args.y_patch}"
            dir_path.mkdir(parents=True, exist_ok=True)

            save_attention_map(
                maps[i, 0], img_vis, dir_path, file_name, args.cmap, args.sigma
            )

    # Save the input image showing the selected patch
    if args.token == "patch":
        highlight = img_vis.copy()
        draw = ImageDraw.Draw(highlight)
        x_px, y_px = args.x_patch * patch_size, args.y_patch * patch_size
        draw.rectangle(
            [x_px, y_px, x_px + patch_size - 1, y_px + patch_size - 1],
            outline=(255, 0, 0),
            width=max(1, patch_size // 8),
        )
        patch_path = (
            output_dir / f"{image_name}_patch_{args.x_patch}_{args.y_patch}.png"
        )
        highlight.save(patch_path)
        print(f"{patch_path} saved.")

    # Save the image with the merged patches overlayed, or the original ViT
    # patch grid when the block comes before the G2TM module
    if hasattr(net, "info") and net.info["unmerge_idx"] is not None:
        # The source matrix is not computed during inference any more: it is
        # rebuilt from the unmerge index, which encodes the same fusions.
        source = rebuild_source(net.info["unmerge_idx"], n_tok)
        vis_out = make_visualization(img_vis, source, patch_size)
        vis_path = output_dir / f"{image_name}_vis_{n_tok}_tokens.png"
    else:
        vis_out = add_grid(img_vis, patch_size)
        vis_path = output_dir / f"{image_name}_grid.png"

    vis_out.save(vis_path)
    print(f"{vis_path} saved.")


if __name__ == "__main__":
    main()
