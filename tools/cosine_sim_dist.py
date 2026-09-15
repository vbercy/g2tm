"""Script to plot the distribution of cosine similarity between neighboring tokens.

Example: see README.md
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pylint: disable=C0413

from main import LightningDataModule, LightningModule
from models.eomt import EoMT
from tools.utils import (
    has_cli_override,
    build_from_yaml,
    unwrap_network,
    iter_images,
    make_input,
)


@torch.no_grad()
def plot_cos_sim_dist(
    model: LightningModule,
    net: EoMT,
    datamodule: LightningDataModule,
    stage: str,
    device: torch.device,
    layer_id: int,
    output_dir: str,
    grid_size: tuple,
    max_images: int = None,
) -> None:
    """Plot the distribution of cosine similarity between neighboring tokens.

    Args:
        model (LightningModule): The EoMT model with the LightningModule wrapper.
        net (EoMT): The unwrapped EoMT network.
        datamodule (LightningDataModule): The data loader containing several splits.
        stage (str): The split of the datamodule to use ('fit' or 'validate').
        device (torch.device): The device to use for computation.
        layer_id (int): The ID of the layer to analyze (0-based).
        output_dir (str): The directory to save the plot.
        grid_size (tuple): The size of the grid of tokens.
        max_images (int | None): Optionally process only the first N images. Defaults
            to all images.
    """
    print(f"Plotting cosine similarity distribution from encoder layer ID {layer_id}.")
    token_sims = []
    grid_h, grid_w = grid_size
    n_tokens = grid_h * grid_w

    num_images = 0
    loader = iter_images(datamodule, stage)
    for img in tqdm(loader, position=0, leave=False):
        num_images += 1
        if max_images is not None and num_images > max_images:
            break
        crops = make_input(model, img, device)

        for crop in crops:
            # Run the encoder forward until we reach the desired layer.
            # The network expects inputs in [0, 1] (see LightningModule.forward)
            x_post_attn = net.get_feature_map(crop / 255.0, layer_id, post_attn=True)[0]
            if layer_id > len(net.encoder.backbone.blocks) - net.num_blocks:
                patch_start = net.encoder.backbone.num_prefix_tokens + net.num_q
            else:
                patch_start = net.encoder.backbone.num_prefix_tokens
            x_post_attn = x_post_attn[:, patch_start:, :]

            # Replicate the fused tokens to get the notion of neighboring back
            # Enables us to plot proper similarity distribution of neighboring tokens
            if model.patch_type != "pure":
                unmerge_idx = net.info["unmerge_idx"]
                if unmerge_idx is not None:
                    x_ = x_post_attn[:1, unmerge_idx[0], :]
                else:
                    x_ = x_post_attn
            else:
                x_ = x_post_attn

            # Compute the cosine similarities between neighboring tokens
            has_right_idxs = (
                (torch.arange(grid_h, device=device)[:, None] * grid_w)
                + torch.arange(grid_w - 1, device=device)
            ).ravel()
            has_bottom_idxs = torch.arange(n_tokens - grid_w, device=device)
            right_sims = F.cosine_similarity(  # pylint: disable=E1102
                x_[:, has_right_idxs + 1, :], x_[:, has_right_idxs, :], dim=-1
            )
            bottom_sims = F.cosine_similarity(  # pylint: disable=E1102
                x_[:, has_bottom_idxs + grid_w, :], x_[:, has_bottom_idxs, :], dim=-1
            )
            if model.patch_type != "pure":
                sims = torch.cat((right_sims.flatten(), bottom_sims.flatten()))
                token_sims += sims[sims <= 0.99].tolist()
            else:
                token_sims += torch.cat(
                    (right_sims.flatten(), bottom_sims.flatten())
                ).tolist()

        # Reset the generator
        loader = iter_images(datamodule, stage)

    # Plotting the histogram
    plt.figure(figsize=(22, 17))
    plt.hist(
        token_sims, bins=100, range=(0, 1.0), color="blue", edgecolor="k", alpha=0.6
    )
    plt.xlabel("Cosine similarity", fontsize=30)
    plt.ylabel("Number", fontsize=30)
    plt.title(
        "Distribution of cosine similarity between neighboring tokens "
        f"at encoder layer n°{layer_id}",
        fontsize=35,
    )

    # Format the y-axis to display in multiples of 10^5
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits(
        (-2, 3)
    )  # Adjust formatting to show scientific notation when needed
    plt.gca().yaxis.set_major_formatter(formatter)

    # Adjust the font size of the scale label (e.g., × 10^5)
    ax = plt.gca()
    offset_text = ax.yaxis.get_offset_text()
    offset_text.set_text("× $10^5$")  # Customize the scale label if necessary
    offset_text.set_fontsize(23)  # Set the font size of the scale label

    # Ensure the same starting point (0) for both axes and only one plotted
    plt.xlim(left=0)  # Set x-axis to start from 0
    plt.ylim(bottom=0)  # Set y-axis to start from 0
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticks([y for y in yticks if y != 0.0])

    # Set tick sizes
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    # Display the grid
    plt.grid()
    plt.tight_layout()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/distrib_block{layer_id}.png")
    plt.close()


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse command line arguments.

    Returns:
        tuple[argparse.Namespace, list[str]]: Parsed arguments and remaining CLI args.
    """
    parser = argparse.ArgumentParser(description="Plot the cosine similarity distribution of neighbouring tokens.")
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
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optionally profile only the first N validation images. "
        "Defaults to all images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the folder where the figures will be saved.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="fit",
        choices=("fit", "validate"),
        help="Split of the dataloader to use for plotting.",
    )
    args, cli_args = parser.parse_known_args()

    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be greater than 0.")

    if args.data_path is not None and not has_cli_override(cli_args, "--data.path"):
        cli_args.extend(["--data.path", args.data_path])

    return args, cli_args


def main():
    """Main function to plot the distribution of cosine similarity between neighboring
    tokens.
    """
    args, cli_args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.", file=sys.stderr)
        args.device = "cpu"

    output_dir = Path(args.output_dir)
    device = torch.device(args.device)
    model, datamodule = build_from_yaml(args.config, cli_args)
    model.eval().to(device)
    net = unwrap_network(model)

    grid_size = net.encoder.backbone.patch_embed.grid_size

    for i in range(len(net.encoder.backbone.blocks)):
        plot_cos_sim_dist(
            model,
            net,
            datamodule,
            args.stage,
            device,
            i,
            output_dir,
            grid_size,
            args.max_images,
        )


if __name__ == "__main__":
    main()
