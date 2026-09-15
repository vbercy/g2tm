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

# Modifications based on code from Daniel Bolya et al. (ToMe)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------


import random
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

import torch
import torch.nn.functional as F


def generate_colormap(n: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a colormap with N floating-point elements.

    Args:
        n (int): Number of colors to generate.
        seed (int): Random seed.

    Returns:
        list[tuple[float]]: List of RGB colors.
    """
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(n)]


def make_visualization(
    img: Image.Image, source: torch.Tensor, patch_size: int = 16
) -> Image.Image:
    """Overlay of the fused patches on the original image.

    This function creates a visualization of the fused patches on top of the
    original image, like in the paper.

    Args:
        img (PIL.Image.Image): Resized image from the dataset.
        source (torch.Tensor): Matrix indicating which tokens have been fused
            with another.
        patch_size (int): Height and width of an image patch in pixels.

    Returns:
        PIL.Image.Image: The expected visualization with the same size as the
            input.
    """
    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    print(f"Number of tokens left: {source.size(1)}.")

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(1, num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.asarray(cmap[i]).reshape((1, 1, 3))

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img


def add_grid(
    image: Image.Image,
    grid_size: int = 16,
    grid_color: tuple = (128, 128, 128),
    thickness: int = 1,
) -> Image.Image:
    """Overlay of the original ViT patches on the image.

    This function adds a regular grid of the specified size to the input image.

    Args:
        image (PIL.Image.Image): Input image.
        grid_size (int): Size of each grid cell (i.e.: size of an image patch
            in pixels).
        grid_color (tuple[int]): Color of the grid lines (default is gray).
        thickness (int): Thickness of the grid lines.

    Returns:
        image_with_grid (PIL.Image.Image): Regular grid overlaying the input
            image.
    """
    # Create a copy of the input image to avoid modifying the original image
    image_with_grid = image.copy()
    draw = ImageDraw.Draw(image_with_grid)

    width, height = image_with_grid.size

    # Draw vertical lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=thickness)

    # Draw horizontal lines
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=grid_color, width=thickness)

    return image_with_grid
