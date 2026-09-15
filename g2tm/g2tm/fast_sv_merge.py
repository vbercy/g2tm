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
"""Functions performing G2TM's token fusion on a token sequence, using a tensorized
Fast SV implementation to support ONNX export.
"""

from functools import partial

import torch

from .utils import FourTensors, cosine_similarity_masks, g2tm_merge
from .connected_components import connected_components_labels


def get_merge_labels(
    x_feat: torch.Tensor,
    threshold: float,
    base_grid_h: int,
    base_grid_w: int,
    b: int,
    n: int,
    device: torch.device,
    num_iters: int = None,
) -> torch.Tensor:
    """Determine the groups of tokens to merge, as component labels.

    This function computes the cosine similarity between all tokens and their
    respective 4-neighbors. Two tokens are neighbors if and only if the patches
    they represent in the image are neighbors. The similarity scores are then
    thresholded. The remaining edges between tokens define a graph whose
    connected components are computed with a fixed number of tensor
    operations (see `connected_components_labels`), so that the whole
    process traces to a static graph at ONNX export time.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        threshold (float): Threshold parameter for G2TM.
        base_grid_h (int): Image height in number of patches.
        base_grid_w (int): Image width in number of patches.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.
        num_iters (int): Number of FastSV iterations.

    Returns:
        torch.Tensor: (b, n) tensor giving, for each token, the smallest token index of
        its connected component. Tokens sharing a label belong to the same group to
        merge; the token whose index equals its label is the group representative
        (i.e.: the resulting mean value).
    """
    # Apply the cosine similarity and thresholding operations to get the remaining
    # edges of the graph between tokens. The remaining edges are represented by two
    # binary masks, for the right and bottom neighbors of each token.
    right_mask, bottom_mask = cosine_similarity_masks(
        x_feat, threshold, base_grid_h, base_grid_w, b, n, device
    )

    return connected_components_labels(right_mask, bottom_mask, base_grid_w, num_iters)


def fast_sv_merge(
    feat: torch.Tensor,
    threshold: float,
    num_iters: int = None,
    distill_token: bool = False,
    unmerge_idx_needed: bool = False,
) -> FourTensors:
    """Applies the entire G2TM processing to a token feature sequence.

    This function determines the groups of token indices to merge with each
    other and then computes the merged token associated to each of these
    groups. It separates the cases where the batch size is 1 or any other
    value to process the token sequence accordingly.

    Args:
        feat (torch.Tensor): Token feature sequence.
        threshold (float): Threshold parameter for G2TM.
        num_iters (int): Number of FastSV iterations.
        distill_token (bool): Whether the sequence contains a distillation
            token.
        unmerge_idx_needed (bool): Whether to compute the unmerge index.

    Returns:
        feat (torch.Tensor): Reduced token feature sequence.
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor|None): Binary mask for reduced tokens.
        unmerge_idx (torch.Tensor): Index expanding the reduced sequence back
            onto the full patch grid (used by the decoder).
    """

    return g2tm_merge(
        feat,
        threshold,
        distill_token=distill_token,
        get_labels=partial(get_merge_labels, num_iters=num_iters),
        unmerge_idx_needed=unmerge_idx_needed,
    )
