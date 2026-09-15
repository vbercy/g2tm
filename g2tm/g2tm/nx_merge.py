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
"""Functions performing G2TM's token fusion on a token sequence using the connected
components implementation from NetworkX.
"""

from typing import List

import torch
import networkx as nx

from .utils import FourTensors, cosine_similarity_masks, g2tm_merge

DoubleIntList = List[List[int]]
TripleIntList = List[List[List[int]]]


def get_mergeable_idxs(
    x_feat: torch.Tensor,
    threshold: float,
    base_grid_h: int,
    base_grid_w: int,
    b: int,
    n: int,
    device: torch.device,
) -> TripleIntList:
    """Determine the indices of the groups of tokens to merge.

    This function computes the cosine similarity between all tokens and their
    respective 4-neighbors. Two tokens are neighbors if and only if the patches
    they represent in the image are neighbors. The similarity scores are then
    thresholded. From the remaining edges between tokens, we create a NetworkX
    graph and search its connected components to group tokens to merge with
    each other into lists of indices.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        threshold (float): Threshold parameter for G2TM.
        base_grid_h (int): Image height in number of patches.
        base_grid_w (int): Image width in number of patches.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.

    Returns:
        connected_components (TripleIntList): Groups of indices corresponding
            to tokens to merge in each batch.
    """
    # Apply the cosine similarity and thresholding operations to get the remaining
    # edges of the graph between tokens. The remaining edges are represented by two
    # binary masks, for the right and bottom neighbors of each token.
    right_mask, bottom_mask = cosine_similarity_masks(
        x_feat, threshold, base_grid_h, base_grid_w, b, n, device
    )

    # Get all the connected components for all the batches in a list
    # (i.e.: List[List[List, ...], ...])
    connected_components = []
    for bb in range(b):
        # Get indices of tokens having a valid connection with its right and
        # bottom neighbors
        right_connections = right_mask[bb].nonzero()  # .cpu() (a bit longer)
        bottom_connections = bottom_mask[bb].nonzero()  # .cpu() (a bit longer)
        # Concatenate the tensors to get the pairs of connected tokens
        src_nodes = torch.cat(  # pylint: disable=E1101
            (right_connections, bottom_connections), dim=0
        )
        dst_nodes = torch.cat(  # pylint: disable=E1101
            (right_connections + 1, bottom_connections + base_grid_w), dim=0
        )
        # Create NetworkX graph and find connected components
        graph = nx.Graph(
            torch.cat((src_nodes, dst_nodes), dim=-1).tolist()  # pylint: disable=E1101
        )
        connected_components.append(list(map(list, nx.connected_components(graph))))

    return connected_components


def get_labels(
    x_feat: torch.Tensor,
    threshold: float,
    base_grid_h: int,
    base_grid_w: int,
    b: int,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Label each token with the representative of its connected component.

    NetworkX returns each connected component as a set, whose iteration order
    is not the index order, so the representative is taken as the minimum of
    the component to match the other implementations.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        threshold (float): Threshold parameter for G2TM.
        base_grid_h (int): Image height in number of patches.
        base_grid_w (int): Image width in number of patches.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.

    Returns:
        torch.Tensor: (b, n) tensor giving, for each token, the index of the
            representative (lowest index) of its connected component .
    """
    connected_components = get_mergeable_idxs(
        x_feat, threshold, base_grid_h, base_grid_w, b, n, device
    )

    # The labels are built on the host, where the components already are, then
    # moved to the device in a single transfer.
    labels = []
    for components in connected_components:
        image_labels = list(range(n))
        for component in components:
            representative = min(component)
            for node in component:
                image_labels[node] = representative
        labels.append(image_labels)

    return torch.tensor(  # pylint: disable=E1101
        labels, dtype=torch.int64, device=device
    )


def nx_merge(
    feat: torch.Tensor,
    threshold: float,
    is_encoder: bool = True,
    distill_token: bool = False,
) -> FourTensors:
    """Applies the entire G2TM processing to a token feature sequence.

    This function determines the groups of token indices to merge with each
    other and then computes the merged token associated to each of these
    groups. It separates the cases where the batch size is 1 or any other
    value to process the token sequence accordingly.

    Args:
        feat (torch.Tensor): Token feature sequence.
        threshold (float): Threshold parameter for G2TM.
        is_encoder (bool): Whether the G2TM have been inserted in a ViT
                           encoder.
        distill_token (bool): Whether the sequence contains a distillation
                              token.

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
        is_encoder=is_encoder,
        distill_token=distill_token,
        get_labels=get_labels,
    )
