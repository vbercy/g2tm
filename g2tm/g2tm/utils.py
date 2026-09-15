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
"""Utility functions for G2TM merge."""

from typing import List, Tuple, Union, Callable
import math

import torch
import torch.nn.functional as F

TwoTensors = Tuple[torch.Tensor, torch.Tensor]
ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
MayBeTensor = Union[torch.Tensor, None]
FourTensors = Tuple[torch.Tensor, torch.Tensor, MayBeTensor, torch.Tensor]
FiveTensors = Tuple[torch.Tensor, torch.Tensor, MayBeTensor, torch.Tensor, torch.Tensor]


def cosine_similarity_masks(
    x_feat,
    threshold: float,
    base_grid_h: int,
    base_grid_w: int,
    b: int,
    n: int,
    device: torch.device,
) -> TwoTensors:
    """Determine the remaining edges of the graph based on features and cosine similarity.

    This function computes the cosine similarity between all tokens and their
    respective 4-neighbors, linked by edges in the graph. Two tokens are neighbors if
    and only if the patches they represent in the image are neighbors. The similarity
    scores are then thresholded. The function returns a binary mask indicating which
    edges remain in the graph after thresholding.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        threshold (float): Threshold parameter for G2TM.
        base_grid_h (int): Image height in number of patches.
        base_grid_w (int): Image width in number of patches.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two binary masks indicating which edges
            remain in the graph after the thresholding operation, for the right and
            bottom neighbors of each token respectively.
    """
    # === COSINE SIMILARITY ===

    # See the tokens as they were placed like the patches in the image and
    # compare each token with its right and bottom neighbors.
    grid = x_feat.reshape(b, base_grid_h, base_grid_w, -1)
    right_sims = F.cosine_similarity(  # pylint: disable=E1102
        grid[:, :, 1:, :], grid[:, :, :-1, :], dim=-1
    )
    bottom_sims = F.cosine_similarity(  # pylint: disable=E1102
        grid[:, 1:, :, :], grid[:, :-1, :, :], dim=-1
    )

    # === CONNECTED COMPONENTS === #

    # Connection masks based on the similarity with the right and bottom neighbors.
    # The last column (resp. row) has no right (resp. bottom) neighbor, hence the
    # padding with False.
    right_mask = torch.cat(  # pylint: disable=E1101
        (
            right_sims > threshold,
            torch.zeros(  # pylint: disable=E1101
                b, base_grid_h, 1, dtype=torch.bool, device=device
            ),
        ),
        dim=2,
    ).reshape(b, n)
    bottom_mask = torch.cat(  # pylint: disable=E1101
        (
            bottom_sims > threshold,
            torch.zeros(  # pylint: disable=E1101
                b, 1, base_grid_w, dtype=torch.bool, device=device
            ),
        ),
        dim=1,
    ).reshape(b, n)

    return right_mask, bottom_mask


def labels_from_idxs(
    scatter_reduce_idxs: List[ThreeTensors], b: int, n: int, device: torch.device
) -> torch.Tensor:
    """Turn scatter-reduce indices into a component label per token.

    A token that has not been merged or that will be assigned the mean value is its own
    representative, so only the tokens merged away are redirected to the representative
    of their connected component.

    Args:
        scatter_reduce_idxs (List[ThreeTensors]): For each image, the
            (mean_idxs, to_reduce_idxs, merge_idxs) triplet describing the
            connected components (see `connected_components_bfs`).
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        device (torch.device): Device where tensors are processed.

    Returns:
        torch.Tensor: (b, n) tensor giving, for each token, the index of the
            representative of its connected component (i.e.: the resulting mean value).
    """
    labels = (
        torch.arange(n, device=device, dtype=torch.int64)  # pylint: disable=E1101
        .unsqueeze(0)
        .repeat(b, 1)
    )
    for k, (mean_idxs, to_reduce_idxs, merge_idxs) in enumerate(scatter_reduce_idxs):
        if to_reduce_idxs.numel():
            labels[k, to_reduce_idxs] = mean_idxs[merge_idxs.reshape(-1)]
    return labels


def merge_from_labels(
    x_feat: torch.Tensor, labels: torch.Tensor, b: int, n: int, d: int, device
) -> ThreeTensors:
    """Merge token features with each other according to component labels.

    This function computes the mean token feature per connected component,
    accumulated at the representative of the component. Tokens merged away
    keep their value and are flagged in a binary mask, for later masking in
    the attention computation. The size vector, which indicates the size of
    each token (i.e.: number of original patches it represents), is obtained
    from the same scatter operation.

    The sequence length is left unchanged, so every tensor keeps a static
    shape: this path is fully batched and ONNX-friendly.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        labels (torch.Tensor): (b, n) representative of each token.
        b (int): Number of images in the batch.
        n (int): Number of tokens in the sequence.
        d (int): Embedding dimension of the tokens.
        device (torch.device): Device where tensors are processed.

    Returns:
        x_feat (torch.Tensor): Reduced token feature sequence.
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor): Binary mask for reduced tokens.
    """
    # Component sizes, accumulated at the representative position. Kept tokens
    # (representatives and unmerged tokens) are exactly the ones receiving at
    # least one count.
    size = torch.zeros(b, n, dtype=torch.int64, device=device)  # pylint: disable=E1101
    size.scatter_add_(1, labels, torch.ones_like(labels))  # pylint: disable=E1101
    mask = size > 0

    # Mean token feature per connected component, accumulated at the
    # representative position.
    sums = torch.zeros_like(x_feat).scatter_add_(  # pylint: disable=E1101
        1, labels.unsqueeze(-1).expand(-1, -1, d), x_feat
    )
    means = sums / size.clamp(min=1).unsqueeze(-1).to(x_feat.dtype)
    x_feat = torch.where(mask.unsqueeze(-1), means, x_feat)  # pylint: disable=E1101

    return x_feat, size, mask


def get_unmerge_idx(
    labels: torch.Tensor, mask: torch.Tensor, sparse: bool
) -> torch.Tensor:
    """Index mapping each original token position to its representative.

    The decoder has to expand the reduced token sequence back onto the full
    patch grid: for every original position ``j`` it needs the position, in
    the *reduced* sequence, of the token that represents ``j``. This used to
    be read as ``source[k].argmax(dim=0)``, which requires the whole source
    matrix and forces an ``ArgMax`` that ONNX Runtime has no CUDA kernel for
    at opset >= 13. Deriving it from the labels instead keeps everything on
    the device and makes the source matrix unnecessary for inference.

    In the padded case the sequence is not shortened, so a token still sits
    at its original position and the label *is* the index. In the sparse case
    merged-away tokens are popped, so the index is the rank of the
    representative among the kept tokens.

    Args:
        labels (torch.Tensor): (b, n) representative of each token.
        mask (torch.Tensor): (b, n) True where a token is kept.
        sparse (bool): Whether the reduced tokens have been popped from the
            sequence (see `sparsify`).

    Returns:
        unmerge_idx (torch.Tensor): (b, n) int64 index into the reduced token
            sequence, for each original token position.
    """
    if not sparse:
        return labels
    # Rank of each kept token in the shortened sequence, then read it at the
    # representative of every original position. The cast is required: ONNX's
    # CumSum has no boolean overload, so accumulating the mask directly
    # produces a graph that fails shape inference.
    rank = mask.to(torch.int64).cumsum(dim=1) - 1
    return rank.gather(1, labels)


def sparsify(
    x_feat: torch.Tensor, size: torch.Tensor, mask: torch.Tensor
) -> TwoTensors:
    """Pop the tokens merged away from the merge outputs.

    This function removes from the sequence the tokens that have been merged
    into a representative, actually shortening the sequence instead of
    masking it. Thus, we do not have to pass a binary mask afterwards.

    The resulting sequence length is data-dependent: the exported ONNX graph
    carries a dynamic dimension (NonZero + Gather), which ONNX Runtime
    supports. Static-shape runtimes must keep the padded sequence.

    WARNING: This function can only be applied when only 1 image per batch is
    processed.

    Args:
        x_feat (torch.Tensor): Token feature sequence (b, n, d).
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor): Binary mask for reduced tokens.

    Returns:
        x_feat (torch.Tensor): Reduced token feature sequence.
        size (torch.Tensor): Vector storing token sizes.
    """
    keep_idxs = mask[0].nonzero().squeeze(1)

    return x_feat.index_select(1, keep_idxs), size.index_select(1, keep_idxs)


def rebuild_source(unmerge_idx: torch.Tensor, n_kept: int) -> torch.Tensor:
    """Rebuild the source matrix from the unmerge index.

    The source matrix is not needed for inference any more (see
    `get_unmerge_idx`), but the visualization and profiling scripts still
    describe fusions with it. It is a one-hot encoding of the unmerge index,
    so it can be rebuilt on demand with a single scatter.

    Args:
        unmerge_idx (torch.Tensor): (b, n) position, in the reduced sequence,
            of the token representing each original token.
        n_kept (int): Number of tokens in the reduced sequence.

    Returns:
        source (torch.Tensor): Binary source matrix (b, n_kept, n), which
            stores True at position (i, j, k) if the fusion that resulted in
            token j has processed the original token k in batch i.
    """
    b, n = unmerge_idx.shape
    source = torch.zeros(  # pylint: disable=E1101
        b, n_kept, n, dtype=torch.int32, device=unmerge_idx.device
    )
    source.scatter_(
        1,
        unmerge_idx.unsqueeze(1),
        torch.ones(  # pylint: disable=E1101
            b, 1, n, dtype=torch.int32, device=unmerge_idx.device
        ),
    )
    return source


def g2tm_merge(
    feat: torch.Tensor,
    threshold: float,
    n_protected: int,
    grid_size: Tuple[int, int],
    get_labels: Callable = None,
) -> FiveTensors:
    """Applies the entire G2TM processing to a token feature sequence.

    This function determines the groups of token indices to merge with each
    other and then computes the merged token associated to each of these
    groups. It separates the cases where the batch size is 1, where the merged
    tokens are removed from the sequence, and any other value, where they are
    kept in place and masked.

    NOTE: This is a template function that requires the user to provide the
    function labelling the connected components. It should not be used
    directly, but rather called from a `xxx_merge.py` file as provided (nx,
    bfs, fast_sv).

    Args:
        feat (torch.Tensor): Token feature sequence.
        threshold (float): Threshold parameter for G2TM.
        n_protected (int): Number of leading tokens that should not be merged
            (e.g.: [CLS] token, distillation token, etc.).
        grid_size (tuple[int, int]): Patch grid height and width.
        get_labels (Callable): Function labelling, for each token, the
            representative of its connected component.

    Returns:
        feat (torch.Tensor): Reduced token feature sequence.
        size (torch.Tensor): Vector storing token sizes.
        mask (torch.Tensor|None): Binary attention mask (b, N, N) hiding the
            merged-away tokens as keys, None when the sequence has been
            sparsified (b == 1) and there is nothing left to mask.
        unmerge_idx (torch.Tensor): Index expanding the reduced sequence back
            onto the full patch grid (used by the decoder).
        alive_tokens (torch.Tensor): (b, n) mask, True for the tokens kept by
            the merge (representatives and unmerged tokens) and False for the
            ones merged away. Protected tokens are excluded, as they are never
            merged.
    """
    x_protected, x_feat = feat[:, :n_protected, :], feat[:, n_protected:, :]
    b, n, d = x_feat.size()
    device = feat.device

    base_grid_h, base_grid_w = grid_size
    assert base_grid_h * base_grid_w == n

    # Get the component label of each token
    with torch.no_grad():
        labels = get_labels(x_feat, threshold, base_grid_h, base_grid_w, b, n, device)

    # Merge tokens (static shapes, batched). The keep mask returned here is
    # exactly the set of tokens still alive after the merge.
    x_feat, size, alive_tokens = merge_from_labels(x_feat, labels, b, n, d, device)

    # Index used by the decoder to expand the reduced sequence back onto the
    # full patch grid. Computed from the labels, so neither the decoder nor
    # this function ever needs the source matrix.
    unmerge_idx = get_unmerge_idx(labels, alive_tokens, sparse=b == 1)

    if b == 1:
        # Reduced tokens are popped
        x_feat, size = sparsify(x_feat, size, alive_tokens)
        mask = None
    else:
        # Creates a 2D binary attention mask as it is expected in EoMT
        # Merged-away tokens are masked as keys. Keeping their rows live
        # prevents all-masked attention rows from producing NaNs.
        mask = alive_tokens.unsqueeze(1).expand(-1, n, -1)

        # As protected tokens will be added to the sequence, we ensure that
        # merge-away tokens are masked as keys for protected tokens also.
        mask = torch.cat(  # pylint: disable=E1101
            (
                torch.ones(
                    b, n, n_protected, dtype=bool, device=device
                ),  # pylint: disable=E1101
                mask,
            ),
            dim=2,
        )
        protected_mask = torch.cat(  # pylint: disable=E1101
            (
                torch.ones(
                    b,
                    n_protected,
                    n_protected,  # pylint: disable=E1101
                    dtype=bool,
                    device=device,
                ),
                alive_tokens.unsqueeze(1).expand(-1, n_protected, -1),
            ),
            dim=2,
        )
        mask = torch.cat((protected_mask, mask), dim=1)

    feat = torch.cat((x_protected, x_feat), dim=1)  # pylint: disable=E1101
    size = torch.cat(  # pylint: disable=E1101
        (
            torch.ones(  # pylint: disable=E1101
                b, n_protected, dtype=torch.int64, device=device
            ),
            size,
        ),
        dim=1,
    )

    return feat, size, mask, unmerge_idx, alive_tokens
