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
"""Connected components functions to find connected tokens, using Breadth-First Search,
Union-Find and FastSV algorithms.

Only FastSV is compatible with ONNX export, as it is fully tensorized and does not use
any Python loop over the nodes of the graph. Among all connected component algorithm
implementations, FastSV is the fastest when running batched training, but is beaten by
the custom BFS implementation (not compatible with ONNX export) when runing inference
at batch size 1 with PyTorch.
Feel free to add more connected component algorithms if you want to compare them!
"""

from typing import Tuple, Optional
import math

import torch


@torch.compiler.disable
def connected_components_uf(
    right_mask: torch.Tensor,
    bottom_mask: torch.Tensor,
    n: int,
    base_grid_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor]:
    """Finds connected components in a graph using the Union-Find algorithm.

    Args:
        right_mask (torch.Tensor): A boolean vector indicating for each node the
            presence of an edge with its right neighbor in the image grid.
        bottom_mask (torch.Tensor): A boolean vector indicating for each node the
            presence of an edge with its bottom neighbor in the image grid.
        n (int): Total number of nodes in the graph/image grid.
        base_grid_w (int): Width of the original image grid.
        device (torch.device): The device on which to perform computations.

    Returns:
        Tuple[torch.Tensor]: Three tensors containing the indices of mean nodes,
            nodes to be reduced, and merge indices for scatter-reduce operations.
    """
    right = right_mask.tolist()
    bottom = bottom_mask.tolist()
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # Path halving
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Union by rank: attach the shallower tree under the deeper one.
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Process every declared edge
    for i in range(n):
        if right[i]:
            union(i, i + 1)
        if bottom[i]:
            union(i, i + base_grid_w)

    # Group nodes by root (one find() per node, full compression)
    connected_comps = [[] for _ in range(n)]
    for i in range(n):
        connected_comps[find(i)].append(i)

    # Build scatter-reduce index lists
    mean_idxs = []
    to_reduce_idxs = []
    merge_idxs = []
    for cc in connected_comps:
        n_nodes = len(cc)
        if n_nodes > 1:
            merge_idxs += [[len(mean_idxs)]] * (n_nodes - 1)
            mean_idxs.append(cc[0])
            to_reduce_idxs += cc[1:]

    return (
        torch.tensor(  # pylint: disable=E1101
            mean_idxs, device=device, dtype=torch.int64
        ),
        torch.tensor(  # pylint: disable=E1101
            to_reduce_idxs, device=device, dtype=torch.int64
        ),
        torch.tensor(  # pylint: disable=E1101
            merge_idxs, device=device, dtype=torch.int64
        ),
    )


@torch.compiler.disable
def connected_components_bfs(
    right_mask: torch.Tensor,
    bottom_mask: torch.Tensor,
    n: int,
    base_grid_w: int,
    device: torch.device,
) -> Tuple[torch.Tensor]:
    """Finds connected components in a graph using the Breadth-First Search algorithm.

    Args:
        right_mask (torch.Tensor): A boolean vector (n, ) indicating for each node the
            presence of an edge with its right neighbor in the image grid.
        bottom_mask (torch.Tensor): A boolean vector (n, ) indicating for each node the
            presence of an edge with its bottom neighbor in the image grid.
        n (int): Total number of nodes in the graph/image grid.
        base_grid_w (int): Width of the original image grid.
        device (torch.device): The device on which to perform computations.

    Returns:
        Tuple[torch.Tensor]: Three tensors containing the indices of mean nodes,
            nodes to be reduced, and merge indices for scatter-reduce operations.
    """

    # Get indices of tokens having a valid connection with its right and
    # bottom neighbors
    has_right_neigh = right_mask.nonzero()
    has_bottom_neigh = bottom_mask.nonzero()
    # Concatenate the tensors to get the pairs of connected tokens
    src_nodes = torch.cat(  # pylint: disable=E1101
        (has_right_neigh, has_bottom_neigh), dim=0
    ).squeeze(1)
    dst_nodes = torch.cat(  # pylint: disable=E1101
        (has_right_neigh + 1, has_bottom_neigh + base_grid_w), dim=0
    ).squeeze(1)

    # Moving tensors on CPU
    src = src_nodes.tolist()
    dst = dst_nodes.tolist()

    # Creating an adjacency dict. Mandatory step to enable BFS algo to check
    # all 4-neighbors for a given source node (not only right and bottom).
    adj = [[] for _ in range(n)]
    for u, v in zip(src, dst):
        adj[u].append(v)
        adj[v].append(u)

    mean_idxs = []
    to_reduce_idxs = []
    merge_idxs = []
    seen = [False] * n

    # Searching for connected components (BFS)
    for source in range(n):
        if adj[source] and not seen[source]:

            seen[source] = True
            component = [source]
            ptr = 0

            while ptr < len(component):
                node = component[ptr]
                ptr += 1
                for neigh in adj[node]:
                    if not seen[neigh]:
                        seen[neigh] = True
                        component.append(neigh)

            # Processing the connected component for later scatter_reduce
            if len(component) > 1:
                merge_idxs += [[len(mean_idxs)]] * (ptr - 1)
                mean_idxs.append(component[0])
                to_reduce_idxs += component[1:]

    return (
        torch.tensor(  # pylint: disable=E1101
            mean_idxs, device=device, dtype=torch.int64
        ),
        torch.tensor(  # pylint: disable=E1101
            to_reduce_idxs, device=device, dtype=torch.int64
        ),
        torch.tensor(  # pylint: disable=E1101
            merge_idxs, device=device, dtype=torch.int64
        ),
    )


def connected_components_labels(
    right_mask: torch.Tensor,
    bottom_mask: torch.Tensor,
    base_grid_w: int,
    num_iters: Optional[int] = None,
) -> torch.Tensor:
    """Find connected components in batched graphs using a tensorized FastSV algorithm.

    Tokens are nodes of a (base_grid_h, base_grid_w) grid flattened in row-major order.
    An edge links node ``i`` to node ``i + 1`` when ``right_mask[:, i]`` is True and
    node ``i`` to node ``i + base_grid_w`` when ``bottom_mask[:, i]`` is True.

    The algorithm is a tensorized FastSV, the scatter/gather variant of the
    Shiloach-Vishkin parallel connected components algorithm. Every node holds a parent
    pointer ``f`` (initially itself); parents form trees that are progressively hooked
    onto each other and flattened until every tree is a star whose root is the
    component minimum. Each iteration:

    1. ``gf = f[f]`` (grandparent);
    2. every node pulls the minimum grandparent ``ngf`` of its (up to 4) neighbors
        across active edges;
    3. stochastic hooking: ``f[f[u]] <- min(f[f[u]], ngf[u])`` (scatter with min
        reduction);
    4. aggressive hooking: ``f[u] <- min(f[u], ngf[u])``;
    5. shortcutting: ``f[u] <- min(f[u], gf[u])``.

    Pointers only decrease, never leave their component, and the component minimum is a
    fixed point, so the only failure mode of an insufficient ``num_iters`` is a
    component split in several groups (never a wrong merge across components).
    Shiloach-Vishkin style hooking converges in at most ``log_{3/2}(n)`` iterations;
    the default budget adds a margin of 2 on top of that bound. Empirically (exhaustive
    small grids, heavy random fuzzing, adversarial snake/spiral/comb patterns)
    convergence never exceeded ``log2(n) + 1`` iterations.

    Everything is expressed with slice / concat / where / minimum / gather / scatter
    ops on tensors of static shape and a Python loop of fixed length, so the function
    traces to a static ONNX graph (the scatter-min reduction requires opset >= 18) and
    is batch-vectorized.

    Args:
        right_mask (torch.Tensor): A boolean tensor (b, n) indicating for each node
            the presence of an edge with its right neighbor in the image grid.
        bottom_mask (torch.Tensor): A boolean tensor (b, n) indicating for each node
            the presence of an edge with its bottom neighbor in the image grid.
        base_grid_w (int): Width of the original image grid.
        num_iters (int, optional): Number of FastSV iterations. Defaults
            to ``ceil(log_{3/2}(n)) + 2``.

    Returns:
        f (Tuple[torch.Tensor]): A int64 tensor containing the smallest node index of
            each node's connected component (it's "label").
    """
    b, n = right_mask.shape
    device = right_mask.device

    if num_iters is None:
        num_iters = math.ceil(math.log(max(n, 2)) / math.log(1.5)) + 2

    f = (
        torch.arange(n, device=device, dtype=torch.int64)  # pylint: disable=E1101
        .unsqueeze(0)
        .expand(b, n)
    )
    # Sentinel label (n > any real label), used for inactive edges so that
    # they never win the minimum.
    sentinel = torch.full(  # pylint: disable=E1101
        (b, n), n, device=device, dtype=torch.int64
    )
    pad_col = sentinel[:, :1]
    pad_row = sentinel[:, :base_grid_w]

    for _ in range(num_iters):
        # Grandparent of each node
        gf = f.gather(1, f)

        # Pull the neighbor grandparent across each active edge (both
        # directions), inactive edges pull the sentinel instead.
        from_right = torch.where(  # pylint: disable=E1101
            right_mask,
            torch.cat((gf[:, 1:], pad_col), dim=1),
            sentinel,  # pylint: disable=E1101
        )
        right_send = torch.where(right_mask, gf, sentinel)  # pylint: disable=E1101
        from_left = torch.cat(  # pylint: disable=E1101
            (pad_col, right_send[:, :-1]), dim=1
        )
        from_bottom = torch.where(  # pylint: disable=E1101
            bottom_mask,
            torch.cat((gf[:, base_grid_w:], pad_row), dim=1),  # pylint: disable=E1101
            sentinel,
        )
        bottom_send = torch.where(bottom_mask, gf, sentinel)  # pylint: disable=E1101
        from_top = torch.cat(  # pylint: disable=E1101
            (pad_row, bottom_send[:, :-base_grid_w]), dim=1
        )

        # Minimum grandparent over connected neighbors (sentinel if none)
        ngf = torch.minimum(  # pylint: disable=E1101
            torch.minimum(from_right, from_left),  # pylint: disable=E1101
            torch.minimum(from_bottom, from_top),  # pylint: disable=E1101
        )

        # Stochastic hooking: the parent slot of u receives the smallest
        # neighbor grandparent, f[f[u]] <- min(f[f[u]], ngf[u]).
        hooked = f.scatter_reduce(1, f, ngf, reduce="amin", include_self=True)

        # Aggressive hooking (min with ngf) + shortcutting (min with gf).
        # Taking the minimum with the previous `f` guarantees the sentinel
        # never survives, so pointers stay valid gather indices.
        f = torch.minimum(  # pylint: disable=E1101
            torch.minimum(hooked, f), torch.minimum(ngf, gf)  # pylint: disable=E1101
        )

    return f
