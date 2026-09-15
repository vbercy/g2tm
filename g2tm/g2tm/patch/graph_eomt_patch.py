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
"""Patching EoMT model by inserting G2TM within one of its encoder's Transformer
block.
"""

from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from models.eomt import EoMT

from g2tm.bfs_merge import bfs_merge
from g2tm.fast_sv_merge import fast_sv_merge

TwoTensors = Tuple[torch.Tensor, torch.Tensor]
ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class G2TMEoMT(EoMT):
    """Modified ViT encoder class for G2TM token reduction method.

    This module instanciates a dictionnary where all the parameters needed
    for the G2TM token reduction process are stored.
    """

    def _reset_g2tm_state(self):
        """Reset the G2TM tensors (that are class attributes) between two distinct
        forward passes.
        """
        self.info["size"] = None
        self.info["mask"] = None
        self.info["unmerge_idx"] = None
        self.info["alive_tokens"] = None
        self.info["selected_layer"] = self.selected_layer
        self.info["threshold"] = self.threshold

    def _predict(self, x: torch.Tensor) -> TwoTensors:
        """EoMT's Mask Module forward, predicts class logits and mask logits.

        Args:
            x (torch.Tensor): Token features (b, n, c).

        Returns:
            mask_logits (torch.Tensor): EoMT's mask logits for segmentation.
            class_logits (torch.Tensor): EoMT's class logits for segmentation.
        """
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        # Unmerge the token sequence
        # We get rid of the source matrix and the argmax, the index is derived from
        # the component labels by the merge.
        idxs = self.info["unmerge_idx"]
        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :].gather(
            1, idxs.unsqueeze(-1).expand(-1, -1, x.size(-1))
        )

        # Reshape the token to the grid size
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(  # pylint: disable=W0221
        self, attn_mask, prob, query_patch_mask
    ):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.size(0), self.num_q, device=attn_mask.device)
                > prob
            )
            patch_start = self.num_q + self.encoder.backbone.num_prefix_tokens
            if isinstance(query_patch_mask, torch.Tensor):
                attn_mask[:, : self.num_q, patch_start:][random_queries] = (
                    query_patch_mask[random_queries]
                )
            else:
                # No merge mask to restore (sparse path or no merge yet):
                # un-gate the sampled queries entirely, as in vanilla EoMT.
                attn_mask[:, : self.num_q, patch_start:][
                    random_queries
                ] = query_patch_mask

        return attn_mask

    def _attn_mask(
        self, x: torch.Tensor, mask_logits: torch.Tensor, i: int
    ) -> torch.Tensor:
        """Update the attention mask when EoMT's mask annealing strategy for training
        is needed.

        Args:
            x (torch.Tensor): Token features.
            mask_logits (torch.Tensor): EoMT's intermediate mask logits.
            i (int): Index of the current Transformer block.

        Returns:
            attn_mask (torch.Tensor): The updated attention mask (b, n, n).
        """
        b, n, _ = x.shape
        patch_start = self.num_q + self.encoder.backbone.num_prefix_tokens

        if self.info["mask"] is None:
            attn_mask = torch.ones(b, n, n, dtype=torch.bool, device=x.device)
            query_patch_mask = True
        else:
            attn_mask = self.info["mask"].clone()
            query_patch_mask = self.info["mask"][:, : self.num_q, patch_start:]

        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        keep = interpolated > 0
        unmerge_idx = self.info["unmerge_idx"]
        n_patch = n - patch_start
        if unmerge_idx is not None and keep.size(-1) != n_patch:
            # Sparse (b == 1) path: the sequence was physically reduced, so
            # project the full-grid mask onto the merged tokens. A merged
            # token stays visible to a query if any of its constituent
            # patches does, which is a max-reduction over the original
            # positions sharing the same representative. Scattering on
            # `unmerge_idx` replaces the former `bmm` with the source matrix,
            # which the merge does not build any more.
            keep = (
                torch.zeros(  # pylint: disable=E1101
                    b, self.num_q, n_patch, dtype=torch.float32, device=x.device
                ).scatter_reduce_(
                    2,
                    unmerge_idx.unsqueeze(1).expand(-1, self.num_q, -1),
                    keep.to(torch.float32),
                    reduce="amax",
                )
                > 0
            )
        attn_mask[:, : self.num_q, patch_start:] = (
            attn_mask[:, : self.num_q, patch_start:] & keep
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
            query_patch_mask,
        )
        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        if rope is not None:
            return module(x, mask, rope)[0]

        b, n, c = x.shape

        qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        dropout_p = module.attn_drop.p if self.training else 0.0

        size = self.info["size"]
        # (Inverse) proportional attention only needs the manual path once
        # token sizes exist, i.e. after the merge layer.
        weighted_attn = (
            self.info["prop_attn"] or self.info["iprop_attn"]
        ) and size is not None

        if module.fused_attn and not weighted_attn:  # NOTE: A False si export ONNX ?
            x = F.scaled_dot_product_attention(  # pylint: disable=E1102
                q, k, v, mask, dropout_p
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            # Using log(0)=-inf for padded to 0 tokens to mask them as keys
            if self.info["prop_attn"] and size is not None:
                attn = attn + size.log()[:, None, None, :]
            # Fixing attention of all padded to 0 tokens to -inf to keep them
            # masked as keys
            elif self.info["iprop_attn"] and size is not None:
                log_size = size.log().masked_fill(size == 0, float("inf"))
                attn = attn - log_size[:, None, None, :]
            # The boolean mask (merge mask + EoMT's query gating) must be
            # applied even when (i)prop attention re-weights the keys
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(b, n, c)))

        return x

    def add_query_tokens(self, x: torch.Tensor) -> TwoTensors:
        """Add query tokens to the token sequence and adapt G2TM size and
        mask tensors accordingly.

        Args:
            x (torch.Tensor): Token features.

        Returns:
            attn_mask (torch.Tensor): The updated attention mask (b, n, n).
        """
        b, n, _ = x.shape
        x = torch.cat(  # pylint: disable=E1101
            (self.q.weight[None, :, :].expand(b, -1, -1), x), dim=1
        )

        # Add size and mask values for new query tokens
        self.info["size"] = torch.cat(  # pylint: disable=E1101
            (
                torch.ones(  # pylint: disable=E1101
                    b, self.num_q, dtype=torch.int32, device=x.device
                ),
                self.info["size"],
            ),
            dim=1,
        )
        if self.info["mask"] is not None:
            # As query tokens will be added to the sequence, we ensure
            # that merge-away tokens are masked as keys for query
            # tokens also.
            mask_ = torch.cat(  # pylint: disable=E1101
                (
                    torch.ones(  # pylint: disable=E1101
                        b,
                        n,
                        self.num_q,
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    self.info["mask"],
                ),
                dim=2,
            )
            alive_tokens_ = torch.cat(  # pylint: disable=E1101
                (
                    torch.ones(  # pylint: disable=E1101
                        b,
                        self.encoder.backbone.num_prefix_tokens,
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    self.info["alive_tokens"],
                ),
                dim=1,
            )
            query_mask = torch.cat(  # pylint: disable=E1101
                (
                    torch.ones(  # pylint: disable=E1101
                        b,
                        self.num_q,
                        self.num_q,
                        dtype=torch.bool,
                        device=x.device,
                    ),
                    alive_tokens_.unsqueeze(1).expand(-1, self.num_q, -1),
                ),
                dim=2,
            )
            self.info["mask"] = torch.cat(  # pylint: disable=E1101
                (query_mask, mask_), dim=1
            )

        attn_mask = self.info["mask"]

        return x, attn_mask

    def forward(self, x: torch.Tensor) -> TwoTensors:
        """EoMT's forward pass, with a G2TM module inserted within one of its
        Transformer blocks.

        Args:
            x (torch.Tensor): Input images (b, 3, h, w).

        Returns:
            mask_logits_per_layer (list): List of predicted mask logits for
                segmentation from the final Transformer blocks (only the last block
                outside training).
            class_logits_per_layer (list): List of predicted class logits for
                segmentation from the final Transformer blocks (only the last block
                outside training).
        """
        self._reset_g2tm_state()
        x, rope = self._embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x, attn_mask = self.add_query_tokens(x)

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            x = self._attn_forward(block, x, attn_mask, rope)

            # G2TM module
            if i == self.info["selected_layer"] - 1:
                if self.info["fast_sv"]:
                    (
                        x,
                        self.info["size"],
                        self.info["mask"],
                        self.info["unmerge_idx"],
                        self.info["alive_tokens"],
                    ) = fast_sv_merge(
                        x,
                        self.info["threshold"],
                        self.encoder.backbone.num_prefix_tokens,
                        self.encoder.backbone.patch_embed.grid_size,
                        self.info["num_iters"],
                    )
                else:
                    (
                        x,
                        self.info["size"],
                        self.info["mask"],
                        self.info["unmerge_idx"],
                        self.info["alive_tokens"],
                    ) = bfs_merge(
                        x,
                        self.info["threshold"],
                        self.encoder.backbone.num_prefix_tokens,
                        self.encoder.backbone.patch_embed.grid_size,
                    )
                attn_mask = self.info["mask"]

            x = self._mlp_forward(block, x)

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )

    def get_feature_map(self, x: torch.Tensor, layer_id: int, post_attn: bool = False):
        self._reset_g2tm_state()
        x, rope = self._embed(x)

        attn_mask = None
        for i, block in enumerate(self.encoder.backbone.blocks[:layer_id]):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x, attn_mask = self.add_query_tokens(x)

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, _ = self._predict(self.encoder.backbone.norm(x))
                attn_mask = self._attn_mask(x, mask_logits, i)

            x = self._attn_forward(block, x, attn_mask, rope)

            # G2TM module
            if i == self.info["selected_layer"] - 1:
                if self.info["fast_sv"]:
                    (
                        x,
                        self.info["size"],
                        self.info["mask"],
                        self.info["unmerge_idx"],
                        self.info["alive_tokens"],
                    ) = fast_sv_merge(
                        x,
                        self.info["threshold"],
                        self.encoder.backbone.num_prefix_tokens,
                        self.encoder.backbone.patch_embed.grid_size,
                        self.info["num_iters"],
                    )
                else:
                    (
                        x,
                        self.info["size"],
                        self.info["mask"],
                        self.info["unmerge_idx"],
                        self.info["alive_tokens"],
                    ) = bfs_merge(
                        x,
                        self.info["threshold"],
                        self.encoder.backbone.num_prefix_tokens,
                        self.encoder.backbone.patch_embed.grid_size,
                    )
                attn_mask = self.info["mask"]

            # Same semantics as EoMT.get_feature_map: with post_attn, skip
            # only the MLP of the last requested block
            if not (post_attn and i == layer_id - 1):
                x = self._mlp_forward(block, x)

        return x, attn_mask, rope


def apply_patch(
    model: EoMT,
    selected_layer: int,
    threshold: float,
    prop_attn: bool = False,
    iprop_attn: bool = False,
    method: str = "bfs",
    num_iters: int = None,
):
    """Apply the modifications for G2TM token reduction method on
    the PyTorch ViT classifier.

    This function initializes the dictionnary storing the G2TM parameters,
    modifies the classes of some Segmenter's modules and inserts the
    dictionnary in these classes.

    Args:
        model (nn.Module): PyTorch model.
        selected_layer (int): Layer to apply G2TM.
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply or not Proportional Attention.
        iprop_attn (bool): Whether to apply or not Inverse Proportional
            Attention.
        method (str): Whether to use the custom BFS ('bfs') or the FastSV version
            ('fastsv', only one compatible with the ONNX export) of G2TM.
        num_iters (int): Number of FastSV iterations.
    """
    # Token reduction patch for EoMT
    model.token_reduction = True

    # Sanity checks
    if (
        selected_layer < 1
        or selected_layer > len(model.encoder.backbone.blocks) - model.num_blocks
    ):
        raise ValueError(
            "selected_layer should be the 1-based index of one of the first "
            "transformer blocks before query tokens are added (i.e.: between "
            f"1 and {len(model.encoder.backbone.blocks) - model.num_blocks}."
        )

    if prop_attn and iprop_attn:
        raise ValueError(
            "Inverse Proportional Attention and Proportional"
            "Attention cannot be activated at the same time."
        )

    if method not in ["nx", "bfs", "fastsv"]:
        raise ValueError(
            f"Method {method} provided is not supported, please select an "
            "implementation of G2TM among: NetworkX ('nx'), custom BFS ('bfs') and "
            "Fast SV ('fastsv')"
        )
    elif method == "nx":
        raise ValueError(
            "The NetworkX implementation ('nx') is deprecated, but you should switch "
            "to the custom BFS one ('bfs') as it is a faster version of the former."
        )

    model.__class__ = G2TMEoMT
    model.selected_layer = selected_layer
    model.threshold = threshold
    model.info = {
        "size": None,
        "mask": None,
        "unmerge_idx": None,
        "alive_tokens": None,
        "prop_attn": prop_attn,
        "iprop_attn": iprop_attn,
        "selected_layer": model.selected_layer,
        "threshold": model.threshold,
        "fast_sv": method == "fastsv",
    }
    if method == "fastsv":
        model.info["num_iters"] = num_iters

    print("FastSV (ONNX) version of G2TM activated: ", model.info["fast_sv"])
    print("Proportional Attention activated: ", model.info["prop_attn"])
    print("Inverse Proportional Attention activated: ", model.info["iprop_attn"])
