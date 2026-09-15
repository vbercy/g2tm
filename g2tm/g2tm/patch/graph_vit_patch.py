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
"""Patching ViT model by inserting G2TM within one of its encoder's Transformer
block.
"""

from typing import Tuple

import torch

from vit.model.blocks import Block, Attention
from vit.model.classifier import Classifier
from vit.model.vit import VisionTransformer

from g2tm.bfs_merge import bfs_merge
from g2tm.fast_sv_merge import fast_sv_merge
from g2tm.attention import (
    MaskedAttention,
    MaskedBlock,
    ProportionalBlock,
    ProportionalAttention,
    InverseProportionalBlock,
    InverseProportionalAttention,
)

TwoTensors = Tuple[torch.Tensor, torch.Tensor]
ThreeTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class G2TMBlock(Block):
    """Modified multi-head self-attention block with application of
    G2TM token reduction.

    This module coresponds to an entire self-attention block, to which we add
    a G2TM module between the masked attention layer and the MLP layer.

    Args:
        dim (int): Embedding dimension of the tokens.
        heads (int): Number of attention heads.
        mlp_dim (int): Hidden dimension of the MLP layer.
        dropout (float): dropout probability applied to attention weights.
        drop_path (float): Probability of dropping the residual path.

    Attributes:
        heads (int): Embedding dimension of the tokens.
        scale (float): Scaling factor applied to attention scores.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> TwoTensors:
        """Modified multi-head self-attention block implementation with
        application of G2TM token reduction.

        This method applies all the layers of the self-attention block to a
        token feature sequence. Between the attention layer and the MLP layer,
        the block applies the G2TM token reduction method to the sequence and
        initializes the source, size and mask tensors.

        Args:
            x (torch.Tensor): Token features (B, N, C).
            return_attention (bool): Wether the features (False) or the
                attention map (True) should be returned.

        Returns:
            x (torch.Tensor): Modified token features.
            OR attn (torch.Tensor): Corresponding attention map.
        """

        y, attn = self.attn(self.norm1(x))
        x = x + self._drop_path1(y)

        # G2TM module
        if self.info["fast_sv"]:
            x, self.info["size"], self.info["mask"], self.info["unmerge_idx"] = (
                fast_sv_merge(
                    x,
                    self.info["threshold"],
                    self.info["num_iters"],
                    self.info["distill_token"],
                    self.info["unmerge_idx_needed"],
                )
            )
        else:
            x, self.info["size"], self.info["mask"], self.info["unmerge_idx"] = (
                bfs_merge(
                    x,
                    self.info["threshold"],
                    self.info["distill_token"],
                    self.info["unmerge_idx_needed"],
                )
            )

        x = x + self._drop_path2(self.mlp(self.norm2(x)))

        # Placed at the end, to get access to size and unmerge_idx in the
        # visualization script
        if return_attention:
            return attn
        return x


class G2TMVisionTransformer(VisionTransformer):
    """Modified ViT encoder class for G2TM token reduction method.

    This module instanciates a dictionnary where all the parameters needed
    for the G2TM token reduction process are stored.
    """

    def forward(self, *args, **kwdargs) -> TwoTensors:

        self.info["size"] = None
        self.info["mask"] = None
        self.info["unmerge_idx"] = None
        self.info["selected_layer"] = self.selected_layer
        self.info["threshold"] = self.threshold

        return super().forward(*args, **kwdargs)

    def get_attention_map(self, *args, **kwdargs) -> TwoTensors:

        self.info["size"] = None
        self.info["mask"] = None
        self.info["unmerge_idx"] = None
        self.info["selected_layer"] = self.selected_layer
        self.info["threshold"] = self.threshold

        return super().get_attention_map(*args, **kwdargs)


def apply_patch(
    model: Classifier,
    selected_layer: int,
    threshold: float,
    prop_attn: bool = False,
    iprop_attn: bool = False,
    fast_sv: bool = False,
    num_iters: int = None,
):
    """Apply the modifications for G2TM token reduction method on
    the PyTorch ViT classifier.

    This function initializes the dictionnary storing the G2TM parameters,
    modifies the classes of some ViT's modules and inserts the
    dictionnary in these classes.

    Args:
        model (nn.Module): PyTorch model.
        selected_layer (int): Layer to apply G2TM.
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply or not Proportional Attention.
        iprop_attn (bool): Whether to apply or not Inverse Proportional
            Attention.
        fast_sv (bool): Whether to use or not the FastSV version of G2TM, compatible
            with the ONNX export.
        num_iters (int): Number of FastSV iterations.
    """
    # Token reduction patch for ViT
    model.token_reduction = True

    # Token reduction patch for vit
    model.vit.__class__ = G2TMVisionTransformer
    model.vit.selected_layer = selected_layer
    model.vit.threshold = threshold
    model.vit.info = {
        "size": None,
        "mask": None,
        "unmerge_idx": None,
        "prop_attn": prop_attn,
        "iprop_attn": iprop_attn,
        "distill_token": False,
        "selected_layer": model.vit.selected_layer,
        "threshold": model.vit.threshold,
        "fast_sv": fast_sv,
        "unmerge_idx_needed": False,
    }
    if fast_sv:
        model.vit.info["num_iters"] = num_iters

    print("FastSV (ONNX) version of G2TM activated: ", model.vit.info["fast_sv"])
    print("Proportional Attention activated: ", model.vit.info["prop_attn"])
    print("Inverse Proportional Attention activated: ", model.vit.info["iprop_attn"])

    if model.vit.info["prop_attn"] and model.vit.info["iprop_attn"]:
        raise ValueError(
            "Inverse Proportional Attention and Proportional"
            "Attention cannot be activated at the same time."
        )

    if hasattr(model.vit, "dist_token") and model.vit.dist_token is not None:
        model.vit.info["distill_token"] = True

    # (G2TMBlock or (Inverse)ProportionalBlock) + G2TMAttention => masked
    # (inverse) proportional attention
    # MaskedBlock + MaskedAttention => masked attention
    print("Selected layer: ", model.vit.selected_layer)
    len_att = 1
    len_block = 1
    for module in model.vit.modules():
        if isinstance(module, Block):
            if len_block == model.vit.selected_layer:
                module.__class__ = G2TMBlock
                module.info = model.vit.info
            elif model.vit.info["prop_attn"]:
                module.__class__ = ProportionalBlock
                module.info = model.vit.info
            elif model.vit.info["iprop_attn"]:
                module.__class__ = InverseProportionalBlock
                module.info = model.vit.info
            else:
                module.__class__ = MaskedBlock
                module.info = model.vit.info
            len_block += 1
        # Before G2TM being applied, both  model.vit.info["mask"] and
        # model.vit.info["size"] are None
        elif isinstance(module, Attention):
            if model.vit.info["prop_attn"]:
                module.__class__ = ProportionalAttention
                module.info = model.vit.info
            elif model.vit.info["iprop_attn"]:
                module.__class__ = InverseProportionalAttention
                module.info = model.vit.info
            else:
                module.__class__ = MaskedAttention
                module.info = model.vit.info
            len_att += 1


def unmerge_idx_needed(model: Classifier, needed: bool = True):
    """Ask G2TM to compute the unmerge index on the following forward passes.

    A classifier reads the [CLS] token, so the reduced token sequence never has
    to be expanded back onto the patch grid: the unmerge index is not computed
    by default. Only the visualization needs it, to rebuild the source matrix
    describing the fusions (see `g2tm.rebuild_source`).

    The G2TM block reads this flag from the dictionnary at every call, so this
    function can be called any time after the model has been patched, and the
    next forward pass will store the index in `model.vit.info["unmerge_idx"]`.

    Args:
        model (Classifier): PyTorch model already patched with G2TM.
        needed (bool): Whether the following forward passes have to compute the
            unmerge index.
    """
    if not getattr(model, "token_reduction", False):
        raise ValueError(
            "The model has no G2TM module: apply `graph_vit_patch` to it first."
        )
    model.vit.info["unmerge_idx_needed"] = needed
