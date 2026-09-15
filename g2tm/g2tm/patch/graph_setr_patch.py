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
"""Patching SETR model by inserting G2TM within one of its encoder's
Transformer block.
"""

from typing import Tuple

import torch

from setr.model.blocks import Block, Attention
from setr.model.setr import SETR
from setr.model.vit import VisionTransformer, MLAVisionTransformer

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
                    self.info["is_encoder"],
                    self.info["distill_token"],
                )
            )
        else:
            x, self.info["size"], self.info["mask"], self.info["unmerge_idx"] = (
                bfs_merge(
                    x,
                    self.info["threshold"],
                    self.info["is_encoder"],
                    self.info["distill_token"],
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


class G2TMMLAVisionTransformer(MLAVisionTransformer):
    """Modified SETR-MLA ViT encoder class for G2TM token reduction method.

    This module instanciates a dictionnary where all the parameters needed
    for the G2TM token reduction process are stored.
    """

    def forward(self, *args, **kwdargs) -> TwoTensors:

        self.info["size"] = None
        self.info["unmerge_idx"] = None
        self.info["mask"] = None
        self.info["selected_layer"] = self.selected_layer
        self.info["threshold"] = self.threshold

        return super().forward(*args, **kwdargs)

    def get_attention_map(self, *args, **kwdargs) -> TwoTensors:

        self.info["size"] = None
        self.info["unmerge_idx"] = None
        self.info["mask"] = None
        self.info["selected_layer"] = self.selected_layer
        self.info["threshold"] = self.threshold

        return super().get_attention_map(*args, **kwdargs)


def apply_patch(
    model: SETR,
    selected_layer: int,
    threshold: float,
    prop_attn: bool = False,
    iprop_attn: bool = False,
    fast_sv: bool = False,
    num_iters: int = None,
):
    """Apply the modifications for G2TM token reduction method on
    the PyTorch SETR model.

    This function initializes the dictionnary storing the G2TM parameters,
    modifies the classes of some SETR's modules and inserts the dictionnary
    in these classes.

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
    # Token reduction patch for SETR
    model.token_reduction = True

    # Token reduction patch for encoder
    if model.encoder.__class__ == VisionTransformer:
        model.encoder.__class__ = G2TMVisionTransformer
    elif model.encoder.__class__ == MLAVisionTransformer:
        model.encoder.__class__ = G2TMMLAVisionTransformer
    else:
        raise ValueError(
            f"Unsupported encoder type {model.encoder.__class__}, expected "
            "VisionTransformer or MLAVisionTransformer from SETR library."
        )
    model.encoder.selected_layer = selected_layer
    model.encoder.threshold = threshold
    model.encoder.info = {
        "size": None,
        "mask": None,
        "unmerge_idx": None,
        "prop_attn": prop_attn,
        "iprop_attn": iprop_attn,
        "is_encoder": model.encoder.cls_token is not None,
        "distill_token": False,
        "selected_layer": model.encoder.selected_layer,
        "threshold": model.encoder.threshold,
        "fast_sv": fast_sv,
    }
    if fast_sv:
        model.encoder.info["num_iters"] = num_iters

    print("FastSV (ONNX) version of G2TM activated: ", model.encoder.info["fast_sv"])
    print("Proportional Attention activated: ", model.encoder.info["prop_attn"])
    print(
        "Inverse Proportional Attention activated: ", model.encoder.info["iprop_attn"]
    )

    if model.encoder.info["prop_attn"] and model.encoder.info["iprop_attn"]:
        raise ValueError(
            "Inverse Proportional Attention and Proportional"
            "Attention cannot be activated at the same time."
        )

    if hasattr(model.encoder, "dist_token") and model.encoder.dist_token is not None:
        model.encoder.info["distill_token"] = True

    # (G2TMBlock or (Inverse)ProportionalBlock) + G2TMAttention => masked
    # (inverse) proportional attention
    # MaskedBlock + MaskedAttention => masked attention
    print("Selected layer: ", model.encoder.selected_layer)
    len_att = 1
    len_block = 1
    for module in model.encoder.modules():
        if isinstance(module, Block):
            if len_block == model.encoder.selected_layer:
                module.__class__ = G2TMBlock
                module.info = model.encoder.info
            elif model.encoder.info["prop_attn"]:
                module.__class__ = ProportionalBlock
                module.info = model.encoder.info
            elif model.encoder.info["iprop_attn"]:
                module.__class__ = InverseProportionalBlock
                module.info = model.encoder.info
            else:
                module.__class__ = MaskedBlock
                module.info = model.encoder.info
            len_block += 1
        # Before G2TM being applied, both  model.encoder.info["mask"] and
        # model.encoder.info["size"] are None
        elif isinstance(module, Attention):
            if model.encoder.info["prop_attn"]:
                module.__class__ = ProportionalAttention
                module.info = model.encoder.info
            elif model.encoder.info["iprop_attn"]:
                module.__class__ = InverseProportionalAttention
                module.info = model.encoder.info
            else:
                module.__class__ = MaskedAttention
                module.info = model.encoder.info
            len_att += 1

    # Token reduction patch for decoder
    model.decoder.info = model.encoder.info

    for module in model.decoder.modules():
        if isinstance(module, Block):
            if model.decoder.info["prop_attn"]:
                module.__class__ = ProportionalBlock
                module.info = model.decoder.info
            elif model.decoder.info["iprop_attn"]:
                module.__class__ = InverseProportionalBlock
                module.info = model.decoder.info
            else:
                module.__class__ = MaskedBlock
                module.info = model.decoder.info
            len_block += 1
        elif isinstance(module, Attention):
            if model.decoder.info["prop_attn"]:
                module.__class__ = ProportionalAttention
                module.info = model.decoder.info
            elif model.decoder.info["iprop_attn"]:
                module.__class__ = InverseProportionalAttention
                module.info = model.decoder.info
            else:
                module.__class__ = MaskedAttention
                module.info = model.decoder.info
            len_att += 1
