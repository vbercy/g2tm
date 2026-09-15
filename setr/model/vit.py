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

# Modifications based on code from Robin Strudel et al. (Segmenter)

# MIT License

# Copyright (c) 2021 Robin Strudel
# Copyright (c) INRIA

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Adapted from 2020 Ross Wightman (https://github.com/rwightman/pytorch-image-models)
"""Vision Transformer PyTorch class, as SETR's encoder."""

import torch
from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import _load_weights

from setr.model.utils import init_weights, resize_pos_embed
from setr.model.blocks import Block


class PatchEmbedding(nn.Module):
    """Patch Embedding layer (i.e. : Conv2D layer)"""

    def __init__(self, image_size, patch_size, embed_dim, channels):
        super().__init__()

        self.image_size = image_size
        if image_size[0] % patch_size != 0 or image_size[1] % patch_size != 0:
            raise ValueError("image dimensions must be divisible by the patch size")
        self.grid_size = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, im):
        """Forward function."""
        # B, C, H, W = im.shape
        x = self.proj(im).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ViT backbone."""

    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size,
            patch_size,
            d_model,
            channels,
        )
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
        self.n_cls = n_cls

        # cls and pos tokens
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)  # pylint: disable=E1101
        )
        self.distilled = distilled
        if self.distilled:
            self.dist_token = nn.Parameter(
                torch.zeros(1, 1, d_model)  # pylint: disable=E1101
            )
            self.pos_embed = nn.Parameter(
                torch.randn(  # pylint: disable=E1101
                    1, self.patch_embed.num_patches + 2, d_model
                )
            )
            self.head_dist = nn.Linear(d_model, n_cls)
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(  # pylint: disable=E1101
                    1, self.patch_embed.num_patches + 1, d_model
                )
            )

        # transformer blocks
        dp_rates = torch.linspace(0, drop_path_rate, n_layers)  # pylint: disable=E1101
        dpr = [x.item() for x in dp_rates]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        # output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_cls)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        if self.distilled:
            trunc_normal_(self.dist_token, std=0.02)
        self.pre_logits = nn.Identity()

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """Load pretrained weights."""
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, im, token_reduction=False):
        """Forward function."""
        b, _, h, w = im.shape
        ps = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # pylint: disable=E1101
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # pylint: disable=E1101

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (h // ps, w // ps),
                num_extra_tokens,
            )
        x = x + pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # remove CLS/DIST tokens for decoding and unmerge
        x = x[:, num_extra_tokens:]
        if token_reduction:
            self.info["size"] = self.info["size"][:, num_extra_tokens:]
            if self.info["mask"] is not None:
                self.info["mask"] = self.info["mask"][:, num_extra_tokens:]
            # We get rid of the source matrix and the argmax, the index is derived from
            # the component labels by the merge.
            idxs = self.info["unmerge_idx"]
            x_ = x.gather(1, idxs.unsqueeze(-1).expand(-1, -1, self.d_model))

            return (x_,)

        return (x,)

    def get_attention_map(self, im, layer_id):
        """Get attention maps from a specified layer."""
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id}"
                f" < {self.n_layers}."
            )
        b, _, h, w = im.shape
        ps = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # pylint: disable=E1101
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # pylint: disable=E1101

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (h // ps, w // ps),
                num_extra_tokens,
            )
        x = x + pos_embed

        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


class MLAVisionTransformer(VisionTransformer):
    """ViT backbone for SETR-MLA."""

    def __init__(
        self,
        image_size,
        patch_size,
        n_layers,
        d_model,
        d_ff,
        n_heads,
        n_cls,
        dropout=0.1,
        drop_path_rate=0.0,
        distilled=False,
        channels=3,
        n_stages=4,
    ):
        super().__init__(
            image_size,
            patch_size,
            n_layers,
            d_model,
            d_ff,
            n_heads,
            n_cls,
            dropout,
            drop_path_rate,
            distilled,
            channels,
        )
        self.n_stages = n_stages
        assert n_layers % n_stages == 0, (
            f"The number of layers ({n_layers}"
            f") is not divisible by the number of MLA stages ({n_stages})."
        )
        self.mla_indices = [n_layers // n_stages * (i + 1) - 1 for i in range(n_stages)]

        self.norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(self.n_stages)]
        )

        self.apply(init_weights)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        """Load pretrained weights."""
        _load_weights(self, checkpoint_path, prefix)
        # Copy the params of the original unique LayerNorm onto the MLA ones
        with torch.no_grad():
            for ln in self.norms:
                ln.weight.copy_(self.norm.weight)
        del self.norm

    def forward(self, im, token_reduction=False):
        """Forward function, feature maps returned from the shallower to the
        deeper one.
        """
        b, _, h, w = im.shape
        ps = self.patch_size

        x = self.patch_embed(im)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        if self.distilled:
            dist_tokens = self.dist_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # pylint: disable=E1101
        else:
            x = torch.cat((cls_tokens, x), dim=1)  # pylint: disable=E1101

        pos_embed = self.pos_embed
        num_extra_tokens = 1 + self.distilled
        if x.shape[1] != pos_embed.shape[1]:
            pos_embed = resize_pos_embed(
                pos_embed,
                self.patch_embed.grid_size,
                (h // ps, w // ps),
                num_extra_tokens,
            )
        x = x + pos_embed
        # x = x[:, 1:]  # drop [CLS] token as in SETR-MLA backbone
        x = self.dropout(x)

        feats = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.mla_indices:
                feats.append(x)

        feat0 = self.norms[0](feats[0])
        feat1 = self.norms[1](feats[1])
        feat2 = self.norms[2](feats[2])
        feat3 = self.norms[3](feats[3])

        # remove CLS/DIST tokens for decoding and unmerge
        feat0 = feat0[:, num_extra_tokens:]
        feat1 = feat1[:, num_extra_tokens:]
        feat2 = feat2[:, num_extra_tokens:]
        feat3 = feat3[:, num_extra_tokens:]
        if token_reduction:
            self.info["size"] = self.info["size"][:, num_extra_tokens:]
            if self.info["mask"] is not None:
                self.info["mask"] = self.info["mask"][:, num_extra_tokens:]
            # We get rid of the source matrix and the argmax, the index is derived from
            # the component labels by the merge.
            idxs = self.info["unmerge_idx"]
            feat0_ = feat0.gather(1, idxs.unsqueeze(-1).expand(-1, -1, self.d_model))
            feat1_ = feat1.gather(1, idxs.unsqueeze(-1).expand(-1, -1, self.d_model))
            feat2_ = feat2.gather(1, idxs.unsqueeze(-1).expand(-1, -1, self.d_model))
            feat3_ = feat3.gather(1, idxs.unsqueeze(-1).expand(-1, -1, self.d_model))

            return (feat0_, feat1_, feat2_, feat3_)

        return (feat0, feat1, feat2, feat3)
