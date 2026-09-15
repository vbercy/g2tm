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
"""SETR model PyTorch class."""

import torch
from torch import nn
import torch.nn.functional as F

from setr.model.utils import padding, unpadding
from setr.model.vit import MLAVisionTransformer
from setr.model.decoder import MLADecoder


class SETR(nn.Module):
    """SETR model."""

    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.token_reduction = False
        self.is_mla = False

        if isinstance(self.decoder, MLADecoder):
            assert isinstance(self.encoder, MLAVisionTransformer)
            self.is_mla = True

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""

        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        """Forward function."""
        h_ori, w_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        h, w = im.size(2), im.size(3)

        # Merging and unmerging done inside the encoder
        x = self.encoder(im, self.token_reduction)

        masks = self.decoder(*x, (h, w))

        masks = F.interpolate(masks, size=(h, w), mode="bilinear")
        masks = unpadding(masks, (h_ori, w_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        """Get attention map from a specified layer of the encoder."""
        return self.encoder.get_attention_map(im, layer_id)
