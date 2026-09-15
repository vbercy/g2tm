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
"""ViT classifier model."""

import torch
from torch import nn

from vit.model.utils import padding


class Classifier(nn.Module):
    """ViT classifier model."""

    def __init__(
        self,
        vit,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = vit.patch_size
        self.vit = vit
        self.token_reduction = False

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""

        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("vit.", self.vit)
        return nwd_params

    def forward(self, im, return_features=False):
        """Forward function."""
        im = padding(im, self.patch_size)
        x = self.vit(im, return_features)

        if self.token_reduction:
            num_extra_tokens = 1 + self.vit.distilled
            self.vit.info["size"] = self.vit.info["size"][:, num_extra_tokens:]
            if self.vit.info["mask"] is not None:
                self.vit.info["mask"] = self.vit.info["mask"][:, num_extra_tokens:]

        return x

    def get_attention_map(self, im, layer_id):
        """Get attention map from a specified layer of the encoder."""
        return self.vit.get_attention_map(im, layer_id)
