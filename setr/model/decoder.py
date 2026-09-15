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

# Modifications based on code from Sixiao Zheng et al. (SETR)

# MIT License

# Copyright (c) Zhang Vision Group, Fudan University.

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
"""Decoder PyTorch classes."""

from math import sqrt
import torch
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, ModuleList, Sequential
import torch.nn.functional as F

from setr.model.utils import init_weights


class NaiveDecoder(Module):
    """Naive Decoder from SETR."""

    def __init__(
        self,
        n_cls,
        d_encoder,
        channels=256,
        n_conv=2,
        upsampling="bilinear",
        align_corners=False,
    ):
        super().__init__()

        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.upsampling = upsampling
        self.align_corners = align_corners

        self.n_conv = n_conv
        self.conv0 = Conv2d(d_encoder, channels, kernel_size=1, stride=1)
        self.conv1 = Conv2d(channels, n_cls, kernel_size=1, stride=1)
        self.bn = BatchNorm2d(channels)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""
        return set()

    def forward(self, x, im_size):
        """Forward function."""
        b, n, c = x.size()
        base_grid = int(sqrt(n))
        assert base_grid**2 == n

        x = x.transpose(1, 2).reshape((b, c, base_grid, base_grid)).contiguous()

        x = self.bn(self.conv0(x))
        x = F.relu(x, inplace=True)
        # Does not exist in the paper, but in the official implementation
        # See https://github.com/fudan-zvg/SETR/blob/main/mmseg/models/decode_heads/vit_up_head.py
        x = F.interpolate(
            x,
            size=x.shape[-1] * 4,
            mode=self.upsampling,
            align_corners=self.align_corners,
        )
        x = self.conv1(x)
        x = F.interpolate(
            x, size=im_size, mode=self.upsampling, align_corners=self.align_corners
        )

        return x


class PUPDecoder(Module):
    """PUP Decoder from SETR-PUP."""

    def __init__(
        self,
        n_cls,
        d_encoder,
        channels=256,
        n_conv=4,
        upsampling="bilinear",
        align_corners=False,
    ):
        super().__init__()

        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.upsampling = upsampling
        self.align_corners = align_corners

        self.n_conv = n_conv
        self.convs = ModuleList(
            [Conv2d(d_encoder, channels, kernel_size=3, stride=1, padding=1)]
            + [
                Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
                for _ in range(n_conv - 1)
            ]
            + [Conv2d(channels, n_cls, kernel_size=1, stride=1)]
        )
        self.bns = ModuleList([BatchNorm2d(channels) for _ in range(n_conv)])

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""
        return set()

    def forward(self, x, im_size):
        """Forward function."""
        b, n, c = x.size()
        base_grid = int(sqrt(n))
        assert base_grid**2 == n

        x = x.transpose(1, 2).reshape((b, c, base_grid, base_grid)).contiguous()

        for i in range(self.n_conv + 1):
            if i == self.n_conv:
                x = self.convs[i](x)
            else:
                x = self.bns[i](self.convs[i](x))
                x = F.relu(x, inplace=True)
                x = F.interpolate(
                    x,
                    size=x.shape[-1] * 2,
                    mode=self.upsampling,
                    align_corners=self.align_corners,
                )

        assert x.shape[-2] == im_size[0] and x.shape[-1] == im_size[1]

        return x


class ConvMLA(Module):
    """Convolutional module from the MLA decoder."""

    def __init__(self, in_channels=1024, mla_channels=512, n_stages=4):
        super().__init__()
        self.n_stages = n_stages

        self.convs = ModuleList(
            [
                Sequential(
                    Conv2d(in_channels, mla_channels, 1, bias=False),
                    BatchNorm2d(mla_channels),  # LayerNorm in official implementation
                    ReLU(),
                )
                for _ in range(n_stages)
            ]
        )

        self.mla_convs = ModuleList(
            [
                Sequential(
                    Conv2d(mla_channels, mla_channels, 3, padding=1, bias=False),
                    BatchNorm2d(mla_channels),
                    ReLU(),
                )
                for _ in range(n_stages)
            ]
        )

    def to_2d(self, x):
        """Reshape the token sequence into a 2D feature map."""
        b, n, c = x.shape
        base_grid = int(sqrt(n))
        assert base_grid**2 == n
        x = x.transpose(1, 2).reshape((b, c, base_grid, base_grid)).contiguous()
        return x

    def forward(self, feat0, feat1, feat2, feat3):
        """Forward function, feature maps given from the shallower to the deeper one."""

        feat0 = self.to_2d(feat0)
        feat1 = self.to_2d(feat1)
        feat2 = self.to_2d(feat2)
        feat3 = self.to_2d(feat3)

        feat0 = self.convs[0](feat0)
        feat1 = self.convs[1](feat1)
        feat2 = self.convs[2](feat2)
        feat3 = self.convs[3](feat3)

        feat3_plus = feat3
        feat2_plus = feat2 + feat3_plus
        feat1_plus = feat1 + feat2_plus
        feat0_plus = feat0 + feat1_plus

        mla0 = self.mla_convs[0](feat0_plus)
        mla1 = self.mla_convs[1](feat1_plus)
        mla2 = self.mla_convs[2](feat2_plus)
        mla3 = self.mla_convs[3](feat3)

        return mla0, mla1, mla2, mla3


class MLADecoder(Module):
    """MLA Decoder from SETR-MLA."""

    def __init__(
        self,
        n_cls,
        d_encoder,
        n_layers,
        n_stages=4,
        upsampling="bilinear",
        align_corners=False,
    ):
        super().__init__()

        self.d_encoder = d_encoder
        self.n_cls = n_cls
        self.n_stages = n_stages
        self.mla_indices = [n_layers // n_stages * (i + 1) - 1 for i in range(n_stages)]
        self.upsampling = upsampling
        self.align_corners = align_corners

        channels = d_encoder // 2
        head_channels = channels // 2
        self.mla = ConvMLA(d_encoder, channels, self.n_stages)
        self.heads = ModuleList(
            [
                Sequential(
                    Conv2d(channels, head_channels, 3, padding=1, bias=False),
                    BatchNorm2d(head_channels),
                    ReLU(),
                )
                for _ in range(self.n_stages)
            ]
        )
        self.final_conv = Conv2d(4 * head_channels, n_cls, 3, padding=1)

        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Modules with no weight decay."""
        return set()

    def forward(self, feat0, feat1, feat2, feat3, im_size):
        """Forward function, feature map given from the shallower to the
        deeper one.
        """
        feat0, feat1, feat2, feat3 = self.mla(feat0, feat1, feat2, feat3)

        scale_factor = 4
        head0 = F.interpolate(
            self.heads[0](feat0),
            scale_factor=scale_factor,
            mode=self.upsampling,
            align_corners=self.align_corners,
        )
        head1 = F.interpolate(
            self.heads[1](feat1),
            scale_factor=scale_factor,
            mode=self.upsampling,
            align_corners=self.align_corners,
        )
        head2 = F.interpolate(
            self.heads[2](feat2),
            scale_factor=scale_factor,
            mode=self.upsampling,
            align_corners=self.align_corners,
        )
        head3 = F.interpolate(
            self.heads[3](feat3),
            scale_factor=scale_factor,
            mode=self.upsampling,
            align_corners=self.align_corners,
        )

        out = self.final_conv(torch.cat([head0, head1, head2, head3], dim=1))
        out = F.interpolate(
            out, size=im_size, mode=self.upsampling, align_corners=self.align_corners
        )

        return out
