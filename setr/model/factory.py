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
"""Utility functions to create SETR models."""

import os
from pathlib import Path

import yaml

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs, _create_vision_transformer
from timm.models.registry import register_model


from setr.model.vit import VisionTransformer
from setr.model.utils import checkpoint_filter_fn
from setr.model.vit import MLAVisionTransformer
from setr.model.decoder import NaiveDecoder, PUPDecoder, MLADecoder
from setr.model.setr import SETR
import setr.utils.torch as ptu


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper
    (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384,
    source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = {
        "patch_size": 8,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        **kwargs,
    }
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg={
            "url": "",
            "input_size": (3, 384, 384),
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "num_classes": 1000,
        },
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    """Instanciate ViT backbone."""
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    _ = model_cfg.pop("normalization")
    model_cfg.setdefault("n_cls", 1000)
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    default_cfg = default_cfgs.get(
        backbone,
        {
            "pretrained": False,
            "num_classes": 1000,
            "drop_rate": 0.0,
            "drop_path_rate": 0.0,
            "drop_block_rate": None,
        },
    )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    vit_cls = MLAVisionTransformer if "n_stages" in model_cfg else VisionTransformer
    model = vit_cls(**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = ptu.load_checkpoint(
            path,
            map_location="cpu",
            weights_only=False,
        )
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    """Instanciate decoder."""
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model

    if name == "naive":
        decoder = NaiveDecoder(**decoder_cfg)
    elif name == "pup":
        decoder = PUPDecoder(**decoder_cfg)
    elif name == "mla":
        decoder = MLADecoder(**decoder_cfg)
    else:
        raise ValueError(
            f"SETR's decoder type should be Naive, PUP or MLA. Got {name}."
        )

    return decoder


def create_setr(model_cfg):
    """Instanciate whole SETR model."""
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    # Add MLA indices to config for encoder
    if decoder_cfg["name"] == "mla":
        model_cfg["n_stages"] = decoder_cfg["n_stages"]
        decoder_cfg["d_encoder"] = model_cfg["d_model"]
        decoder_cfg["n_layers"] = model_cfg["n_layers"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = SETR(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path):
    """Instanciate the model according to the configuration file and load a
    checkpoint into the model.
    """
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r", encoding="utf-8") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_setr(net_kwargs)
    data = ptu.load_checkpoint(
        model_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant
