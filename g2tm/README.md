# G2TM: Single-Module Graph-Guided Token Merging for Efficient Semantic Segmentation

![G2TM Token Visualizations](./figs/visualisations.png)

<p align="center">
  <br> <strong>Authors:</strong> <a href="https://orcid.org/0009-0006-0682-8927" style="color: #4752C4">Victor BERCY</a>, <a href="https://orcid.org/0000-0002-5102-7735" style="color: #4752C4">Martyna POREBA</a>, <a href="https://orcid.org/0009-0000-9061-4396" style="color: #4752C4">Michal SZCZEPANSKI</a>, <a href="https://orcid.org/0000-0002-2860-8128" style="color: #4752C4">Samia BOUCHAFA</a>
</p>

<p align="center">
  <!-- Python version -->
  <img src="https://img.shields.io/badge/python-3.12-blue.svg" alt="Python versions">

  <!-- Pytorch version -->
  <img src="https://img.shields.io/badge/torch-2.4.1-red.svg" alt="Pytorch">
  
  <!-- Licence -->
  <img src="https://img.shields.io/badge/license-Apache2.0-green.svg" alt="License">

  <!-- Linting -->
  <a href="https://github.com/vbercy/g2tm/actions/workflows/pylint.yml">
    <img src="https://github.com/vbercy/g2tm/actions/workflows/pylint.yml/badge.svg?branch=main" alt="Pylint"/>
  </a>

  <!-- Conda installation -->
  <a href="https://github.com/vbercy/g2tm/actions/workflows/python-package-conda.yml">
    <img src="https://github.com/vbercy/g2tm/actions/workflows/python-package-conda.yml/badge.svg?branch=main" alt="Conda installation"/>
  </a>
</p>

<p align="center">
  <!-- Article -->
  <a href="https://www.scitepress.org/Link.aspx?doi=10.5220/0014267600004084">
    <img src="https://img.shields.io/badge/%F0%9F%93%83-Editor-yellow" alt="Article">
  </a>
  <!-- Article -->
  <a href="https://cea.hal.science/cea-05578363">
    <img src="https://img.shields.io/badge/%F0%9F%93%83-Open--source-brightgreen" alt="Article">
  </a>
  <!-- Repository -->
  <a href="https://github.com/vbercy/g2tm">
    <img src="https://img.shields.io/badge/GitHub-gray.svg?style=flat&logo=github" alt="Repository">
  </a>
</p>

Graph-Guided Token Merging (G2TM) is a lightweight one-shot module designed to eliminate redundant tokens early in the ViT architecture. It performs a single merging step after a shallow attention block, enabling all subsequent layers to operate on a compact token set. It leverages graph theory to identify groups of semantically redundant patches.

![G2TM Overview](./figs/g2tm.png)

## G2TM applied to Vision Transformer

In this repository, Graph-Guided Token Merging (G2TM) is applied to
[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby, ICLR 2020, by extending its code with token merging modules.

## Installation

In this section, we will explain how to set the environment up for this repository (ViT + G2TM) step by step. 

**1. Clone the repository:**

``` bash
git clone https://github.com/vbercy/g2tm
cd g2tm
```

**2. Setting up a conda environment:**

For this project, we recommand using [PyTorch](https://pytorch.org/) 2.11.0 built for CUDA 12.6, along with torchvision 0.26.0 and [NetworkX](https://github.com/networkx/networkx) 3.6.1.

``` bash
# Create environment
conda create -n g2tm_vit python==3.11.*
conda activate g2tm_vit
# Install the packaging helpers (the build backend requires setuptools < 80.9.0)
pip install setuptools==80.8.0 wheel
# PyTorch installation
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

**3. Installing the Vision Transformer requirements:**

``` bash
# set up the ViT package
pip install -v -e .
```

**4. Installing the G2TM requirements:**

``` bash
# set up the G2TM package (identical on both devices)
cd g2tm/ && pip install -v -e . && cd ../
```

The environment is ready when the following command prints the shape of a classification prediction:

``` bash
python -c "
import torch
from vit.model.vit import VisionTransformer
from vit.model.classifier import Classifier
from g2tm.patch import graph_vit_patch
vit = VisionTransformer((128, 128), 16, 4, 192, 768, 3, 19)
model = Classifier(vit, n_cls=19).eval()
graph_vit_patch(model, selected_layer=2, threshold=0.88)
print('CUDA:', torch.cuda.is_available(), '| output:', tuple(model(torch.randn(1, 3, 128, 128)).shape))
"
# CUDA: True | output: (1, 19, 128, 128)
```

**5. Prepare the datasets**

Define an OS environment variable pointing to the directory corresponding to the dataset you want to use:

```bash
export DATASET=/path/to/dataset/dir
```

## How to use?

### Training

To train a ViT model (size tiny, small, base or large) with G2TM on a specific dataset (whose path is provided by `DATASET`), use the command provided below. For example, we chose to apply G2TM at the 2nd layer with a threshold of 0.88 and without any modified attention formulation. We recommand using the Fast SV implementation of G2TM here, as it is the fastest implementation when running a batched training.

**Note:** a log file and a tensorboard directory will automatically be created for you to monitor your training.

```bash
python ./vit/train.py --log-dir <model_dir> \
                       --dataset <dataset_name> \
                       --backbone vit_<size>_patch16_384 \
                       --patch-type graph \
                       --selected-layer 2 \
                       --threshold 0.88 \
                       --batch-size 8 \
                       --fast-sv \
```

We explain here the specific options for G2TM:
- `--patch-type graph`: Applies the G2TM token reduction method.
- `--selected-layer 2`: Specifies which layers of the network to apply G2TM. In this case, the 2nd layer.
- `--threshold 0.88`: Sets the similarity threshold for merging tokens in G2TM.
- `--fast-sv`: Selects the Fast SV implementation of G2TM, instead of the BFS one (in this case `--bfs`)

**All training commands** can be run with or without G2TM using the `patch-type` option, as well as with or without (Inverse) Proportional Attention using the `prop-attn` or `iprop-attn` options.

For more examples of training commands (e.g.: with curriculum, with Inverse Proportional Attention, etc.), see [TRAINING](./TRAINING.md).

### Inference

You can download a checkpoint with its configuration in a common folder, in the [Results and Models](#results-and-models) part.

To perform an evaluation (Top-1 accuracy) of a ViT model with G2TM on the dataset it has been trained on, execute the following command. Make sure that the directory provided for the `model-path` option contains the checkpoint AND the `variant.yaml` file. Here, we choose to evaluate the model with G2TM applied at the 2nd layer with a threshold of 0.88, as it has been trained in the previous part.

**NOTE:** Please use the correct values for the `selected-layer` and `threshold` options for the evaluated model. You can find these values in the `variant.yaml` file of each model.

```bash
python ./vit/test.py <ckpt_file> \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.88
```

**All evaluation commands** can be run with or without G2TM using the `patch-type` option, as well as with or without (Inverse) Proportional Attention using the `prop-attn` or `iprop-attn` options.

### Benchmarking

To calculate the throughput and GFLOPs of a model, execute the following commands. Again, ensure that the directory provided for the `model-path` option contains the checkpoint AND the `variant.yaml` file.

**NOTE:** Please use the correct values for the `selected-layer` and `threshold` options for the evaluated model. You can find these values in the `variant.yaml` file of each model.

```bash
# Im/sec
python ./vit/speedtest.py <ckpt_file> <dataset_name> \
       --batch-size 1 \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.88
```
```bash
# GFLOPs
python ./vit/flops.py <ckpt_file> <dataset_name> \
       --batch-size 1 \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.88
```

To profile model activity during inference on CPU and GPU using PyTorch tools, use the following command:

```bash
python ./vit/profile_model.py <ckpt_file> <dataset_name> \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.88
```

**All benchmarking commands** can be run with or without G2TM using the `patch-type` option, as well as with or without (Inverse) Proportional Attention using the `prop-attn` or `iprop-attn` options.

### Token visualization

To visualize attention maps as well as the tokens at a specified layer for a specific image, execute the following command. It supports visualizations for both models with and without token reduction. For more details on the outputs, see the function documentation. In the example below, we generate visualization for a ViT model with G2TM applied at the 2nd layer with a threshold of 0.88.

```bash
python ./vit/show_attn_map.py <ckpt_file> <img_path> \
       <output_dir> <dataset_cmap> \
       --cls --enc --layer-id <layer> \
       --patch-type graph \
       --selected-layer 1 \
       --threshold 0.95
```

We explain here the specific options for G2TM:
- `--cls`: The attention maps provided are so with respect to the [CLS] token, otherwise (`--patch`) you have to provide the coordinate of the reference patch (`--x-patch <x> --y-patch <y>`).
- `--enc`: Whether the visualization is made in the encoder or in the decoder (`--dec`).
- `--layer-id <layer>`: The index of the layer (starting from 0) for visualization.

To get some statistics on the remaining tokens after merging, please run the following command:

```bash
python ./vit/token_stats.py <ckpt_file> <dataset> \
       --layer-id <layer> \
       --patch-type graph \
       --selected-layer 1 \
       --threshold 0.95
```

We explain here the specific options for G2TM:
- `--layer-id <layer>`: The index of the layer (starting from 0) where to measure the token statistics (measured after the merging operation if the Transformer block contains a G2TM module).

**All token commands** can be run with or without G2TM using the `patch-type` option, as well as with or without (Inverse) Proportional Attention using the `prop-attn` or `iprop-attn` options.

### ONNX export

To export a specific model into the ONNX format and evaluate it using ONNX Runtime, use the command provided below. In the example below, we convert a a ViT model with G2TM applied at the 2nd layer with a threshold of 0.88 into an ONNX file. G2TM hyperparameters will be frozen inside the file.

**Note:** The "ONNX-friendly" implementation of G2TM, using FastSV algorithm, will be automatically selected. In pratice, on ImageNet-1k, 6 iterations are enough to retain the same accuracy as the original implementation of G2TM on a ViT (see next command).

```bash
python ./vit/export_onnx.py <model_path> \
       --onnx-name <file_name> \
       --patch-type graph \
       --selected-layer 2 \
       --threshold 0.88 \
       --batch-size 8 \
       --num-iters 6 \
       --eval-onnx
```

We explain here the specific options for G2TM:
- `--onnx_name <file_name>`: Name of the ONNX file (without the .onnx extension).
- `--num-iters <n_iters>`: Number of Fast SV iterations.
- `--eval-onnx`: Whether to evaluate the exported ONNX model or not.

**All export commands** can be run with or without G2TM using the `patch-type` option, as well as with or without (Inverse) Proportional Attention using the `prop-attn` or `iprop-attn` options.

As the Fast SV version is an iterative algorithm, it needs a certain number of iterations to achieve the same fusions as the BFS version, otherwise it can leave connected components split. To determine the minimum number of iterations needed by FastSV for given model and dataset, you can run the command below.

```bash
python ./vit/fastsv_iters.py <model_path> <dataset_name> \
       --selected-layer 2 \
       --threshold 0.88 \
       --max-iters 16 \
       [--n-images N]
```

## Results and Models

See [RESULTS](./RESULTS.md) for some comparative results for ViT + G2TM and the corresponding model checkpoints.

**NOTE:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

**WARNING: For now, results above are given using the NetworkX implementation of G2TM, therefore differences in the throughput scores can occur if you use other implementations.**

## Upcoming Features 

```
- [x] Training and Inference scripts
- [x] Flops and Speedtest scripts
- [x] Token and attention map visualization scripts
- [x] Results on ImageNet-1k dataset
- [x] ONNX export script
- [ ] Nvidia Jetson running scripts
```

## Acknowledgements

This code rewrite the official [Vision Transformer](https://github.com/google-research/vision_transformer) code (under [Apache 2.0 Licence](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE))  using pure PyTorch and integrate it within the training/inference framework developed by the official [Segmenter](https://github.com/rstrudel/segmenter) code (under [MIT Licence](https://github.com/rstrudel/segmenter/blob/master/LICENSE)). It uses the repository structure and some utils functions from [ToMe](https://github.com/facebookresearch/ToMe) (under [CC-BY-NC licence](https://github.com/facebookresearch/ToMe/blob/main/LICENSE)), as well as utils functions from [AGLM](https://github.com/tue-mps/algm-segmenter).

Inheriting from the Segmenter repository, the Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models) library (under [Apache 2.0 Licence](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE)).

All files covered by Segmenter's or ToMe's licences include a header indicating the licence and whether the file has been modified. You can find such files from Segmenter's repository in the [`vit`](./vit/) directory and from ToMe's repository in the [`patch`](./g2tm/g2tm/patch/) and [`vis`](./g2tm/g2tm/vis/) folders.

Below are other Python librairies, along with their corresponding licenses, used in this work:
- [Click](https://github.com/pallets/click) under [BSD-3-Clause License](https://github.com/pallets/click/blob/main/LICENSE.txt)
- [einops](https://github.com/arogozhnikov/einops) under [MIT License](https://github.com/arogozhnikov/einops/blob/main/LICENSE)
- [FVCore](https://github.com/facebookresearch/fvcore) under [Apache 2.0 License](https://github.com/facebookresearch/fvcore/blob/main/LICENSE)
- [Matplotlib](https://github.com/matplotlib/matplotlib) under [PSF License](https://matplotlib.org/stable/project/license.html)
- [NetworkX](https://github.com/networkx/networkx) under [BSD-3-Clause License](https://github.com/networkx/networkx/blob/main/LICENSE.txt)
- [Numpy](https://github.com/numpy/numpy) under [BSD-3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt)
- [ONNX](https://github.com/onnx/onnx) under [Apache 2.0 License](https://github.com/onnx/onnx/blob/main/LICENSE)
- [ONNXRuntime](https://github.com/microsoft/onnxruntime) under [MIT License](https://github.com/microsoft/onnxruntime/blob/main/LICENSE)
- [Pillow](https://github.com/python-pillow/Pillow) under [MIT-CMU License](https://github.com/python-pillow/Pillow/blob/main/LICENSE)
- [PyTorch](https://github.com/pytorch/pytorch) under [BSD-3-Clause License](https://github.com/pytorch/pytorch/blob/main/LICENSE)
- [Scipy](https://github.com/scipy/scipy) under [BSD-3-Clause License](https://github.com/scipy/scipy/blob/main/LICENSE.txt)
- [Tqdm](https://github.com/tqdm/tqdm) under [MPL v. 2.0 and MIT Licenses](https://github.com/tqdm/tqdm/blob/master/LICENCE)

## License and Contributing

By contributing to G2TM, you agree that your contributions will be licensed under the [LICENSE file](./LICENSE) in the root directory of this source tree.

```
   Copyright © 2025 Commissariat à l'Energie Atomique et aux Energies Alternatives (CEA) 

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

## Citation

If you cite G2TM or use this repository in your work, please cite:

```
@conference{bercy2026g2tm,
       author={Victor Bercy and Martyna Poreba and Michal Szczepanski and Samia Bouchafa},
       title={G2TM: Single-Module Graph-Guided Token Merging for Efficient Semantic Segmentation},
       booktitle={Proceedings of the 21st International Conference on Computer Vision Theory and Applications - Volume 2: VISAPP},
       year={2026},
       pages={43-54},
       publisher={SciTePress},
       organization={INSTICC},
       doi={10.5220/0014267600004084},
       isbn={978-989-758-804-4},
       issn={2184-4321},
}
```
