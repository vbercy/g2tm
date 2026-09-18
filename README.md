# G2TM: Single-Module Graph-Guided Token Merging for Efficient Semantic Segmentation

![G2TM Token Visualizations](./figs/visualisations.png)

<p align="center">
  <br> <strong>Authors:</strong> <a href="https://orcid.org/0009-0006-0682-8927" style="color: #4752C4">Victor BERCY</a>, <a href="https://orcid.org/0000-0002-5102-7735" style="color: #4752C4">Martyna POREBA</a>, <a href="https://orcid.org/0009-0000-9061-4396" style="color: #4752C4">Michal SZCZEPANSKI</a>, <a href="https://orcid.org/0000-0002-2860-8128" style="color: #4752C4">Samia BOUCHAFA</a>
</p>

<p align="center">
  <!-- Python version -->
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python versions">

  <!-- Pytorch version -->
  <img src="https://img.shields.io/badge/torch-2.11.0-red.svg" alt="Pytorch">
  
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
  <a href="https://cea.hal.science/cea-05578363">
    <img src="https://img.shields.io/badge/%F0%9F%93%83-Original%20version-yellow" alt="Article">
  </a>
  <!-- Extension -->
  <a href="https://arxiv.org/abs/2609.18279">
    <img src="https://img.shields.io/badge/%F0%9F%93%83-Extended%20version-yellow" alt="Article">
  </a>
  <!-- Repository -->
  <a href="https://github.com/vbercy/g2tm">
    <img src="https://img.shields.io/badge/GitHub-gray.svg?style=flat&logo=github" alt="Repository">
  </a>
</p>

Graph-Guided Token Merging (G2TM) is a lightweight one-shot module designed to eliminate redundant tokens early in the ViT architecture. It performs a single merging step after a shallow attention block, enabling all subsequent layers to operate on a compact token set. It leverages graph theory to identify groups of semantically redundant patches.

![G2TM Overview](./figs/g2tm.png)

## G2TM applied to EoMT

In this repository, Graph-Guided Token Merging (G2TM) is applied to
[Your ViT is Secretly an Image Segmentation Model ](https://arxiv.org/abs/2503.19108)
by Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus, CVPR 2025, by extending its code with token merging modules.

## Installation

In this section, we will explain how to set the environment up for this repository (EoMT + G2TM) step by step. 

**1. Clone the repository:**

``` bash
git clone https://github.com/vbercy/g2tm
cd g2tm
```

**2. Setting up a conda environment:**

For this project, we recommand using [PyTorch](https://pytorch.org/) 2.11.0 built for CUDA 12.6, along with torchvision 0.26.0 and [NetworkX](https://github.com/networkx/networkx) 3.6.1.

``` bash
# Create environment
conda create -n g2tm_eomt python==3.11.*
conda activate g2tm_eomt
# PyTorch installation
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu126
```

**3. Installing the EoMT requirements:**

``` bash
# set up the EoMT package
pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

**4. Installing the G2TM requirements:**

``` bash
# set up the G2TM package (identical on both devices)
cd g2tm/ && pip install -v -e . && cd ../
```

The environment is ready when the following command, run from the root of the repository, prints the shape of the predicted masks and class logits. It builds an EoMT model with a ViT-T backbone, inserts G2TM at the 2nd layer and forwards a random 512×512 image, without any dataset, checkpoint or weight download:

``` bash
python -c "
import torch
from models.vit import ViT
from models.eomt import EoMT
from g2tm.patch import graph_eomt_patch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = ViT((512, 512), backbone_name='vit_tiny_patch16_384', ckpt_path='none')
net = EoMT(encoder, num_classes=150, num_q=100, masked_attn_enabled=False)
graph_eomt_patch(net, selected_layer=2, threshold=0.0)
net.eval().to(device)
with torch.inference_mode():
    masks, classes = net(torch.rand(1, 3, 512, 512, device=device))
merged = net.info['size'].size(1) < net.num_q + 1 + net.n_patch_tokens
print(f'CUDA: {torch.cuda.is_available()} | G2TM merged: {merged} | masks: {tuple(masks[-1].shape)} | classes: {tuple(classes[-1].shape)}')
"
# CUDA: True | G2TM merged: True | masks: (1, 100, 128, 128) | classes: (1, 100, 151)
```

**5. Prepare the datasets**

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.

## How to use?

This repository keeps EoMT's entry point for training and evaluation through `main.py` with a config from [`configs/`](./configs/).
The analysis, export and utility scripts can be found in [`tools/`](./tools/). Every command below accepts the usual [LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) overrides, so G2TM is configured with plain `--model.*` arguments.

### G2TM options

These apply to **every** command in this section, `main.py` and `tools/` alike:

| Option | Description |
| --- | --- |
| `--model.patch_type graph` | Enables G2TM (`pure`, the default, disables it). |
| `--model.selected_layer <layer_id>` | 1-based index of the block after which tokens are merged. |
| `--model.threshold <tau>>` | Cosine-similarity threshold above which neighbours are merged. |
| `--model.method <implementation>` | Connected-components implementation: `bfs` (default, fastest with PyTorch during inference) or `fastsv` (tensorized, the only one exportable to ONNX). |
| `--model.num_iters <n_iters>` | Number of FastSV iterations (for `fastsv` only). |
| `--model.prop_attn <true\|false>` | Proportional Attention (mutually exclusive with the below). |
| `--model.iprop_attn <true\|false>` | Inverse Proportional Attention (mutually exclusive with the above). |

**NOTE:** Define a environment variable `DATASET` containing the path to the directory containing the dataset zip files and which will be used in the commands below. Also, replace `<ckpt_file>` with the path to your checkpoint file.

### Training

To train an EoMT-Large model with G2TM applied at the 2nd layer with a threshold of 0.88, on ADE20K at resolution 512×512, run:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.method fastsv
```

We recommand using the Fast SV implementation (option `--model.method fastsv`) of G2TM here, as it is the fastest implementation when running a batched training.

**NOTE:** Keep the total batch size at `devices × batch_size = 16`, as in EoMT.

Runs are logged to [Weights & Biases](https://wandb.ai/); use `tools/sync_wandb.py` to push an offline run (see [Utilities](#utilities)).

To fine-tune from an existing checkpoint, add:

```bash
  --model.ckpt_path <ckpt_file> \
  --model.load_ckpt_class_head False
```

`--model.load_ckpt_class_head False` skips the classification head when fine-tuning on a dataset with a different number of classes.

**NOTE:** With **DINOv3** configs, the code expects delta weights by default; add `--model.delta_weights False` to use absolute weights.

**Threshold curriculum.** To start from a permissive threshold and tighten it during training, add `--model.curric_thresh true --model.start_thresh 0.95 --model.curric_warmup 6 --model.curric_period 3`: the threshold then decreases by 0.01 every `curric_period` epochs after `curric_warmup`, down to `--model.threshold`.

**NOTE:** `torch.compile` is enabled by default. G2TM is compatible with it, but the BFS connected components algorithm is not (custom and NetworkX implementation); pass `--compile_disabled` to turn the compilation off entirely.

For more training examples (other datasets and resolutions, the three G2TM
implementations, (Inverse) Proportional Attention, threshold curriculum, fine-tuning), see [TRAINING](./TRAINING.md).

### Evaluation

To perform an evaluation (mIoU) of an EoMT model with G2TM on the dataset it has been trained on, run the following command.

```bash
python main.py validate \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--model.ckpt_path <ckpt_file> \
	--model.network.masked_attn_enabled False \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88
```

**NOTE:** The `selected_layer`, `threshold` and `method` values the model was trained with are stored in the checkpoint under the `merging` key and printed when loading.

`--model.network.masked_attn_enabled False` is EoMT's inference behaviour: mask annealing has driven the attention masks off by the end of training, and it avoids re-running the Mask Module before each of the last blocks.

### Benchmarking

All benchmarking scripts process **one sliding-window crop at a time** (batch size 1), so G2TM runs on its sparse path where merged tokens are removed from the sequence. The padded path used for batched training never shortens it, and would report no gain.

```bash
# Throughput (crops/s)
python tools/fps_bench.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--model.ckpt_path <ckpt_file> \
	--model.network.masked_attn_enabled False \
	--warmup 50 --repeat 5 \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

```bash
# GFLOPs (fvcore, reported as MACs)
python tools/compute_flops.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--model.ckpt_path <ckpt_file> \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

Options specific to these scripts:
- `--max_images N`: profile only the first `N` validation images.
- `--warmup` / `--repeat`: warm-up iterations, and timed repetitions per crop.
- `--amp`: benchmark under FP16 autocast (`fps_bench.py` only).
- `--head`: count only the last blocks and the Mask Module (`compute_flops.py` only).
- `--mode crop|image`: report per crop (default) or summed over an image's crops.

`compute_flops.py` forces `masked_attn_enabled` off on its own, and counts fused attention through a dedicated handler, which `fvcore` does not support.

### Token analysis and visualization

To get statistics on the tokens remaining after merging:

```bash
python tools/token_stats.py \
    -c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
    --data_path $DATASET \
    --model.ckpt_path <ckpt_file> \
    --model.network.masked_attn_enabled False \
    --layer_id 2 \
    --model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

- `--layer_id <layer>`: 1-based index of the block at which tokens are counted (after the merge, if that block holds the G2TM module).

To plot the distribution of cosine similarity between neighbouring tokens, block by block, useful for choosing a threshold:

```bash
python tools/cosine_sim_dist.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--model.ckpt_path <ckpt_file> \
	--output_dir ./figs/sim_distrib \
	--stage validate \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

- `--stage fit|validate`: split to draw images from.
- One `distrib_block<i>.png` is written per encoder block.

To predict and save colorized segmentation maps for your own images:

```bash
python tools/predict_seg_maps.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--ckpt_path <ckpt_file> \
	--images_dir ./vis/ade20k/images \
	--cmap ./vis/ade20k/cmap.yml \
	--output_dir ./vis/ade20k/preds \
	--overlay \
	--model.network.masked_attn_enabled False \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

- `--images_dir` / `--cmap`: input images and the class colormap (default to the ADE20K ones shipped in [`vis/`](./vis/)).
- `--overlay`: additionally save a blend of each image with its segmentation map.

To visualize attention maps as well as the tokens at a specified block for a specific image, run the following command. It supports visualizations for both models with and without token reduction. For more details on the outputs, see the function documentation. In the example below, we generate visualization for an EoMT model with G2TM applied at the 2nd layer with a threshold of 0.88.

```bash
python tools/show_attn_map.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--ckpt_path <ckpt_file> \
	--image_path ./vis/ade20k/images/ADE_val_00000001.jpg \
	--output_dir ./vis/ade20k/attn \
	--layer_id 5 --token patch --x_patch 16 --y_patch 16 \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88
```

- `--layer_id <layer>`: 0-based index of the Transformer block to visualize.
- `--token patch|cls|query`: token the attention is read from. `patch` uses the patch at `--x_patch <x> --y_patch <y>`, `cls` uses the [CLS] token, and `query` uses the query tokens of the Mask Module.
- `--query_id <i>`: with `--token query`, restrict the output to that single query. By default, one map is saved per query that predicts an actual class, in a folder named after it (the class names are read from `--cmap_file`, default `./vis/ade20k/cmap.yml`).
- `--cmap <name>` / `--sigma <px>`: matplotlib colormap and standard deviation of the gaussian smoothing of the attention maps.

Segmentation maps are not produced here, `tools/predict_seg_maps.py` above already covers them.

**NOTE:** EoMT is encoder-only, so there is no encoder/decoder split to choose from. What plays the role of a decoder is the last `num_blocks` blocks, the only ones where the query tokens are part of the sequence: `--token query` is therefore rejected before block `len(blocks) - num_blocks` (block 20 for a ViT-L).

**NOTE:** `--data_path` is still required here: it is only used to instantiate the datamodule from the config, no dataset image is read.

### ONNX export

To export an EoMT model into the ONNX format and validate it with ONNX Runtime:

```bash
python tools/export_onnx.py \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data_path $DATASET \
	--ckpt_path <ckpt_file> \
	--eval_onnx \
	--model.patch_type graph --model.selected_layer 2 --model.threshold 0.88 \
	--model.method fastsv --model.num_iters 8
```

G2TM hyperparameters will be frozen inside the file.

**Note:** The "ONNX-friendly" implementation of G2TM, using FastSV algorithm, will be automatically selected. In pratice, on ADE20K, 8 iterations are enough to retain the same accuracy as the original implementation of G2TM on a Segmenter (see next command).

**NOTE:** G2TM **must** use `--model.method fastsv` option. The `bfs` and `nx` implementations cannot be converted to ONNX because of their connected components algorithm made in pure Python.

Options specific to this script:
- `--onnx_name <name>`: name of the ONNX file, without the `.onnx` extension. Defaults to a name describing the config and the G2TM parameters.
- `--opset <n>`: ONNX opset, 18 or above (the FastSV connected components need the min-reduction `ScatterElements` introduced there).
- `--eval_onnx`: evaluate the exported model with ONNX Runtime and report its mIoU.
- `--max_images N`: restrict that evaluation to the first `N` validation images.

The exported graph takes one `float32` crop `(1, 3, H, W)` with values in `[0, 255]` and returns per-pixel class logits `(1, num_classes, H, W)`. The batch axis is deliberately static: a symbolic batch makes the token count a run-time scalar, which forces host/device copies inside the G2TM block.

As FastSV is iterative, it needs enough iterations to reach the same merges as the BFS version, otherwise it can leave connected components split. Leaving `--model.num_iters` unset makes it use the theoretical value, which is often too high.

### Utilities

To push an offline W&B run to your account:

```bash
export WANDB_API_KEY=<your_api_key>
python tools/sync_wandb.py <run_folder> --project <project> --entity <entity>
```

**All commands** in this section can be run with or without G2TM through
`--model.patch_type`, and with or without (Inverse) Proportional Attention through `--model.prop_attn` or `--model.iprop_attn`.

## Results and Models

See [RESULTS](./RESULTS.md) for some comparative results for EoMT + G2TM and the corresponding model checkpoints.

**NOTE:** We are still looking for a solution to host all model checkpoints, in the meantime do not hesitate to request the checkpoints by contacting one of the authors.

## Upcoming Features 

```
- [x] Training and Inference scripts
- [x] Flops and Speedtest scripts
- [x] Token and attention map visualization scripts
- [x] Results on ADE20K and Cityscapes datasets
- [x] ONNX export script
- [ ] Nvidia Jetson running scripts
```

## Acknowledgements

This code extends the official [EoMT](https://github.com/tue-mps/eomt) code (under [MIT Licence](https://github.com/tue-mps/eomt/blob/master/LICENSE)). It uses the repository structure and some utils functions from [ToMe](https://github.com/facebookresearch/ToMe) (under [CC-BY-NC licence](https://github.com/facebookresearch/ToMe/blob/main/LICENSE)).

All files covered by EoMT's or ToMe's licences include a header indicating the licence and whether the file has been modified. In the G2TM package, you can find such files from ToMe's repository in the [`patch`](./g2tm/g2tm/patch/) and [`vis`](./g2tm/g2tm/vis/) folders.

Below are other Python librairies, along with their corresponding licenses, used in this work:
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
