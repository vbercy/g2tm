# Training commands

Below, we provide the training commands for training EoMT + G2TM.

Training goes through EoMT's entry point, `main.py fit`, with a configuration file from
[`configs/`](./configs/). Every option of the configuration can be overridden on the
command line through the
[LightningCLI](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) syntax,
so G2TM is enabled with plain `--model.*` arguments.

Configuration files follow the path convention below:

```
configs/<pre-training>/<dataset>/<task>/eomt_<size>_<resolution>.yaml
```

The following pre-trainings are available:
* **AugReg** (`augreg`): supervised ImageNet-21k ViTs, the pre-training used in the G2TM
  paper for Segmenter.
* **DINOv2** (`dinov2`): the pre-training of the original EoMT paper.
* **DINOv3** (`dinov3`): the most recent EoMT models.

The following datasets and tasks can be used, depending on the configuration:
* [ADE20K](https://ade20k.csail.mit.edu/): `semantic`, `panoptic`
* [COCO](https://cocodataset.org/): `panoptic`, `instance`
* [Cityscapes](https://www.cityscapes-dataset.com/): `semantic`

Different pre-training recipes and backbone sizes are available for EoMT, selected by the configuration file: `configs/<pretraining>/<dataset>/<task>/eomt_<backbone_size>_<resolution>.yaml`.
Run `ls configs/**/*.yaml` for the exact list, and see
[main.py](main.py) and [training/lightning_module.py](training/lightning_module.py) for
all the available options.

Do not forget to define the `DATASET` environment variable, pointing at the directory
that holds the dataset **zip** files (they are read directly, no need to unzip):

```bash
export DATASET=/path/to/dataset
```

## Single GPU on ADE20K

The following commands enable single-GPU training with different ViT backbone sizes.
G2TM is applied at layer 2 with a threshold of 0.88, using the custom BFS implementation
(the fastest one for the batched training path).

EoMT + G2TM with a ViT-B backbone (AugReg pre-training):

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_base_512.yaml \
	--data.path $DATASET \
	--trainer.devices 1 \
	--data.batch_size 16 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.method bfs
```

EoMT + G2TM with a ViT-L backbone (AugReg pre-training):

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 1 \
	--data.batch_size 16 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.method bfs
```

EoMT + G2TM with a ViT-L backbone (DINOv2 pre-training):

```bash
python main.py fit \
	-c configs/dinov2/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 1 \
	--data.batch_size 16 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.method bfs
```

**NOTE:** DINOv2 and DINOv3 backbones carry register tokens (5 prefix tokens instead of
1). G2TM protects them automatically, and the merge only ever applies to patch tokens.

## Multi GPU

Multi-GPU training is handled by Lightning: simply raise `--trainer.devices`. There is no
`torchrun` wrapper to add.

EoMT + G2TM at layer 2 with threshold 0.88, ViT-L backbone, with 2 GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88
```

**NOTE:** Keep the total batch size at `devices × batch_size = 16`: the learning-rate schedule and
the attention-mask annealing steps of the configurations assume it.

## Other datasets and resolutions

The input resolution and the task are set by the configuration file, which directly
impacts the number of tokens in the sequence, and therefore how much G2TM can merge.

EoMT + G2TM at layer 2 with threshold 0.94, ViT-L backbone, on Cityscapes at 1024×1024:

```bash
python main.py fit \
	-c configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.94
```

EoMT + G2TM at layer 2 with threshold 0.88, ViT-L backbone, on COCO panoptic at 640×640:

```bash
python main.py fit \
	-c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
	--data.path $DATASET \
	--trainer.devices 4 \
	--data.batch_size 4 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88
```

**NOTE:** the threshold is not transferable between datasets or pre-trainings. The
distribution of the cosine similarity between neighbouring tokens shifts with both, so a
threshold tuned on ADE20K will not merge the same proportion of tokens on Cityscapes. Use
`tools/cosine_sim_dist.py` to inspect that distribution before choosing a value.

## G2TM implementations

Three implementations of the connected-components search are available through
`--model.method`, and they all produce the same merges:

| Method | Description |
| --- | --- |
| `bfs` (default) | Custom Breadth-First Search. The fastest one, recommended for training. |
| `fastsv` | Tensorized FastSV, the only implementation that can be exported to ONNX. |
| `nx` | The original NetworkX implementation, kept for reference. Deprecated: `apply_patch` rejects it, use `bfs` instead. |

EoMT + G2TM at layer 2 with threshold 0.88, using FastSV with 8 iterations:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.method fastsv \
	--model.num_iters 8
```

**NOTE:** FastSV is iterative, so it needs enough iterations to reach the same merges as
BFS, otherwise it leaves connected components split. Leaving `--model.num_iters` unset
runs it to convergence.

## (Inverse) Proportional Attention

The following commands enable Proportional Attention and Inverse Proportional Attention
respectively, for all the attention layers that follow the merging module. They weight
the attention logits by the (inverse) log-size of each merged token, so that a token
standing for many patches is not under- (or over-) represented.

EoMT + G2TM at layer 2 with threshold 0.88, with Proportional Attention:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.prop_attn true
```

EoMT + G2TM at layer 2 with threshold 0.88, with Inverse Proportional Attention:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.iprop_attn true
```

**NOTE:** the two options are mutually exclusive, enabling both raises an error. Both
force the un-fused attention path in the blocks that follow the merge, which is slower
than the fused kernel used otherwise.

## Curriculum

The following command lets the threshold vary during training, by activating the
`curric_thresh` option. You can tweak the curriculum with the options below:
* `--model.threshold`: the final threshold value, at the end of training
* `--model.start_thresh`: the starting threshold value
* `--model.curric_warmup`: the epoch at which the curriculum starts (0-based)
* `--model.curric_period`: the number of epochs between two curriculum steps

**NOTE:** between two curriculum steps, the threshold is decreased by 0.01. You therefore
have to tweak the options above carefully so that the final value is actually reached
before the end of training: the curriculum needs
`curric_warmup + curric_period × (start_thresh − threshold) / 0.01` epochs.

EoMT + G2TM at layer 2 with a threshold decreasing from 0.95 to 0.88, ViT-L backbone:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88 \
	--model.curric_thresh true \
	--model.start_thresh 0.95 \
	--model.curric_warmup 3 \
	--model.curric_period 4
```

Here, the model is trained with a starting threshold of 0.95 during 3 epochs. From the
4th epoch, the threshold is decreased by 0.01 every 4 epochs, reaching 0.88 at epoch 31,
which is the number of epochs of that configuration.

The curriculum state is stored in the checkpoint under the `merging` key, so a resumed
run picks the curriculum back up where it stopped.

## Fine-tuning and resuming

To fine-tune from an existing EoMT checkpoint:

```bash
python main.py fit \
	-c configs/augreg/ade20k/semantic/eomt_large_512.yaml \
	--data.path $DATASET \
	--trainer.devices 2 \
	--data.batch_size 8 \
	--model.ckpt_path <ckpt_file> \
	--model.load_ckpt_class_head False \
	--model.patch_type graph \
	--model.selected_layer 2 \
	--model.threshold 0.88
```

🔧 Replace `<ckpt_file>` with the checkpoint to fine-tune.

`--model.load_ckpt_class_head False` skips the classification head when fine-tuning on a dataset with a different number of classes.
With **DINOv3** configurations, the code expects delta weights relative to the DINOv3 weights by default; add `--model.delta_weights False` to use absolute weights instead.

A checkpoint trained **without** G2TM can be fine-tuned **with** it, and vice versa: the
merging module holds no parameter. A warning is printed when the G2TM state of the
checkpoint does not match the one requested on the command line.

## Miscellaneous

**Compilation.** `torch.compile` is enabled by default. G2TM is compatible with it, but
the connected-components search always falls back to eager (it is deliberately hidden
from the compiler). Pass `--compile_disabled` to turn compilation off entirely.

**Logging.** Runs are logged to [Weights & Biases](https://wandb.ai/) with the project and
run name set by the configuration. Offline runs can be pushed afterwards with
`tools/sync_wandb.py`.

**Checkpoints.** Checkpoints are written under `--trainer.default_root_dir`, keeping the
three best by validation mIoU plus the last one.
