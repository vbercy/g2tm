"""Utility functions for the evaluation scripts."""

from typing import Iterable, Tuple, List
import sys

import torch
from torch import nn
from lightning.pytorch.callbacks import ModelSummary

from main import LightningCLI, LightningDataModule, LightningModule
from models.eomt import EoMT
from g2tm.patch import G2TMEoMT


def has_cli_override(cli_args: list[str], name: str) -> bool:
    """Check cli arguments."""
    return (
        any(arg == name or arg.startswith(f"{name}=") for arg in cli_args)
        and len(cli_args) > 0
    )


def build_from_yaml(
    cfg_path: str,
    cli_args: list[str],
) -> tuple[LightningModule, LightningDataModule]:
    """Instantiate the Lightning model and datamodule using LightningCLI without
    running fit/validate.
    """
    cli_kwargs = dict(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
        args=["--config", cfg_path, *cli_args],
        run=False,  # DO NOT launch fit/validate/test
    )

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]]
        cli = LightningCLI(**cli_kwargs)
    except SystemExit as e:
        raise RuntimeError(
            f"LightningCLI failed to parse/build from config '{cfg_path}'. "
            f"Please check the path and YAML contents."
        ) from e
    finally:
        sys.argv = old_argv

    if not hasattr(cli, "model") or cli.model is None:
        raise RuntimeError(
            "LightningCLI did not instantiate a model from the provided config."
        )
    if not hasattr(cli, "datamodule") or cli.datamodule is None:
        raise RuntimeError(
            "LightningCLI did not instantiate a datamodule from the provided config."
        )

    return cli.model, cli.datamodule


def unwrap_network(maybe_wrapped) -> torch.nn.Module:
    """If the LightningModule wraps the actual nn.Module in `.network`, return that.
    Otherwise, return the object itself.
    """
    net = getattr(maybe_wrapped, "network", None)
    return net if isinstance(net, torch.nn.Module) else maybe_wrapped


def iter_images(
    datamodule: LightningDataModule, stage: str = "validate"
) -> Iterable[torch.Tensor]:
    """Iterate over the a specific set of the LightningDataModule."""
    if getattr(datamodule, "path", None) is None:
        raise RuntimeError(
            "The config did not set data.path. Pass the dataset root with "
            "--data-path /path/to/dataset or --data.path /path/to/dataset. "
            "For ADE20K, that directory should contain ADEChallengeData2016.zip."
        )

    datamodule.setup(stage)
    if stage == "validate":
        loaders = datamodule.val_dataloader()
    elif stage == "fit":
        loaders = datamodule.train_dataloader()
    else:
        raise ValueError(
            "Unexpected stage '{stage}', choose among "
            "'validate' or 'fit' to iterate over the validation"
            "or the training split respectively."
        )
    if not isinstance(loaders, (list, tuple)):
        loaders = (loaders,)

    for loader in loaders:
        for batch in loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch

            if isinstance(imgs, torch.Tensor):
                if imgs.ndim == 3:
                    yield imgs
                elif imgs.ndim == 4:
                    yield from imgs
                else:
                    raise RuntimeError(
                        f"Expected image tensor with 3 or 4 dims, got {imgs.shape}."
                    )
            elif isinstance(imgs, (list, tuple)):
                yield from imgs
            else:
                raise RuntimeError(
                    f"Could not extract images from validation batch of type {type(batch)}."
                )


def make_input(
    model: LightningModule, img: torch.Tensor, device: torch.device
) -> List[torch.Tensor]:
    """Build the list of model inputs for a single image, one crop per entry.

    Each entry has shape (1, 3, H, W) so that every forward pass runs at batch
    size 1. This keeps G2TM on its sparse path, where merged tokens are
    physically removed from the sequence: token/FLOP/latency measurements then
    reflect the real reduction. Stacking the sliding-window crops into one
    batch would silently fall back to the padded path, which never shrinks the
    sequence.
    """
    img = img.to(device, non_blocking=device.type == "cuda")

    if hasattr(model, "ignore_idx"):
        crops, _ = model.window_imgs_semantic((img,))
        return list(crops.split(1))

    if hasattr(model, "resize_and_pad_imgs_instance_panoptic"):
        return [model.resize_and_pad_imgs_instance_panoptic((img,))]

    return [img[None, ...]]


class EoMTHead(G2TMEoMT):
    """EoMT's (head only) PyTorch class."""

    def __init__(
        self,
        blocks: nn.ModuleList,
        class_head: nn.Module,
        mask_head: nn.Module,
        upscale: nn.Module,
        norm: nn.Module,
        info: dict,
        num_q: int,
        num_prefix_tokens: int,
        grid_size: tuple,
    ):
        # Skip EoMT.__init__ (it requires encoder/num_classes/num_q and would
        # rebuild the whole network): only nn.Module initialisation is needed,
        # the submodules are injected below.
        super(EoMT, self).__init__()  # pylint: disable=E1003
        self.blocks = blocks
        self.class_head = class_head
        self.mask_head = mask_head
        self.upscale = upscale
        self.norm = norm
        self.info = info
        self.grid_size = grid_size
        self.num_prefix_tokens = num_prefix_tokens
        self.num_q = num_q

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """EoMT's Mask Module forward, predicts class logits and mask logits.

        Args:
            x (torch.Tensor): Token features (B, N, C).

        Returns:
            mask_logits (torch.Tensor): EoMT's mask logits for segmentation.
            class_logits (torch.Tensor): EoMT's class logits for segmentation.
        """
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        # Unmerge the token sequence if needed (i.e.:  if G2TM)
        # `info` is always set (a vanilla model gets a stub dict), so the merge is
        # detected by the presence of an unmerge index, not by the attribute.
        idxs = self.info.get("unmerge_idx")
        if idxs is not None:
            # We get rid of the source matrix and the argmax, the index is derived from
            # the component labels by the merge.
            x = x[:, self.num_q + self.num_prefix_tokens :, :].gather(
                1, idxs.unsqueeze(-1).expand(-1, -1, x.size(-1))
            )
        else:
            x = x[:, self.num_q + self.num_prefix_tokens :, :]

        # Reshape the token to the grid size
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *self.grid_size)

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None, rope: torch.Tensor | None
    ) -> Tuple[List]:
        """Forward method."""
        mask_logits_per_layer, class_logits_per_layer = [], []

        for block in self.blocks:
            x = self._attn_forward(block, x, attn_mask, rope)
            x = self._mlp_forward(block, x)

        mask_logits, class_logits = self._predict(self.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
