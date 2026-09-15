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
"""Engine utility functions for training and evaluation of the model."""

import math

import torch

from vit.utils.logger import MetricLogger
from vit.metrics import gather_data, compute_metrics
import vit.utils.torch as ptu


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    best_checkpoint,
    amp_autocast,
    loss_scaler,
    writer,
    clip_grad=None,
):
    """Train the model for a single epoch."""
    criterion = torch.nn.CrossEntropyLoss()
    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    mb = 1024.0 * 1024.0
    torch.cuda.reset_peak_memory_stats()

    model.train()
    data_loader.set_epoch(epoch)

    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        cls_gt = batch["label"].long().to(ptu.device)

        optimizer.zero_grad()

        with amp_autocast():
            cls_pred = model.forward(im)
            loss = criterion(cls_pred, cls_gt)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training", force=True)

        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
            )
        else:
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        lr_scheduler.step()

        torch.cuda.synchronize()
        logger.update(
            loss=loss_value,
            learning_rate=optimizer.param_groups[0]["lr"],
        )

        num_updates = lr_scheduler.last_epoch
        if num_updates == 1 or num_updates % print_freq == 0:
            writer.add_scalar(
                "train_loss", logger.loss.value, num_updates, new_style=True
            )
            writer.add_scalar(
                "learning_rate", logger.learning_rate.value, num_updates, new_style=True
            )
            if torch.cuda.is_available():
                writer.add_scalar(
                    "train_cuda_mem",
                    torch.cuda.max_memory_allocated() / mb,
                    num_updates,
                    new_style=True,
                )
                torch.cuda.reset_peak_memory_stats()

    # Log the last batch of the epoch if not already logged
    if num_updates % print_freq > 0:
        writer.add_scalar("train_loss", logger.loss.value, num_updates, new_style=True)
        writer.add_scalar(
            "learning_rate", logger.learning_rate.value, num_updates, new_style=True
        )
        if torch.cuda.is_available():
            writer.add_scalar(
                "train_cuda_mem",
                torch.cuda.max_memory_allocated() / mb,
                num_updates,
                new_style=True,
            )
            torch.cuda.reset_peak_memory_stats()

    return logger, best_checkpoint


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    amp_autocast,
):
    """Evaluate the model using accuracy metrics of classification."""
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module
    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    freq = 1000

    val_cls_pred = {}
    val_cls_gt = {}
    model.eval()
    for batch in logger.log_every(data_loader, freq, header):
        im = batch["im"].to(ptu.device)
        labels = batch["label"].long().tolist()
        idxs = batch["idx"].long().tolist()

        with amp_autocast():
            cls_pred = model_without_ddp(im)
            cls_pred = cls_pred.detach().cpu().tolist()

        for idx, pred, label in zip(idxs, cls_pred, labels):
            val_cls_pred[int(idx)] = pred
            val_cls_gt[int(idx)] = int(label)

    val_cls_pred = gather_data(val_cls_pred)
    val_cls_gt = gather_data(val_cls_gt)
    scores = compute_metrics(
        val_cls_pred,
        val_cls_gt,
        distributed=ptu.distributed,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger
