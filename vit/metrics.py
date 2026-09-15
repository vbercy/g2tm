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
"""Evaluation and metrics utility functions for image classification task."""

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil

import torch
import torch.distributed as dist

import vit.utils.torch as ptu


def accuracy(output, target, topk=(1,)):
    """Computes the ImageNet classifcation accuracy over the k top predictions
    for the specified values of k borrowed from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            correct_k /= batch_size
            res[f"top{k}"] = correct_k
        return res


def gather_data(cls_pred, tmp_dir=None):
    """Distributed data gathering prediction and ground truth are stored in a
    common tmp directory and loaded on the master node to compute metrics.
    """
    if tmp_dir is None:
        tmpprefix = os.path.expandvars("./temp")
    else:
        tmpprefix = os.path.expandvars(tmp_dir)
    max_len = 512
    # 32 is whitespace
    dir_tensor = torch.full(
        (max_len,), 32, dtype=torch.uint8, device=ptu.device  # pylint: disable=E1101
    )
    if ptu.dist_rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=tmpprefix)
        tmpdir = torch.tensor(  # pylint: disable=E1101
            bytearray(tmpdir.encode()), dtype=torch.uint8, device=ptu.device
        )
        dir_tensor[: len(tmpdir)] = tmpdir
    # broadcast tmpdir from 0 to to the other nodes
    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    tmpdir = Path(tmpdir)
    dist.barrier()

    # Save results in temp file and load them on main process
    tmp_file = tmpdir / f"part_{ptu.dist_rank}.pkl"
    with open(tmp_file, "wb") as f:
        pkl.dump(cls_pred, f)
    dist.barrier()
    cls_pred = {}
    if ptu.dist_rank == 0:
        for i in range(ptu.world_size):
            with open(tmpdir / f"part_{i}.pkl", "rb") as f:
                part_cls_pred = pkl.load(f)
            cls_pred.update(part_cls_pred)
        shutil.rmtree(tmpdir)
    return cls_pred


def compute_metrics(
    cls_pred,
    cls_gt,
    distributed=False,
):
    """Compute top-k accuracy metrics for classification."""
    ret_metrics = torch.zeros(2, dtype=float, device=ptu.device)
    if ptu.dist_rank == 0:
        list_cls_pred = []
        list_cls_gt = []
        keys = sorted(cls_pred.keys())
        for k in keys:
            list_cls_pred.append(cls_pred[k])
            list_cls_gt.append(cls_gt[k])
        output = torch.tensor(list_cls_pred, dtype=torch.float32)
        target = torch.tensor(list_cls_gt, dtype=torch.long)
        if output.ndim != 2:
            raise ValueError(
                "Expected classification logits with shape [N, C], "
                f"got shape {tuple(output.shape)}."
            )
        if output.shape[1] < 5:
            print(
                "Warning: Number of classes is less than 5,"
                "Top-5 accuracy score cannot be computed."
            )
            metrics = accuracy(
                output=output,
                target=target,
                topk=(1,),
            )
            metrics["top5"] = 0
        else:
            metrics = accuracy(
                output=output,
                target=target,
                topk=(1, 5),
            )
        ret_metrics[0] = metrics["top1"] * 100
        ret_metrics[1] = metrics["top5"] * 100
    # broadcast metrics from 0 to all nodes
    if distributed:
        dist.broadcast(ret_metrics, 0)
    top1_acc, top5_acc = ret_metrics
    ret = {"top1_accuracy": top1_acc, "top5_accuracy": top5_acc}
    return ret
