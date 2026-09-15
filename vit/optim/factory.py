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
"""Optimizer and scheduler instantiation functions."""

from timm import scheduler
from timm import optim
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def create_scheduler(opt_args, optimizer):
    """Instanciate scheduler."""
    if opt_args.sched == "cosine":
        warmup = LinearLR(
            optimizer=optimizer,
            start_factor=opt_args.start_factor,
            end_factor=1.0,
            total_iters=opt_args.warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=opt_args.steps - opt_args.warmup_steps,
            eta_min=0.0,
        )
        lr_scheduler = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, cosine],
            milestones=[opt_args.warmup_steps],
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler


def create_optimizer(opt_args, model):
    """Instanciate optimizer."""
    if opt_args.opt == "sgd" and opt_args.sched == "cosine":
        return SGD(
            model.parameters(),
            lr=opt_args.lr,
            weight_decay=opt_args.weight_decay,
            momentum=opt_args.momentum,
        )
    else:
        return optim.create_optimizer(opt_args, model)
