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
"""Export model with or without G2TM into ONNX format and evaluate it using
ONNXRuntime on the same device.

Example: see README.md
"""

from pathlib import Path
import warnings

import click

import torch

from torch.utils.data import Dataset
import onnx
import onnxruntime as ort
import onnxscript.optimizer

from vit.utils.logger import MetricLogger
from vit.model.factory import load_model
from vit.data.factory import create_dataset
from vit.metrics import gather_data, compute_metrics
from vit.utils import distributed
import vit.utils.torch as ptu

import g2tm

warnings.filterwarnings("ignore")

model_prefixes = {
    "pure": "",
    "graph": "g2tm_",
}
size_letters = {
    "tiny": "T",
    "small": "S",
    "base": "B",
    "large": "L",
}


def eval_onnx_model(onnx_file: str, data_loader: Dataset) -> MetricLogger:
    """Evaluate an ONNX model by computing the Top-1 accuracy score.

    This function computes the same top-k metrics as the PyTorch evaluator.

    Args:
        onnx_file (Path): Path to the ONNX model file.
        data_loader (torch.utils.data.Dataset): PyTorch dataloader.

    Returns:
        logger (MetricLogger): Evaluation logger.
    """
    val_cls_pred = {}
    val_cls_gt = {}

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]
    ort_session = ort.InferenceSession(onnx_file, providers=providers)
    print("Available providers:", ort.get_available_providers())
    print("Active providers for the session:", ort_session.get_providers())

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 100

    for batch in logger.log_every(data_loader, print_freq, header):
        ort_inputs = {ort_session.get_inputs()[0].name: batch["im"].numpy()}
        cls_pred = ort_session.run(None, ort_inputs)[0]
        for idx, pred, label in zip(
            batch["idx"].tolist(), cls_pred.tolist(), batch["label"].tolist()
        ):
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


@torch.no_grad()
@click.command()
@click.argument("model_path", type=str)
@click.option("--onnx-name", type=str, default=None)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=1, type=int)
@click.option("--threshold", default=0.88, type=float)
@click.option("--prop-attn/--no-prop-attn", default=False, is_flag=True)
@click.option("--iprop-attn/--no-iprop-attn", default=False, is_flag=True)
@click.option("--num-iters", default=None, type=int)
@click.option("--eval-onnx/--no-eval-onnx", default=True, is_flag=True)
def main(
    model_path: str,
    onnx_name: str,
    patch_type: str,
    selected_layer: int,
    threshold: float,
    prop_attn: bool,
    iprop_attn: bool,
    num_iters: int,
    eval_onnx: bool,
):
    """Export PyTorch model into ONNX format.

    Args:
        model_path (str): Path to PyTorch model.
        onnx_name (str): Name to give to the ONNX model.
        patch_type (str): Token reduction method (pure => no reduction).
        selected_layer (int): Layer to apply token reduction (1-based).
        threshold (float): Threshold parameter for G2TM.
        prop_attn (bool): Whether to apply Proportional Attention.
        iprop_attn (bool): Whether to apply Inverse Proportional Attention.
        num_iters (int): Number of FastSV iterations.
        eval_onnx (bool): Whether to evaluate the exported ONNX model or not.
    """

    ptu.set_gpu_mode(True)
    distributed.init_process()
    verbose = False

    model, variant = load_model(model_path)
    input_size = variant["dataset_kwargs"]["crop_size"]

    if patch_type == "graph":
        g2tm.graph_vit_patch(
            model, selected_layer, threshold, prop_attn, iprop_attn, True, num_iters
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    with torch.no_grad():
        if onnx_name is None:
            size_letter = size_letters[variant["net_kwargs"]["backbone"].split("_")[1]]
            onnx_name = (
                model_prefixes[patch_type]
                + f"vit_{size_letter}"
                + (
                    f"_L{selected_layer}_T{threshold:.2f}_{num_iters}it"
                    if patch_type == "graph"
                    else ""
                )
            )
        onnx_file = Path(model_path).parent / (onnx_name + ".onnx")
        dummy_input = torch.randn(  # pylint: disable=E1101
            1, 3, input_size, input_size, device=ptu.device
        )
        # Opset >= 18 needed for the min-reduction ScatterElements used by the G2TM
        # connected components (FastSV hooking step).
        # Batch axis deliberately left static (batch size 1, inference anyway), as a
        # symbolic batch forces the token count to be read from the tensor shapes at
        # run time, and that CPU-side scalar feeds GPU ops in the G2TM block, which
        # insert a MemcpyFromHost. Pinning it folds those away entirely.
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
            opset_version=18,
            verbose=verbose,
            input_names=["input"],
            output_names=["output"],
        )

    # Constant folding: removes the dead `If` wrappers PyTorch emits around
    # scatter ops (their "empty src" branch is statically false here).
    onnx_model = onnxscript.optimizer.optimize(onnx.load(onnx_file))
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_file)
    if verbose:
        print(onnx.printer.to_text(onnx_model.graph))

    print("Model exported to ONNX !")

    if eval_onnx:
        print("Testing the model using ONNXRuntime...")

        dataset_kwargs = variant["dataset_kwargs"]
        dataset_kwargs["batch_size"] = 1
        dataset_kwargs["split"] = "val"
        dataset_kwargs["crop"] = False
        validation_loader = create_dataset(dataset_kwargs)

        logger = eval_onnx_model(onnx_file, validation_loader)

        print("Metrics:", logger, flush=True)
        print(str(logger.top1_accuracy).split(" ", maxsplit=1)[0])
        print("")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
