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
import numpy as np

import torch

# def pin_torch_cudnn():
#     """Pin cuDNN to the copy bundled with PyTorch, before ORT loads its own.

#     Two complete cuDNN 9 installations are visible here: the one shipped
#     inside ``torch/lib`` and the system one in ``/lib/x86_64-linux-gnu``.
#     cuDNN 9 is split across several shared objects (``libcudnn_ops``,
#     ``libcudnn_graph``, ``libcudnn_cnn``, ...) that are only compatible
#     within a single release, and they are loaded lazily by soname. So the
#     process can end up mixing them: importing torch loads its
#     ``libcudnn_graph.so.9``, then ONNX Runtime's CUDA provider pulls the
#     *system* ``libcudnn_ops.so.9``, which then fails to resolve a symbol
#     against the older graph library:

#         libcudnn_ops.so.9: undefined symbol:
#         _ZN5cudnn5graph13LibraryLoader11getInstanceEv,
#         version libcudnn_graph.so.9

#     ONNX Runtime reports this as "Failed to create CUDAExecutionProvider"
#     and silently falls back to CPU. Loading torch's complete set with
#     ``RTLD_GLOBAL`` first claims every cuDNN soname for one consistent
#     release, so the system copies are never pulled in and the mix cannot
#     happen. Whether it triggers otherwise depends on load order, which is
#     why it appears intermittently.

#     Best effort: failures here are non-fatal (e.g. CPU-only installs).
#     """
#     torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
#     # Sub-libraries first, then the dispatcher, so that every soname it may
#     # dlopen later is already resolved to torch's copy.
#     cudnn_libs = sorted(
#         glob.glob(os.path.join(torch_lib, "libcudnn_*.so.9"))
#     ) + glob.glob(os.path.join(torch_lib, "libcudnn.so.9"))
#     for lib in cudnn_libs:
#         try:
#             ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
#         except OSError:
#             pass


# pin_torch_cudnn()

from torch.utils.data import Dataset
import onnx
import onnxruntime as ort
import onnxscript.optimizer

from segm.utils.logger import MetricLogger
from segm.model.factory import load_model
from segm.model.utils import resize, sliding_window, merge_windows
from segm.data.utils import IGNORE_LABEL
from segm.data.factory import create_dataset
from segm.metrics import gather_data, compute_metrics
from segm.utils import distributed
import segm.utils.torch as ptu

import g2tm

warnings.filterwarnings("ignore")

model_names = {
    "pure": "segmenter",
    "graph": "g2tm",
}
size_letters = {
    "tiny": "T",
    "small": "S",
    "base": "B",
    "large": "L",
}


def onnx_inference(
    ort_session: ort.InferenceSession,
    batch: dict,
    window_size: int,
    window_stride: int,
    n_cls: int,
) -> np.array:
    """Sliding window inference from an ONNX model

    This function runs an inference for the first image of the batch. The
    inference is made using an ONNX model and uses a sliding fixed-size window
    to process image whose size is greater than the model input.

    Args:
        ort_session (ort.InferenceSession): ONNX model.
        batch (dict): Dictionnary containing the image, the groundtruth and all
        the information of the input batch.
        window_size (int): Size (height and width) of a window in pixels.
        window_stride (int): Step for each slidein pixels.
        n_cls (int): Number of classes in the dataset.

    Returns:
        im_seg_map (np.array): Predicted segmentation map.
    """
    img = torch.tensor(batch["im"][0])  # pylint: disable=E1101
    img_metas = batch["im_metas"][0]
    ori_shape = img_metas["ori_shape"]
    ori_shape = (ori_shape[0].item(), ori_shape[1].item())
    flip = False

    img = resize(img, window_size)
    windows = sliding_window(img, flip, window_size, window_stride)
    crops = torch.stack(windows.pop("crop"))[:, 0]  # pylint: disable=E1101
    n_crops = len(crops)
    seg_maps = torch.zeros(  # pylint: disable=E1101
        (n_crops, n_cls, window_size, window_size)
    )

    for i in range(0, n_crops):
        ort_inputs = {ort_session.get_inputs()[0].name: crops[i : i + 1].numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        seg_maps[i] = torch.tensor(ort_outputs[0])  # pylint: disable=E1101

    windows["seg_maps"] = seg_maps
    im_seg_map = merge_windows(windows, window_size, ori_shape)

    return im_seg_map.numpy()


def eval_onnx_model(
    onnx_file: str, data_loader: Dataset, window_size: int, window_stride: int
) -> MetricLogger:
    """Evaluate an ONNX model by computing the mIoU score.

    This functions computes the mIoU score of an ONNX model, the same way
    it is done for PyTorch models. In this way, we can validate the ONNX
    model export made.

    Args:
        onnx_file (Path): Path to the ONNX model file.
        data_loader (torch.utils.data.Dataset): PyTorch dataloader.
        window_size (int): Size (height and width) of a window in pixels.
        window_stride (int): Step for each slidein pixels.

    Returns:
        logger (MetricLogger): Evaluation logger.
    """
    val_seg_gt = data_loader.dataset.get_gt_seg_maps()
    val_seg_pred = {}

    dataset_meta = {
        "classes": data_loader.unwrapped.names,
        "label_map": {},
        "reduce_zero_label": data_loader.unwrapped.reduce_zero_label,
    }

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
        filename = batch["im_metas"][0]["img_path"][0]
        seg_pred = onnx_inference(
            ort_session, batch, window_size, window_stride, data_loader.unwrapped.n_cls
        )
        seg_pred = np.argmax(seg_pred, axis=0)
        val_seg_pred[filename] = seg_pred

    val_seg_pred = gather_data(val_seg_pred)
    scores = compute_metrics(
        val_seg_pred,
        val_seg_gt,
        dataset_meta,
        ignore_index=IGNORE_LABEL,
    )

    for k, v in scores.items():
        logger.update(**{f"{k}": v, "n": 1})

    return logger


@torch.no_grad()
@click.command()
@click.argument("model_path", type=str)
@click.option("--onnx-name", type=str, default=None)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layer", default=2, type=int)
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
        g2tm.graph_segmenter_patch(
            model, selected_layer, threshold, prop_attn, iprop_attn, True, num_iters
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(ptu.device)

    with torch.no_grad():
        if onnx_name is None:
            onnx_name = (
                model_names[patch_type]
                + "_"
                + f"seg{size_letters[variant['net_kwargs']['backbone'].split('_')[1]]}"
                + f"_L{selected_layer}_T{threshold:.2f}"
                + (f"_{num_iters}it" if patch_type == "graph" else "")
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

        window_size = variant["inference_kwargs"]["window_size"]
        window_stride = variant["inference_kwargs"]["window_stride"]
        logger = eval_onnx_model(
            onnx_file, validation_loader, window_size, window_stride
        )

        print("Metrics:", logger, flush=True)
        print(str(logger.mean_iou).split(" ", maxsplit=1)[0])
        print("")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
