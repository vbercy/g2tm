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
"""Base dataset wrapper class."""

from torch.utils.data import Dataset
from torchvision import transforms

from vit.config import dataset_dir
from vit.data.utils import STATS, default_class_names


class BaseClassificationDataset(Dataset):
    """Common wrapper for torchvision classification datasets."""

    def __init__(
        self,
        image_size,
        crop_size,
        split,
        normalization="vit",
        data_root=None,
        num_classes=None,
        label_prefix="class",
        **kwargs,  # pylint: disable=W0613
    ):
        super().__init__()
        self.image_size = image_size
        self.crop_size = crop_size
        self.split = split
        self.normalization = STATS[normalization]
        self.label_prefix = label_prefix

        try:
            self.data_root = dataset_dir()
        except ValueError as e:
            print("Warning: ", str(e), " Fallback to dataset config file.")
            self.data_root = data_root

        self.dataset = None
        self.transform = self.build_transform()
        self.names = self._infer_class_names()
        self.n_cls = num_classes
        if len(self.names) != num_classes:
            self.names = default_class_names(self.n_cls, prefix=self.label_prefix)

    def build_transform(self):
        """Build standard train/eval transforms for classification."""
        if self.split in {"train", "trainval"}:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomCrop(self.crop_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(**self.normalization),
                ]
            )

        resize_size = max(self.image_size, self.crop_size)
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(**self.normalization),
            ]
        )

    def _build_dataset(self, **kwargs):
        raise NotImplementedError

    def _infer_class_names(self):
        classes = getattr(self.dataset, "classes", None)
        if classes:
            return list(classes)

        class_to_idx = getattr(self.dataset, "class_to_idx", None)
        if class_to_idx:
            return [
                class_name
                for class_name, _ in sorted(
                    class_to_idx.items(), key=lambda item: item[1]
                )
            ]

        targets = self._all_targets()
        if not targets:
            return []
        return default_class_names(len(set(targets)), prefix=self.label_prefix)

    def _all_targets(self):
        for attribute in ("targets", "_labels", "labels"):
            targets = getattr(self.dataset, attribute, None)
            if targets is not None:
                return [int(target) for target in targets]
        return None

    def __getitem__(self, idx):
        im, label = self.dataset[idx]

        if hasattr(label, "item"):
            label = int(label.item())

        return {"im": im, "label": label, "idx": idx}

    def get_gt_labels(self):
        """Get only groundtruth classification labels."""
        labels = {}

        for idx, data in enumerate(self.dataset):
            _, labels[idx] = data

        return labels

    def __len__(self):
        return len(self.dataset)

    @property
    def unwrapped(self):
        """Unwrap."""
        return self

    def set_epoch(self, epoch):
        """Set number of epochs."""

    def get_diagnostics(self, logger):
        """Get diagnostics from logger."""

    def get_snapshot(self):
        """Get snapshot."""
        return {}

    def end_epoch(self, epoch):
        """Hook run at the end of an epoch."""
