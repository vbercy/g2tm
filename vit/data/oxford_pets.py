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
"""Oxford-IIIT Pet dataset wrapper class."""

from torchvision import datasets

from vit.data.base import BaseClassificationDataset
from vit.data.config.oxford_pets import DATASET_CONFIG


class OxfordPetsDataset(BaseClassificationDataset):
    """Oxford-IIIT Pet dataset wrapper."""

    def __init__(
        self,
        image_size,
        crop_size,
        split,
        normalization="vit",
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            crop_size=crop_size,
            split=split,
            normalization=normalization,
            data_root=DATASET_CONFIG["data_root"],
            num_classes=DATASET_CONFIG["num_classes"],
            label_prefix=DATASET_CONFIG["label_prefix"],
            **kwargs,
        )

        self.dataset = self._build_dataset()

    def _build_dataset(self, **kwargs):
        return datasets.OxfordIIITPet(
            root=self.data_root,
            split=DATASET_CONFIG["split_map"][self.split],
            target_types="category",
            transform=self.transform,
            download=DATASET_CONFIG["download"],
        )
