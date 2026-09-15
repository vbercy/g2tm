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
"""Oxford Flowers dataset configuration."""

DATASET_CONFIG = {
    "name": "oxford_flowers",
    "data_root": "/data1/is156025/vb282713/Oxford_Flowers-102",
    "num_classes": 102,
    "label_prefix": "oxford_flower",
    "download": False,
    "split_map": {
        "train": "train",
        "trainval": "train",
        "val": "val",
        "test": "test",
    },
}
