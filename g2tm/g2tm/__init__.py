"""
G2TM - Graph-Guided Token Merging

This package provide utilities for inserting and running G2TM inside a Segmenter model.
It includes PyTorch modules, utilities and visualization functions.
"""

from . import nx_merge
from . import fast_sv_merge
from . import bfs_merge
from .utils import rebuild_source
from .connected_components import (
    connected_components_bfs,
    connected_components_uf,
    connected_components_labels,
)
from .version import __version__, version_info
from .vis import *

# NOTE: `.patch` is deliberately NOT imported here. The model patches depend on
# the host framework (e.g. `models.eomt` for EoMT), which is not a dependency of
# this package. Importing it eagerly would make `import g2tm` fail outside that
# repository. Import the patch explicitly instead:
#     from g2tm.patch import G2TMEoMT, graph_eomt_patch

__all__ = [
    "nx_merge",
    "rebuild_source",
    "fast_sv_merge",
    "bfs_merge",
    "connected_components_bfs",
    "connected_components_uf",
    "connected_components_labels",
    "__version__",
    "version_info",
]
