"""
G2TM - Graph-Guided Token Merging

This package provide utilities for inserting and running G2TM inside a ViT model.
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
from .patch import *
from .vis import *

__all__ = [
    "nx_merge",
    "rebuild_source",
    "unmerge_idx_needed",
    "fast_sv_merge",
    "bfs_merge",
    "connected_components_bfs",
    "connected_components_uf",
    "connected_components_labels",
    "__version__",
    "version_info",
]
