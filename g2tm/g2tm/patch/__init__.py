"""
Patching function for ViT models by inserting G2TM within one of its encoder's
Transformer blocks.

Example: Inserting G2TM@2[0.88] within a ViT model
    >>> from vit.model.factory import create_vit
    >>> from g2tm.graph import graph_vit_patch
    >>> model = create_vit({...})
    >>> selected_layer = 2
    >>> threshold = 0.88
    >>> graph_vit_patch(model, selected_layer, threshold)
"""

from .graph_vit_patch import apply_patch as graph_vit_patch
from .graph_vit_patch import unmerge_idx_needed

__all__ = ["graph_vit_patch", "unmerge_idx_needed"]
