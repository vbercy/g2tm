"""
Patching function for SETR models by inserting G2TM within one of its encoder's
Transformer blocks.

Example: Inserting G2TM@2[0.88] within a SETR model
    >>> from setr.model.factory import create_setr
    >>> from g2tm.graph import graph_setr_patch
    >>> model = create_setr({...})
    >>> selected_layer = 2
    >>> threshold = 0.88
    >>> graph_setr_patch(model, selected_layer, threshold)
"""

from .graph_setr_patch import apply_patch as graph_setr_patch

__all__ = ["graph_setr_patch"]
