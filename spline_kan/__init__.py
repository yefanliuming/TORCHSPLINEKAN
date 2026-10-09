"""
spline_kan - B-spline KAN (Kolmogorov-Arnold Network) layers for PyTorch.
"""

from .kan_linear import KANLinear
from .kan import KAN
from .pruning import PruneResult, prune_nodes

__version__ = "0.1.0"

__all__ = ["KANLinear", "KAN", "PruneResult", "prune_nodes", "__version__"]
