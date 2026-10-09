"""
spline_kan - B-spline KAN (Kolmogorov-Arnold Network) layers for PyTorch.
"""

from .kan_linear import KANLinear
from .kan import KAN

__version__ = "0.1.0"

__all__ = ["KANLinear", "KAN", "__version__"]
