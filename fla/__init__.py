"""Lightweight FLA namespace shim for local kernel development.

The installed ``fla`` package imports optional model/layer integrations at top
level. In this environment that path trips a Transformers/kernels version
incompatibility before ``fla.ops`` can be imported.  Keep the package namespace
open so imports such as ``fla.ops.utils.cumsum`` still resolve to the installed
Flash Linear Attention modules, without importing the optional public API.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
__version__ = "0.5.1"

__all__: list[str] = []
