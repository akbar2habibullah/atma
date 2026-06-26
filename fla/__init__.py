"""Lightweight FLA namespace shim for local kernel development.

The installed ``fla`` package imports optional model/layer integrations at top
level. In this environment that path trips a Transformers/kernels version
incompatibility before ``fla.ops`` can be imported. Keep the package namespace
open so imports such as ``fla.ops.utils.cumsum`` still resolve to the installed
Flash Linear Attention modules, without importing the optional public API.
"""

from pathlib import Path
from pkgutil import extend_path
import site

__path__ = extend_path(__path__, __name__)

for site_dir in [*site.getsitepackages(), site.getusersitepackages()]:
    fla_dir = Path(site_dir) / __name__
    if fla_dir.is_dir() and str(fla_dir) not in __path__:
        __path__.append(str(fla_dir))

__version__ = "0.5.1"

__all__: list[str] = []
