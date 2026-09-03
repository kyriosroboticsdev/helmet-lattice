"""Per-machine configuration for the Helmet Lattice Fusion add-in.

The add-in has to shell out to a system Python that has numpy/skimage/trimesh
and the ML stack installed, because Fusion's bundled interpreter has none of
them (see fusion_addin/README.md). Those paths differ per researcher, so they
must not be hardcoded in version-controlled source.

Resolution order, first hit wins:

  1. local_config.py  next to this file (gitignored; copy local_config.example.py)
  2. environment variable  HELMET_LATTICE_<NAME>
  3. the default passed by the caller

Nothing here raises: a missing or broken local_config.py falls through to the
env var and then the default, so the add-in still loads.
"""

import os

_HERE = os.path.dirname(__file__)
_LOCAL = None
_LOCAL_ERR = None


def _local_module():
    """Load local_config.py by path. Cached; never raises."""
    global _LOCAL, _LOCAL_ERR
    if _LOCAL is not None or _LOCAL_ERR is not None:
        return _LOCAL
    path = os.path.join(_HERE, "local_config.py")
    if not os.path.exists(path):
        _LOCAL_ERR = "not present"
        return None
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hl_local_config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LOCAL = mod
    except Exception as exc:          # a typo in local_config must not kill the add-in
        _LOCAL_ERR = str(exc)
    return _LOCAL


def get(name, default=""):
    """Return setting `name`, or `default` if it is set nowhere."""
    mod = _local_module()
    if mod is not None:
        value = getattr(mod, name, None)
        if value:
            return value
    return os.environ.get("HELMET_LATTICE_" + name, default)


def describe():
    """Human-readable source of the current config, for error dialogs."""
    if _local_module() is not None:
        return "local_config.py"
    return "environment / defaults (local_config.py: %s)" % (_LOCAL_ERR or "not loaded")
