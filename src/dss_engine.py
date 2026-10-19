"""DSS engine import shim.

The task prompt names ``OpenDSSDirect.py``. That distribution name is not
resolvable on this machine's package index for Python 3.13 (no wheel and no
sdist; see docs/environment.md). ``dss-python`` is the maintained successor of
the same project by the same authors (dss-extensions) and exposes the same
engine through ``dss.IDSS``.

This module provides a single import point so that the rest of the code does
not depend on which of the two distributions is installed, and records which
one was actually loaded so provenance is unambiguous.
"""

from __future__ import annotations

import importlib

ENGINE_PACKAGE = None
dss = None
_import_error = None

try:  # preferred: the maintained successor
    from dss import dss as _dss_mod  # type: ignore

    dss = _dss_mod
    ENGINE_PACKAGE = "dss-python"
except Exception as exc:  # pragma: no cover
    _import_error = exc
    try:
        import opendssdirect as _od  # type: ignore

        dss = _od
        ENGINE_PACKAGE = "OpenDSSDirect.py"
    except Exception as exc2:
        _import_error = exc2

if dss is None:  # pragma: no cover
    raise ImportError(
        "No DSS engine available. Install `dss-python` (maintained successor of "
        "OpenDSSDirect.py). Original error: %r" % (_import_error,)
    )


def engine_version() -> str:
    try:
        v = getattr(dss, "Version", None)
        return str(v) if v is not None else "unknown"
    except Exception:
        return "unknown"


class _Engine:
    """Normalise the two public APIs onto one attribute surface.

    ``dss-python`` exposes a flat engine object (``dss.Text``, ``dss.Loads``,
    ``dss.Version``) whereas ``OpenDSSDirect.py`` exposes an ``opendssdirect``
    module whose top level is already flat in the same way. Both also provide a
    ``to_opendssdirect``-compatible surface, so we simply forward attribute
    access to whichever engine was loaded.
    """

    def __init__(self, mod):
        object.__setattr__(self, "_mod", mod)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, "_mod"), item)


engine = _Engine(dss)


def engine_info() -> dict:
    info = {"engine_package": ENGINE_PACKAGE, "engine_version": engine_version()}
    for pkg in ("dss-python", "dss_python", "OpenDSSDirect", "opendssdirect"):
        mod = importlib.import_module(pkg) if _has(pkg) else None
        if mod is not None:
            info.setdefault("packages", {})[pkg] = getattr(mod, "__version__", "n/a")
    return info


def _has(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


__all__ = ["dss", "ENGINE_PACKAGE", "engine_version", "engine_info"]
