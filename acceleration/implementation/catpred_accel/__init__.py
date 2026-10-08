"""Explicit CatPred runtime integration; no import-time model or device work."""
from .config import RuntimeConfig
__version__ = "0.1.0"
__all__ = ["RuntimeConfig", "Runtime"]


def __getattr__(name):
    if name == "Runtime":
        from .api import Runtime
        return Runtime
    raise AttributeError(name)
