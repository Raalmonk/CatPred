"""Private loading for byte-preserved modules with legacy absolute imports."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
import types
import uuid
from .source import verify_files
import json


@contextmanager
def _aliases(values):
    absent = object()
    old = {name: sys.modules.get(name, absent) for name in values}
    try:
        sys.modules.update(values)
        yield
    finally:
        for name, value in old.items():
            if value is absent:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def load_private():
    root = Path(__file__).resolve().parent
    expected = json.loads((root / "accepted_sources.json").read_text())
    verify_files(root / "_accepted", expected)
    prefix = "catpred_accel._session_" + uuid.uuid4().hex
    package = types.ModuleType(prefix)
    package.__path__ = [str(root / "_accepted")]
    sys.modules[prefix] = package
    names, modules = [prefix], {}
    try:
        for name in ("packing", "reuse", "probes", "streaming"):
            qualified = prefix + "." + name
            spec = importlib.util.spec_from_file_location(qualified, root / "_accepted" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified] = module
            names.append(qualified)
            with _aliases(modules):
                spec.loader.exec_module(module)
            modules[name] = module
        return modules, names
    except BaseException:
        for name in reversed(names):
            sys.modules.pop(name, None)
        raise


def unload_private(names):
    for name in reversed(names):
        sys.modules.pop(name, None)
