"""Read-only byte identity checks; never import verified code here."""
from pathlib import Path
import hashlib


class SourceMismatch(RuntimeError):
    pass


def verify_files(root, expected):
    root = Path(root).resolve()
    actual = {}
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if root not in path.parents:
            raise SourceMismatch("Source path escapes its declared package")
        if not path.is_file():
            raise SourceMismatch("Missing source: " + relative)
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != digest:
            raise SourceMismatch("Source digest mismatch: " + relative)
        actual[relative] = value
    return actual
