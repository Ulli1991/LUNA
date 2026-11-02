"""Reference solutions from the luna-reference-solutions repository, sha256-checked against its MANIFEST.json."""
# Lookup: $LUNA_REFERENCE_SOLUTIONS, a sibling checkout ../luna-reference-solutions of the LUNA repository,
# ~/luna-reference-solutions. A missing repository, file or manifest entry, or a sha256 mismatch, raises (never skips).
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "LUNA_REFERENCE_SOLUTIONS"
REPO_NAME = "luna-reference-solutions"
_LUNA_ROOT = Path(__file__).resolve().parents[2]    # src/luna/reference_data.py -> the LUNA checkout


class ReferenceDataError(RuntimeError):
    """A reference solution is missing, unlisted or altered."""


def _candidates():
    """The roots searched, in order, with how each was chosen."""
    env = os.environ.get(ENV)
    if env:
        return [(Path(env).expanduser(), f"${ENV}")]
    return [(_LUNA_ROOT.parent / REPO_NAME, "sibling of the LUNA checkout"), (Path.home() / REPO_NAME, "home")]


def reference_root():
    """The luna-reference-solutions checkout (the directory holding MANIFEST.json)."""
    tried = []
    for root, how in _candidates():
        if (root / "MANIFEST.json").is_file():
            return root
        tried.append(f"{root} ({how})")
    raise ReferenceDataError(
        f"luna-reference-solutions not found; looked in: {'; '.join(tried)}. Clone it and set "
        f"{ENV}=/path/to/{REPO_NAME} (or place it next to the LUNA checkout).")


@lru_cache(maxsize=None)
def _manifest(root):
    """{relative path: manifest entry} of a checkout."""
    with open(Path(root) / "MANIFEST.json") as f:
        return {e["path"]: e for e in json.load(f)["files"]}


@lru_cache(maxsize=None)
def _sha256(path, mtime_ns, size):
    """sha256 hex digest of a file (cached per path, mtime and size)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def reference_path(rel):
    """Absolute path of `rel` (e.g. "athenak/richtmyer_meshkov/rm_ppmx_rk4_512.npz"), after its sha256 check."""
    root = reference_root()
    entry = _manifest(str(root)).get(rel)
    if entry is None:
        raise ReferenceDataError(f"{rel} is not listed in {root / 'MANIFEST.json'} (${ENV} or checkout out of date?)")
    path = root / rel
    if not path.is_file():
        raise ReferenceDataError(f"{path} is missing although listed in MANIFEST.json; set {ENV} to a complete checkout")
    st = path.stat()
    got = _sha256(str(path), st.st_mtime_ns, st.st_size)
    if got != entry["sha256"]:
        raise ReferenceDataError(f"{path}: sha256 {got} != MANIFEST.json {entry['sha256']}")
    return str(path)


def reference_named(name):
    """reference_path of the one manifest entry whose file name is `name` (e.g. "a5_ppmx_rk4_4096.npz")."""
    root = reference_root()
    hits = [p for p in _manifest(str(root)) if os.path.basename(p) == name]
    if len(hits) != 1:
        raise ReferenceDataError(f"{name}: {len(hits)} entries in {root / 'MANIFEST.json'} (need exactly one): {hits}")
    return reference_path(hits[0])
