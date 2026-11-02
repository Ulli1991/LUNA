"""Reference solutions from the luna-reference-solutions repository, pinned by reference-solutions.lock."""
# Lookup: $LUNA_REFERENCE_SOLUTIONS, a sibling checkout ../luna-reference-solutions of the LUNA repository,
# ~/luna-reference-solutions. The checkout must be the locked commit (its git HEAD, else its REVISION file) with
# the locked MANIFEST.json, and every file must match its MANIFEST.json sha256; anything else raises (never
# skips). LUNA_REFERENCE_SOLUTIONS_UNLOCKED=1 accepts another commit or manifest, for work on new references;
# each checkout accepted that way is logged (logger luna.reference_data, and stderr).
import hashlib
import json
import logging
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

ENV = "LUNA_REFERENCE_SOLUTIONS"
UNLOCK_ENV = "LUNA_REFERENCE_SOLUTIONS_UNLOCKED"
REPO_NAME = "luna-reference-solutions"
_LUNA_ROOT = Path(__file__).resolve().parents[2]    # src/luna/reference_data.py -> the LUNA checkout
LOCK = _LUNA_ROOT / "reference-solutions.lock"
_log = logging.getLogger(__name__)


class ReferenceDataError(RuntimeError):
    """A reference solution is missing, unlisted, altered, or not the locked revision."""


def read_lock(path=None):
    """{key: value} of the lock file (repository, commit, manifest_sha256, tag)."""
    path = Path(path or LOCK)
    if not path.is_file():
        raise ReferenceDataError(f"{path} not found: the reference-data lock comes with a LUNA clone")
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    if len(out.get("commit", "")) != 40 or len(out.get("manifest_sha256", "")) != 64:
        raise ReferenceDataError(f"{path}: needs a full 40-hex `commit` and a 64-hex `manifest_sha256`")
    return out


def checkout_commit(root):
    """(commit, how) of a checkout: its own git HEAD, else the first word of its REVISION file, else (None, why)."""
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        top, _, head = p.stdout.strip().partition("\n")
        if p.returncode == 0 and Path(top).resolve() == Path(root).resolve():   # not an enclosing repository
            return head.strip(), "git rev-parse HEAD"
    except (OSError, subprocess.TimeoutExpired):
        pass
    rev = Path(root) / "REVISION"
    if rev.is_file() and rev.read_text().split():
        return rev.read_text().split()[0], str(rev)
    return None, "not a git checkout and no REVISION file"


@lru_cache(maxsize=None)
def _check_lock(root, lock_path, unlocked):
    """Refuse a checkout that is not the locked commit and manifest; if unlocked, log it instead."""
    lock = read_lock(lock_path)
    commit, how = checkout_commit(root)
    with open(Path(root) / "MANIFEST.json", "rb") as f:
        manifest = hashlib.sha256(f.read()).hexdigest()
    wrong = []
    if commit != lock["commit"]:
        wrong.append(f"commit {commit} ({how}) != locked {lock['commit']}")
    if manifest != lock["manifest_sha256"]:
        wrong.append(f"MANIFEST.json sha256 {manifest} != locked {lock['manifest_sha256']}")
    if not wrong:
        return
    msg = f"{root}: {'; '.join(wrong)} (lock: {lock_path})"
    if not unlocked:
        raise ReferenceDataError(
            f"{msg}. Check out the locked commit {lock['commit']} of {REPO_NAME}, or, working on new references, "
            f"set {UNLOCK_ENV}=1 (logged) and update the lock in the commit that needs the new data.")
    note = f"{UNLOCK_ENV}=1, using an UNLOCKED reference checkout: {msg}"
    _log.warning(note)
    print(f"[luna.reference_data] WARNING {note}", file=sys.stderr)


def _candidates():
    """The roots searched, in order, with how each was chosen."""
    env = os.environ.get(ENV)
    if env:
        return [(Path(env).expanduser(), f"${ENV}")]
    return [(_LUNA_ROOT.parent / REPO_NAME, "sibling of the LUNA checkout"), (Path.home() / REPO_NAME, "home")]


def reference_root():
    """The luna-reference-solutions checkout (the directory holding MANIFEST.json), checked against the lock."""
    tried = []
    for root, how in _candidates():
        if (root / "MANIFEST.json").is_file():
            _check_lock(str(root), str(LOCK), os.environ.get(UNLOCK_ENV, "") == "1")
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
