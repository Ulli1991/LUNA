"""luna.reference_data: a locked, listed file resolves; a missing checkout, an unlisted or altered file, or a checkout
that is not the locked commit or manifest raises; the developer override is logged."""
import hashlib
import json
import logging
import subprocess

import pytest

from luna import reference_data as rd

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _fake_repo(root, content=b"abc", revision=COMMIT):
    """A one-file luna-reference-solutions checkout at root, its commit in a REVISION file (no git)."""
    (root / "luna" / "p").mkdir(parents=True)
    (root / "luna" / "p" / "f.txt").write_bytes(content)
    entry = dict(path="luna/p/f.txt", sha256=hashlib.sha256(b"abc").hexdigest(), size=3)
    (root / "MANIFEST.json").write_text(json.dumps(dict(schema=1, files=[entry])))
    if revision:
        (root / "REVISION").write_text(revision + "\n")
    return root


def _manifest_sha(root):
    """sha256 of a checkout's MANIFEST.json."""
    return hashlib.sha256((root / "MANIFEST.json").read_bytes()).hexdigest()


def _lock(tmp_path, monkeypatch, commit, manifest_sha):
    """Point luna.reference_data at a lock file with this commit and manifest sha256."""
    p = tmp_path / "test.lock"
    p.write_text(f"# test lock\ncommit = {commit}\nmanifest_sha256 = {manifest_sha}\ntag =\n")
    monkeypatch.setattr(rd, "LOCK", p)
    monkeypatch.delenv(rd.UNLOCK_ENV, raising=False)
    rd._check_lock.cache_clear()
    return p


def _git(root, *args):
    """Run git in root (a throwaway repository under tmp_path) and return its stdout."""
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake checkout at the locked COMMIT (REVISION file), selected by the env var."""
    root = _fake_repo(tmp_path / "ref")
    _lock(tmp_path, monkeypatch, COMMIT, _manifest_sha(root))
    monkeypatch.setenv(rd.ENV, str(root))
    return root


def test_listed_file_resolves(repo):
    """The env var wins and a locked checkout's file matching its sha256 comes back as an absolute path."""
    assert rd.reference_path("luna/p/f.txt") == str(repo / "luna" / "p" / "f.txt")
    assert rd.reference_named("f.txt") == str(repo / "luna" / "p" / "f.txt")


def test_missing_checkout_names_the_env_var(tmp_path, monkeypatch):
    """No checkout: the error names LUNA_REFERENCE_SOLUTIONS and the path looked in (an error, not a skip)."""
    monkeypatch.setenv(rd.ENV, str(tmp_path / "absent"))
    with pytest.raises(rd.ReferenceDataError, match=f"{rd.ENV}.*absent|absent.*{rd.ENV}"):
        rd.reference_path("luna/p/f.txt")


def test_unlisted_and_altered_files_raise(repo):
    """A path not in MANIFEST.json, and a file whose bytes changed, both raise."""
    with pytest.raises(rd.ReferenceDataError, match="not listed"):
        rd.reference_path("luna/p/g.txt")
    (repo / "luna" / "p" / "f.txt").write_bytes(b"abd")
    with pytest.raises(rd.ReferenceDataError, match="sha256"):
        rd.reference_path("luna/p/f.txt")


def test_wrong_git_commit_refuses(tmp_path, monkeypatch):
    """A git checkout whose HEAD is not the locked commit is refused, naming both; teeth: at the lock it resolves."""
    root = _fake_repo(tmp_path / "ref", revision=None)
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "v1")
    head = _git(root, "rev-parse", "HEAD")
    monkeypatch.setenv(rd.ENV, str(root))
    _lock(tmp_path, monkeypatch, COMMIT, _manifest_sha(root))
    with pytest.raises(rd.ReferenceDataError, match=f"commit {head} .*!= locked {COMMIT}.*{rd.UNLOCK_ENV}"):
        rd.reference_path("luna/p/f.txt")
    _lock(tmp_path, monkeypatch, head, _manifest_sha(root))
    assert rd.reference_path("luna/p/f.txt") == str(root / "luna" / "p" / "f.txt")


def test_wrong_revision_file_refuses(tmp_path, monkeypatch):
    """Without git, the REVISION file is the commit: another commit there is refused."""
    root = _fake_repo(tmp_path / "ref", revision="f" * 40)
    monkeypatch.setenv(rd.ENV, str(root))
    _lock(tmp_path, monkeypatch, COMMIT, _manifest_sha(root))
    with pytest.raises(rd.ReferenceDataError, match=f"commit {'f' * 40} .*REVISION.*!= locked {COMMIT}"):
        rd.reference_path("luna/p/f.txt")


def test_no_commit_at_all_refuses(tmp_path, monkeypatch):
    """No git checkout of its own (only an enclosing repository) and no REVISION file: refused."""
    _git(tmp_path, "init", "-q")                       # an enclosing repository must not lend its HEAD
    root = _fake_repo(tmp_path / "ref", revision=None)
    monkeypatch.setenv(rd.ENV, str(root))
    _lock(tmp_path, monkeypatch, COMMIT, _manifest_sha(root))
    with pytest.raises(rd.ReferenceDataError, match="commit None .*no REVISION file"):
        rd.reference_path("luna/p/f.txt")


def test_regenerated_manifest_refuses(repo):
    """The locked commit with a re-generated MANIFEST.json (substituted data) is refused."""
    m = json.loads((repo / "MANIFEST.json").read_text())
    m["files"][0]["sha256"] = hashlib.sha256(b"abd").hexdigest()
    (repo / "MANIFEST.json").write_text(json.dumps(m))
    (repo / "luna" / "p" / "f.txt").write_bytes(b"abd")
    rd._check_lock.cache_clear()
    with pytest.raises(rd.ReferenceDataError, match="MANIFEST.json sha256 .* != locked"):
        rd.reference_path("luna/p/f.txt")


def test_unlocked_override_is_logged(tmp_path, monkeypatch, capsys, caplog):
    """LUNA_REFERENCE_SOLUTIONS_UNLOCKED=1 accepts another commit, and says so on stderr and in the log."""
    root = _fake_repo(tmp_path / "ref", revision="f" * 40)
    monkeypatch.setenv(rd.ENV, str(root))
    _lock(tmp_path, monkeypatch, COMMIT, _manifest_sha(root))
    monkeypatch.setenv(rd.UNLOCK_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="luna.reference_data"):
        assert rd.reference_path("luna/p/f.txt") == str(root / "luna" / "p" / "f.txt")
    assert "UNLOCKED" in capsys.readouterr().err
    assert any("UNLOCKED" in r.getMessage() for r in caplog.records)


def test_the_lock_file_pins_a_full_commit_and_manifest():
    """The committed reference-solutions.lock parses: a 40-hex commit and a 64-hex manifest sha256."""
    lock = rd.read_lock()
    int(lock["commit"], 16), int(lock["manifest_sha256"], 16)
    assert lock["repository"].endswith("luna-reference-solutions.git")
