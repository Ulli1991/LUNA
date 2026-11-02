"""luna.reference_data: a listed file resolves; a missing checkout, an unlisted path or an altered file raises."""
import hashlib
import json

import pytest

from luna import reference_data as rd


def _fake_repo(tmp_path, content=b"abc"):
    """A one-file luna-reference-solutions checkout under tmp_path."""
    (tmp_path / "luna" / "p").mkdir(parents=True)
    (tmp_path / "luna" / "p" / "f.txt").write_bytes(content)
    entry = dict(path="luna/p/f.txt", sha256=hashlib.sha256(b"abc").hexdigest(), size=3)
    (tmp_path / "MANIFEST.json").write_text(json.dumps(dict(schema=1, files=[entry])))
    return tmp_path


def test_listed_file_resolves(tmp_path, monkeypatch):
    """The env var wins and a file matching its sha256 comes back as an absolute path."""
    monkeypatch.setenv(rd.ENV, str(_fake_repo(tmp_path)))
    assert rd.reference_path("luna/p/f.txt") == str(tmp_path / "luna" / "p" / "f.txt")
    assert rd.reference_named("f.txt") == str(tmp_path / "luna" / "p" / "f.txt")


def test_missing_checkout_names_the_env_var(tmp_path, monkeypatch):
    """No checkout: the error names LUNA_REFERENCE_SOLUTIONS and the path looked in (an error, not a skip)."""
    monkeypatch.setenv(rd.ENV, str(tmp_path / "absent"))
    with pytest.raises(rd.ReferenceDataError, match=f"{rd.ENV}.*absent|absent.*{rd.ENV}"):
        rd.reference_path("luna/p/f.txt")


def test_unlisted_and_altered_files_raise(tmp_path, monkeypatch):
    """A path not in MANIFEST.json, and a file whose bytes changed, both raise."""
    monkeypatch.setenv(rd.ENV, str(_fake_repo(tmp_path, content=b"abd")))
    with pytest.raises(rd.ReferenceDataError, match="not listed"):
        rd.reference_path("luna/p/g.txt")
    with pytest.raises(rd.ReferenceDataError, match="sha256"):
        rd.reference_path("luna/p/f.txt")
