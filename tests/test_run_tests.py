"""tools/run_tests.py itself: per-test counting on tiny fixture files, and --check-coverage. No jax."""
import json
import textwrap

import pytest

from tools import run_tests as rt

FIXTURES = {
    "pass": ("def test_a():\n    pass\n\ndef test_b():\n    pass\n",
             "pass", {"passed": 2}),
    "skip": ("import pytest\n\n@pytest.mark.skip(reason='x')\ndef test_a():\n    pass\n\n"
             "def test_b():\n    pytest.skip('y')\n",
             "skip", {"skipped": 2}),
    "modskip": ("import pytest\npytest.skip('no devices', allow_module_level=True)\n\n"
                "def test_a():\n    pass\n",
                "skip", {"skipped": 1}),
    "partial": ("import pytest\n\ndef test_a():\n    pass\n\n"
                "@pytest.mark.skipif(True, reason='x')\ndef test_b():\n    pass\n",
                "partial", {"passed": 1, "skipped": 1}),
    "xfail": ("import pytest\n\n@pytest.mark.xfail(reason='known')\ndef test_a():\n    assert 0\n\n"
              "@pytest.mark.xfail(reason='known', strict=False)\ndef test_b():\n    pass\n",
              "pass", {"xfailed": 1, "xpassed": 1}),
    "xpass_strict": ("import pytest\n\n@pytest.mark.xfail(reason='known', strict=True)\n"
                     "def test_a():\n    pass\n",
                     "fail", {"failed": 1}),
    "fail": ("def test_a():\n    pass\n\ndef test_b():\n    assert 1 == 2\n",
             "fail", {"passed": 1, "failed": 1}),
    "setup_error": ("import pytest\n\n@pytest.fixture\ndef broken():\n    raise RuntimeError\n\n"
                    "def test_a(broken):\n    pass\n",
                    "fail", {"error": 1}),
    "collect_error": ("import no_such_module_xyz\n\ndef test_a():\n    pass\n",
                      "fail", {"error": 1}),
    "empty": ("x = 1\n", "fail", {}),
    # exit code 0 without a finished session: the old runner called this a pass
    "crash": ("import os\n\ndef test_a():\n    os._exit(0)\n", "fail", {}),
}


@pytest.fixture
def fixture_dir(tmp_path):
    """A directory of fixture test files with its own (empty) pytest config."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    for name, (src, _, _) in FIXTURES.items():
        (tmp_path / f"test_{name}.py").write_text(src)
    return tmp_path


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_counting(fixture_dir, name):
    """Each fixture file gets the status and the per-outcome counts pytest reports."""
    _, status, expected = FIXTURES[name]
    r = rt.run_one(f"test_{name}.py", 120, cwd=str(fixture_dir))
    assert r["status"] == status, r["tail"]
    assert r["counts"] == {**dict.fromkeys(rt.OUTCOMES, 0), **expected}, r["tail"]


def test_fully_skipped_is_not_ok_as_pass(fixture_dir, tmp_path):
    """The report counts a fully skipped file as skip, not pass, and sums the test outcomes."""
    recs = [rt.run_one(f"test_{n}.py", 120, cwd=str(fixture_dir)) for n in ("pass", "skip", "partial")]
    rec = rt.write_report(str(tmp_path / "r.json"), "files", recs, "now", "fixture")
    assert rec["counts"] == {"pass": 1, "partial": 1, "skip": 1, "fail": 0, "known": 0, "timeout": 0}
    assert rec["tests"]["passed"] == 3 and rec["tests"]["skipped"] == 3
    assert rec["ok"]


def _report(path, label, files):
    """A minimal report: {file: {test id: outcome}}."""
    recs = [{"file": f, "status": "pass", "seconds": 0.0, "tests": t} for f, t in files.items()]
    path.write_text(json.dumps({"label": label, "counts": {}, "files": recs}))
    return str(path)


@pytest.fixture
def two_configs(tmp_path):
    """Two configs: a.py::t1 skipped in both, a.py::t2 runs in one, b.py skipped at module
    level in one and run in the other."""
    cpu = _report(tmp_path / "cpu.json", "cpu", {
        "a.py": {"a.py::t1": "skipped", "a.py::t2": "skipped"}, "b.py": {"b.py": "skipped"}})
    gpu = _report(tmp_path / "gpu.json", "gpu", {
        "a.py": {"a.py::t1": "skipped", "a.py::t2": "passed"}, "b.py": {"b.py::t": "xfailed"}})
    return [cpu, gpu]


def _allow(tmp_path, tests):
    """An allow-list file with these entries."""
    p =tmp_path / "allow.json"
    p.write_text(json.dumps({"tests": tests}))
    return str(p)


def test_coverage_fails_on_skip_everywhere(two_configs, tmp_path, capsys):
    """a.py::t1 is skipped in every config: the check fails and names it."""
    assert rt.check_coverage(two_configs, _allow(tmp_path, {}), ["a.py", "b.py"]) == 1
    out = capsys.readouterr().out
    assert "SKIPPED EVERYWHERE  a.py::t1" in out
    assert "SKIPPED EVERYWHERE  a.py::t2" not in out and "SKIPPED EVERYWHERE  b.py" not in out


def test_coverage_allowlist_with_reason(two_configs, tmp_path):
    """Allow-listing the one test with a reason makes the check pass."""
    allow = _allow(tmp_path, {"a.py::t1": "needs hardware no standard config has"})
    assert rt.check_coverage(two_configs, allow, ["a.py", "b.py"]) == 0


def test_coverage_allowlist_needs_reason(two_configs, tmp_path):
    """An allow-list entry without a written reason is refused."""
    with pytest.raises(ValueError, match="reason"):
        rt.check_coverage(two_configs, _allow(tmp_path, {"a.py::t1": ""}), ["a.py", "b.py"])


def test_coverage_stale_allowlist(two_configs, tmp_path, capsys):
    """An allow-listed test that runs somewhere is stale: the check fails."""
    allow = _allow(tmp_path, {"a.py::t1": "needs hardware no standard config has",
                              "a.py::t2": "needs hardware no standard config has"})
    assert rt.check_coverage(two_configs, allow, ["a.py", "b.py"]) == 1
    assert "STALE ALLOW-LIST ENTRY (it passes)  a.py::t2" in capsys.readouterr().out


def test_coverage_file_in_no_report(two_configs, tmp_path, capsys):
    """A test file that no report ran fails the check."""
    allow = _allow(tmp_path, {"a.py::t1": "needs hardware no standard config has"})
    assert rt.check_coverage(two_configs, allow, ["a.py", "b.py", "c.py"]) == 1
    assert "IN NO REPORT  c.py" in capsys.readouterr().out


def test_coverage_ignores_a_file_no_longer_in_the_tree(two_configs, tmp_path, capsys):
    """a.py removed from the tree: its skips in older reports are named and ignored; teeth: with
    a.py in the tree the same reports fail on a.py::t1."""
    assert rt.check_coverage(two_configs, _allow(tmp_path, {}), ["b.py"]) == 0
    assert "NOT IN THE TREE (ignored)  a.py" in capsys.readouterr().out
    assert rt.check_coverage(two_configs, _allow(tmp_path, {}), ["a.py", "b.py"]) == 1


def test_coverage_refuses_old_report(tmp_path, capsys):
    """A report from the old runner (no per-test outcomes) is refused, not read as covered."""
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"counts": {}, "files": [{"file": "a.py", "status": "pass"}]}))
    assert rt.check_coverage([str(p)], _allow(tmp_path, {}), ["a.py"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_coverage_end_to_end(fixture_dir, tmp_path):
    """Real runs: the skip fixture skipped in both configs fails; run it somewhere and it passes."""
    run = [rt.run_one(f"test_{n}.py", 120, cwd=str(fixture_dir)) for n in ("pass", "skip")]
    a = str(tmp_path / "a.json")
    rt.write_report(a, "files", run, "now", "a")
    files = ["test_pass.py", "test_skip.py"]
    assert rt.check_coverage([a, a], _allow(tmp_path, {}), files) == 1
    (fixture_dir / "test_skip.py").write_text(textwrap.dedent("""
        def test_a():
            pass

        def test_b():
            pass
    """))
    b = str(tmp_path / "b.json")
    rt.write_report(b, "files", [rt.run_one("test_skip.py", 120, cwd=str(fixture_dir))], "now", "b")
    assert rt.check_coverage([a, b], _allow(tmp_path, {}), files) == 0


def test_shard_partition_and_balance(monkeypatch):
    """Shards partition the files, keep X_a/X_b together, and the largest stays alone."""
    costs = {"t_big.py": 100.0, "t_r_a.py": 10.0, "t_r_b.py": 10.0, "t_1.py": 30.0, "t_2.py": 30.0,
             "t_3.py": 25.0}
    monkeypatch.setattr(rt, "load_costs", lambda: costs)
    files = sorted(costs)
    parts = [rt.shard(files, i, 3) for i in (1, 2, 3)]
    assert sorted(sum(parts, [])) == files
    assert any({"t_r_a.py", "t_r_b.py"} <= set(p) for p in parts)
    assert ["t_big.py"] in parts
    loads = sorted(sum(costs[f] for f in p) for p in parts)
    assert loads == [50.0, 55.0, 100.0]


def test_merge_reports_refuses_overlap(tmp_path, capsys):
    """Merging two shards that both ran a file is refused; disjoint shards merge."""
    a = _report(tmp_path / "a.json", "cpu8", {"a.py": {"a.py::t": "passed"}})
    b = _report(tmp_path / "b.json", "cpu8", {"a.py": {"a.py::t": "passed"}})
    c = _report(tmp_path / "c.json", "cpu8", {"c.py": {"c.py::t": "skipped"}})
    for p in (a, b, c):
        rec = json.loads(open(p).read())
        rec["started"] = "now"
        open(p, "w").write(json.dumps(rec))
    assert rt.merge_reports(str(tmp_path / "m.json"), [a, b]) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert rt.merge_reports(str(tmp_path / "m.json"), [a, c]) == 0
    assert rt.merge_reports(str(tmp_path / "m.json"), [str(tmp_path / "none.json")]) == 1


def test_exclude_needs_reason(tmp_path):
    """An excluded file without a written reason is refused."""
    p = tmp_path / "ex.json"
    p.write_text(json.dumps({"files": {"tests/test_x.py": ""}}))
    with pytest.raises(ValueError, match="reason"):
        rt.load_exclude(str(p))


def test_known_failures_are_loud_and_cannot_rot(fixture_dir):
    """A listed failing file is 'known' with its cause; a listed passing file is a fail."""
    known = {"test_fail.py": "cause at x.py:1", "test_pass.py": "cause at y.py:2"}
    f = rt.apply_known(rt.run_one("test_fail.py", 120, cwd=str(fixture_dir)), known)
    p = rt.apply_known(rt.run_one("test_pass.py", 120, cwd=str(fixture_dir)), known)
    s = rt.apply_known(rt.run_one("test_skip.py", 120, cwd=str(fixture_dir)), known)
    assert f["status"] == "known" and f["known_cause"] == "cause at x.py:1"
    assert p["status"] == "fail" and "did not fail" in p["tail"]
    assert s["status"] == "skip"


def test_coverage_collection_error_is_not_a_run(tmp_path, capsys):
    """A file that errors at collection everywhere (e.g. missing data) ran nothing: the check fails."""
    p = _report(tmp_path / "a.json", "gpu", {"c.py": {"c.py": "error"}})
    assert rt.check_coverage([p], _allow(tmp_path, {}), ["c.py"]) == 1
    assert "SKIPPED EVERYWHERE  c.py" in capsys.readouterr().out


def test_coverage_a_test_that_only_failed_is_not_covered(tmp_path, capsys):
    """A test that failed (or errored, or xpassed) in every config that ran it is not covered; teeth: passing it
    in one config covers it."""
    for bad in ("failed", "error", "xpassed"):
        a = _report(tmp_path / "a.json", "gpu", {"a.py": {"a.py::t": bad}})
        b = _report(tmp_path / "b.json", "cpu", {"a.py": {"a.py::t": "skipped"}})
        assert rt.check_coverage([a, b], _allow(tmp_path, {}), ["a.py"]) == 1, bad
        out = capsys.readouterr().out
        assert f"NEVER PASSED  a.py::t  [gpu:{bad}, cpu]" in out, out
        c = _report(tmp_path / "c.json", "gpu4", {"a.py": {"a.py::t": "passed"}})
        assert rt.check_coverage([a, b, c], _allow(tmp_path, {}), ["a.py"]) == 0, bad
        capsys.readouterr()


def test_coverage_strict_xfail_counts(tmp_path):
    """A strict xfail that xfailed is covered (its bug raised what it is scoped to)."""
    a = _report(tmp_path / "a.json", "gpu", {"a.py": {"a.py::t": "xfailed"}})
    assert rt.check_coverage([a], _allow(tmp_path, {}), ["a.py"]) == 0


def test_coverage_module_id_needs_a_passing_test(tmp_path, capsys):
    """A module-level skip is covered by a passing test of that file elsewhere, not by a failing one."""
    a = _report(tmp_path / "a.json", "cpu", {"a.py": {"a.py": "skipped"}})
    b = _report(tmp_path / "b.json", "gpu", {"a.py": {"a.py::t": "failed"}})
    assert rt.check_coverage([a, b], _allow(tmp_path, {}), ["a.py"]) == 1
    assert "SKIPPED EVERYWHERE  a.py  [cpu]" in capsys.readouterr().out
    c = _report(tmp_path / "c.json", "gpu", {"a.py": {"a.py::t": "passed"}})
    assert rt.check_coverage([a, c], _allow(tmp_path, {}), ["a.py"]) == 0


def test_coverage_allowlist_does_not_excuse_a_failure(tmp_path, capsys):
    """The skip allow-list excuses a skip, never a test that ran and failed."""
    a = _report(tmp_path / "a.json", "gpu", {"a.py": {"a.py::t": "failed"}})
    allow = _allow(tmp_path, {"a.py::t": "needs hardware no standard config has"})
    assert rt.check_coverage([a], allow, ["a.py"]) == 1
    assert "NEVER PASSED  a.py::t" in capsys.readouterr().out


def test_coverage_end_to_end_failing_test(fixture_dir, tmp_path, capsys):
    """Real run: test_fail.py::test_b fails in every config and is named; test_a passed and is covered."""
    a = str(tmp_path / "a.json")
    rt.write_report(a, "files", [rt.run_one("test_fail.py", 120, cwd=str(fixture_dir))], "now", "a")
    assert rt.check_coverage([a, a], _allow(tmp_path, {}), ["test_fail.py"]) == 1
    out = capsys.readouterr().out
    assert "NEVER PASSED  test_fail.py::test_b" in out and "test_fail.py::test_a" not in out


def test_xfails_are_strict():
    """pyproject makes every xfail strict and no test opts out, which is what lets xfailed count as covered."""
    import pathlib
    import tomllib
    root = pathlib.Path(rt.ROOT)
    assert tomllib.loads((root / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]["xfail_strict"]
    lax = [str(p) for p in sorted((root / "tests").glob("*.py"))
           if p.name != "test_run_tests.py" and "strict=False" in p.read_text()]
    assert not lax, lax
