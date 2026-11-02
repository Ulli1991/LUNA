"""Run the test suite in tiers and leave a machine-readable report.

    python tools/run_tests.py                  # the fast tier
    python tools/run_tests.py --tier full      # everything
    python tools/run_tests.py --tier full --update-costs   # and re-time it
    python tools/run_tests.py --check-report --max-age-h 30
    python tools/run_tests.py --check-coverage REPORT...   # nothing skipped in every config

One pytest process per FILE: a wedged GPU runtime takes its own process with
it, a crash names the file it happened in, and a per-file timeout bounds the
damage (a stall is the failure mode that costs a whole batch its wall clock).

Tiers come from MEASURED cost, in tests/tiers.json, against a budget in
seconds: the fast tier is the cheapest files that fit, everything else is the
full tier. A file with no recorded cost counts as slow until it is timed, so a
new test can never silently inflate the fast tier.

This knows nothing about any machine: paths, the Python, the budget and the
timeout come from the environment or the flags. How it is invoked -- by hand,
by a batch job, by a timer -- is per-site and belongs outside this file (see
submit/examples/nightly_tests.sbatch for one).

`--devices N` runs the files on N virtual CPU devices instead of whatever
conftest pins, which is how the sharded paths get covered without a GPU:
`--devices 2` is the smallest mesh that exercises a halo exchange.

`--lint` adds `ruff check` (configured in pyproject, and until now never run)
as one more file-shaped result. It FAILS when ruff is missing rather than
skipping quietly, because a lint that silently does nothing is worse than no
lint.

Counting is per TEST, not per exit code: a pytest plugin
(tools/pytest_plugins/luna_outcomes.py) records each test's outcome -- passed, failed, skipped,
xfailed, xpassed, error, pytest's own categories, so a strict xfail that passes is failed --
and a file's status is
    pass     every test ran (passed / xfailed / xpassed),
    partial  some ran, some were skipped,
    skip     nothing ran (all skipped, or a module-level skip),
    fail     a failure, an error, a non-zero exit, or no outcomes at all (a crash),
    timeout  the per-file limit.
A fully skipped file used to count as a pass; now it is visible, and `ok` still means no fail or
timeout, because which tests skip depends on the config (a GPU test skips on CPU).

Coverage is judged across configs with `--check-coverage REPORT...`: it fails if any test id
passed in none of the reports given (skipped, not collected, or failed / errored wherever it ran),
or if a test file is in none of them, unless the id is in tests/skip_allowlist.json with a
written reason. Covered means passed, or xfailed for a strict xfail (COVERED); xpassed is not. The standard configs, one report
each (`--label` names it in the report):
    CPU-8 / --devices 8   --devices 8 --tier fast, and --tier full (no JAX_PLATFORMS set is the
                          same: conftest pins CPU with 8 devices)
    --devices 2           --devices 2 --tier fast (and full when there is time)
    1 GPU                 JAX_PLATFORMS=cuda|rocm, one GPU visible, --tier full
    2 / 4 GPU             2 / 4 GPUs visible, --tier multidevice (the files that ask jax for
                          the device count; on fewer devices they skip)
    python tools/run_tests.py --check-coverage dev8_full.json dev2.json gpu1.json gpu2.json ...
submit/examples/nightly_tests.sbatch produces them.

Environment: LUNA_TEST_BUDGET_S, LUNA_TEST_TIMEOUT_S, LUNA_TEST_REPORT.
Exit status: 0 only when no selected file failed or timed out.
"""
import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSTS = os.path.join(ROOT, "tests", "tiers.json")
LOCAL_COSTS = os.path.join(ROOT, "tests", ".costs.local.json")   # gitignored; --update-costs writes here
DEFAULT_REPORT = os.path.join(ROOT, "test_report.json")
ALLOWLIST = os.path.join(ROOT, "tests", "skip_allowlist.json")
PLUGIN_DIR = os.path.join(ROOT, "tools", "pytest_plugins")   # luna_outcomes.py
UNKNOWN_COST = 1e9          # an untimed file is slow until measured
OUTCOMES = ("passed", "failed", "skipped", "xfailed", "xpassed", "error")
RAN = ("passed", "failed", "xfailed", "xpassed", "error")       # the test body was reached
# Coverage evidence: a test is covered only where it PASSED. A strict xfail that xfailed counts (its
# registered bug raised the AssertionError it is scoped to: the expected, asserted outcome). xpassed does
# not: under xfail_strict (pyproject) a strict xpass is reported "failed", and only a non-strict xfail can
# xpass, and a non-strict xfail asserts nothing either way. failed / error never count.
COVERED = ("passed", "xfailed")
MULTIDEVICE = re.compile(r"device_count\(|len\(jax\.(local_)?devices\(")


def _read_costs(path):
    try:
        with open(path) as f:
            return json.load(f)["seconds"]
    except (OSError, KeyError, ValueError):
        return {}


def load_costs():
    """{test file: measured seconds}: the committed table, overridden by local measurements."""
    return {**_read_costs(COSTS), **_read_costs(LOCAL_COSTS)}


def test_files():
    """Every test file, repo-relative, sorted."""
    return sorted(os.path.relpath(f, ROOT)
                  for f in glob.glob(os.path.join(ROOT, "tests", "test_*.py")))


def select(tier, budget_s):
    """The files of a tier: 'fast' is the cheapest that fit in `budget_s`
    (measured), 'full' is every file, 'slow' is what fast leaves out,
    'multidevice' the files that ask jax for the device count."""
    costs = load_costs()
    files = test_files()
    if tier == "full":
        return files
    if tier == "multidevice":
        return [f for f in files if MULTIDEVICE.search(open(os.path.join(ROOT, f)).read())]
    cheap, total = [], 0.0
    for f in sorted(files, key=lambda f: costs.get(f, UNKNOWN_COST)):
        c = costs.get(f, UNKNOWN_COST)
        if total + c > budget_s:
            break
        cheap.append(f)
        total += c
    if tier == "fast":
        return sorted(cheap)
    return sorted(set(files) - set(cheap))          # slow


def classify(returncode, outcomes):
    """(status, counts) of one file from pytest's exit code and the plugin's outcomes."""
    counts = dict.fromkeys(OUTCOMES, 0)
    if outcomes is None:
        return "fail", counts                       # crashed before the session ended
    for o in list(outcomes["tests"].values()) + list(outcomes["collect"].values()):
        counts[o] = counts.get(o, 0) + 1
    ran = counts["passed"] + counts["xfailed"] + counts["xpassed"]
    if counts["failed"] or counts["error"] or returncode not in (0, 5):
        return "fail", counts
    if ran == 0:
        return ("skip" if counts["skipped"] else "fail"), counts   # rc 5 with nothing at all: fail
    return ("partial" if counts["skipped"] else "pass"), counts


def run_one(path, timeout_s, python=sys.executable, devices=0, cwd=ROOT):
    """One pytest process for one file. Returns the record for the report."""
    t0 = time.time()
    fd, outfile = tempfile.mkstemp(prefix="luna_outcomes_", suffix=".json")
    os.close(fd)
    os.remove(outfile)             # the plugin creates it; its absence means the session died
    cmd = [python, "-X", "faulthandler", "-m", "pytest", path, "-q", "--no-header",
           "-p", "no:cacheprovider", "-p", "luna_outcomes",
           "-o", f"faulthandler_timeout={max(timeout_s - 30, 30)}"]
    env = dict(os.environ)
    env["LUNA_TEST_OUTCOMES"] = outfile
    env["PYTHONPATH"] = os.pathsep.join([p for p in [env.get("PYTHONPATH")] if p] + [PLUGIN_DIR])
    if devices:
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={devices}"
    outcomes, rc = None, None
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout_s, env=env,
                           capture_output=True, text=True)
        rc, tail = p.returncode, (p.stdout or "")[-2000:]
        try:
            with open(outfile) as f:
                outcomes = json.load(f)
        except (OSError, ValueError):
            tail += "\n[run_tests] no outcomes written: the pytest session did not finish"
        status, counts = classify(rc, outcomes)
    except subprocess.TimeoutExpired as e:
        status, counts = "timeout", dict.fromkeys(OUTCOMES, 0)
        tail = (e.stdout or b"").decode(errors="replace")[-2000:] if e.stdout else ""
    finally:
        if os.path.exists(outfile):
            os.remove(outfile)
    tests = {}
    if outcomes is not None:
        tests = {**outcomes["collect"], **outcomes["tests"]}
    return {"file": path, "status": status, "seconds": round(time.time() - t0, 2),
            "returncode": rc, "counts": counts, "tests": tests, "tail": tail}


def run_lint(timeout_s):
    """`ruff check` as one more record. Missing ruff is a failure, not a skip."""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=ROOT,
                           timeout=timeout_s, capture_output=True, text=True)
        status, tail = ("pass" if p.returncode == 0 else "fail"), (p.stdout or "")[-2000:]
    except FileNotFoundError:
        status, tail = "fail", "ruff is not installed (pip install ruff)"
    except subprocess.TimeoutExpired:
        status, tail = "timeout", ""
    if status == "fail" and "No module named" in tail:
        tail = "ruff is not installed (pip install ruff)"
    return {"file": "ruff check", "status": status,
            "seconds": round(time.time() - t0, 2), "tail": tail}


STATUSES = ("pass", "partial", "skip", "fail", "known", "timeout")
MARK = {"pass": "ok  ", "partial": "part", "skip": "SKIP", "fail": "FAIL", "known": "KNWN", "timeout": "STALL"}


def write_report(path, tier, records, started, label="", excluded=None, commit=None):
    """The report every consumer reads: one record per file plus the counts.
    Written even when the run fails, so its ABSENCE means the runner never
    finished -- that is the alarm."""
    counts = {s: sum(1 for r in records if r["status"] == s) for s in STATUSES}
    tests = {o: sum(r.get("counts", {}).get(o, 0) for r in records) for o in OUTCOMES}
    rec = {"tier": tier, "label": label, "started": started,
           "finished": datetime.datetime.now().isoformat(timespec="seconds"),
           "seconds": round(sum(r["seconds"] for r in records), 1),
           "counts": counts, "tests": tests, "ok": counts["fail"] == 0 and counts["timeout"] == 0,
           "host": os.uname().nodename, "python": sys.version.split()[0],
           "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
           "xla_flags": os.environ.get("XLA_FLAGS", ""), "commit": _commit() if commit is None else commit,
           "excluded": excluded or {}, "files": records}
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def _commit():
    """The checkout's short commit, '' outside git."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _fmt_counts(c):
    """'3 passed, 1 skipped' from a counts dict, zeros left out."""
    return ", ".join(f"{c[o]} {o}" for o in OUTCOMES if c.get(o)) or "no tests"


def check_report(path, max_age_h):
    """A report that is missing, stale or failing is a non-zero exit. Silence
    is the failure this suite actually produces, so it must be loud here."""
    if not os.path.exists(path):
        print(f"NO REPORT at {path}: the runner never finished")
        return 1
    rec = json.load(open(path))
    age_h = (time.time() - os.path.getmtime(path)) / 3600.0
    state = "ok" if rec.get("ok") else "FAILING"
    print(f"{state}: tier={rec['tier']} {rec['counts']} tests {rec.get('tests', '?')} "
          f"in {rec['seconds']}s on {rec['host']}, {age_h:.1f} h old")
    if age_h > max_age_h:
        print(f"STALE: older than {max_age_h} h")
        return 1
    return 0 if rec.get("ok") else 1


def load_allowlist(path):
    """{test id: reason}. A missing file is an empty list; an entry without a reason is refused."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        tests = json.load(f).get("tests", {})
    bad = [k for k, v in tests.items() if not isinstance(v, str) or len(v.strip()) < 10]
    if bad:
        raise ValueError(f"{path}: every allow-listed test needs a written reason; missing for {bad}")
    return tests


def tree_ids_from_reports(paths):
    """The test ids of the current tree, from reports run on it: every id with `::` per file."""
    ids = set()
    for p in paths:
        with open(p) as f:
            for r in json.load(f)["files"]:
                ids.update(t for t in r.get("tests", {}) if "::" in t)
    return ids


def collect_tree_ids(python=sys.executable, cwd=ROOT, path="tests"):
    """The test ids of the current tree from `pytest --collect-only -q` (one process, CPU)."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    p = subprocess.run([python, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", path],
                       cwd=cwd, capture_output=True, text=True, env=env)
    return {line.strip() for line in p.stdout.splitlines() if "::" in line and not line.startswith(" ")}


def check_coverage(paths, allowlist_path=ALLOWLIST, all_files=None, tree_ids=None):
    """Exit status 1 if a test passed in no report (skipped, not collected, or only failed / errored
    everywhere), or a test file is in none of them, unless allow-listed with a reason; an allow-list
    entry that passed somewhere is stale and also fails. A test id `f::Class` or `f` (class /
    module-level skip or error) is covered when any test under it passed. Covered = COVERED.
    `tree_ids`: the ids of the current tree; an id of a file with known tree ids that is not among
    them (a removed test or parametrisation, in an older report) is named and ignored."""
    allow = load_allowlist(allowlist_path)
    passed, failed, skipped, files = set(), {}, {}, set()
    print(f"{'config':<28}{'files':>6}{'pass':>6}{'part':>6}{'skip':>6}{'fail':>6}{'stall':>6}"
          f"{'tests ok':>10}{'failed':>8}{'skipped':>9}")
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        label = (rec.get("label") or os.path.basename(p)) + (f"@{rec['commit']}" if rec.get("commit") else "")
        n_ok = n_fail = n_skip = 0
        for r in rec["files"]:
            if r["file"] == "ruff check" or r["status"] == "timeout":
                continue
            if "tests" not in r:
                print(f"REFUSED: {p} has no per-test outcomes (written by the old runner)")
                return 1
            files.add(r["file"])
            for tid, o in r["tests"].items():
                if o in COVERED:
                    passed.add(tid)
                    n_ok += 1
                elif o in RAN and not (o == "error" and "::" not in tid):   # reached, but no evidence
                    failed.setdefault(tid, []).append(f"{label}:{o}")
                    n_fail += 1
                else:                                                       # skipped, or a collection error
                    skipped.setdefault(tid, []).append(label)
                    n_skip += 1
        c = rec["counts"]
        print(f"{label:<28}{len(rec['files']):>6}{c.get('pass', 0):>6}{c.get('partial', 0):>6}"
              f"{c.get('skip', 0):>6}{c.get('fail', 0):>6}{c.get('timeout', 0):>6}{n_ok:>10}{n_fail:>8}{n_skip:>9}")

    def covered(tid):
        return tid in passed or any(r.startswith(tid + "::") for r in passed)

    current = set(test_files() if all_files is None else all_files)
    seen = set(skipped) | set(failed)
    gone = sorted({t.split("::")[0] for t in seen} - current)   # in an older report, not in the tree
    for f in gone:
        print(f"  NOT IN THE TREE (ignored)  {f}")
    known = {t.split("::")[0] for t in tree_ids or ()}            # files whose current ids are known

    def in_tree(t):
        f = t.split("::")[0]
        if f not in current:
            return False
        if "::" not in t or f not in known:                      # a whole file, or ids unknown: keep it
            return True
        return t in tree_ids or any(i.startswith(t + "::") for i in tree_ids)

    gone_ids = sorted(t for t in seen if t.split("::")[0] in current and not in_tree(t))
    for t in gone_ids:
        print(f"  NOT IN THE TREE (ignored)  {t}")
    never = sorted(t for t in seen if not covered(t) and in_tree(t))
    missing = sorted(current - files)
    stale = sorted(t for t in allow if covered(t))
    bad_never = [t for t in never if t not in allow or t in failed]   # the allow-list excuses skips only
    bad_missing = [f for f in missing if f not in allow]
    n_failed = n_skipped = 0
    for t in never:
        if t in allow and t not in failed:
            print(f"  allowed  {t}  ({allow[t]})")
            continue
        how = "NEVER PASSED" if t in failed else "SKIPPED EVERYWHERE"   # reached somewhere, never passed
        n_failed += t in failed
        n_skipped += t not in failed
        where = failed.get(t, []) + skipped.get(t, [])
        print(f"  {how}  {t}  [{', '.join(where)}]")
    for f in missing:
        print(f"  {'allowed' if f in allow else 'IN NO REPORT'}  {f}")
    for t in stale:
        print(f"  STALE ALLOW-LIST ENTRY (it passes)  {t}")
    ok = not (bad_never or bad_missing or stale)
    print(f"\ncoverage {'ok' if ok else 'FAILING'}: {len(passed)} test ids passed somewhere; "
          f"{n_failed} never passed where they ran, {n_skipped} skipped in every config, "
          f"{len(bad_missing)} files in no report, {len(stale)} stale allow-list entries ({len(allow)} allowed)")
    return 0 if ok else 1


def shard(files, i, n):
    """Shard i of n (1-based): longest-first greedy on measured cost, deterministic. A pair
    X_a.py / X_b.py (b reads what a wrote) stays in one shard, a before b."""
    costs = load_costs()
    groups = {}
    for f in files:
        groups.setdefault(re.sub(r"_[ab]\.py$", "", f), []).append(f)
    load, out = [0.0] * n, [[] for _ in range(n)]
    for key in sorted(groups, key=lambda k: (-sum(costs.get(f, UNKNOWN_COST) for f in groups[k]), k)):
        k = min(range(n), key=lambda j: (load[j], j))
        load[k] += sum(costs.get(f, UNKNOWN_COST) for f in groups[key])
        out[k] += groups[key]
    return sorted(out[i - 1])


def apply_known(r, known):
    """A listed file that fails is 'known' (its cause printed); one that does not fail is a 'fail'."""
    if r["file"] not in known or r["status"] == "timeout":
        return r
    if r["status"] == "fail":
        return {**r, "status": "known", "known_cause": known[r["file"]]}
    return {**r, "status": "fail",
            "tail": f"[run_tests] listed in the known failures ({known[r['file']]}) but it did not "
                    f"fail: remove it from the list\n" + r["tail"]}


def load_exclude(path):
    """{file: reason} of files a site cannot run (e.g. data it does not have); a reason is required."""
    with open(path) as f:
        files = json.load(f)["files"]
    bad = [k for k, v in files.items() if not isinstance(v, str) or len(v.strip()) < 10]
    if bad:
        raise ValueError(f"{path}: every excluded file needs a written reason; missing for {bad}")
    return files


def merge_reports(out, paths, label=""):
    """One report from the shards of ONE config (tails dropped): what --check-coverage reads."""
    records, excluded, started, label0, commits = [], {}, None, "", set()
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"REFUSED: no shard reports to merge into {out}")
        return 1
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        records += [{k: v for k, v in r.items() if k != "tail"} for r in rec["files"]]
        excluded.update(rec.get("excluded", {}))
        started = min(started or rec["started"], rec["started"])
        label0 = label0 or rec.get("label", "")
        commits.add(rec.get("commit", "?"))
    dup = sorted({r["file"] for r in records if sum(q["file"] == r["file"] for q in records) > 1})
    if dup:
        print(f"REFUSED: files in more than one shard: {dup}")
        return 1
    rec = write_report(out, "merged", sorted(records, key=lambda r: r["file"]), started,
                       label or label0, excluded=excluded, commit=",".join(sorted(commits)))
    print(f"merged {len(paths)} reports, {len(records)} files -> {out}: {rec['counts']} "
          f"tests {_fmt_counts(rec['tests'])}")
    return 0 if rec["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", default="fast", choices=("fast", "slow", "full", "multidevice"))
    ap.add_argument("--files", nargs="+", help="run these test files instead of a tier")
    ap.add_argument("--label", default="", help="the config's name in the report (e.g. gpu2)")
    ap.add_argument("--budget-s", type=float,
                    default=float(os.environ.get("LUNA_TEST_BUDGET_S", 120)),
                    help="wall-clock budget that defines the fast tier")
    ap.add_argument("--timeout-s", type=float,
                    default=float(os.environ.get("LUNA_TEST_TIMEOUT_S", 900)),
                    help="per-file timeout; a stall fails that file only")
    ap.add_argument("--report", default=os.environ.get("LUNA_TEST_REPORT", DEFAULT_REPORT))
    ap.add_argument("--update-costs", action="store_true",
                    help="write the measured seconds to tests/.costs.local.json (gitignored)")
    ap.add_argument("--check-report", action="store_true",
                    help="report on the last report instead of running")
    ap.add_argument("--max-age-h", type=float, default=30.0)
    ap.add_argument("--check-coverage", nargs="+", metavar="REPORT",
                    help="fail if a test passed in none of these reports")
    ap.add_argument("--tree-ids", nargs="+", metavar="REPORT",
                    help="with --check-coverage: reports run on the current tree; their test ids are the "
                    "tree's, and an id of an older report that is not among them is named and ignored")
    ap.add_argument("--collect-tree", action="store_true",
                    help="with --check-coverage: take the tree's test ids from pytest --collect-only")
    ap.add_argument("--allowlist", default=ALLOWLIST,
                    help="tests allowed to skip everywhere, each with a reason")
    ap.add_argument("--devices", type=int, default=0,
                    help="run on N virtual CPU devices (2 exercises a halo exchange)")
    ap.add_argument("--shard", metavar="I/N",
                    help="run shard I of N (1-based), balanced by measured cost")
    ap.add_argument("--known-failures", metavar="JSON",
                    help="files that fail for a known, cited cause: reported as known, and a "
                    "listed file that does not fail is itself a failure (the list cannot rot)")
    ap.add_argument("--exclude", metavar="JSON",
                    help="leave out the files listed (each with a reason), e.g. tests/ci_exclude.json")
    ap.add_argument("--merge-reports", nargs="+", metavar=("OUT", "REPORT"),
                    help="merge the reports of one config's shards into OUT (tails dropped)")
    ap.add_argument("--lint", action="store_true", help="also run ruff check")
    ap.add_argument("--list", action="store_true", help="print the tier and stop")
    a = ap.parse_args(argv)

    if a.check_report:
        return check_report(a.report, a.max_age_h)
    if a.check_coverage:
        tree = None
        if a.tree_ids:
            tree = tree_ids_from_reports(a.tree_ids)
        elif a.collect_tree:
            tree = collect_tree_ids()
        return check_coverage(a.check_coverage, a.allowlist, tree_ids=tree)

    if a.merge_reports:
        return merge_reports(a.merge_reports[0], a.merge_reports[1:], a.label)

    files = a.files if a.files else select(a.tier, a.budget_s)
    excluded = load_exclude(a.exclude) if a.exclude else {}
    known = load_exclude(a.known_failures) if a.known_failures else {}
    files = [f for f in files if f not in excluded]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        if not 1 <= i <= n:
            print(f"--shard {a.shard}: expected I/N with 1 <= I <= N")
            return 2
        files = shard(files, i, n)
    absent = [f for f in files if not os.path.exists(os.path.join(ROOT, f))]
    if absent:
        print(f"no such test file: {absent}")
        return 2
    if a.list:
        costs = load_costs()
        print("\n".join(f"{costs.get(f, UNKNOWN_COST):9.1f}  {f}" for f in files))
        print(f"{sum(costs.get(f, UNKNOWN_COST) for f in files):9.1f}  total measured seconds")
        return 0
    started = datetime.datetime.now().isoformat(timespec="seconds")
    print(f"{'files' if a.files else 'tier=' + a.tier}"
          + (f" shard={a.shard}" if a.shard else "") + (f" excluded={len(excluded)}" if excluded else "")
          + f" files={len(files)} budget={a.budget_s}s "
          f"timeout={a.timeout_s}s/file"
          + (f" devices={a.devices}" if a.devices else ""), flush=True)
    records = []
    for f in files:
        r = run_one(f, a.timeout_s, devices=a.devices)
        r = apply_known(r, known)
        records.append(r)
        print(f"  {MARK[r['status']]} {r['seconds']:7.1f}s  {f}  {_fmt_counts(r['counts'])}",
              flush=True)
    if a.lint:
        r = run_lint(a.timeout_s)
        records.append(r)
        print(f"  {'ok  ' if r['status'] == 'pass' else 'FAIL'} {r['seconds']:7.1f}s  "
              f"ruff check", flush=True)
    rec = write_report(a.report, "files" if a.files else a.tier, records, started, a.label,
                       excluded=excluded)
    print(f"\nfiles: {rec['counts']}\ntests: {_fmt_counts(rec['tests'])}\n"
          f"in {rec['seconds']}s -> {a.report}")

    if a.update_costs:
        costs = _read_costs(LOCAL_COSTS)
        costs.update({r["file"]: r["seconds"] for r in records
                      if r["status"] in ("pass", "partial", "skip")})
        with open(LOCAL_COSTS, "w") as f:
            json.dump({"note": "local measured seconds per test file (gitignored); "
                               "copy into tests/tiers.json to commit them", "seconds": costs}, f,
                      indent=1, sort_keys=True)
        print(f"costs updated: {len(costs)} files -> {LOCAL_COSTS}")
    for r in records:
        if r["status"] == "known":
            print(f"KNOWN FAILURE {r['file']}: {r['known_cause']}")
    for r in records:
        if r["status"] in ("fail", "timeout"):
            print(f"\n--- {r['file']} ({r['status']}) ---\n{r['tail']}")
    return 0 if rec["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
