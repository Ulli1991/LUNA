"""Pytest plugin for tools/run_tests.py: write {test id: outcome} of one run to $LUNA_TEST_OUTCOMES.

Outcomes are pytest's own terminal categories (pytest_report_teststatus): passed, failed, skipped,
xfailed, xpassed, error. A strict xfail that passes is "failed", as pytest reports it. A failing
setup or teardown is "error" and wins over the call outcome. A module-level skip or a collection
error is recorded under the file's id, so a file that runs nothing still leaves a trace.
"""
import json
import os

_CONFIG = None
_PHASES = {}       # test id -> categories of its non-empty phases, in order
_COLLECT = {}      # file id -> "skipped" | "error"


def pytest_configure(config):
    """Remember the config: the category comes from the config's own teststatus hook."""
    global _CONFIG
    _CONFIG = config


def pytest_runtest_logreport(report):
    """Record the category of each phase (setup, call, teardown) of each test."""
    cat = _CONFIG.hook.pytest_report_teststatus(report=report, config=_CONFIG)[0]
    if cat:
        _PHASES.setdefault(report.nodeid, []).append(cat)


def pytest_collectreport(report):
    """Record a module-level skip or a collection error under the collector's id."""
    if report.failed:
        _COLLECT[report.nodeid] = "error"
    elif report.skipped:
        _COLLECT[report.nodeid] = "skipped"


def _outcome(cats):
    """One outcome per test: an error or failure in any phase wins, else the last phase's category."""
    for bad in ("error", "failed"):
        if bad in cats:
            return bad
    return cats[-1]


def pytest_sessionfinish(session, exitstatus):
    """Write the outcomes once the session is over (a crashed process writes nothing: that is a fail)."""
    path = os.environ.get("LUNA_TEST_OUTCOMES")
    if not path:
        return
    out = {"collect": _COLLECT, "tests": {k: _outcome(v) for k, v in _PHASES.items()},
           "exitstatus": int(exitstatus)}
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
