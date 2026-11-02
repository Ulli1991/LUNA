# Contributing to LUNA

Bug reports, questions and pull requests are welcome through GitHub issues and pull requests.
Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Set up

```bash
git clone https://github.com/Ulli1991/LUNA.git
git clone https://github.com/Ulli1991/luna-reference-solutions.git    # next to LUNA
cd LUNA
pip install -e ".[plot,test]"
```

The tests compare against reference solutions kept in
[luna-reference-solutions](https://github.com/Ulli1991/luna-reference-solutions). LUNA finds
that repository through `$LUNA_REFERENCE_SOLUTIONS`, else as `../luna-reference-solutions`
next to the clone, else as `~/luna-reference-solutions`, and checks every file's sha256 against
its `MANIFEST.json`. A missing or altered file fails the test; it is never skipped. The
checkout must be the commit in `reference-solutions.lock` (`git -C ../luna-reference-solutions
checkout <commit>`); another commit or a re-generated `MANIFEST.json` is refused. Working on new
references, set `LUNA_REFERENCE_SOLUTIONS_UNLOCKED=1` (each use is logged) and update the lock
in the commit that needs the new data.

## Run the tests

```bash
python tools/run_tests.py                 # the fast tier on CPU, about 2 minutes, as CI runs it
python tools/run_tests.py --devices 2     # the same on two virtual devices: the halo exchange
python tools/run_tests.py --tier full     # everything (meant for a GPU node)
ruff check .
```

The runner starts one pytest process per file and writes `test_report.json`. Tiers come from
measured cost in `tests/tiers.json`; a new test file counts as slow until it is timed. The GPU,
multi-GPU and multi-node tiers and the environment variables they need are described in
[docs/TESTING.md](docs/TESTING.md). Some tests need optional software (AMUSE, a Torch clone at
`$TORCH_SRC`, chemistry data at `$LUNA_DATA`); without it they fail and say what is missing.

## Rules for code and tests

These are the project's working rules; a pull request is reviewed against them.

- **A test has a source and teeth.** Compare against an exact or analytic solution, published
  numbers (cite paper and equation or table) or another code's output. Every bound states where
  it comes from (the measured value and the margin, or the paper). No tolerance is widened to
  make a test pass. A test that would still pass with the feature broken is not a test: show
  that it fails when the thing it checks is wrong. A test without an independent reference is
  labelled "regression only".
- **Refuse, don't fall back.** Code that cannot do what was asked raises an error that names the
  problem; it does not quietly switch to another method. Fallbacks that exist (for example
  `allow_first_order_faces`) are opt-in, counted and reported. See
  `src/luna/setup/run_kind.py` for the pattern.
- **Quiet is not correct.** A NaN or a run that stops short must fail loudly, not end "normally".
- **Refactors are bitwise.** A change that only reorganises code leaves results bitwise
  unchanged, and a run split over N devices equals the run on one device.
- **Measure before claiming.** Performance and accuracy numbers name the hardware, problem size
  and precision, and come from a finished run.
- **Every run goes through `luna` / `main.py`** (a run file plus `field=value` overrides); no
  standalone runners.
- **One-line docstrings**; details go in `docs/`.
- **Commits are one short plain sentence** (about 50 characters, no prefix such as `fix:`);
  the explanation goes in the pull request or the docs.
- `ruff check .` passes.

New modules: read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first. The `CLAUDE.md` files in
the tree are the same rules written for coding agents.

## Pull requests

Open the pull request against `main`, describe what changed and why, and list the tests you ran
with their result (CPU fast tier at least; GPU results with the hardware named). New behaviour
comes with a test. Add a line to the "Unreleased" section of [CHANGELOG.md](CHANGELOG.md).

## Releasing (maintainers)

1. Set the version in `pyproject.toml` (and `CITATION.cff`), move the "Unreleased" entries of
   CHANGELOG.md under the version, commit, tag `vX.Y.Z` and push the tag.
2. `.github/workflows/release.yml` then builds the sdist and wheel (README links made absolute
   at the tag by `tools/pypi_readme.py`), runs `twine check` and a clean-venv smoke run, and
   uploads to PyPI with Trusted Publishing (no API token).

The one-time setup below is done (the v0.1.0 tag created `luna-hydro` on PyPI, 2026-11-02);
it is kept for a move of the repository or a second maintainer. On pypi.org, as the project owner:

1. Account settings, Publishing, "Add a new pending publisher" (the project does not exist yet):
   PyPI project name `luna-hydro`, owner `Ulli1991`, repository `LUNA`, workflow
   `release.yml`, environment `pypi`.
2. On GitHub, Settings, Environments: create the environment `pypi` (optionally with required
   reviewers, so every upload needs an approval).
3. Optional dry run: the same pending publisher on test.pypi.org, and a copy of the workflow
   whose publish step sets `repository-url: https://test.pypi.org/legacy/`.

A GitHub release also makes a new Zenodo version (the repository's Zenodo switch is on); the
concept DOI 10.5281/zenodo.23874689 always resolves to the latest. If the repository or workflow
is renamed, update the publisher on PyPI to match.
