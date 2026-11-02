## What and why

<!-- What this changes and why. Link the issue if there is one. -->

## Tests

<!-- What you ran and the result. Hardware and precision for GPU runs. -->
- [ ] `python tools/run_tests.py` (CPU fast tier)
- [ ] `python tools/run_tests.py --devices 2`
- [ ] `ruff check .`
- [ ] New behaviour has a test against an independent reference, with each bound's source stated
- [ ] A refactor leaves results bitwise unchanged (say how you checked)

## Checklist

- [ ] Errors refuse loudly; no silent fallback added
- [ ] CHANGELOG.md "Unreleased" updated
- [ ] Docs updated where behaviour or settings changed (docs/CONFIG.md for new `Config` fields)
