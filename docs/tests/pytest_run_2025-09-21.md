# Pytest Regression Suite – 2025-09-21

Executed inside the project `.venv` from repo root.

```bash
.venv/bin/pytest
```

Outcome: `17 passed, 1 skipped` (runtime 2.79s).

No regressions observed; the skip is expected (`tests/test_kernel_litmus.py` requires the optional `sep_quantum` extension).
