# TODO

## Step 1: Fix duplicate test collection
- Current issue: pytest import file mismatch because duplicate basenames exist:
  - `tests/test_env.py` vs `tests/tests/test_env.py`
  - same for `test_models.py`, `test_replay.py`, `test_reward.py`
- Approach: remove/disable the duplicate `tests/tests/*` copies OR add pytest config to ignore one folder.

## Step 2: Re-run tests
- Run `python -m pytest -q` after the fix.

## Step 3: Fix any remaining failures
- If tests fail beyond collection, inspect and patch implementation/imports.


