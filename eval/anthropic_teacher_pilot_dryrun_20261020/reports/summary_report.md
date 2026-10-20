# Catbird Anthropic Teacher Harness — Run Report

- **Run ID**: `catbird-anthropic-teacher-run-v1` (Generation `gen-20261020-v1`)
- **Config Bundle SHA-256**: `73b188e20a0eb953e923d2b58e2cff4621f16bd8498d7b29c8dd1e15a44dd644`
- **Simulated Dry-Run Responses Present**: `True` (Never usable as real teacher inference)
- **Paid API Calls Executed**: `0`
- **Ledger Hash Chain Verified**: `True`

## Budget & Liability Ledger
- **Total Cash Ceiling**: `$200.000000`
- **Total Settled Cost**: `$0.020662`
- **Total Outstanding Reservations**: `$0.000000`
- **Remaining Cash Ceiling**: `$199.979338`
- **Budget Halted**: `False`

## Request Statuses & Multi-Teacher Disagreement
- **Request Status Counts**: `{"succeeded": 33}`
- **Schema-Valid Non-Truncated Rate**: `1.0000` (Gate `>= 0.99`)
- **Multi-Teacher Items Compared**: `9`
- **Disagreement Breakdown**: `{"resolved_or_single_teacher": 9}`

## Explicit Future Pilot Launch Command (Capped at $8 Pilot / $2 Outstanding)
```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python catbird-intel/ops/anthropic_teacher_harness.py submit --run-dir /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/anthropic_teacher_pilot_dryrun_20261020 --stage pilot --model-tier haiku --transport live --no-offline --authorize-paid-submission --authorize-corpus-transmission
```

## Resume & Crash Recovery Commands
```sh
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python catbird-intel/ops/anthropic_teacher_harness.py recover-partials --run-dir /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/anthropic_teacher_pilot_dryrun_20261020
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python catbird-intel/ops/anthropic_teacher_harness.py reconcile --run-dir /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/anthropic_teacher_pilot_dryrun_20261020
PYTHONDONTWRITEBYTECODE=1 /tmp/catbird-teacher-venv/bin/python catbird-intel/ops/anthropic_teacher_harness.py collect --run-dir /Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/anthropic_teacher_pilot_dryrun_20261020
```
