#!/usr/bin/env python3
"""No-cloud Linux smoke for the diagnostic watchdog and process-group kill."""
from __future__ import annotations

import json
import os
import select
import sys
import time

from gcs_drive_diagnostic import supervise_worker


def main() -> int:
    if not sys.platform.startswith("linux"):
        print(json.dumps({"event": "watchdog_smoke_unsupported", "requires": "Linux"}))
        return 2
    reader, writer = os.pipe()
    ready_reader, ready_writer = os.pipe()
    try:
        def stalled_worker():
            descendant = os.fork()
            if descendant == 0:
                os.write(ready_writer, b"ready")
                time.sleep(0.5)
                os.write(writer, b"late")
                os._exit(0)
            time.sleep(0.5)
            return 0
        result = supervise_worker(stalled_worker, time.monotonic() + 0.2)
        os.close(writer)
        writer = -1
        os.close(ready_writer)
        ready_writer = -1
        ready_available, _, _ = select.select([ready_reader], [], [], 0)
        descendant_started = bool(ready_available and os.read(ready_reader, 16) == b"ready")
        readable, _, _ = select.select([reader], [], [], 0.55)
        late_write = bool(readable and os.read(reader, 16))
        passed = result == 124 and descendant_started and not late_write
        print(json.dumps({"event": "watchdog_smoke", "passed": passed,
                          "worker_exit": result, "descendant_started": descendant_started,
                          "late_descendant_write": late_write}))
        return 0 if passed else 1
    finally:
        os.close(reader)
        os.close(ready_reader)
        if writer >= 0:
            os.close(writer)
        if ready_writer >= 0:
            os.close(ready_writer)


if __name__ == "__main__":
    raise SystemExit(main())
