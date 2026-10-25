"""CLI: python3 -m labeling {serve,new-session,build-pools,pilot-report}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .axes import COMPONENT_AXES


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m labeling")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_serve = sub.add_parser("serve", help="serve the labeling page on 127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.add_argument("--sessions-root", help="alternate sessions dir (for trying the tool without touching gold)")
    p_new = sub.add_parser("new-session", help="create a session and print its id")
    p_new.add_argument("--kind", choices=["axis", "trap", "narrate"], default="axis")
    p_new.add_argument("--axis", choices=COMPONENT_AXES)
    p_new.add_argument("--pool", default="holdout600")
    p_new.add_argument("--subset")
    p_new.add_argument("--seed", type=int)
    sub.add_parser("build-pools", help="build extra_v1 and trap_pairs_v1 (needs pyarrow)")
    p_rep = sub.add_parser("pilot-report", help="timings and notes for the pilot50 sessions")
    p_rep.add_argument("--out", help="write the JSON report here as well as printing it")
    args = parser.parse_args()

    if args.cmd == "serve":
        from .server import serve

        serve(args.port, sessions_root=Path(args.sessions_root) if args.sessions_root else None)
    elif args.cmd == "new-session":
        from .service import new_session

        session = new_session(kind=args.kind, axis=args.axis, pool=args.pool, subset=args.subset, seed=args.seed)
        print(session.id)
    elif args.cmd == "build-pools":
        from .pool_builder import build_pools

        print(json.dumps(build_pools(), indent=2))
    elif args.cmd == "pilot-report":
        from .pilot_report import pilot_report

        report = pilot_report()
        text = json.dumps(report, indent=2, ensure_ascii=False)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
        print(text)


if __name__ == "__main__":
    main()
