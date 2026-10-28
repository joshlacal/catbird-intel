"""Rules for the us-west1 pass (plan §3.2, §4.2-§4.7).

The values the pass needs from S1's pre-registration (cutoffs, caps, samples,
the frozen Task A feature list, the O3 shrink order, the list split) live in one
JSON file. `rules_draft.json` holds the plan's stated defaults so S8 can run;
S1 must commit its own file and S9 refuses a draft unless told otherwise
(`--allow-draft-rules`). The sha256 of the file used goes into every receipt.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
DRAFT = HERE / "rules_draft.json"

CUTOFF_NAMES = ("T_fit", "T_val", "T", "T_test", "T_now")

# Per-type stats a feature may use (§4.5).
STATS = ("dec", "nzdays", "nposts", "dsl", "dsf", "n7d", "early", "late", "dl")
# A+ engagement types that carry per-type stats.
A_TYPES = ("LO", "LR", "LU", "RP", "CR", "URf", "URn", "Qf", "Qn")
PAIR_FEATURES = ("fol", "fol_age", "fol_mbulk", "mutual", "ack_share", "reply_nolike_share", "maint", "lst")
USER_FEATURES = ("d_u_log", "vol_pct", "light", "missing")
PROD_FEATURES = ("followers_log", "originals_log", "originals_share", "curators_log", "blockers_1k")


def epoch(day: str) -> int:
    """'2026-08-15' or an ISO timestamp -> Unix seconds (UTC). A bare date is the
    end of that day (cutoffs are inclusive of their date: window (C-90d, C])."""
    if "T" in day:
        return int(dt.datetime.fromisoformat(day.replace("Z", "+00:00")).timestamp())
    d = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp()) + 86400 - 1


def day_start(day: str) -> int:
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp())


def draft_features() -> list[dict]:
    feats: list[dict] = []
    for t in A_TYPES:
        for h in (14, 45, 120):
            feats.append({"name": f"{t}_dec{h}", "type": t, "stat": "dec", "h": h})
        for s in ("nzdays", "nposts", "dsl", "dsf", "n7d", "early", "late"):
            feats.append({"name": f"{t}_{s}", "type": t, "stat": s})
        if t in ("LO", "LR", "LU", "RP"):
            feats.append({"name": f"{t}_dl", "type": t, "stat": "dl"})
    for name in PAIR_FEATURES:
        feats.append({"name": name, "level": "pair"})
    for name in USER_FEATURES:
        feats.append({"name": name, "level": "user"})
    for name in PROD_FEATURES:
        feats.append({"name": name, "level": "producer"})
    return feats


def draft_rules() -> dict:
    feats = draft_features()
    return {
        "status": "DRAFT: plan defaults, not the S1 pre-registration. S1 replaces this file and commits its sha256 before S9.",
        "n_partitions": 16,
        "cutoffs": {"T_fit": "2026-07-18", "T_val": "2026-08-01", "T": "2026-08-15", "T_test": "2026-09-20", "T_now": "2026-10-18"},
        "window_days": 90,
        "label_days_A": 14,
        "label_days_B": [7, 30],
        "label_days_C": 30,
        "label_cap_pre_T": "2026-08-15",
        "label_cap_final": "2026-10-18",
        "m1_label_end": "2026-09-12",
        "m2_cohort": ["2026-08-15", "2026-09-08"],
        "m2_outcome_days": 30,
        "o5_follow_window": ["2026-08-15", "2026-09-18"],
        "o5_outcome_days": 30,
        "twelve_month_start": "2025-10-18",
        "snapshot_end": "2026-10-24T23:59:59Z",
        "tid_valid": ["2022-11-01T00:00:00Z", "2026-10-25T00:00:00Z"],
        "light_max_producer_follows": 3000,
        "burst_window_s": 30,
        "burst_threshold": 10,
        "bulk_div": 10,
        "like_day_cap": 500,
        "repost_day_cap": 100,
        "like_pair_day_cap": 3,
        "cr_window_s": 259200,
        "hostile_window_s": 172800,
        "early_s": 300,
        "late_s": 7200,
        "maint": {"n_min": 5, "recip_min": 0.5, "cov_min": 0.5, "orig_min": 5},
        "a_u_k": 50,
        "a_u_alphas": [0.5, 1.0],
        "k_min": 50,
        "samples": {
            "fit_mod": 50, "fit_rem": 7, "fit_halvings": 0,
            "m1_mod": 100, "m1_rem": 13, "m1_halvings": 0,
            "pc_mod": 1000, "pc_lt": 20,
            "s8_actor_mod": [200, 400],
        },
        "o3": {
            "max_followed": 100, "max_evidenced": 30, "n_random": 20, "m1_max_pairs": 300,
            "budget_bytes": 8000000000, "min_fit_users": 150000, "min_m1_users": 100000,
            "drop_t_test": False, "trim_features": [],
            "trim_order": [f["name"] for f in feats if f.get("h") == 120],
        },
        "o3b_default_scales": {"dec": 6.0, "nzdays": 90, "nposts": 255, "dsl": 91, "dsf": 91, "n7d": 255,
                               "early": 1.0, "late": 1.0, "dl": 1.0, "pair": 8.0, "user": 16.0, "producer": 20.0},
        "o3b_scales": {},
        "list_split": {"salt": "|listsplit", "mod": 100, "A": [0, 50], "B1a": [50, 67], "B1b": [67, 84], "B2": [84, 100]},
        "curate": {"max_members": 500, "max_lists_per_owner": 3, "seed": "7"},
        "o6_recent_days": 30,
        "o8": {"max_posts": 64, "per_month": 6},
        "o12": {"emit": True, "per_stratum": 10000, "random_nonfollower_quotes": 200},
        "teacher": {"min_engagements": 2, "min_days": 2, "top_k": 200, "min_light_engagers": 50,
                    "follower_idf_alpha": 1.0, "sample": 1000, "cap": 500, "min_cos": 0.02, "seed": 42},
        "positive_control": {"like_day": "2026-10-20", "snapshot_min": "2026-10-21T00:00:00Z"},
        "backfill_filter": None,
        "status_list_complete": False,
        "p2_user_subchunks": 4,
        "clickhouse": {"max_memory_usage": 420000000000, "max_bytes_before_external_group_by": 160000000000,
                       "max_bytes_before_external_sort": 160000000000},
        "features": feats,
    }


class Rules:
    def __init__(self, data: dict, sha256: str, path: str):
        self.d = data
        self.sha256 = sha256
        self.path = path
        self.draft = str(data.get("status", "")).upper().startswith("DRAFT")
        self.N = int(data["n_partitions"])
        self.cut = {k: epoch(v) for k, v in data["cutoffs"].items()}
        self.cut_names = list(CUTOFF_NAMES)
        self.W = int(data["window_days"]) * 86400
        self.features = [f for f in data["features"] if f["name"] not in set(data["o3"].get("trim_features", []))]
        self.validate()

    def __getitem__(self, k):
        return self.d[k]

    def validate(self) -> None:
        missing = [c for c in CUTOFF_NAMES if c not in self.d["cutoffs"]]
        if missing:
            raise ValueError(f"rules: missing cutoffs {missing}")
        names = [f["name"] for f in self.d["features"]]
        if len(names) != len(set(names)):
            raise ValueError("rules: duplicate feature names")
        for f in self.d["features"]:
            if "type" in f:
                if f["type"] not in A_TYPES or f["stat"] not in STATS:
                    raise ValueError(f"rules: bad feature {f}")
                if f["stat"] == "dec" and not f.get("h"):
                    raise ValueError(f"rules: decayed feature needs h: {f}")
            elif f.get("level") == "pair" and f["name"] not in PAIR_FEATURES:
                raise ValueError(f"rules: unknown pair feature {f}")
            elif f.get("level") == "user" and f["name"] not in USER_FEATURES:
                raise ValueError(f"rules: unknown user feature {f}")
            elif f.get("level") == "producer" and f["name"] not in PROD_FEATURES:
                raise ValueError(f"rules: unknown producer feature {f}")

    def cidx(self, name: str) -> int:
        return self.cut_names.index(name)

    def label_cap(self, cname: str) -> int:
        """B'/C' label cap [Vf40]: labels at T_fit/T_val cut at T; at T_test cut at Oct 18."""
        if cname in ("T_fit", "T_val"):
            return epoch(self.d["label_cap_pre_T"])
        return epoch(self.d["label_cap_final"])


def load(path: str | pathlib.Path | None = None) -> Rules:
    p = pathlib.Path(path) if path else DRAFT
    raw = p.read_bytes()
    return Rules(json.loads(raw), hashlib.sha256(raw).hexdigest(), str(p))


if __name__ == "__main__":  # regenerate the draft file
    DRAFT.write_text(json.dumps(draft_rules(), indent=1) + "\n")
    print(DRAFT, hashlib.sha256(DRAFT.read_bytes()).hexdigest())
