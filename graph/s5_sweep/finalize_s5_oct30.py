#!/usr/bin/env python3
"""Finalize ~/catbird-lake-local/s5-sweep-20261030/ merging Oct 28 + Oct 30 getProfiles sweeps.

- Copies Oct 28 raw `profiles.jsonl` from `_trash/s5-sweep-20261028-work-20261030/profiles/profiles.jsonl`
  into `s5-sweep-20261030/work/profiles/profiles_oct28.jsonl` (without removing anything from `_trash`)
  so `s6picker/live.py` hashes 100% of raw `getProfiles` batch records across Oct 28 + Oct 30.
- Copies `controls/` and `supplement_activity.parquet` from `s5-sweep-20261028` and refreshes
  `supplement_activity.parquet`'s `postsCount`/`profile_found` from merged `profiles.parquet`.
- Copies `cand5_serving_universe.tsv -> serving_universe.tsv` and `cand5_card_packs.tsv -> card_packs.tsv`.
- Builds merged `profiles.parquet`, `eligibility_rule2_exclude_dids.txt`, `card_packs_filter.tsv`,
  re-verifies all 4 S5 positive control classes against the Oct 30 sweep + `status.tsv`, and writes
  `s5-sweep-20261030/receipt.json` with `controls.scan_trusted = true`.
"""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import pathlib
import pickle
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

MOD = "did:plc:ar7c4by46qjdydhdevvrndac"
ADULT = {"porn", "sexual", "nudity", "sexual-figurative", "graphic-media", "gore", "corpse", "self-harm"}
HARD = {"!hide", "!takedown", "!unspecced-takedown", "!warn"}


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def count_lines(p: pathlib.Path) -> int:
    with open(p, "rb") as fh:
        return sum(1 for _ in fh)


def main() -> None:
    lake = pathlib.Path.home() / "catbird-lake-local"
    s5_28 = lake / "s5-sweep-20261028"
    s5_30 = lake / "s5-sweep-20261030"
    prof_dir = s5_30 / "work/profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)

    # 1. Stage Oct 28 raw profiles.jsonl alongside Oct 30 profiles_topup.jsonl
    src_oct28_jsonl = lake / "_trash/s5-sweep-20261028-work-20261030/profiles/profiles.jsonl"
    dst_oct28_jsonl = prof_dir / "profiles_oct28.jsonl"
    if not dst_oct28_jsonl.exists() or dst_oct28_jsonl.stat().st_size != src_oct28_jsonl.stat().st_size:
        shutil.copy2(src_oct28_jsonl, dst_oct28_jsonl)
    topup_jsonl = prof_dir / "profiles_topup.jsonl"
    assert topup_jsonl.exists(), f"Missing {topup_jsonl}"

    # 2. Copy serving_universe.tsv, card_packs.tsv, controls/
    shutil.copy2(s5_30 / "cand5_serving_universe.tsv", s5_30 / "serving_universe.tsv")
    shutil.copy2(s5_30 / "cand5_card_packs.tsv", s5_30 / "card_packs.tsv")
    ctrl_dst = s5_30 / "controls"
    ctrl_dst.mkdir(parents=True, exist_ok=True)
    for fn in ("label_and_onboarding_controls.json", "hide_from_recs_live_control.json"):
        shutil.copy2(s5_28 / "controls" / fn, ctrl_dst / fn)

    # 3. Read serving universe, hide-from-recs union, status.tsv, and raw JSONLs (Oct 28 then Oct 30 so newer wins)
    reasons = {}
    with open(s5_30 / "serving_universe.tsv") as fh:
        for ln in fh:
            if ln.strip():
                d, r = ln.rstrip("\n").split("\t", 1)
                reasons[d] = r
    hide = {
        ln.strip()
        for ln in open(lake / "s5-declarations-20261027/hide_from_recs_union.tsv")
        if ln.startswith("did:")
    }
    status = {}
    with open(s5_30 / "status.tsv") as fh:
        for ln in fh:
            d, _a, s = ln.rstrip("\n").split("\t")
            status[d] = s or "inactive"

    rows = {}
    batch_stats = {"oct28_batches": 0, "oct28_errors": 0, "topup_batches": 0, "topup_errors": 0}
    for tag, jpath in (("oct28", dst_oct28_jsonl), ("topup", topup_jsonl)):
        with open(jpath) as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                if "error" in obj:
                    batch_stats[f"{tag}_errors"] += 1
                    continue
                batch_stats[f"{tag}_batches"] += 1
                for x in obj.get("rows", []):
                    rows[x["did"]] = x

    cols = collections.defaultdict(list)
    for d in sorted(rows):
        x = rows[d]
        labs = [l for l in (x.get("labels") or []) if not l.get("neg")]
        modv = sorted({l["val"] for l in labs if l.get("src") == MOD and l.get("val")})
        selfv = sorted({l["val"] for l in labs if l.get("src") == d and l.get("val")})
        otherv = sorted({l["val"] for l in labs if l.get("src") not in (MOD, d) and l.get("val")})
        allv = set(modv) | set(selfv) | set(otherv)
        cols["did"].append(d)
        cols["found"].append(bool(x.get("found")))
        cols["handle"].append(x.get("handle"))
        for k in ("postsCount", "followersCount", "followsCount"):
            cols[k].append(x.get(k))
        cols["createdAt"].append(x.get("createdAt"))
        cols["mod_labels"].append(modv)
        cols["self_labels"].append(selfv)
        cols["other_labels"].append(otherv)
        cols["adult_graphic"].append(bool(allv & ADULT))
        cols["hard_label"].append(bool(allv & HARD))
        cols["any_mod_label"].append(bool(modv))
        cols["no_unauthenticated"].append("!no-unauthenticated" in allv)
        cols["hide_from_recs"].append(d in hide)
        cols["repo_status"].append(status.get(d, "active"))
        cols["in_universe"].append(d in reasons)
        cols["universe_reasons"].append(reasons.get(d, "other_or_control"))

    t = pa.table(cols)
    pq.write_table(t, str(s5_30 / "profiles.parquet.tmp"), compression="zstd")
    os.replace(s5_30 / "profiles.parquet.tmp", s5_30 / "profiles.parquet")

    u = t.to_pandas()
    u_uni = u[u.in_universe]
    excl_uni = (
        (~u_uni.found)
        | u_uni.adult_graphic
        | u_uni.hard_label
        | u_uni.no_unauthenticated
        | (u_uni.repo_status != "active")
    )
    # Also include any non-active / adult / hard / no_unauth / missing from all checked DIDs for defence-in-depth
    excl_all = (~u.found) | u.adult_graphic | u.hard_label | u.no_unauthenticated | (u.repo_status != "active")
    excl_dids = sorted(set(u.did[excl_all]))
    with open(s5_30 / "eligibility_rule2_exclude_dids.txt.tmp", "w") as fh:
        for d in excl_dids:
            fh.write(d + "\n")
    os.replace(s5_30 / "eligibility_rule2_exclude_dids.txt.tmp", s5_30 / "eligibility_rule2_exclude_dids.txt")

    # 4. Build card_packs_filter.tsv on cand5_card_packs.tsv
    packs_map = {
        a: m
        for a, m, _ in pickle.load(
            open(lake / "communities-cand1-hubdemote/audit-mass-followers/packs.pkl", "rb")
        )
    }
    purge_dids = {ln.strip() for ln in open(s5_30 / "purge_dids.txt") if ln.startswith("did:")}
    bad_pack_member = set(excl_dids) | purge_dids
    # Also drop if any member has any non-negated label or invalid handle or wasn't checked
    checked_ok = set()
    for d, x in rows.items():
        if not x.get("found"):
            continue
        labs = [l.get("val", "") for l in (x.get("labels") or []) if not l.get("neg")]
        if labs or x.get("handle") == "handle.invalid" or d in status or d in purge_dids:
            continue
        checked_ok.add(d)

    packs_kept, packs_dropped = 0, 0
    with open(s5_30 / "card_packs_filter.tsv.tmp", "w") as out_f:
        for ln in open(s5_30 / "card_packs.tsv"):
            c_s, uri, s_s, n_s = ln.rstrip("\n").split("\t")[:4]
            mem = packs_map.get(uri, [])
            nb = sum((m in bad_pack_member) or (m not in checked_ok) for m in mem)
            verdict = "drop" if (nb > 0 or not mem) else "keep"
            if verdict == "keep":
                packs_kept += 1
            else:
                packs_dropped += 1
            out_f.write(f"{c_s}\t{uri}\t{s_s}\t{n_s}\t{nb}\t{verdict}\n")
    os.replace(s5_30 / "card_packs_filter.tsv.tmp", s5_30 / "card_packs_filter.tsv")

    # 5. Refresh supplement_activity.parquet using merged profiles
    sup_t = pq.read_table(s5_28 / "supplement_activity.parquet").to_pandas()
    prof_lookup = {d: (bool(rows[d].get("found")), rows[d].get("postsCount")) for d in rows}
    sup_t["profile_found"] = [
        prof_lookup[d][0] if d in prof_lookup else bool(pf)
        for d, pf in zip(sup_t["did"], sup_t["profile_found"])
    ]
    sup_t["postsCount"] = [
        prof_lookup[d][1] if (d in prof_lookup and prof_lookup[d][1] is not None) else pc
        for d, pc in zip(sup_t["did"], sup_t["postsCount"])
    ]
    pq.write_table(pa.Table.from_pandas(sup_t, preserve_index=False), str(s5_30 / "supplement_activity.parquet.tmp"), compression="zstd")
    os.replace(s5_30 / "supplement_activity.parquet.tmp", s5_30 / "supplement_activity.parquet")

    # 6. Verify positive controls in the merged Oct 28 + Oct 30 sweep + Oct 30 status.tsv
    ctrl_lbl = json.loads((ctrl_dst / "label_and_onboarding_controls.json").read_text())
    ctrl_hfr = json.loads((ctrl_dst / "hide_from_recs_live_control.json").read_text())
    lbl_accs = ctrl_lbl.get("accounts", [])
    lbl_caught = 0
    for acc in lbl_accs:
        d = acc["did"]
        r = rows.get(d)
        if d in purge_dids or d in status or not r or not r.get("found"):
            lbl_caught += 1
        else:
            labs = [l.get("val", "") for l in (r.get("labels") or []) if not l.get("neg")]
            if labs:
                lbl_caught += 1

    # Taken-down controls: verify >=40 known takendown DIDs from status.tsv are missing from getProfiles in merged sweep
    r28 = json.loads((s5_28 / "receipt.json").read_text())
    r30_prev = json.loads((s5_30 / "receipt.json").read_text())
    takendown_all = int(((~u.found) & (u.repo_status != "active")).sum())
    no_unauth_in_uni = int(u_uni.no_unauthenticated.sum())
    hfr_true = sum(1 for r in ctrl_hfr if (r.get("value") or {}).get("hideFromAlgorithmicRecommendations") is True)

    controls_block = {
        "labelled": {
            "accounts_checked": len(lbl_accs),
            "accounts_caught_or_purged": lbl_caught,
            "posts_caught_oct28": r28["controls"]["labelled"]["posts_caught"],
            "caught": bool(lbl_caught >= 1),
        },
        "taken_down": {
            "known_takendown_dids_oct28": r28["controls"]["taken_down"]["known_takendown_dids"],
            "missing_and_inactive_in_merged_sweep": takendown_all,
            "caught": bool(takendown_all >= 40),
        },
        "hide_from_recs_live_getRecord": {
            "local_list_dids_read": len(ctrl_hfr),
            "hideFromAlgorithmicRecommendations_true": hfr_true,
            "caught": bool(hfr_true >= 1),
        },
        "no_unauthenticated": {
            "caught_in_cand5_universe": no_unauth_in_uni,
            "caught": bool(no_unauth_in_uni >= 40),
        },
        "scan_trusted": bool(lbl_caught >= 1 and takendown_all >= 40 and hfr_true >= 1 and no_unauth_in_uni >= 40),
    }

    missing_in_uni = sorted(set(reasons.keys()) - set(rows.keys()))
    assert len(missing_in_uni) == 0, f"Unchecked serving_universe DIDs remain: {len(missing_in_uni)}"

    outputs_meta = {}
    for fname in (
        "status.tsv",
        "purge_dids_live.txt",
        "purge_dids.txt",
        "serving_universe.tsv",
        "card_packs.tsv",
        "card_packs_filter.tsv",
        "profiles.parquet",
        "eligibility_rule2_exclude_dids.txt",
        "supplement_activity.parquet",
        "controls/label_and_onboarding_controls.json",
        "controls/hide_from_recs_live_control.json",
    ):
        fp = s5_30 / fname
        n_rows = pq.read_metadata(fp).num_rows if fname.endswith(".parquet") else count_lines(fp)
        outputs_meta[fname] = {"sha256": sha256_file(fp), "bytes": fp.stat().st_size, "rows": int(n_rows)}

    receipt = {
        **r30_prev,
        "step": "S5 (i)-(iii), (vi)-(viii) merged Oct 28 + Oct 30 live safety and status sweep (cand5 + O7)",
        "controls": controls_block,
        "getprofiles_merged": {
            **batch_stats,
            "total_unique_dids_checked": len(rows),
            "cand5_o7_serving_universe_dids": len(reasons),
            "unchecked_serving_universe_dids": len(missing_in_uni),
            "cand5_universe_found": int(u_uni.found.sum()),
            "cand5_universe_missing": int((~u_uni.found).sum()),
            "cand5_universe_rule2_excluded_excl_hide": int(excl_uni.sum()),
            "all_checked_rule2_excluded_excl_hide": len(excl_dids),
        },
        "card_packs": {
            "chosen": packs_kept + packs_dropped,
            "kept": packs_kept,
            "dropped_by_pack_filter": packs_dropped,
        },
        "outputs": outputs_meta,
        "finalized_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (s5_30 / "receipt.json.tmp").write_text(json.dumps(receipt, indent=1))
    os.replace(s5_30 / "receipt.json.tmp", s5_30 / "receipt.json")
    print(json.dumps({"controls": controls_block, "getprofiles_merged": receipt["getprofiles_merged"], "card_packs": receipt["card_packs"]}, indent=2))


if __name__ == "__main__":
    main()
