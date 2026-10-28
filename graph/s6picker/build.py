"""S6 baseline cold-start picker: build the localhost-only page from cand1 and S5 (i)'s live sweep.

  build.py [--s5-dir DIR] [--out DIR] [--dry-run-unverified-controls]

One command: reads the frozen cand1 artifacts, the S1 A-half lists, the Claude labels and S5 (i)'s
raw getProfiles responses; writes OUT/index.html (static, no external requests), OUT/cards.json,
OUT/receipt.json (input hashes, every raw AppView response hash, counts, the bar) and OUT/verify.json.
Serve it with serve.py (127.0.0.1 only). Plan: docs/execution-plans/2026-10-25-whole-network-signals-plan.md
row S6 and §3.3; rules: docs/execution-plans/2026-10-28-whole-network-preregistration.md §10 (frozen).

Provisional eligibility (S6 only, not the registered grid): a candidate is a rank-0 KnownFor member of
the card's community, is not in a follow-back/follow-train raw community, is on >=1 A-half curated
list, is not a user-intents opt-out, is not flagged likely_minors or adult by Haiku or Sonnet, and
passes S5's live check (returned by getProfiles, no label, not !no-unauthenticated, valid handle,
postsCount >= 3). hide-from-recs accounts stay candidates (amendment A1: posts only). A candidate
with no live check is never shown; a card with < 25 live-checked eligible candidates is not shown.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from s6picker import inputs, keywords, live as live_mod  # noqa: E402

BUNDLE = 25
POP_LIMIT, POP_MAX = 500_000, 2
MIN_POSTS = 3
PACKS_SHOWN = 3
FEEDS_SHOWN = 3
EXTRA = 10  # candidates kept beyond 25 so the in-page multi-card solver can honour the ceiling
HIDE_CARD_CLASSES = {"likely_minors"}
DROP_PROFILE_CLASSES = {"likely_minors", "adult"}


def card_order_key(c):
    return (c["sensitive"], -c["engagers"], c["community"])


def build(s5_dir, out, allow_unverified_controls=False):
    out = pathlib.Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    t0 = datetime.datetime.now(datetime.timezone.utc)

    lv = live_mod.load(s5_dir)
    controls_ok = bool(lv.controls) and all(lv.controls.values())
    if lv.present and not controls_ok and not allow_unverified_controls:
        sys.exit(f"S5 receipt at {s5_dir} does not show every control class caught ({lv.controls or 'none found'}); "
                 "plan S5: otherwise the scan is untrusted and nothing is shown. Re-run with "
                 "--dry-run-unverified-controls only for a dry run.")

    comm, pid0, com0, did0, producer_flagged, flagged_c = inputs.load_graph()
    followers, light = inputs.load_quality()
    home = dict(zip(did0, com0.tolist()))
    pid_of = dict(zip(did0, pid0.tolist()))
    pid_home = np.full(len(producer_flagged), -1, np.int64)
    pid_home[pid0] = com0
    a_owners, a_names = inputs.a_lists(cache, home)
    all_names = inputs.all_list_names()
    claude_c, claude_p = inputs.claude_labels()
    hfr = inputs.did_set(inputs.PATHS["hide_from_recs_union"])
    snap = inputs.profile_snapshot()
    optout = inputs.did_set(inputs.PATHS["user_intents_optout"])
    offers, offers_sha = inputs.s5_card_packs(s5_dir)
    packs = inputs.packs_by_community(cache, pid_home, offers)
    pnames = inputs.pack_names()
    feeds_by_c = collections.defaultdict(list)
    for fg in inputs.feedgens():
        if fg["did"] in home:
            feeds_by_c[home[fg["did"]]].append(fg)

    members = collections.defaultdict(list)
    for d, c in home.items():
        members[c].append(d)

    stats = collections.Counter()
    drop_member = collections.Counter()
    cards, excluded = [], collections.defaultdict(list)
    for r in comm:
        c = r["community"]
        cl = claude_c.get(c) or {}
        fl = flagged_c[c]
        a_top = [n for n, _k in a_names.get(c, [])]
        names_screened = a_top[:6] + all_names.get(c, [])[:6]
        kw = sorted({k for n in names_screened for k in keywords.classes(n)})
        csens = sorted(cl.get("sensitivity") or [])
        if fl["follow_back"] or fl["follow_train"]:
            excluded["flagged"].append(c); continue
        if HIDE_CARD_CLASSES & (set(kw) | set(csens)):
            excluded["likely_minors"].append(c); continue
        if not a_top:
            excluded["unlabelled"].append(c); continue
        pre, elig = [], []
        for d in members[c]:
            p = pid_of[d]
            if producer_flagged[p]:
                drop_member["producer_in_flagged_raw_community"] += 1; continue
            if a_owners.get(d, 0) < 1:
                drop_member["no_A_list"] += 1; continue
            if d in optout:
                drop_member["user_intents_optout"] += 1; continue
            if DROP_PROFILE_CLASSES & claude_p.get(d, set()):
                drop_member["claude_profile_minors_or_adult"] += 1; continue
            pre.append(d)
            st = lv.status(d)
            if st != "ok":
                drop_member[f"live_{st}"] += 1; continue
            prof = lv.profiles[d]
            if (prof.get("postsCount") or 0) < MIN_POSTS:
                drop_member["live_postsCount_lt_3"] += 1; continue
            elig.append(d)
        if len(elig) < BUNDLE:
            unchecked = sum(lv.status(d) == "unchecked" for d in pre)
            if len(pre) >= BUNDLE and unchecked:
                excluded["missing_live_check"].append(c)
            else:
                excluded["lt25_eligible"].append(c)
            continue

        def score(d):
            p = pid_of[d]
            return (-(light[p] / math.sqrt(max(followers[p], 1))), -a_owners.get(d, 0), -int(followers[p]), d)

        ranked = sorted(elig, key=score)
        bundle, big = [], 0
        for d in ranked:
            fc = lv.profiles[d].get("followersCount") or 0
            if fc > POP_LIMIT:
                if big >= POP_MAX:
                    continue
                big += 1
            bundle.append(d)
            if len(bundle) >= BUNDLE + EXTRA:
                break
        sensitive = bool(kw or csens)
        card = {
            "community": c, "scene": r["scene_id"], "size": r["size"], "eligible": len(elig),
            "engagers": int(sum(light[pid_of[d]] for d in members[c])),
            "label": " · ".join(a_top[:3]), "list_names": a_names.get(c, [])[:6],
            "claude": {k: cl.get(k) for k in ("label", "description", "community_type", "onboarding_fit", "confidence",
                                             "primary_language", "model")} if cl else None,
            "language": cl.get("primary_language") or "?",
            "sensitive": sensitive, "keyword_classes": kw, "claude_sensitivity": csens,
            "members": [] if sensitive else [acct(lv.profiles[d], d in hfr, snap.get(d)) for d in bundle],
            "packs": [], "feeds": [],
        }
        if not sensitive:
            card["packs"], pstat = card_packs(packs.get(c, []), lv, pnames)
            card["feeds"], fstat = card_feeds(feeds_by_c.get(c, []), lv, hfr)
            stats.update(pstat); stats.update(fstat)
        cards.append(card)

    cards.sort(key=card_order_key)
    scenes = collections.defaultdict(list)
    for cd in cards:
        scenes[cd["scene"]].append(cd)
    scene_list = []
    for s, cs in scenes.items():
        cnt = collections.Counter()
        for cd in cs:
            for n, k in cd["list_names"]:
                cnt[n] += k
        scene_list.append({"scene": s, "name": " / ".join(n for n, _ in cnt.most_common(2)), "cards": [cd["community"] for cd in cs],
                           "engagers": sum(cd["engagers"] for cd in cs if not cd["sensitive"])})
    scene_list.sort(key=lambda s: (-s["engagers"], s["scene"]))  # scenes led by cards that show members

    shown = [cd for cd in cards]
    data = {
        "generated_at": t0.isoformat(), "dry_run": (not lv.present) or not controls_ok,
        "live_present": lv.present, "controls": lv.controls,
        "scenes": scene_list, "cards": {str(cd["community"]): cd for cd in shown},
        "counts": {"cards": len(shown), "with_members": sum(not cd["sensitive"] for cd in shown),
                   "member_list_hidden_sensitive": sum(cd["sensitive"] for cd in shown),
                   "excluded": {k: len(v) for k, v in excluded.items()}},
    }
    (out / "cards.json").write_text(json.dumps(data, ensure_ascii=False))
    (out / "index.html").write_text(render(data))
    ver = verify(data, lv, hfr, claude_p, claude_c, excluded)
    (out / "verify.json").write_text(json.dumps(ver, indent=1))
    receipt = make_receipt(data, lv, s5_dir, out, stats, drop_member, excluded, ver, t0)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False))
    return data, ver, receipt


def acct(p, hide_from_recs, snap=None):
    name, bio = snap or ("", "")
    return {"did": p["did"], "handle": p.get("handle", ""), "name": (p.get("displayName") or name or "")[:60],
            "bio": " ".join((p.get("description") or bio or "").split())[:110], "followers": p.get("followersCount") or 0,
            "posts": p.get("postsCount") or 0, "hfr": hide_from_recs}


def card_packs(cands, lv, pnames):
    """§3.3 / S1 §10 pack filter: drop if any member is taken down, missing, labelled or
    !no-unauthenticated (hide-from-recs members do not drop a pack, A1). Unchecked members drop it too."""
    kept, st = [], collections.Counter()
    for p in cands:
        statuses = collections.Counter(lv.status(m) for m in p["members"])
        bad = {k: v for k, v in statuses.items() if k != "ok"}
        if bad:
            st["packs_dropped_" + "_".join(sorted(bad))] += 1
            continue
        st["packs_kept"] += 1
        creator = p["uri"].split("/")[2]
        handle = lv.profiles[creator]["handle"] if lv.status(creator) == "ok" else ""
        kept.append({"uri": p["uri"], "name": pnames.get(p["uri"], ""), "creator_handle": handle,
                     "members": len(p["members"]), "in_card": p["in_card"], "member_dids": p["members"]})
        if len(kept) >= PACKS_SHOWN:
            break
    return kept, st


def card_feeds(fgs, lv, hfr):
    """Feed filter: drop if the creator is hide-from-recs, !no-unauthenticated, labelled, taken down,
    missing, or has no live check."""
    kept, st = [], collections.Counter()
    for f in sorted(fgs, key=lambda f: (f["did"], f["rkey"])):
        s = lv.status(f["did"])
        if f["did"] in hfr:
            st["feeds_dropped_creator_hide_from_recs"] += 1; continue
        if s != "ok":
            st[f"feeds_dropped_creator_{s}"] += 1; continue
        st["feeds_kept"] += 1
        kept.append({"uri": f"at://{f['did']}/app.bsky.feed.generator/{f['rkey']}", "name": f["display_name"][:60],
                     "description": " ".join(f["description"].split())[:140], "creator": f["did"],
                     "creator_handle": lv.profiles[f["did"]]["handle"]})
        if len(kept) >= FEEDS_SHOWN:
            break
    return kept, st


def verify(data, lv, hfr, claude_p, claude_c, excluded):
    """Independent re-check of everything the page shows. Any entry in `violations` fails the S6 bar."""
    v = []
    for key, cd in data["cards"].items():
        c = cd["community"]
        if "likely_minors" in (claude_c.get(c, {}).get("sensitivity") or []):
            v.append(f"card {c}: Claude likely_minors card shown")
        if cd["sensitive"] and (cd["members"] or cd["packs"] or cd["feeds"]):
            v.append(f"card {c}: sensitive card discloses members/packs/feeds")
        if not cd["sensitive"] and len(cd["members"]) < BUNDLE:
            v.append(f"card {c}: bundle {len(cd['members'])} < {BUNDLE}")
        if sum(m["followers"] > POP_LIMIT for m in cd["members"][:BUNDLE]) > POP_MAX:
            v.append(f"card {c}: popularity ceiling broken")
        for m in cd["members"]:
            s = lv.status(m["did"])
            if s != "ok":
                v.append(f"card {c}: member {m['did']} live status {s}")
            if DROP_PROFILE_CLASSES & claude_p.get(m["did"], set()):
                v.append(f"card {c}: member {m['did']} Claude profile minors/adult")
        for p in cd["packs"]:
            for m in p["member_dids"]:
                if lv.status(m) != "ok":
                    v.append(f"card {c}: pack {p['uri']} member {m} {lv.status(m)}")
        for f in cd["feeds"]:
            if f["creator"] in hfr or lv.status(f["creator"]) != "ok":
                v.append(f"card {c}: feed {f['uri']} creator not allowed")
    hidden = set(excluded.get("likely_minors", [])) | set(excluded.get("flagged", []))
    page_dids = {m["did"] for cd in data["cards"].values() for m in cd["members"]}
    sens_dids = set()
    for cd in data["cards"].values():
        if cd["sensitive"]:
            sens_dids.add(cd["community"])
    return {"violations": v, "cards": len(data["cards"]), "shown_member_dids": len(page_dids),
            "hidden_cards_present": sorted(int(k) for k in data["cards"] if int(k) in hidden),
            "sensitive_cards": len(sens_dids)}


def make_receipt(data, lv, s5_dir, out, stats, drop_member, excluded, ver, t0):
    ins = {k: {"path": str(p), "sha256": inputs.sha256(p)} for k, p in inputs.PATHS.items() if pathlib.Path(p).exists()}
    for name in ("card_packs.tsv", "serving_universe.tsv"):
        p = pathlib.Path(s5_dir).expanduser() / "work/universe" / name
        if p.exists():
            ins["s5_" + name] = {"path": str(p), "sha256": inputs.sha256(p)}
    for p in inputs.FEEDGEN_DAYS + inputs.PACK_DAYS:
        ins[str(p.relative_to(inputs.LAKE))] = {"path": str(p), "sha256": inputs.sha256(p)}
    code = {p.name: inputs.sha256(p) for p in sorted(pathlib.Path(__file__).resolve().parent.glob("*.py"))}
    rh = hashlib.sha256("\n".join(lv.response_hashes).encode()).hexdigest()
    (out / "raw_response_hashes.txt").write_text("".join(h + "\n" for h in lv.response_hashes))
    bar = {
        "cards_ge_30": data["counts"]["cards"] >= 30,
        "zero_safety_violations": not ver["violations"],
        "all_controls_caught": bool(lv.controls) and all(lv.controls.values()),
        "raw_appview_responses_hashed": bool(lv.response_hashes),
        "labelled_baseline_not_evaluated": "baseline, not evaluated" in (out / "index.html").read_text(),
    }
    return {
        "step": "S6 baseline cold-start picker (Josh-only, 127.0.0.1, never published)",
        "plan": "docs/execution-plans/2026-10-25-whole-network-signals-plan.md (row S6, §3.3)",
        "rules": "docs/execution-plans/2026-10-28-whole-network-preregistration.md §10 (frozen; A1)",
        "started": t0.isoformat(), "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "dry_run": data["dry_run"], "bar": bar, "bar_pass": all(bar.values()),
        "s5": {"dir": str(s5_dir), "present": lv.present, "receipt_sha256": lv.receipt_sha256, "controls": lv.controls,
               "files": lv.files, "profiles_parsed": len(lv.profiles), "dids_requested": len(lv.requested),
               "dids_only_in_failed_calls": len(lv.errored_only),
               "raw_response_note": "S5 (i) stores each getProfiles response as one slimmed JSON line (did, found, handle, "
                                    "counts, labels); the hashes are sha256 of those lines as written by S5, one per call",
               "raw_response_records": len(lv.response_hashes), "raw_response_hashes_sha256": rh,
               "raw_response_hashes_file": "raw_response_hashes.txt"},
        "counts": data["counts"], "member_drops": dict(drop_member), "packs_feeds": dict(stats),
        "excluded_cards": {k: sorted(v) for k, v in excluded.items()},
        "verify": {k: v for k, v in ver.items() if k != "violations"}, "violations": ver["violations"][:200],
        "inputs": ins, "code_sha256": code,
        "rules_in_force": {"bundle": BUNDLE, "popularity_ceiling": f"<= {POP_MAX} accounts > {POP_LIMIT:,} followers",
                           "min_live_postsCount": MIN_POSTS, "hide_card_classes": sorted(HIDE_CARD_CLASSES),
                           "drop_profile_classes": sorted(DROP_PROFILE_CLASSES),
                           "rank": "light_engagers/sqrt(followers) (provisional proxy for D(p)/sqrt(followers)), "
                                   "then distinct A-half curators, then followers"},
    }


def render(data):
    tpl = (pathlib.Path(__file__).resolve().parent / "page.html").read_text()
    page = dict(data, cards={k: dict(cd, packs=[{x: y for x, y in p.items() if x != "member_dids"} for p in cd["packs"]])
                             for k, cd in data["cards"].items()})
    blob = json.dumps(page, ensure_ascii=False).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", blob)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--s5-dir", default=str(inputs.LAKE / "s5-sweep-20261028"))
    ap.add_argument("--out", default=str(inputs.LAKE / "s6-picker-20261028"))
    ap.add_argument("--dry-run-unverified-controls", action="store_true",
                    help="build even if the S5 receipt does not show every control caught (dry run only)")
    a = ap.parse_args()
    data, ver, receipt = build(a.s5_dir, a.out, a.dry_run_unverified_controls)
    print(json.dumps({"cards": data["counts"], "violations": len(ver["violations"]), "bar": receipt["bar"],
                      "dry_run": data["dry_run"], "out": a.out}, indent=1))
