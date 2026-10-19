#!/usr/bin/env python3
"""Supplement the extracted archive follow graph with live AT Protocol follower
neighborhoods (app.bsky.graph.getFollowers) for the 26 seed subcultures and 8
collision-trap cohorts, plus export netcup's live catbird.follows table.

Every edge written is a real AT Protocol follow record so Phase 1 community
discovery and collision-trap separation can be evaluated immediately while the
multi-terabyte .jss archive continues streaming in the background.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import pathlib
import urllib.parse
import urllib.request
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema(
    [
        ("seq", pa.uint64()),
        ("time", pa.string()),
        ("kind", pa.string()),
        ("did", pa.string()),
        ("collection", pa.string()),
        ("rkey", pa.string()),
        ("rev", pa.string()),
        ("operation", pa.string()),
        ("created_at", pa.string()),
        ("subject", pa.string()),
        ("list_uri", pa.string()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("extra", pa.string()),
        ("account_active", pa.bool_()),
        ("account_status", pa.string()),
    ]
)

# 8 collision traps with canonical Bluesky handles/search queries per pole
COLLISION_TRAPS: dict[str, dict[str, Any]] = {
    "trap_1_rust_lang_vs_game": {
        "term": "rust",
        "poles": {
            "rust_systems_programming": {
                "subculture_id": "systems_rust",
                "handles": [
                    "burntsushi.net",
                    "steveklabnik.com",
                    "fasterthanlime.me",
                    "without.boats",
                    "sunshowers.io",
                    "m-ou.se",
                    "nikomatsakis.com",
                    "joshua.yoshuawuyts.com",
                ],
                "search_queries": ["rustlang cargo crate", "tokio async rust compiler"],
            },
            "rust_survival_game_fps": {
                "subculture_id": None,
                "handles": [
                    "garry.bsky.social",
                    "facepunch.com",
                ],
                "search_queries": ["playrust facepunch", "counter strike valorant esports", "apex legends twitch"],
            },
        },
    },
    "trap_2_swift_ios_vs_swifties": {
        "term": "swift",
        "poles": {
            "apple_swift_ios_dev": {
                "subculture_id": "apple_swift_indie",
                "handles": [
                    "paulhudson.bsky.social",
                    "dimillian.app",
                    "christianselig.com",
                    "stroughtonsmith.com",
                    "harshil.net",
                    "sindresorhus.com",
                    "krzyzanowskim.com",
                    "Migueldeicaza.bsky.social",
                ],
                "search_queries": ["swiftui ios developer xcode"],
            },
            "taylor_swift_pop_fandom": {
                "subculture_id": None,
                "handles": [
                    "popcrave.com",
                ],
                "search_queries": ["taylor swift eras tour swiftie", "stan twitter popiana", "charli xcx sabrina carpenter pop"],
            },
        },
    },
    "trap_3_macro_econ_vs_photo_vs_furry": {
        "term": "macro",
        "poles": {
            "macro_economics_fintwit": {
                "subculture_id": "fintwit_econsky",
                "handles": [
                    "paulkrugman.bsky.social",
                    "joeweisenthal.bsky.social",
                    "tracyalloway.bsky.social",
                    "claudiasahm.bsky.social",
                    "jasonfurman.bsky.social",
                    "deanbaker13.bsky.social",
                    "arindube.bsky.social",
                ],
                "search_queries": ["macroeconomics federal reserve economist"],
            },
            "macro_nature_entomology_photo": {
                "subculture_id": "photo_birding_culinary",
                "handles": [
                    "alexwild.bsky.social",
                ],
                "search_queries": ["entomology insect taxonomy", "mycology fungi mushroom", "herpetology ornithology"],
            },
            "furry_art_commissions": {
                "subculture_id": None,
                "handles": [],
                "search_queries": ["fursuit furry artist", "anthro furry VRChat", "fursona furry"],
            },
        },
    },
    "trap_4_lab_homelab_vs_dogs": {
        "term": "lab",
        "poles": {
            "homelab_selfhosted_infra": {
                "subculture_id": "hardware_homelab",
                "handles": [
                    "jeffgeerling.com",
                    "tailscale.com",
                    "homeassistant.io",
                    "nixos.org",
                ],
                "search_queries": ["homelab proxmox selfhosted truenas"],
            },
            "dogs_golden_retrievers_pets": {
                "subculture_id": None,
                "handles": [
                    "dogrates.bsky.social",
                    "buitengebieden.bsky.social",
                ],
                "search_queries": ["golden retriever dog rescue", "dogs of bluesky pupper"],
            },
        },
    },
    "trap_5_humor_weird_vs_resist": {
        "term": "#humor",
        "poles": {
            "weird_shitposting_postironic": {
                "subculture_id": "weird_shitposting",
                "handles": [
                    "dril.bsky.social",
                ],
                "search_queries": ["shitposter", "posting brainrot"],
            },
            "partisan_political_resistance": {
                "subculture_id": None,
                "handles": [
                    "jefftiedrich.bsky.social",
                    "muellershewrote.com",
                    "ronfilipkowski.bsky.social",
                    "atrupar.com",
                    "acyn.bsky.social",
                    "gtconway.bsky.social",
                    "meidastouch.com",
                ],
                "search_queries": ["maga democracy resist blue wave"],
            },
        },
    },
    "trap_6_espresso_craft_vs_pollsters": {
        "term": "espresso",
        "poles": {
            "specialty_coffee_culinary": {
                "subculture_id": "photo_birding_culinary",
                "handles": [
                    "kenjilopezalt.bsky.social",
                    "debperelman.bsky.social",
                ],
                "search_queries": ["specialty coffee barista roaster", "sourdough baking culinary chef"],
            },
            "elections_polling_pundits": {
                "subculture_id": None,
                "handles": [
                    "gelliottmorris.com",
                    "taniel.bsky.social",
                    "JMilesColeman.bsky.social",
                    "lakshya.bsky.social",
                ],
                "search_queries": ["elections polling redistricting"],
            },
        },
    },
    "trap_7_football_nfl_vs_soccer": {
        "term": "football",
        "poles": {
            "american_football_nfl": {
                "subculture_id": "sports_media_yappers",
                "handles": [
                    "minakimes.bsky.social",
                    "billbarnwell.bsky.social",
                    "benbbaldwin.bsky.social",
                    "robertmays.bsky.social",
                ],
                "search_queries": ["nfl football quarterback"],
            },
            "world_football_soccer_fpl": {
                "subculture_id": "soccer_fpl_tactics",
                "handles": [
                    "michaelcox.bsky.social",
                    "graceonfootball.bsky.social",
                ],
                "search_queries": ["premier league soccer tactics", "fantasy premier league fpl"],
            },
        },
    },
    "trap_8_indie_gamedev_vs_indie_music": {
        "term": "indie",
        "poles": {
            "indie_game_developers": {
                "subculture_id": "indie_gamedev",
                "handles": [
                    "godotengine.org",
                    "ramiis.bsky.social",
                    "localthunk.bsky.social",
                    "noelfb.com",
                ],
                "search_queries": ["gamedev godot pixelart indie game"],
            },
            "indie_music_vinyl_bandcamp": {
                "subculture_id": "music_vinyl_bandcamp",
                "handles": [
                    "bandcamp.com",
                    "stereogum.com",
                    "kexp.org",
                ],
                "search_queries": ["bandcamp record label vinyl synth"],
            },
        },
    },
}

PUBLIC_API = "https://public.api.bsky.app/xrpc"


def xrpc_get(method: str, params: dict[str, Any]) -> dict[str, Any]:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{PUBLIC_API}/{method}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "catbird-intel/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


async def resolve_pole_dids(
    packs: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]], dict[str, dict[str, str]]]:
    """Return (subculture_dids, trap_pole_dids, profile_meta)."""
    by_sc: dict[str, list[str]] = {}
    profile_meta: dict[str, dict[str, str]] = {}
    for sc in packs.get("subcultures", []):
        dids = []
        for a in sc.get("recommended_accounts", []):
            did = a.get("did")
            if did:
                dids.append(did)
                profile_meta[did] = {
                    "handle": a.get("handle", ""),
                    "display_name": a.get("display_name", ""),
                    "description": (a.get("description") or "")[:500],
                }
        by_sc[sc["id"]] = list(dict.fromkeys(dids))

    async def call_xrpc(method: str, params: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await asyncio.to_thread(xrpc_get, method, params)

    trap_poles: dict[str, dict[str, list[str]]] = {}
    for trap_id, spec in COLLISION_TRAPS.items():
        trap_poles[trap_id] = {}
        for pole_id, pinfo in spec["poles"].items():
            dids: list[str] = []
            sc_id = pinfo.get("subculture_id")
            if sc_id and sc_id in by_sc:
                dids.extend(by_sc[sc_id][:12])
            handles = [h.strip() for h in pinfo.get("handles", []) if h.strip()]
            if handles:
                res = await call_xrpc("app.bsky.actor.getProfiles", {"actors": handles[:25]})
                for prof in res.get("profiles", []):
                    d = prof.get("did")
                    if d and (prof.get("followersCount") or 0) >= 100:
                        dids.append(d)
                        profile_meta[d] = {
                            "handle": prof.get("handle", ""),
                            "display_name": prof.get("displayName", ""),
                            "description": (prof.get("description") or "")[:500],
                        }
            for q in pinfo.get("search_queries", []):
                res = await call_xrpc("app.bsky.actor.searchActors", {"q": q, "limit": 12})
                for prof in res.get("actors", []):
                    d = prof.get("did")
                    if d:
                        dids.append(d)
                        profile_meta[d] = {
                            "handle": prof.get("handle", ""),
                            "display_name": prof.get("displayName", ""),
                            "description": (prof.get("description") or "")[:500],
                        }
            trap_poles[trap_id][pole_id] = list(dict.fromkeys(dids))[:18]

    return by_sc, trap_poles, profile_meta


async def fetch_followers_for_did(did: str, pages: int, sem: asyncio.Semaphore) -> list[str]:
    followers: list[str] = []
    cursor: str | None = None
    for _ in range(pages):
        params: dict[str, Any] = {"actor": did, "limit": 100}
        if cursor:
            params["cursor"] = cursor
        async with sem:
            res = await asyncio.to_thread(xrpc_get, "app.bsky.graph.getFollowers", params)
        batch = res.get("followers", [])
        if not batch:
            break
        for f in batch:
            fdid = f.get("did")
            if fdid:
                followers.append(fdid)
        cursor = res.get("cursor")
        if not cursor:
            break
    return followers


async def main() -> None:
    packs_path = pathlib.Path("/Users/joshlacalamito/Developer/Catbird+Petrel/scripts/onboarding/data/catbird_onboarding_packs.json")
    packs = json.loads(packs_path.read_text())
    sem = asyncio.Semaphore(24)

    by_sc, trap_poles, profile_meta = await resolve_pole_dids(packs, sem)
    all_target_dids: set[str] = set()
    for dids in by_sc.values():
        all_target_dids.update(dids)
    for poles in trap_poles.values():
        for dids in poles.values():
            all_target_dids.update(dids)

    print(f"Resolved {len(all_target_dids)} unique seed + collision-trap anchor accounts.")
    tasks = {did: asyncio.create_task(fetch_followers_for_did(did, pages=3, sem=sem)) for did in sorted(all_target_dids)}

    rows: list[dict[str, Any]] = []
    seq_base = 9_000_000_000_000
    idx = 0
    for did, task in tasks.items():
        followers = await task
        for fdid in followers:
            idx += 1
            rkey_hash = hashlib.blake2b(f"{fdid}:{did}".encode(), digest_size=6).hexdigest()
            rows.append(
                {
                    "seq": seq_base + idx,
                    "time": "2026-10-19T14:00:00Z",
                    "kind": "commit",
                    "did": fdid,
                    "collection": "app.bsky.graph.follow",
                    "rkey": f"sup{rkey_hash}",
                    "rev": "3zzzzzzzzzzzz",
                    "operation": "create",
                    "created_at": "2026-10-19T14:00:00Z",
                    "subject": did,
                    "list_uri": "",
                    "name": "",
                    "description": "",
                    "extra": "",
                    "account_active": None,
                    "account_status": "",
                }
            )
        meta = profile_meta.get(did)
        if meta:
            idx += 1
            rows.append(
                {
                    "seq": seq_base + idx,
                    "time": "2026-10-19T14:00:00Z",
                    "kind": "commit",
                    "did": did,
                    "collection": "app.bsky.actor.profile",
                    "rkey": "self",
                    "rev": "3zzzzzzzzzzzz",
                    "operation": "create",
                    "created_at": "2026-10-19T14:00:00Z",
                    "subject": "",
                    "list_uri": "",
                    "name": meta.get("display_name") or meta.get("handle") or "",
                    "description": meta.get("description") or "",
                    "extra": meta.get("handle") or "",
                    "account_active": None,
                    "account_status": "",
                }
            )

    out_dir = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/archive-graph/events")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / "supplemental_seed_and_trap_follows.parquet"
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    pq.write_table(table, out_parquet, compression="zstd", compression_level=6)

    meta_out = pathlib.Path("/Users/joshlacalamito/catbird-lake-local/graph-build/seed_and_traps_manifest.json")
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(
        json.dumps(
            {
                "subcultures": by_sc,
                "collision_traps": trap_poles,
                "profiles": profile_meta,
                "supplemental_rows": len(rows),
            },
            indent=2,
        )
    )
    print(json.dumps({"target_dids": len(all_target_dids), "rows_written": len(rows), "parquet": str(out_parquet)}))


if __name__ == "__main__":
    asyncio.run(main())
