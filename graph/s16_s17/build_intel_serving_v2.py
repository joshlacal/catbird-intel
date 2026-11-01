#!/usr/bin/env python3
"""Build S17 rev 6 Community Intelligence Serving Bundle (`intel_serving_v2.json`).

Compiles clean S17 rev 5 lake artifacts into a fast, zero-PII, viewer-hydration-ready
serving bundle for `catbird-intel`:
1. All 104 scenes mapped to bespoke human titles, unique slugs, 12 macro-categories, and Apple SF Symbols.
2. All 1,016 communities with 100%-backfilled clean sample post AT-URIs, up to 16 clean preview post AT-URIs,
   curated-list consensus tags, L2-normalized int8 256d post centroids, and mined aggregate TF-IDF + concept tokens.
3. All 25,400 organic members classified into 5 Community Roles (`anchor_creator`, `visual_maker`,
   `conversationalist`, `scene_curator`, `project_or_studio`), calibrated `recommendedMode`
   (`originals_only` when `repostShareBps >= 4500`), weekly cadence, and non-PII `whyPicked` rationales.
4. Precomputed contrastive `tasteDuels` (8 classic ambiguity traps + ~56 sibling sub-scene splitters).
5. Compact `warmStartIndex` mapping top producers to `(communityId, weightBps)` for instant orbit seeding,
   plus `highHostilityPairs` (`S14`) and `noUnauthDids` (`S5` `!no-unauthenticated`).
"""
from __future__ import annotations

import base64
import collections
import datetime as dt
import hashlib
import json
import math
import pathlib
import re
import sqlite3
from typing import Any

import numpy as np
import pyarrow.parquet as pq

LAKE = pathlib.Path("/Users/joshlacalamito/catbird-lake-local")
S17_DIR = LAKE / "s17-picker-v1-20261030"
S16_DIR = LAKE / "s16-s17-20261030"
OUT_PATH = S17_DIR / "intel_serving_v2.json"

CATALOG_VERSION = "s17_rev6_intel_v2_20261031"
_B32_SET = set("abcdefghijklmnopqrstuvwxyz234567")


def expand_did(s: str) -> str:
    if s.startswith("did:"):
        return s
    pad = "=" * ((4 - len(s) % 4) % 4)
    raw15 = base64.urlsafe_b64decode(s + pad)
    return "did:plc:" + base64.b32encode(raw15).decode("ascii").lower()


def expand_pack_uri(s: str) -> str:
    if s.startswith("at://"):
        return s
    cdid, rkey = s.split("/", 1)
    return f"at://{expand_did(cdid)}/app.bsky.graph.starterpack/{rkey}"


def expand_feed_uri(s: str) -> str:
    if s.startswith("at://"):
        return s
    cdid, rkey = s.split("/", 1)
    return f"at://{expand_did(cdid)}/app.bsky.feed.generator/{rkey}"


# Complete, human-curated taxonomy for all 104 S17 scenes:
# scene_id -> (sceneTitle, sceneSlug, macroCategory, sfSymbolName)
SCENE_TAXONOMY_104: dict[int, tuple[str, str, str, str]] = {
    6: ("Turkish Literature & Pop Culture", "turkish-lit-culture", "world_languages", "globe.europe.africa.fill"),
    8: ("Japanese Everyday Life & Photography", "jp-daily-photo", "world_languages", "camera.macro"),
    11: ("Fursuit Makers & Convention Craft", "fursuit-makers", "cities_craft_food", "scissors"),
    13: ("Warcraft & Character Concept Art", "warcraft-character-art", "arts_illustration", "wand.and.stars"),
    14: ("Indie English VTubers & Art Streamers", "en-vtubers-art", "film_tv_anime", "sparkles.tv.fill"),
    19: ("Pokémon, Undertale & Indie Fan Art", "pokemon-undertale-art", "arts_illustration", "paintpalette.fill"),
    21: ("Spanish Comics, Games & Humor", "es-comics-games", "world_languages", "bubble.left.and.bubble.right.fill"),
    22: ("Urbanism, Cycling & Active Mobility", "urbanism-cycling", "cities_craft_food", "bicycle"),
    24: ("Fate/FGO & Touken Ranbu Illustrators", "jp-fgo-touken-art", "arts_illustration", "paintbrush.pointed.fill"),
    25: ("Nature, Bird & Street Photography", "nature-street-photo", "science_nature", "camera.aperture"),
    26: ("Star Fox & Kemono Creature Artists", "kemono-starfox-art", "arts_illustration", "pawprint.circle.fill"),
    27: ("Brazilian Illustration, RPGs & Indie Games", "br-art-indie-games", "world_languages", "gamecontroller.fill"),
    29: ("Fantasy & Concept Illustration", "fantasy-concept-art", "arts_illustration", "paintbrush.pointed.fill"),
    30: ("German Culture, Law & Football", "de-culture-football", "world_languages", "globe.europe.africa.fill"),
    31: ("Sci-Fi, Fantasy, Horror & Romance Books", "sff-horror-romance-books", "books_writing_comics", "book.closed.fill"),
    32: ("Brazilian Dev & BookSky", "br-dev-booksky", "world_languages", "chevron.left.forwardslash.chevron.right"),
    33: ("MusicSky, Composers & Indie Labels", "musicsky-composers", "music_audio", "music.quarternote.3"),
    34: ("Software Engineering, Rust, Swift & Design", "software-systems-design", "software_ai_tech", "terminal.fill"),
    35: ("Weird Internet Humor & Shitposting", "weird-humor-shitposting", "queer_fandom_hangout", "face.smiling.inverse"),
    36: ("Stand-Up Comedy, Banter & Doctor Who", "comedy-banter-drwho", "film_tv_anime", "mic.fill"),
    37: ("Indie Animation & Minecraft Creators", "indie-anim-minecraft", "games_interactive", "cube.fill"),
    38: ("Indie Game Dev, Shaders & Game Writing", "indie-gamedev-craft", "games_interactive", "gamecontroller.fill"),
    40: ("French Bande Dessinée, Tech & Culture", "fr-bd-culture", "world_languages", "text.book.closed.fill"),
    41: ("Japanese Manga & Anime Illustrators", "jp-manga-illustrators", "arts_illustration", "scribble.variable"),
    42: ("US Sports, NBA, MLB & Analytics", "us-sports-analytics", "sports_motorsport", "baseball.fill"),
    43: ("Philosophy & International Law", "philosophy-intl-law", "news_civics_history", "building.columns.fill"),
    44: ("Brazilian Music & Pop Journalism", "br-music-journalism", "music_audio", "headphones"),
    45: ("Brazilian TV & Pop Commentary", "br-tv-pop", "film_tv_anime", "tv.fill"),
    46: ("World Football, Premier League & Tactics", "world-football-soccer", "sports_motorsport", "soccerball"),
    48: ("Japanese Original-Character & Doujin Art", "jp-oc-doujin-art", "arts_illustration", "sparkles"),
    49: ("Bravo & Reality TV Fandom", "reality-tv-bravo", "film_tv_anime", "popcorn.fill"),
    50: ("K-Pop Girl Groups & SHINee Fandom", "kpop-fandom", "music_audio", "music.mic"),
    51: ("Montreal Canadiens & Canadian Culture", "montreal-habs-culture", "sports_motorsport", "hockey.puck.fill"),
    52: ("Furry, Creature & Dragon Illustration", "furry-creature-art", "arts_illustration", "pawprint.fill"),
    53: ("Film Critics, Cinephiles & Folk Horror", "film-critics-cinephiles", "film_tv_anime", "film.stack.fill"),
    54: ("Jeopardy! & Trivia Quizzers", "jeopardy-trivia", "queer_fandom_hangout", "questionmark.bubble.fill"),
    55: ("Brasília Everyday Community", "brasilia-community", "world_languages", "sun.max.fill"),
    56: ("Fine Art Painting, Landscapes & Fiber Craft", "fine-art-fiber-craft", "arts_illustration", "swatchpalette.fill"),
    57: ("Markets, UK Broadcasters & Public Policy", "markets-finance-policy", "news_civics_history", "chart.line.uptrend.xyaxis"),
    58: ("Formula 1, IndyCar & Automotive Culture", "f1-motorsport-cars", "sports_motorsport", "flag.checkered"),
    59: ("Indie Cartoonists & Sonic Fan Art", "cartoonists-sonic-art", "books_writing_comics", "pencil.and.outline"),
    60: ("Star Wars, Tolkien & BG3 Fandom", "starwars-tolkien-bg3", "film_tv_anime", "moon.stars.fill"),
    61: ("Folklore, Forteana & Weird Mysteries", "paranormal-weird-lore", "queer_fandom_hangout", "sparkles.rectangle.stack.fill"),
    62: ("Darkwave, Nerd Music & Historical Craft", "darkwave-filk-sca", "music_audio", "guitars.fill"),
    63: ("Mathematics, Libraries & KidLit", "math-libraries-kidlit", "books_writing_comics", "function"),
    64: ("History, Archaeology & Medieval Studies", "history-archaeology", "news_civics_history", "scroll.fill"),
    65: ("Destiny, Speedrunning & Elder Scrolls", "destiny-speedrun-rpg", "games_interactive", "bolt.horizontal.fill"),
    67: ("AT Protocol, Bluesky & Creative Coding", "atproto-creative-code", "software_ai_tech", "network"),
    69: ("Uruguayan & Eurovision Pop Culture", "uruguay-eurovision", "world_languages", "music.note.tv.fill"),
    70: ("JRPG Streamers & Anime Rhythm Fandom", "jrpg-streamers", "games_interactive", "dpad.fill"),
    71: ("Neuroscience, Genetics & Cell Biology", "neuro-cell-biology", "science_nature", "atom"),
    72: ("Linguistics & Cognitive Science", "linguistics-cogsci", "science_nature", "character.bubble.fill"),
    75: ("College Rock, R.E.M. & U2 Fans", "college-alt-rock", "music_audio", "opticaldisc.fill"),
    76: ("Webcomics & Indie Comic Strips", "webcomics-indie-comics", "books_writing_comics", "rectangle.split.2x2.fill"),
    78: ("Ball-Jointed Dolls & HoYoverse Art", "bjd-hoyoverse-art", "arts_illustration", "heart.text.square.fill"),
    79: ("Entomology, Birding, Fossils & Earth Science", "bugs-birds-earth-science", "science_nature", "ladybug.fill"),
    80: ("Character & Multiverse Roleplay", "character-roleplay", "queer_fandom_hangout", "theatermasks.fill"),
    81: ("Emergency, Critical Care & Clinical Medicine", "clinical-medicine", "science_nature", "stethoscope"),
    83: ("Cats, Dogs & Animal Rescues", "cats-dogs-pets", "science_nature", "cat.fill"),
    84: ("Retro Gaming, CRT Computing & Video Essays", "retro-gaming-essays", "games_interactive", "desktopcomputer"),
    85: ("Public Transit, Walkability & Kiwi Tech", "public-transit-walkable", "cities_craft_food", "tram.fill"),
    86: ("Railways & European Train Enthusiasts", "railways-trains", "cities_craft_food", "train.side.front.car"),
    87: ("Soulsborne & Korean Game Illustration", "soulsborne-kr-art", "games_interactive", "shield.lefthalf.filled"),
    88: ("Poetry, Haiku & Literary Small Presses", "poetry-haiku-lit", "books_writing_comics", "quote.opening"),
    89: ("Eastern European Fandom & Humor", "ru-fandom-humor", "world_languages", "bubble.left.fill"),
    90: ("Zelda, Fire Emblem & Atlus Fan Art", "zelda-fe-atlus-art", "games_interactive", "sparkles"),
    91: ("Clean Energy & Climate Transition Policy", "clean-energy-policy", "science_nature", "wind"),
    92: ("Dropout, Alt-Comedy & Actual-Play Shows", "dropout-alt-comedy", "film_tv_anime", "theatermasks.circle.fill"),
    93: ("Old School RuneScape Community", "osrs-runescape", "games_interactive", "crown.fill"),
    96: ("Tabletop RPGs & Warhammer Miniatures", "ttrpg-warhammer", "games_interactive", "die.face.6.fill"),
    97: ("Indie Animation & Voice Acting", "animation-voice-actors", "film_tv_anime", "waveform.and.mic"),
    100: ("Genshin Impact & Anime Fan Art", "genshin-anime-art", "arts_illustration", "sparkle.magnifyingglass"),
    101: ("Gunpla, Transformers & Sci-Fi Mecha", "gunpla-mecha-scifi", "arts_illustration", "gearshape.2.fill"),
    102: ("Poker, Strategy & Miami/Austin Sports", "poker-strategy-sports", "sports_motorsport", "suit.spade.fill"),
    104: ("Magic: The Gathering & Tabletop Card Games", "mtg-tcg-community", "games_interactive", "rectangle.portrait.on.rectangle.portrait.angled.fill"),
    105: ("Brazilian BTS ARMY", "br-bts-army", "music_audio", "heart.circle.fill"),
    107: ("Final Fantasy XIV Players & Artists", "ffxiv-community", "games_interactive", "sparkles"),
    109: ("Dutch Books, Nature & Everyday Life", "nl-culture-nature", "world_languages", "tulip"),
    110: ("Swifties & Pop Music Fandoms", "swifties-pop-fandom", "music_audio", "star.fill"),
    112: ("Phish & Live Jam Bands", "phish-jam-bands", "music_audio", "music.note.list"),
    113: ("AI, Machine Learning & HCI Research", "ai-ml-hci-research", "software_ai_tech", "cpu.fill"),
    114: ("Nordic & Finnish Ecology, Art & Football", "nordic-fi-community", "world_languages", "snowflake"),
    115: ("Cosplay Makers & Anime Fandom", "cosplay-anime-makers", "film_tv_anime", "person.crop.artframe"),
    116: ("Vocaloid Producers & Rhythm Game Art", "vocaloid-producers", "music_audio", "waveform.path.ecg"),
    117: ("Food Writing, Sourdough, Beer & Midwest Sports", "food-cooking-beer", "cities_craft_food", "fork.knife"),
    118: ("Japanese Pokémon & Monster Fan Art", "jp-pokemon-digimon-art", "arts_illustration", "bolt.circle.fill"),
    119: ("Southeast Asian & Taiwanese VTuber Art", "sea-tw-vtuber-art", "arts_illustration", "paintbrush.fill"),
    120: ("Fly Fishing, Angling & Mountain Outdoors", "fly-fishing-outdoors", "science_nature", "fish.fill"),
    122: ("Star Trek, RiffTrax & Cult Sci-Fi TV", "startrek-cult-tv", "film_tv_anime", "sparkles.tv"),
    123: ("Irish Literature & Wrexham Football", "irish-lit-wrexham", "books_writing_comics", "leaf.fill"),
    124: ("Pro Wrestling, AEW & Indie Promotions", "pro-wrestling-aew", "sports_motorsport", "figure.wrestling"),
    125: ("Norwegian & Danish Community & Football", "no-dk-community", "world_languages", "globe.europe.africa.fill"),
    126: ("Fighting Game Community (FGC) & Arcade Art", "fgc-fighting-games", "games_interactive", "arcade.stick.console.fill"),
    127: ("Economics Research & EconSky", "economics-econsky", "news_civics_history", "chart.bar.xaxis"),
    128: ("Pacific Northwest Sports & Sounders", "pnw-seattle-sports", "sports_motorsport", "sportscourt.fill"),
    129: ("Astronomy, Planetary Science & Spaceflight", "astronomy-spaceflight", "science_nature", "telescope.fill"),
    132: ("Hazbin Hotel & Hellaverse Fandom", "hellaverse-fandom", "film_tv_anime", "flame.fill"),
    133: ("Hungarian Bluesky Community", "hungarian-community", "world_languages", "globe.europe.africa.fill"),
    134: ("New Orleans Culture & Saints", "new-orleans-nola", "cities_craft_food", "music.quarternote.3"),
    136: ("Survivor, Big Brother & Reality Strategy", "survivor-big-brother", "film_tv_anime", "flame.circle.fill"),
    137: ("Touhou Project & Japanese Pixel Indie", "touhou-jp-pixel", "games_interactive", "sparkles"),
    138: ("Nijisanji & Japanese Streamer Fan Art", "nijisanji-jp-streamers", "film_tv_anime", "play.tv.fill"),
    139: ("Buffalo Bills & Sabres Supporters", "buffalo-bills-sabres", "sports_motorsport", "american.football.fill"),
    150: ("Croatian Bluesky Community", "croatian-community", "world_languages", "globe.europe.africa.fill"),
}

# Rich subcultural concept synonyms injected when title/description match key themes
CONCEPT_SYNONYM_RULES: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\brust programming\b|\brustlang\b", re.I), ["cargo", "tokio", "systems programming", "borrow checker", "crates", "rustlang"]),
    (re.compile(r"\bios\b|\bswift developers?\b|\bswiftui\b|\bapple-platform\b", re.I), ["swiftui", "xcode", "apple developer", "macos", "indie app", "uikit"]),
    (re.compile(r"\bnetwork engineering\b|\bcloud infrastructure\b|\bsysadmins?\b|\bfree software\b", re.I),
     ["homelab", "self-hosted", "linux", "kubernetes", "proxmox", "tailscale", "nixos", "devops", "sysadmin", "docker", "nas"]),
    (re.compile(r"\btools for thought\b|\blocal-first\b", re.I),
     ["obsidian", "pkms", "crdt", "sync", "homelab", "self-hosted", "notetaking"]),
    (re.compile(r"\bhome cooking\b|\bsourdough\b|\bfood writers\b", re.I),
     ["coffee", "espresso", "pourover", "baking", "culinary", "recipes", "kitchen", "fermentation"]),
    (re.compile(r"\beconomists?\b|\beconsky\b|\bfinance and markets\b", re.I),
     ["macro", "macroeconomics", "fed", "inflation", "monetary policy", "fintwit", "gdp", "labor market"]),
    (re.compile(r"\bentomology\b|\bnature & street photographers\b|\bnature and wildlife\b", re.I),
     ["macro", "macro photography", "insects", "bugs", "wildlife lens", "close-up"]),
    (re.compile(r"\bdogs of bluesky\b|\bterrier\b", re.I),
     ["lab", "labrador", "golden retriever", "puppies", "canine", "doggo"]),
    (re.compile(r"\burbanism\b|\bpublic transport\b|\brailway\b", re.I),
     ["transit", "walkable cities", "trains", "bike lanes", "yimby", "housing", "metro", "subways", "headways"]),
    (re.compile(r"\bindependent musicians\b|\bindie rock\b|\bpost-punk\b|\bexperimental electronic\b", re.I),
     ["bandcamp", "vinyl", "synths", "modular", "tape", "album", "diy music"]),
    (re.compile(r"\bpixel art\b|\bgodot\b|\bindie game\b|\btech artists\b", re.I),
     ["gamedev", "shaders", "aseprite", "love2d", "bevy", "unity", "unreal", "sprites", "solo dev"]),
    (re.compile(r"\bat protocol\b|\bbluesky team\b", re.I),
     ["atproto", "lexicon", "pds", "did", "xrpc", "decentralized social", "fediverse"]),
]

STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "have", "has", "had",
    "not", "but", "you", "your", "all", "can", "her", "his", "they", "their", "out", "about",
    "who", "what", "when", "where", "why", "how", "just", "like", "more", "some", "into",
    "over", "also", "new", "one", "two", "its", "our", "many", "most", "other", "such",
    "posting", "posts", "accounts", "account", "community", "bluesky", "fans", "fan",
    "general", "chatter", "sharing", "sharers", "http", "https", "www", "com", "org", "bsky",
}
WORD_RE = re.compile(r"[a-z0-9#+._-]{3,24}")


def tokenize_clean(text: str) -> list[str]:
    return [
        w.strip("._-")
        for w in WORD_RE.findall(text.lower())
        if len(w.strip("._-")) >= 3 and w.strip("._-") not in STOPWORDS
    ]


def classify_member(
    meta: dict[str, Any],
    sig: dict[str, Any] | None,
) -> tuple[str, str, int, int, int, str]:
    """Return (role, recommendedMode, repostShareBps, mediaShareBps, estPostsPerWeek, whyPicked)."""
    lf = meta.get("lf") or {}
    acct_type = str((sig or {}).get("claude_acct_type") or meta.get("at") or "")
    c5_type = str((sig or {}).get("c5_pred_type") or "")
    c5_brand = float((sig or {}).get("c5_p_brand") or 0.0)
    curator_boost = float((sig or {}).get("curator_boost") or 1.0)
    img_frac = float((sig or {}).get("slab_img_frac") if sig and sig.get("slab_img_frac") is not None else (meta.get("img") or 0.0))
    ppd = float((sig or {}).get("posts_per_day") if sig and sig.get("posts_per_day") is not None else (meta.get("ppd") or 0.6))
    pq_posts = float((sig or {}).get("pq_posts") or 0.0)
    pq_orig = float((sig or {}).get("pq_originals") or 0.0)
    pq_mutual = float((sig or {}).get("pq_mutual_share") or 0.0)

    # Estimate repost fraction: prefer live hydrated repost_frac if n_items >= 5, else pre-cutoff non-original share
    live_rp = lf.get("rp")
    if live_rp is not None:
        repost_frac = float(live_rp)
    elif pq_posts >= 10:
        repost_frac = max(0.0, min(0.85, (1.0 - pq_orig / max(pq_posts, 1.0)) * 0.55))
    else:
        repost_frac = 0.12

    reply_frac = float(lf.get("ry")) if lf.get("ry") is not None else (0.32 if pq_mutual >= 0.35 else 0.15)

    repost_bps = int(round(max(0.0, min(1.0, repost_frac)) * 10000))
    media_bps = int(round(max(0.0, min(1.0, img_frac)) * 10000))
    est_ppw = int(round(max(1.0, min(350.0, ppd * 7.0))))

    # 5-role classification calibrated against real candidate_signals quantiles:
    # - project_or_studio: organizations, news/publications, studios, brands
    # - visual_maker: media-first artists, photographers, makers (img >= 35%, or creator_or_artist with img >= 18%)
    # - scene_curator: heavy reposters/highlighters (>=42% reposts) or top-decile curator_boost (>=550) with >=24% reposts
    # - conversationalist: high mutual/reply community regulars
    # - anchor_creator: original essayists, thread authors, and core domain voices
    org_types = {"project_or_studio", "organization", "news_media", "brand_or_commercial", "media_outlet", "brand_or_org"}
    if c5_type in org_types or acct_type in org_types or c5_brand >= 0.42:
        role = "project_or_studio"
    elif media_bps >= 3500 or (
        (c5_type == "creator_or_artist" or acct_type == "creator_or_artist") and media_bps >= 1800
    ):
        role = "visual_maker"
    elif repost_bps >= 4200 or (curator_boost >= 550.0 and repost_bps >= 2200):
        role = "scene_curator"
    elif reply_frac >= 0.26 or pq_mutual >= 0.36 or (ppd >= 2.8 and media_bps < 2000):
        role = "conversationalist"
    else:
        role = "anchor_creator"

    mode = "originals_only" if repost_bps >= 4500 else "follow_all"

    # Crisp, non-PII algorithmic "Why picked" explanation
    if role == "visual_maker":
        pct = max(42, min(99, round(media_bps / 100)))
        why = f"Visual maker · {pct}% media posts · ~{est_ppw}/wk"
    elif role == "scene_curator":
        if mode == "originals_only":
            why = f"Scene curator · shares community highlights · Originals Only tames reposts"
        else:
            why = f"Scene curator · surfaces standout voices · ~{est_ppw}/wk"
    elif role == "project_or_studio":
        why = f"Project, studio, or publication feed · ~{est_ppw}/wk"
    elif role == "conversationalist":
        why = f"Conversational regular · active community threads · ~{est_ppw}/wk"
    else:
        why = f"Core scene anchor · strong original posts · ~{est_ppw}/wk"

    if mode == "originals_only" and "Originals Only" not in why:
        why = f"{why} · Originals Only suggested"

    return role, mode, repost_bps, media_bps, est_ppw, why[:108]


def build_bundle() -> dict[str, Any]:
    cards_v1 = json.loads((S17_DIR / "cards_v1.json").read_text())
    descs = json.loads((S17_DIR / "cards_descriptions.json").read_text())
    insp = json.loads((S17_DIR / "inspector_index.json").read_text())

    # Safety & exclusion sets
    purge_dids = set((LAKE / "s5-sweep-20261030/purge_dids.txt").read_text().splitlines())
    non_org = set((S17_DIR / "non_organic_exclude_dids.txt").read_text().splitlines())
    overrides = json.loads((S17_DIR / "manual_overrides.json").read_text())
    manual_excl = set(overrides.get("excluded_dids", []))

    prof = pq.read_table(
        LAKE / "s5-sweep-20261030/profiles.parquet",
        columns=["did", "hide_from_recs", "any_mod_label", "no_unauthenticated"],
    )
    hfr_dids: set[str] = set()
    mod_dids: set[str] = set()
    no_unauth_all: set[str] = set()
    for d, hfr, mod, nou in zip(
        prof["did"].to_pylist(),
        prof["hide_from_recs"].to_pylist(),
        prof["any_mod_label"].to_pylist(),
        prof["no_unauthenticated"].to_pylist(),
    ):
        if hfr:
            hfr_dids.add(d)
        if mod:
            mod_dids.add(d)
        if nou:
            no_unauth_all.add(d)

    unsafe_dids = hfr_dids | purge_dids | mod_dids | non_org | manual_excl

    # Expand S17 account table (25,400 accounts)
    acct_dids = [expand_did(a[0]) for a in cards_v1["accts"]]
    acct_followers = [int(a[2]) for a in cards_v1["accts"]]
    acct_meta_list: list[dict[str, Any]] = insp["acct_meta"]

    # Load candidate_signals for catalog DIDs
    catalog_did_set = set(acct_dids)
    sig_tbl = pq.read_table(
        S17_DIR / "candidate_signals.parquet",
        columns=[
            "did",
            "slab_img_frac",
            "posts_per_day",
            "curator_boost",
            "pq_posts",
            "pq_originals",
            "pq_mutual_share",
            "claude_acct_type",
            "c5_pred_type",
            "c5_p_brand",
            "s13_score",
        ],
    )
    sig_by_did: dict[str, dict[str, Any]] = {}
    sig_cols = {col: sig_tbl[col].to_pylist() for col in sig_tbl.column_names}
    for idx, d in enumerate(sig_cols["did"]):
        if d in catalog_did_set:
            sig_by_did[d] = {col: sig_cols[col][idx] for col in sig_tbl.column_names if col != "did"}

    # Load author_feeds.sqlite for clean post URI backfill + aggregate community TF-IDF terms
    con = sqlite3.connect(S17_DIR / "author_feeds.sqlite")
    sqlite_orig_uris: dict[str, list[str]] = {}
    sqlite_author_text: dict[str, str] = {}
    for did, pjson, sflag in con.execute(
        "SELECT did, posts_json, live_safety_flag FROM author_feeds WHERE http_status = 200"
    ):
        if sflag or did in unsafe_dids:
            continue
        arr = json.loads(pjson or "[]")
        uris: list[str] = []
        texts: list[str] = []
        for p in arr:
            if not isinstance(p, dict):
                continue
            u = str(p.get("uri") or "")
            if (
                u.startswith(f"at://{did}/app.bsky.feed.post/")
                and not p.get("is_repost")
                and not p.get("is_reply")
            ):
                uris.append(u)
                t = str(p.get("text") or "")
                if t:
                    texts.append(t[:240])
        if uris:
            sqlite_orig_uris[did] = uris[:6]
        if texts:
            sqlite_author_text[did] = " ".join(texts[:8])
    con.close()

    # Load medoids.parquet for catalog DIDs
    med_tbl = pq.read_table(LAKE / "s12-closeout-20261030/medoids.parquet", columns=["did", "uris"])
    med_by_did: dict[str, list[str]] = {}
    for d, uris in zip(med_tbl["did"].to_pylist(), med_tbl["uris"].to_pylist()):
        if d in catalog_did_set and d not in unsafe_dids and uris:
            clean_u = [u for u in uris if isinstance(u, str) and u.startswith(f"at://{d}/app.bsky.feed.post/")]
            if clean_u:
                med_by_did[d] = clean_u[:3]

    # Load oct22_preview_posts.parquet for preview pools & medoid backfill
    oct_tbl = pq.read_table(
        S16_DIR / "oct22_preview_posts.parquet",
        columns=["community", "uri", "did", "weighted_residual", "politicalness", "heat"],
    )
    oct_by_cid: dict[int, list[tuple[str, str, float]]] = collections.defaultdict(list)
    oct_by_did: dict[str, list[str]] = collections.defaultdict(list)
    for cid, uri, did, wres, pol, heat in zip(
        oct_tbl["community"].to_pylist(),
        oct_tbl["uri"].to_pylist(),
        oct_tbl["did"].to_pylist(),
        oct_tbl["weighted_residual"].to_pylist(),
        oct_tbl["politicalness"].to_pylist(),
        oct_tbl["heat"].to_pylist(),
    ):
        if did in unsafe_dids or not str(uri).startswith(f"at://{did}/app.bsky.feed.post/"):
            continue
        if (pol is not None and float(pol) > 0.45) or (heat is not None and float(heat) > 0.45):
            continue
        oct_by_cid[int(cid)].append((str(uri), str(did), float(wres or 0.0)))
        if did in catalog_did_set and len(oct_by_did[did]) < 4:
            oct_by_did[did].append(str(uri))

    # Load 256d centroids (card_centroids.npz)
    npz = np.load(S17_DIR / "card_centroids.npz")
    cid_to_cent_idx = {int(c): i for i, c in enumerate(npz["cids"].tolist())}
    cent_posts = npz["cent_posts"].astype(np.float32)

    # First pass: compute document frequency of aggregate community terms across 1,016 communities
    comm_raw_tokens: dict[int, collections.Counter[str]] = {}
    df_counter: collections.Counter[str] = collections.Counter()
    for cid_str, c in cards_v1["cards"].items():
        cid = int(cid_str)
        member_dids = [acct_dids[m] for m in c["m"] if acct_dids[m] not in unsafe_dids]
        agg_text = " ".join(sqlite_author_text[d] for d in member_dids if d in sqlite_author_text)
        cnt = collections.Counter(tokenize_clean(agg_text))
        comm_raw_tokens[cid] = cnt
        for tok, freq in cnt.items():
            if freq >= 2:
                df_counter[tok] += 1

    # Build scenes list
    scenes_out: list[dict[str, Any]] = []
    scene_lookup: dict[int, dict[str, Any]] = {}
    for sc in cards_v1["scenes"]:
        sid = int(sc["s"])
        if sid in SCENE_TAXONOMY_104:
            title, slug, macro, sym = SCENE_TAXONOMY_104[sid]
        else:
            first_c = cards_v1["cards"][str(sc["cs"][0])]
            title = f"{first_c['t']} & Related"
            slug = f"scene-{sid}"
            macro = "queer_fandom_hangout"
            sym = "sparkles"
        valid_cids = [int(x) for x in sc["cs"] if str(x) in cards_v1["cards"]]
        total_el = sum(int(cards_v1["cards"][str(x)]["el"]) for x in valid_cids)
        entry = {
            "sceneId": sid,
            "sceneSlug": slug,
            "sceneTitle": title,
            "macroCategory": macro,
            "symbolName": sym,
            "cardCount": len(valid_cids),
            "totalEligibleAccounts": total_el,
            "topCommunityIds": valid_cids[:12],
        }
        scenes_out.append(entry)
        scene_lookup[sid] = entry

    # Build 1,016 communities
    communities_out: list[dict[str, Any]] = []
    referenced_no_unauth: set[str] = set()

    for cid_str, c in cards_v1["cards"].items():
        cid = int(cid_str)
        sid = int(c["s"])
        sc_meta = scene_lookup[sid]
        desc_obj = descs.get(cid_str) or {}
        description = str(desc_obj.get("d") or c["t"]).strip()
        health = insp["card_health"].get(cid_str) or {}

        coh_posts_bps = int(round(max(0.0, min(1.0, float(health.get("coh_posts") or 0.65))) * 10000))
        coh_social_bps = int(round(max(0.0, min(1.0, float(health.get("coh_t1") or 0.65))) * 10000))

        all_lists = health.get("all_lists") or c.get("ln") or []
        curated_tags = [str(item[0]).strip() for item in all_lists if isinstance(item, list) and len(item) >= 2 and int(item[1]) >= 2][:8]

        # Members (strictly filtered against unsafe_dids)
        members_out: list[dict[str, Any]] = []
        raw_member_indices = [int(m) for m in c["m"]]
        for rank_idx, aid in enumerate(raw_member_indices):
            did = acct_dids[aid]
            if did in unsafe_dids:
                continue
            if did in no_unauth_all:
                referenced_no_unauth.add(did)
            meta = acct_meta_list[aid]
            sig = sig_by_did.get(did)
            role, mode, rp_bps, media_bps, est_ppw, why = classify_member(meta, sig)
            # Normalized rank score in basis points (top member ~9800, 25th ~6200)
            s13_raw = float((sig or {}).get("s13_score") or meta.get("s13") or 100.0)
            decay = max(0.55, 1.0 - 0.016 * rank_idx)
            rank_bps = int(round(min(9950.0, max(5000.0, (6800.0 + min(3000.0, math.log1p(max(s13_raw, 1.0)) * 340.0)) * decay))))

            # Pick 1 clean sample post URI for this member if available
            m_post_uri: str | None = None
            for cand_u in (
                ([f"at://{did}/app.bsky.feed.post/{rk}" for rk in (meta.get("md") or []) if rk])
                + sqlite_orig_uris.get(did, [])
                + med_by_did.get(did, [])
                + oct_by_did.get(did, [])
            ):
                if cand_u.startswith(f"at://{did}/app.bsky.feed.post/"):
                    m_post_uri = cand_u
                    break

            members_out.append({
                "did": did,
                "followers": acct_followers[aid],
                "role": role,
                "recommendedMode": mode,
                "whyPicked": why,
                "rankScoreBps": rank_bps,
                "repostShareBps": rp_bps,
                "mediaShareBps": media_bps,
                "estimatedPostsPerWeek": est_ppw,
                "samplePostUri": m_post_uri,
            })

        # Medoid backfill -> guarantee >= 3 clean samplePostUris from distinct safe authors when available
        sample_uris: list[str] = []
        used_authors: set[str] = set()

        def try_add_sample(uri: str) -> None:
            if len(sample_uris) >= 4 or not uri.startswith("at://"):
                return
            parts = uri.split("/")
            if len(parts) != 5 or parts[3] != "app.bsky.feed.post":
                return
            owner = parts[2]
            if owner in unsafe_dids or owner in used_authors or uri in sample_uris:
                return
            if owner in no_unauth_all:
                referenced_no_unauth.add(owner)
            sample_uris.append(uri)
            used_authors.add(owner)

        # 1. Existing card medoids
        for pair in c.get("md") or []:
            if isinstance(pair, list) and len(pair) == 2:
                aid, rkey = int(pair[0]), str(pair[1])
                if 0 <= aid < len(acct_dids) and rkey:
                    try_add_sample(f"at://{acct_dids[aid]}/app.bsky.feed.post/{rkey}")
        # 2. Member samplePostUris (sqlite / medoids / oct22)
        for m in members_out:
            if m["samplePostUri"]:
                try_add_sample(m["samplePostUri"])
        # 3. Community-level oct22 preview posts
        for uri, owner, _wres in sorted(oct_by_cid.get(cid, []), key=lambda x: -x[2]):
            try_add_sample(uri)
        # 4. Relax 1-per-author cap if still < 3
        if len(sample_uris) < 3:
            for m in members_out:
                d = m["did"]
                for u in sqlite_orig_uris.get(d, []) + med_by_did.get(d, []) + oct_by_did.get(d, []):
                    if u not in sample_uris and len(sample_uris) < 3:
                        sample_uris.append(u)

        # Build rich previewPostUris pool (up to 16 clean URIs, <= 2 per author)
        preview_uris: list[str] = []
        preview_author_counts: collections.Counter[str] = collections.Counter()

        def try_add_preview(uri: str) -> None:
            if len(preview_uris) >= 16 or uri in preview_uris:
                return
            parts = uri.split("/")
            if len(parts) != 5 or parts[3] != "app.bsky.feed.post":
                return
            owner = parts[2]
            if owner in unsafe_dids or preview_author_counts[owner] >= 2:
                return
            if owner in no_unauth_all:
                referenced_no_unauth.add(owner)
            preview_uris.append(uri)
            preview_author_counts[owner] += 1

        for u in sample_uris:
            try_add_preview(u)
        for uri, _owner, _wres in sorted(oct_by_cid.get(cid, []), key=lambda x: -x[2]):
            try_add_preview(uri)
        for m in members_out:
            d = m["did"]
            for u in sqlite_orig_uris.get(d, [])[:2] + med_by_did.get(d, [])[:2]:
                try_add_preview(u)

        # Feeds & Starter Packs (expanded AT-URIs, safe owners only)
        feed_uris: list[str] = []
        for fd in c.get("fd") or []:
            if isinstance(fd, list) and fd:
                u = expand_feed_uri(str(fd[0]))
                owner = u.split("/")[2]
                if owner not in unsafe_dids:
                    if owner in no_unauth_all:
                        referenced_no_unauth.add(owner)
                    feed_uris.append(u)

        pack_uris: list[str] = []
        for pk in c.get("pk") or []:
            if isinstance(pk, list) and pk:
                u = expand_pack_uri(str(pk[0]))
                owner = u.split("/")[2]
                if owner not in unsafe_dids:
                    if owner in no_unauth_all:
                        referenced_no_unauth.add(owner)
                    pack_uris.append(u)

        neighbors = [int(n) for n in (c.get("nb") or []) if str(n) in cards_v1["cards"]][:6]

        # Quantize 256d cent_posts vector to int8 (-127..127)
        cidx = cid_to_cent_idx.get(cid)
        if cidx is not None:
            vec = cent_posts[cidx]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec = vec / norm
            qvec = np.clip(np.round(vec * 127.0), -127, 127).astype(np.int8)
            cent_b64 = base64.b64encode(qvec.tobytes()).decode("ascii")
        else:
            cent_b64 = base64.b64encode(np.zeros(256, dtype=np.int8).tobytes()).decode("ascii")

        # Mine top aggregate TF-IDF terms + concept synonyms for hybrid search
        tfidf_scored: list[tuple[float, str]] = []
        for tok, freq in comm_raw_tokens[cid].items():
            if freq < 2:
                continue
            df = df_counter[tok]
            if df > 180:
                continue
            score = (1.0 + math.log(freq)) * math.log(1017.0 / (1.0 + df))
            tfidf_scored.append((score, tok))
        tfidf_scored.sort(reverse=True)
        mined_terms = [tok for _, tok in tfidf_scored[:18]]

        blob_for_rules = f"{c['t']} {description} {' '.join(curated_tags)}"
        synonyms: list[str] = []
        for pat, syns in CONCEPT_SYNONYM_RULES:
            if pat.search(blob_for_rules):
                for s_tok in syns:
                    if s_tok not in synonyms:
                        synonyms.append(s_tok)

        communities_out.append({
            "communityId": cid,
            "title": str(c["t"]),
            "description": description,
            "sceneId": sid,
            "sceneSlug": sc_meta["sceneSlug"],
            "sceneTitle": sc_meta["sceneTitle"],
            "macroCategory": sc_meta["macroCategory"],
            "symbolName": sc_meta["symbolName"],
            "eligibleCount": int(c["el"]),
            "postCoherenceBps": coh_posts_bps,
            "socialCoherenceBps": coh_social_bps,
            "curatedListTags": curated_tags,
            "neighborCommunityIds": neighbors,
            "samplePostUris": sample_uris[:3],
            "previewPostUris": preview_uris,
            "sampleAccountDids": [m["did"] for m in members_out[:6]],
            "feedGeneratorUris": feed_uris[:3],
            "starterPackUris": pack_uris[:2],
            "searchTerms": synonyms[:14] + mined_terms[:14],
            "centPostsInt8B64": cent_b64,
            "members": members_out,
        })

    comm_by_id = {c["communityId"]: c for c in communities_out}

    # Build Contrastive Taste Duels:
    # Part A: 8 classic ambiguity / collision traps grounded in real S17 rev 6 communities
    classic_traps = [
        (
            "trap_1_rust",
            "When you say 'Rust' or systems craft, which world belongs on your feed?",
            2294,
            "Rust, Cargo, Tokio & Systems Programming",
            651,
            "Indie Game Dev, Survival Worlds & Game Craft",
        ),
        (
            "trap_2_swift",
            "Which 'Swift' timeline are you actually looking for?",
            497,
            "iOS/macOS App Craft, SwiftUI & Indie Apple Devs",
            0,
            "Taylor Swift Fandom, Eras Tour & Pop StanSky",
        ),
        (
            "trap_3_macro",
            "Pick the 'Macro' lens you love reading:",
            207,
            "Macroeconomics, Applied Econ & Policy Research",
            240,
            "Entomology, Close-Up Bugs & Wildlife Science",
        ),
        (
            "trap_4_lab",
            "Which 'Lab' fits your morning scroll?",
            1510,
            "Network Engineering, Homelabs & Infrastructure",
            269,
            "Dogs of Bluesky, Retrievers & Pet Companions",
        ),
        (
            "trap_5_humor",
            "What register of writing & wit fits your vibe?",
            383,
            "Weird Internet Shitposting & Post-Ironic Jokes",
            163,
            "Haiku, Micropoetry & Reflective Literary Notes",
        ),
        (
            "trap_6_morning",
            "Saturday morning energy check:",
            2311,
            "Sourdough Baking, Espresso & Home Cooking",
            431,
            "Walkable Cities, Bike Lanes & Transit Headways",
        ),
        (
            "trap_7_football",
            "Matchday preference:",
            143,
            "World Football, Premier League & FPL Tactics",
            140,
            "NBA Basketball, Analytics & North American Sports",
        ),
        (
            "trap_8_indie",
            "Which indie creative scene pulls you in more?",
            587,
            "Pixel Art, Shaders & Solo Indie Game Makers",
            808,
            "Indie Rock, Post-Punk & Bandcamp Releases",
        ),
    ]

    duels_out: list[dict[str, Any]] = []
    for idx, (sw_id, prompt, cid_a, sub_a, cid_b, sub_b) in enumerate(classic_traps, start=1):
        ca = comm_by_id[cid_a]
        cb = comm_by_id[cid_b]
        duels_out.append({
            "swipeId": sw_id,
            "step": idx,
            "kind": "collision_trap",
            "prompt": prompt,
            "optionA": {
                "communityId": cid_a,
                "title": ca["title"],
                "subtitle": sub_a,
                "sceneTitle": ca["sceneTitle"],
                "samplePostUris": ca["samplePostUris"][:2],
            },
            "optionB": {
                "communityId": cid_b,
                "title": cb["title"],
                "subtitle": sub_b,
                "sceneTitle": cb["sceneTitle"],
                "samplePostUris": cb["samplePostUris"][:2],
            },
        })

    # Part B: Dynamic sub-scene splitters across multi-card scenes
    seen_pairs: set[tuple[int, int]] = {(min(a[2], a[4]), max(a[2], a[4])) for a in classic_traps}
    for sc in sorted(scenes_out, key=lambda x: -x["totalEligibleAccounts"]):
        cids = [cid for cid in sc["topCommunityIds"] if cid in comm_by_id and len(comm_by_id[cid]["samplePostUris"]) >= 2]
        if len(cids) < 2:
            continue
        ca = comm_by_id[cids[0]]
        # Pick a sibling card in the same scene with a distinct title
        cb = None
        for cand_cid in cids[1:]:
            cand = comm_by_id[cand_cid]
            pair_key = (min(ca["communityId"], cand_cid), max(ca["communityId"], cand_cid))
            if cand["title"].lower() != ca["title"].lower() and pair_key not in seen_pairs:
                cb = cand
                seen_pairs.add(pair_key)
                break
        if cb is None:
            continue
        duels_out.append({
            "swipeId": f"split_s{sc['sceneId']}_{ca['communityId']}_{cb['communityId']}",
            "step": len(duels_out) + 1,
            "kind": "subscene_split",
            "prompt": f"Inside {sc['sceneTitle']}, which corner feels more like home?",
            "optionA": {
                "communityId": ca["communityId"],
                "title": ca["title"],
                "subtitle": ca["description"][:96],
                "sceneTitle": ca["sceneTitle"],
                "samplePostUris": ca["samplePostUris"][:2],
            },
            "optionB": {
                "communityId": cb["communityId"],
                "title": cb["title"],
                "subtitle": cb["description"][:96],
                "sceneTitle": cb["sceneTitle"],
                "samplePostUris": cb["samplePostUris"][:2],
            },
        })
        if len(duels_out) >= 64:
            break

    # Build compact Warm-Start Orbit Index:
    # Map every catalog member DID (plus top rank-1 known_for producers in active communities) -> [(cid, weightBps)]
    active_cid_set = set(comm_by_id.keys())
    warm_start: dict[str, list[list[int]]] = collections.defaultdict(list)
    for comm in communities_out:
        cid = comm["communityId"]
        for m in comm["members"]:
            warm_start[m["did"]].append([cid, m["rankScoreBps"]])

    kf_tbl = pq.read_table(
        LAKE / "s3/matched/cand5-full-matched/known_for.parquet",
        columns=["did", "community", "score", "rank", "hide_from_recs"],
    )
    extra_added = 0
    for d, cid, sc_val, rk, hfr in zip(
        kf_tbl["did"].to_pylist(),
        kf_tbl["community"].to_pylist(),
        kf_tbl["score"].to_pylist(),
        kf_tbl["rank"].to_pylist(),
        kf_tbl["hide_from_recs"].to_pylist(),
    ):
        if extra_added >= 45000:
            break
        cid_i = int(cid)
        if hfr or rk > 1 or cid_i not in active_cid_set or d in unsafe_dids or d in warm_start:
            continue
        bps = int(round(max(2500.0, min(9000.0, float(sc_val) * 10000.0))))
        warm_start[d].append([cid_i, bps])
        extra_added += 1

    # High hostility pairs within active 1,016 communities
    high_h_pairs = [
        [int(a), int(b)]
        for a, b in (cards_v1.get("high_h_pairs") or [])
        if int(a) in active_cid_set and int(b) in active_cid_set
    ]

    bundle = {
        "catalogVersion": CATALOG_VERSION,
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sourceCardsSha256": hashlib.sha256((S17_DIR / "cards_v1.json").read_bytes()).hexdigest(),
        "totalScenes": len(scenes_out),
        "totalCommunities": len(communities_out),
        "scenes": scenes_out,
        "communities": communities_out,
        "tasteDuels": duels_out,
        "highHostilityPairs": high_h_pairs,
        "noUnauthDids": sorted(referenced_no_unauth),
        "warmStartIndex": dict(warm_start),
    }

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")))
    tmp.replace(OUT_PATH)
    print(
        f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.2f} MB): "
        f"{len(scenes_out)} scenes, {len(communities_out)} communities, "
        f"{sum(len(c['members']) for c in communities_out)} member entries, "
        f"{sum(1 for c in communities_out if len(c['samplePostUris']) >= 3)}/1016 >=3 sample posts, "
        f"{len(duels_out)} taste duels, {len(warm_start)} warm-start DIDs."
    )
    return bundle


if __name__ == "__main__":
    build_bundle()
