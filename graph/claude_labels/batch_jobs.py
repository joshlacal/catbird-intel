"""Submit and collect Claude Message Batches for the community, profile and hostile-stance label jobs.

  batch_jobs.py submit JOB INPUT OUT_DIR [--offset N] [--limit N] [--exclude DIDS_TXT]
  batch_jobs.py collect BATCH_ID OUT_DIR

JOB is one of JOBS below. INPUT is community_prompts.jsonl (community*), profile_inputs.parquet
(profile_*: pid, did, name, description, community_label) or reply_pairs.parquet (hostile_*). --exclude drops pairs whose parent or child author is listed (applied again at
submit time so the final opt-out union is honoured even if INPUT was built earlier).
submit writes OUT_DIR/<batch_id>.submit.json (job, model, custom_ids, input sha256);
collect writes OUT_DIR/<batch_id>.results.jsonl and a usage/cost summary at Batch API prices.
The key is read from ANTHROPIC_API_KEY or the workspace .env and is never printed.
"""
import argparse
import hashlib
import json
import os
import pathlib

import anthropic
import pyarrow.parquet as pq
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

ENV = pathlib.Path(__file__).resolve().parents[3] / ".env"
# Batch API prices, $ per 1M tokens (input, output): half of list.
PRICES = {"claude-opus-5-5": (2.0, 10.0), "claude-sonnet-5-5": (1.0, 5.0), "claude-haiku-5-5": (0.05, 0.25)}

SENSITIVE = ["news", "politics", "religion", "sexuality_gender", "health", "ethnicity_nationality", "adult", "likely_minors"]
COMMUNITY_TYPES = ["topical_scene", "fandom", "professional_or_academic", "regional_or_local", "language_general",
                   "follow_back_circle", "follow_train_or_political_amplification", "adult_content", "roleplay",
                   "spam_or_commercial", "mixed_or_unclear"]

COMMUNITY_SYSTEM = """You name communities of Bluesky accounts. Each community was found by clustering who-follows-whom among Bluesky's ~2M most-followed accounts. A new user will pick communities ("scenes") during onboarding and get a bundle of accounts to follow, so the name must say what the accounts actually post about, in words a member would recognise.

You get: the community's size, the languages of a 4-day post sample, the most common names of curated lists that contain its members, the most-followed members' display names and bios, and sample recent posts. Lists named like "follow back", "mutuals", "favs" or "crushes" say little about topic. Judge from the evidence only; don't guess beyond it.

Fields:
- label: 2-5 words in English naming the scene (e.g. "Spanish gay Twitter refugees", "UK ornithology", "Splatoon fan artists").
- label_native: the same label in the community's main language when that isn't English, else "".
- description: one sentence on who is in it and what they post.
- primary_language: ISO 639-1 code, or "multi".
- community_type: the best fit. follow_back_circle = accounts that mainly follow each other for follower counts; follow_train_or_political_amplification = mass mutual-follow activism or follow trains; adult_content = mainly sexual or explicit-content accounts; roleplay = character or RP accounts.
- sensitivity: every class that applies to membership itself (being listed as a member would reveal it): news, politics, religion, sexuality_gender, health, ethnicity_nationality, adult, likely_minors. Never infer ethnicity_nationality from language alone. Use likely_minors only for explicit age or school-age cues.
- onboarding_fit: good = a newcomer who picks this gets on-topic, active, real accounts; ok = usable with filtering; poor = circle, spam, adult or incoherent.
- evidence: up to 5 short quotes or terms from the input that support the label.
- confidence: high, medium or low."""

HOSTILE_SYSTEM = """You label the stance of a Bluesky post (the CHILD) that replies to or quotes another post (the PARENT). Labels are used in aggregate to measure how often replies and quotes are hostile; nothing is shown to users.

Fields:
- stance: supportive (agrees, praises, jokes along); neutral (informational, a question, unrelated chatter); civil_disagreement (disagrees without insult); hostile (insults, mocks, threatens, harasses or dunks on the parent or its author); unclear (not enough context, or not understandable).
- dunk: true when the child holds the parent up for ridicule for an audience (typical of quote posts), else false.
- target: who any negativity is aimed at: parent_author, parent_content, third_party or none.
- confidence: high, medium or low.
Posts may be in any language. Judge only what is written; sarcasm counts if it is clear."""

ACCOUNT_TYPES = ["person", "organization", "news_media", "brand_or_commercial", "creator_or_artist", "adult_creator",
                 "bot_or_automated", "fan_or_roleplay", "unclear"]

PROFILE_SYSTEM = """You classify Bluesky accounts from their display name and bio. Labels train a small classifier used in aggregate for account recommendations; nothing is shown to users. You may also get the label of the follow-graph community the account belongs to, which is context, not proof.

Fields:
- account_type: person (an individual posting as themselves); organization (non-commercial group, institution, government, school, NGO); news_media (outlet, journalist account run as an outlet, news aggregator); brand_or_commercial (company, shop, product, paid service, promoter); creator_or_artist (individual whose account mainly promotes their art, writing, music, streams or other work); adult_creator (sexual or explicit content, including links to paid adult content); bot_or_automated (feeds, reposters, automated or generated posting); fan_or_roleplay (fan account, character or roleplay account, parody); unclear (not enough to tell).
- primary_language: ISO 639-1 code of the bio, "multi", or "" when there is no text.
- topics: up to 3 short English topic tags the account is about (e.g. "ornithology", "US politics", "K-pop"), or [].
- sensitivity: every class the bio itself reveals: news, politics, religion, sexuality_gender, health, ethnicity_nationality, adult, likely_minors. Never infer ethnicity_nationality from language alone. Use likely_minors only for an explicit age under 18 or school-age cues.
- confidence: high, medium or low.
Bios may be in any language and may use stylised Unicode letters."""

COMMUNITY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["label", "label_native", "description", "primary_language", "community_type",
                 "sensitivity", "onboarding_fit", "evidence", "confidence"],
    "properties": {
        "label": {"type": "string"}, "label_native": {"type": "string"}, "description": {"type": "string"},
        "primary_language": {"type": "string"},
        "community_type": {"type": "string", "enum": COMMUNITY_TYPES},
        "sensitivity": {"type": "array", "items": {"type": "string", "enum": SENSITIVE}},
        "onboarding_fit": {"type": "string", "enum": ["good", "ok", "poor"]},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}
HOSTILE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["stance", "dunk", "target", "confidence"],
    "properties": {
        "stance": {"type": "string", "enum": ["supportive", "neutral", "civil_disagreement", "hostile", "unclear"]},
        "dunk": {"type": "boolean"},
        "target": {"type": "string", "enum": ["parent_author", "parent_content", "third_party", "none"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}
PROFILE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["account_type", "primary_language", "topics", "sensitivity", "confidence"],
    "properties": {
        "account_type": {"type": "string", "enum": ACCOUNT_TYPES},
        "primary_language": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "sensitivity": {"type": "array", "items": {"type": "string", "enum": SENSITIVE}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
}
JOBS = {
    "community_high": ("claude-opus-5-5", "high", 12000, COMMUNITY_SYSTEM, COMMUNITY_SCHEMA),
    "profile_haiku": ("claude-haiku-5-5", "low", 1000, PROFILE_SYSTEM, PROFILE_SCHEMA),
    "profile_sonnet": ("claude-sonnet-5-5", "low", 1000, PROFILE_SYSTEM, PROFILE_SCHEMA),
    "community": ("claude-opus-5-5", "medium", 8000, COMMUNITY_SYSTEM, COMMUNITY_SCHEMA),
    "hostile_sonnet": ("claude-sonnet-5-5", "low", 2000, HOSTILE_SYSTEM, HOSTILE_SCHEMA),
    "hostile_haiku": ("claude-haiku-5-5", "low", 2000, HOSTILE_SYSTEM, HOSTILE_SCHEMA),
}


def client():
    if "ANTHROPIC_API_KEY" not in os.environ:
        for line in ENV.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
    return anthropic.Anthropic()


def items(job, path, excl):
    if job.startswith("community"):
        for line in open(path):
            r = json.loads(line)
            yield r["custom_id"], r["prompt"]
        return
    t = pq.read_table(path)
    if job.startswith("profile"):
        for r in t.to_pylist():
            if r["did"] in excl:
                continue
            text = f"Display name: {(r['name'] or '').strip()[:100]}\nBio: {' '.join((r['description'] or '').split())[:600]}"
            if r.get("community_label"):
                text += f"\nFollow-graph community: {r['community_label']}"
            yield r["pid"], text
        return
    for r in t.to_pylist():
        if r["parent_did"] in excl or r["child_did"] in excl:
            continue
        how = "replies to" if r["kind"] == "reply" else "quotes"
        yield r["pair_id"], (f"PARENT:\n{' '.join(r['parent_text'].split())[:1000]}\n\n"
                             f"CHILD ({how} the parent):\n{' '.join(r['child_text'].split())[:1000]}")


def submit(a):
    model, effort, max_tokens, system, schema = JOBS[a.job]
    excl = {l.strip() for l in open(a.exclude) if l.strip()} if a.exclude else set()
    todo = list(items(a.job, a.input, excl))[a.offset:a.offset + a.limit if a.limit else None]
    reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(
        model=model, max_tokens=max_tokens, system=system,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": text}])) for cid, text in todo]
    batch = client().messages.batches.create(requests=reqs)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{batch.id}.submit.json").write_text(json.dumps({
        "batch_id": batch.id, "job": a.job, "model": model, "effort": effort, "n": len(reqs),
        "offset": a.offset, "limit": a.limit, "excluded_dids": len(excl),
        "input": str(a.input), "input_sha256": hashlib.sha256(open(a.input, "rb").read()).hexdigest(),
        "custom_ids": [c for c, _ in todo]}, indent=1))
    print(batch.id, batch.processing_status, len(reqs))


def collect(a):
    c = client()
    b = c.messages.batches.retrieve(a.batch_id)
    print(b.processing_status, b.request_counts)
    if b.processing_status != "ended":
        return
    out = pathlib.Path(a.out_dir)
    tot = {"input": 0, "output": 0, "cache_read": 0, "cost": 0.0, "by_type": {}, "stop": {}}
    with open(out / f"{a.batch_id}.results.jsonl", "w") as f:
        for r in c.messages.batches.results(a.batch_id):
            rec = {"custom_id": r.custom_id, "type": r.result.type}
            tot["by_type"][r.result.type] = tot["by_type"].get(r.result.type, 0) + 1
            if r.result.type == "succeeded":
                m = r.result.message
                u = m.usage
                pin, pout = PRICES[m.model]
                cost = (u.input_tokens + (u.cache_creation_input_tokens or 0) * 1.25
                        + (u.cache_read_input_tokens or 0) * 0.1) * pin / 1e6 + u.output_tokens * pout / 1e6
                tot["input"] += u.input_tokens
                tot["output"] += u.output_tokens
                tot["cache_read"] += u.cache_read_input_tokens or 0
                tot["cost"] += cost
                tot["stop"][m.stop_reason] = tot["stop"].get(m.stop_reason, 0) + 1
                text = next((x.text for x in m.content if x.type == "text"), "")
                rec.update(model=m.model, stop_reason=m.stop_reason, input_tokens=u.input_tokens,
                           output_tokens=u.output_tokens, cost=cost)
                try:
                    rec["label"] = json.loads(text) if m.stop_reason == "end_turn" else None
                except json.JSONDecodeError:
                    rec["raw"] = text
            else:
                rec["error"] = str(getattr(r.result, "error", ""))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = max(sum(tot["stop"].values()), 1)
    tot["per_request"] = {"input": tot["input"] / n, "output": tot["output"] / n, "cost": tot["cost"] / n}
    (out / f"{a.batch_id}.usage.json").write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot, indent=1))


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
s = sp.add_parser("submit")
s.add_argument("job", choices=JOBS)
s.add_argument("input")
s.add_argument("out_dir")
s.add_argument("--offset", type=int, default=0)
s.add_argument("--limit", type=int, default=0)
s.add_argument("--exclude")
c_ = sp.add_parser("collect")
c_.add_argument("batch_id")
c_.add_argument("out_dir")
a = ap.parse_args()
submit(a) if a.cmd == "submit" else collect(a)
