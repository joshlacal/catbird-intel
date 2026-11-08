#!/usr/bin/env python3
"""Ship cand5 Opus 5.5 + S17 rev 6 community classification metadata to netcup ClickHouse."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import duckdb

LAKE = os.path.expanduser("~/catbird-lake-local")
LABELS_PARQUET = f"{LAKE}/claude-labels-cand5-20261028/community_labels.parquet"
S17_BUNDLE = f"{LAKE}/s17-picker-v1-20261030/intel_serving_v2.json"
SQL_SCHEMA = os.path.join(os.path.dirname(__file__), "../sql/006_community_meta.sql")

FURRY_RE = re.compile(
    r"\b(furry|furries|fursuit|fursuiter|fursuiting|fursona|anthro|kemono|therian|otherkin|scalie|protogen|yiff|paw\s+day)\b",
    re.I,
)


def main() -> None:
    s17 = json.load(open(S17_BUNDLE))
    s17_by_cid = {int(c["communityId"]): c for c in s17["communities"]}

    con = duckdb.connect()
    rows = con.execute(
        f"""
        SELECT community,
               coalesce(label, ''),
               coalesce(label_native, ''),
               coalesce(description, ''),
               coalesce(primary_language, ''),
               coalesce(community_type, ''),
               coalesce(onboarding_fit, ''),
               sensitivity,
               evidence
        FROM '{LABELS_PARQUET}'
        ORDER BY community
        """
    ).fetchall()

    out_lines: list[str] = []
    for cid, label, label_nat, desc, lang, ctype, fit, sens, evid in rows:
        cid_int = int(cid)
        sens_list = list(sens or [])
        evid_str = " ".join(evid or [])
        blob = f"{label} {label_nat} {desc} {evid_str}"

        is_pol = 1 if ("politics" in sens_list or ctype == "follow_train_or_political_amplification") else 0
        is_ad = 1 if ("adult" in sens_list or ctype == "adult_content" or "likely_minors" in sens_list) else 0
        is_fur = 1 if (FURRY_RE.search(blob) or ctype == "roleplay") else 0
        is_jk = (
            1
            if ctype
            in (
                "follow_train_or_political_amplification",
                "follow_back_circle",
                "spam_or_commercial",
                "adult_content",
                "roleplay",
            )
            else 0
        )
        s17_entry = s17_by_cid.get(cid_int)
        macro_cat = s17_entry["macroCategory"] if s17_entry else ""
        scene_slug = s17_entry["sceneSlug"] if s17_entry else ""
        is_s17_clean = (
            1
            if (
                s17_entry is not None
                and not is_ad
                and not is_fur
                and not is_pol
                and not is_jk
                and macro_cat != "news_civics_history"
            )
            else 0
        )

        out_lines.append(
            json.dumps(
                {
                    "community": cid_int,
                    "label": s17_entry["title"] if s17_entry else label,
                    "primary_language": lang,
                    "community_type": ctype,
                    "onboarding_fit": fit,
                    "macro_category": macro_cat,
                    "scene_slug": scene_slug,
                    "is_political": is_pol,
                    "is_adult": is_ad,
                    "is_furry": is_fur,
                    "is_junk": is_jk,
                    "is_s17_clean": is_s17_clean,
                }
            )
        )

    schema_sql = open(SQL_SCHEMA).read()
    subprocess.run(
        ["ssh", "netcup", "clickhouse-client --multiquery"],
        input=schema_sql + "\nTRUNCATE TABLE catbird.community_meta;\n",
        text=True,
        check=True,
    )
    subprocess.run(
        [
            "ssh",
            "netcup",
            "clickhouse-client --query 'INSERT INTO catbird.community_meta FORMAT JSONEachRow'",
        ],
        input="\n".join(out_lines) + "\n",
        text=True,
        check=True,
    )
    verify = subprocess.run(
        [
            "ssh",
            "netcup",
            "clickhouse-client --query 'SELECT count() AS total, sum(is_political) AS pol, sum(is_adult) AS adult, sum(is_furry) AS furry, sum(is_junk) AS junk, sum(is_s17_clean) AS s17_clean FROM catbird.community_meta FINAL FORMAT PrettyCompactMonoBlock'",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print(verify.stdout, end="")


if __name__ == "__main__":
    main()
