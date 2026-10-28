"""S12m projection: wall time and dollars for the full multimodal slab on one GCE Spot L4.

  projection.py [--bench bench.json] [--posts 44.9e6] [--image-share 0.23] [--images-per-post 1.232]
                [--gate-pass 1.0] [--usd-per-hour R] [--json]

Model: one VM, one IP (no parallel VMs: several IPs fetching at once would be IP rotation, which the etiquette
rules out). The CDN fetch and the GPU run concurrently, so wall = max(fetch h, GPU h) x 1.10 + 0.5 h setup;
GPU h = every post's text pass (text posts' vector, image posts' text-only companion) + the image posts'
interleaved pass. Without a bench file the GPU rates are FLOP-based L4 priors [E], replaced by measurement.
"""
from __future__ import annotations

import argparse
import json
import math

import mmlib as L

POSTS, IMAGE_SHARE, IMAGES_PER_POST = 44.9e6, 0.23, 1.232
OVERHEAD, SETUP_H, MAX_RUN_H = 1.10, 0.5, 24
# L4 priors [E, unsourced]: ~121 dense bf16 TFLOP/s at ~30% utilisation. Text tower ~130M non-embedding
# params -> 0.26 GFLOP/token x 47 tokens; vision tower 170M params over 9 patches per soft token (630 patches at
# 70, 2,520 at 280) plus the soft tokens through the LM; x 1.232 images per image post.
PRIOR = {"text_posts_per_s": 2000.0, "mm70_posts_per_s": 120.0, "mm280_posts_per_s": 32.0}


def rates(bench=None):
    r, src = dict(PRIOR), {k: "prior [E]" for k in PRIOR}
    if bench:
        for k, key in (("text_posts_per_s", "text"), ("mm70_posts_per_s", "mm70"), ("mm280_posts_per_s", "mm280")):
            v = (bench.get(key) or {}).get("posts_per_s")
            if v:
                r[k], src[k] = float(v), f"measured on {bench.get('stack', {}).get('device_name', '?')}"
    return r, src


def project(r, *, posts=POSTS, image_share=IMAGE_SHARE, images_per_post=IMAGES_PER_POST, gate_pass=1.0,
            usd_per_hour=None, rps_list=(50, 100, 200), softs=(70, 280)):
    usd = usd_per_hour or L.g2_standard_8_spot_usd_per_hour()
    img_posts = posts * image_share * gate_pass
    images = img_posts * images_per_post
    text_h = posts / r["text_posts_per_s"] / 3600
    rows = []
    for s in softs:
        mm_h = img_posts / r[f"mm{s}_posts_per_s"] / 3600
        gpu_h = text_h + mm_h
        for rps in rps_list:
            fetch_h = images / rps / 3600
            wall = max(fetch_h, gpu_h) * OVERHEAD + SETUP_H
            rows.append({"soft_tokens": s, "rps": rps, "image_posts": round(img_posts), "images": round(images),
                         "fetch_h": round(fetch_h, 1), "gpu_text_h": round(text_h, 1), "gpu_mm_h": round(mm_h, 1),
                         "bound": "fetch" if fetch_h >= gpu_h else "gpu", "wall_h": round(wall, 1),
                         "vm_sessions_at_24h": math.ceil(wall / MAX_RUN_H), "usd": round(wall * usd, 2)})
    return {"usd_per_hour": round(usd, 4), "posts": posts, "image_share": image_share,
            "images_per_image_post": images_per_post, "gate_pass": gate_pass, "rows": rows}


def dollars_from_bench(res, usd_per_hour):
    out = {}
    for key in ("text", "mm70", "mm280"):
        v = (res.get(key) or {}).get("posts_per_s")
        if v:
            d, arith = L.usd_per_10m(v, usd_per_hour)
            out[f"{key}_usd_per_10m_posts"] = round(d, 2)
            out[f"{key}_arith"] = arith
    f = (res.get("fetch") or {}).get("achieved_rps")
    if f:
        d, arith = L.usd_per_10m(f / IMAGES_PER_POST, usd_per_hour)
        out["fetch_bound_usd_per_10m_image_posts_at_achieved_rps"] = round(d, 2)
        out["fetch_arith"] = f"{f:.1f} req/s / {IMAGES_PER_POST} images per post -> " + arith
    return out


def project_from_bench(res, usd_per_hour):
    r, src = rates(res)
    p = project(r, usd_per_hour=usd_per_hour)
    p["rate_sources"] = src
    p["rates"] = r
    return p


def table(p):
    lines = ["| soft tokens | req/s | images | fetch h | GPU h (text + mm) | bound | wall h | 24 h sessions | $ |",
             "|---|---|---|---|---|---|---|---|---|"]
    for x in p["rows"]:
        lines.append(f"| {x['soft_tokens']} | {x['rps']} | {x['images'] / 1e6:.1f}M | {x['fetch_h']} | "
                     f"{x['gpu_text_h'] + x['gpu_mm_h']:.1f} ({x['gpu_text_h']} + {x['gpu_mm_h']}) | {x['bound']} | "
                     f"{x['wall_h']} | {x['vm_sessions_at_24h']} | {x['usd']:.2f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--posts", type=float, default=POSTS)
    ap.add_argument("--image-share", type=float, default=IMAGE_SHARE)
    ap.add_argument("--images-per-post", type=float, default=IMAGES_PER_POST)
    ap.add_argument("--gate-pass", type=float, default=1.0)
    ap.add_argument("--usd-per-hour", type=float)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    bench = json.load(open(a.bench)) if a.bench else None
    r, src = rates(bench)
    p = project(r, posts=a.posts, image_share=a.image_share, images_per_post=a.images_per_post, gate_pass=a.gate_pass,
                usd_per_hour=a.usd_per_hour)
    p["rate_sources"], p["rates"] = src, r
    print(json.dumps(p, indent=1) if a.json else table(p) + f"\n\nrates: {r}\nsources: {src}\n$/h: {p['usd_per_hour']}")


if __name__ == "__main__":
    main()
