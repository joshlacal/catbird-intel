# Local M4 Max Prototype Report — `Cloudflare/clef-flash` (9B) + `Political_DEBATE_large_v1.0`

**Date:** `2026-10-21T15:34:47.684261+00:00`  
**Host:** `macOS-27.0.1-arm64-arm-64bit-Mach-O` (`mps`)  
**Prepared Run Directory:** `catbird-intel/eval/anthropic_teacher_pilot_v1` (`2,000` frozen pilot posts)  
**Hydration Context:** `catbird-intel/eval/pilot_holdout_parent_quote_hydration_20261021.jsonl` (`36/64` context-enriched posts in slice)  
**Scorable Post Retention:** **`60/64` (`93.8%`)** (up from `5/64 = 7.8%` prior to parent/quote hydration and `SYSTEMONE_INSUFFICIENT_NOUL_CRITERIA` option-key disambiguation)  
**Prototype Slice:** `64` deterministic stratified posts (`0` paid API calls, `$0.00` spend)

## 1. Executive Summary

We executed the 64-post stratified local M4 Max prototype comparing Cloudflare's newly released open-weight **System One** bounded-decision model (`Cloudflare/clef-flash`, `Qwen3.5-9B` + `JointSchemaHead`, Apache-2.0) against cached Teacher-1 (`mlburnham/Political_DEBATE_large_v1.0`) and the legacy `regex_baseline_*` columns across all five v3 health components (`politicalness`, `outraged`, `critical`, `engagement_bait`, `spam`).

### Throughput & Memory on M4 Max (`mps`, `bfloat16`)

| Model | Role | Parameters | Steady-State Latency | Throughput | Peak MPS Allocated | Projected 2,000-Post Pilot Time |
|---|---|---:|---:|---:|---:|---:|
| `Cloudflare/clef-flash` (`f0b7d3cd65`) | All 5 v3 components + `insufficient` + topic/context/stance (`14` simultaneous questions, `0` output tokens) | `9,531,576,564` | `4170.3 ms/post` (p50 `4094.4 ms`, p90 `4564.1 ms`) | `0.24 posts/s` | `18180 MB` (driver `55273 MB`) | `139.0 min` (`$0`) |
| `mlburnham/Political_DEBATE_large_v1.0` (`1a3aff1ecb`) | Teacher-1 binary `politicalness` entailment | `~435M` | `14.2 ms/post` | `70.60 posts/s` | `1660 MB` | `0.5 min` (`$0`) |

## 2. Ordinal Score & Brier-Calibrated Confidence Distributions (`64` Stratified Posts)

| Component | Score `0` | Score `1` | Score `2` | Score `3` | `insufficient` | Mean Continuous `[0,1]` | Mean Confidence | Mean Norm Entropy | Mean `noul` (`insufficient`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `politicalness` | 44 | 5 | 10 | 1 | 4 | `0.1813` | `0.8188` | `0.3451` | `0.1634` |
| `outraged` | 53 | 5 | 3 | 3 | 0 | `0.1254` | `0.8477` | `0.2859` | `0.0532` |
| `critical` | 52 | 1 | 5 | 5 | 1 | `0.1537` | `0.8229` | `0.3232` | `0.1291` |
| `engagement_bait` | 57 | 7 | 0 | 0 | 0 | `0.1109` | `0.7990` | `0.4182` | `0.1505` |
| `spam` | 54 | 4 | 6 | 0 | 0 | `0.1243` | `0.8376` | `0.3232` | `0.1139` |

## 3. Cross-Teacher Agreement & Construct Separation

- **`Clef-Flash` vs. `Political_DEBATE_large_v1.0` (`politicalness`):**
  - Binary agreement at `>= 0.50`: **`93.3%`** (`60` scorable posts)
  - Pearson correlation ($r$): **`0.8341`**
  - Spearman rank correlation ($\rho$): **`0.5983`**
- **Comparison against legacy `regex_baseline_politicalness`:**
  - `Clef-Flash` vs. `regex_baseline`: Pearson $r =$ `0.5988`
  - `Political_DEBATE` vs. `regex_baseline`: Pearson $r =$ `0.521`
- **Decomposed `critical` vs. `outraged` separation:**
  - Posts where `critical` exceeds `outraged` by $\ge 0.15$ (calm/substantive critique without outrage): **`5`**
  - Posts where `outraged` exceeds `critical` by $\ge 0.15$: **`1`**

### Topic & Context Routing Distributions

- **Topics (`topic`):** `{"other": 14, "none_or_everyday": 14, "politics_policy": 11, "culture_media": 10, "commercial_promo": 4, "sports_games": 4, "science_tech": 3, "economy_labor": 2, "elections_campaigns": 1, "international_conflict": 1}`
- **Stance ambiguity (`stance_ambiguity`):** `{"none": 55, "quoted_or_reported": 8, "satire_or_irony": 1}`
- **Missing context kind (`missing_context_kind`):** `{"none": 35, "multi_post_history": 9, "media_visual": 8, "external_link": 8, "reply_parent": 4}`

## 4. Side-by-Side Post Inspection Examples

### Example 1: High Political Agreement (Clef-Flash + Political_DEBATE)
- **URI:** `at://did:plc:j2w2cfun7kam2chtajg22te4/app.bsky.feed.post/3myackikjps2f` | **Stratum:** `natural_probability` | **Type:** `reply` (`text_only`)
- **Text:** > Trump's sinking approval leaves GOP candidates in a bind  Democrats lead Republicans by 8 points in the Post poll, 9 points in Fox's and 10 points in NBC's among registered voters.  The economy is now Trump's main vulnerability as he continues to mock "affordability" www.axios...
- **Politicalness Comparison:** `Clef-Flash` = **`0.834`** (ord `3`, conf `0.53`) | `Political_DEBATE` = **`1.000`** | `regex_baseline` = `0.000`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.018`** (ord `0`) | `critical` = **`0.309`** (ord `0`) | `engagement_bait` = **`0.038`** (ord `0`) | `spam` = **`0.113`** (ord `0`)
- **Metadata:** `topic=elections_campaigns` (`0.50`), `stance=quoted_or_reported`, `context_dep_noul=0.10` (`missing=reply_parent`)

### Example 2: Teacher Disagreement Candidate (|Clef-Flash - Political_DEBATE| for Claude Escalation)
- **URI:** `at://did:plc:h6hj5tgybu7lylcqduokcar5/app.bsky.feed.post/3mybdko3qgk27` | **Stratum:** `natural_probability` | **Type:** `reply` (`context_enriched`)
- **Text:** > Juuri näin!  T. kielipoliisi
- **Politicalness Comparison:** `Clef-Flash` = **`0.070`** (ord `0`, conf `0.83`) | `Political_DEBATE` = **`1.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.045`** (ord `0`) | `critical` = **`0.021`** (ord `0`) | `engagement_bait` = **`0.045`** (ord `0`) | `spam` = **`0.012`** (ord `0`)
- **Metadata:** `topic=none_or_everyday` (`0.35`), `stance=none`, `context_dep_noul=0.27` (`missing=none`)

### Example 3: Construct Separation (`critical` > `outraged`)
- **URI:** `at://did:plc:p5yoii26kayabauhkym3vtms/app.bsky.feed.post/3myak7go4bswc` | **Stratum:** `moderation_enrichment` | **Type:** `original` (`text_only`)
- **Text:** > the idea that a certain region can secede from a polity by a vote of its own residents is wildly unjust
- **Politicalness Comparison:** `Clef-Flash` = **`0.756`** (ord `2`, conf `0.54`) | `Political_DEBATE` = **`1.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.315`** (ord `0`) | `critical` = **`0.685`** (ord `2`) | `engagement_bait` = **`0.022`** (ord `0`) | `spam` = **`0.028`** (ord `0`)
- **Metadata:** `topic=politics_policy` (`0.83`), `stance=none`, `context_dep_noul=0.22` (`missing=external_link`)

### Example 4: High Outrage (`outraged` Signal)
- **URI:** `at://did:plc:fcdkkibpwqgqggyagy5j2oju/app.bsky.feed.post/3myavb2d3q22c` | **Stratum:** `editorial_boundary` | **Type:** `reply` (`context_enriched`)
- **Text:** > You think it’s my responsibility to educate you and not yours? Go fuck yourself.
- **Politicalness Comparison:** `Clef-Flash` = **`0.027`** (ord `0`, conf `0.94`) | `Political_DEBATE` = **`0.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.828`** (ord `3`) | `critical` = **`0.561`** (ord `3`) | `engagement_bait` = **`0.147`** (ord `0`) | `spam` = **`0.009`** (ord `0`)
- **Metadata:** `topic=none_or_everyday` (`0.57`), `stance=none`, `context_dep_noul=0.27` (`missing=multi_post_history`)

### Example 5: Engagement Bait Signal (`engagement_bait`)
- **URI:** `at://did:plc:v5iq6pi3hqpvrz74ntvu2tll/app.bsky.feed.post/3myahl7ify22g` | **Stratum:** `post_type_community_balanced` | **Type:** `quote` (`context_enriched`)
- **Text:** > Share a GIF from a HORROR tv series! 😱🖤  Scooby-Doo, Where Are You?  S2 Ep 05 - ‘Haunted House Hang-Up’ (1970)
- **Politicalness Comparison:** `Clef-Flash` = **`0.010`** (ord `0`, conf `0.98`) | `Political_DEBATE` = **`0.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.013`** (ord `0`) | `critical` = **`0.014`** (ord `0`) | `engagement_bait` = **`0.388`** (ord `0`) | `spam` = **`0.028`** (ord `0`)
- **Metadata:** `topic=culture_media` (`0.86`), `stance=none`, `context_dep_noul=0.24` (`missing=media_visual`)

### Example 6: Spam / Promotional Signal (`spam`)
- **URI:** `at://did:plc:tye2pif2qgwvtzcv5rvx4y5c/app.bsky.feed.post/3myawc6zjzl2c` | **Stratum:** `moderation_enrichment` | **Type:** `original` (`text_only`)
- **Text:** > 🔥 Whey Isolate Protein Fuse 900g Dark Lab Sabor Baunilha  🚨 R$ 48 🎟️ 40OFERTAS1910  🔗 https://meli.la/23dibRL  #Afazenda #ExercitoSegueEmCampo #Rebeca #promoção #oferta
- **Politicalness Comparison:** `Clef-Flash` = **`0.010`** (ord `0`, conf `0.98`) | `Political_DEBATE` = **`0.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.014`** (ord `0`) | `critical` = **`0.014`** (ord `0`) | `engagement_bait` = **`0.273`** (ord `1`) | `spam` = **`0.721`** (ord `2`)
- **Metadata:** `topic=commercial_promo` (`0.90`), `stance=none`, `context_dep_noul=0.07` (`missing=none`)

### Example 7: Context-Dependent / Missing Context (`context_dependent` / `noul`)
- **URI:** `at://did:plc:kfybkwhqgia5r65hz2f72pnm/app.bsky.feed.post/3myatgozx5c26` | **Stratum:** `post_type_community_balanced` | **Type:** `original` (`text_only`)
- **Text:** > x.com/allchronolog...
- **Politicalness Comparison:** `Clef-Flash` = **`0.068`** (ord `insufficient`, conf `0.87`) | `Political_DEBATE` = **`0.000`** | `regex_baseline` = `0.087`
- **Clef-Flash Decomposed Health:** `outraged` = **`0.011`** (ord `0`) | `critical` = **`0.015`** (ord `0`) | `engagement_bait` = **`0.127`** (ord `0`) | `spam` = **`0.564`** (ord `2`)
- **Metadata:** `topic=other` (`0.24`), `stance=none`, `context_dep_noul=0.86` (`missing=external_link`)

## 5. Next-Step Recommendation for the 2,000-Post Pilot & $200 Budget

1. **Run `Cloudflare/clef-flash` + `Political_DEBATE_large_v1.0` locally over all 2,000 prepared pilot posts (`$0` API cost):**
   - At `4170.3 ms/post` (`0.24 posts/s`), scoring all 2,000 pilot posts on M4 Max takes **`~139.0 minutes`** locally with zero token spend.
2. **Use Brier-calibrated entropy & teacher disagreement to route Claude 4.5 / 5.5 Batch escalation:**
   - Posts where `Clef-Flash` and `Political_DEBATE` agree with high confidence (`confidence >= 0.80`, `|delta| < 0.15`) are already well-characterized.
   - Route the high-entropy (`normalized_entropy >= 0.55`), context-ambiguous (`noul >= 0.35`), and cross-teacher disagreement posts (`|Clef - Political_DEBATE| >= 0.25`), plus the mandatory `15%` uniform random audit floor, to the paid Anthropic Batch tiers (`Haiku` / `Sonnet` / `Opus`).
