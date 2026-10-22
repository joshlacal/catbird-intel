# 30-Post Teacher Prototype Qualification Probe — `Cloudflare/clef-flash` + `Political_DEBATE_large_v1.0`

- **Date:** `2026-10-22T13:32:00Z`
- **Host:** `macOS-27.0.1-arm64-arm-64bit-Mach-O` (`mps`, `bfloat16`)
- **Prepared Corpus:** `catbird-intel/eval/anthropic_teacher_pilot_v1/state.sqlite3` (`57,815` eligible train rows; `2,000` stratified pilot rows)
- **Alt-Text Verification Source:** `netcup:catbird.posts FINAL` (`has_alt_text == 0` verified on all 10 `image_only_no_alt` posts)
- **Machine-Verifiable Receipt:** [`catbird-intel/eval/clef_flash_30post_insufficient_probe_20261022.json`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/eval/clef_flash_30post_insufficient_probe_20261022.json)

---

## 1. Probe Design (`N = 30` Real Corpus Posts, `30` Distinct Authors)

We constructed a deterministic 30-post probe set from `catbird-intel/eval/anthropic_teacher_pilot_v1/state.sqlite3` across 3 groups of 10 original posts:
1. **`link_only` (`n = 10`):** 5 standard single-domain bare URLs (`instagram.com`, `youtu.be`, `youtube.com`, `threads.com`, `ketv.com`) + 5 multi-label subdomain / compound-TLD bare URLs (`deanblundell.substack.com`, `www.cbc.ca`, `www.telegraph.co.uk`, `www.ibtimes.co.uk`, `www.cidrap.umn.edu`).
2. **`image_only_no_alt` (`n = 10`, all verified `has_alt_text == 0` and `image_count >= 1` on `netcup` `catbird.posts FINAL`):** 6 emoji / date+credit-only image posts (`10/19/26\n🎥JaysWorld411`, `🐈‍⬛️🎃🐈‍⬛️`, `😋🎂🍒🥧💦`, `👏🏻👏🏻👏🏻`, `✔️✔️✔️🔥`, `👹👹👹👹😡`) + 4 hashtag-only image posts (`#streetphotography #WashingtonDC`, `#Sky #Clouds #Photography`, `#YouCanSeeEverything`, `#minimalMonday\n#aofoto`).
3. **`normal_prose` (`n = 10`):** 9 standard Latin substantive-prose posts (5 political, 4 non-political) + 1 Mathematical Italic Unicode substantive-prose post with an attached image (`𝐴𝑟𝑒 𝑤𝑒 𝑝𝑟𝑒𝑜𝑟𝑑𝑎𝑖𝑛𝑒𝑑 𝑡𝑜 𝑏𝑒 𝑝𝑟𝑜𝑑𝑢𝑐𝑡𝑠 𝑜𝑓 𝑜𝑢𝑟 𝑛𝑎𝑡𝑢𝑟𝑒?...`).

---

## 2. Four Defects Found & Fixed in [`catbird-intel/teacher_harness/transport.py`](file:///Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/teacher_harness/transport.py)

1. **`_SYSTEMONE_URL_RE` Subdomain / Compound-TLD / Query-Suffix Failure (`250` real corpus posts missed):**
   - **Before:** `(?:www\.|[a-zA-Z0-9-]+\.(?:com|...))` had `www\.` as a separate alternation branch and matched only a single domain label before a limited TLD list with `/\S*`. Across the `57,815` eligible train rows in `state.sqlite3`, `250` single-token original link-only posts (`open.substack.com/...`, `www.bbc.co.uk/...`, `www.cbc.ca/...`, `www.telegraph.co.uk/...`, `www.cidrap.umn.edu/...`, `plan.lavote.gov?utm_content...`) left unstripped tokens (`1–4` words) and got `is_link_or_mention_only = False`.
   - **After:** Updated `_SYSTEMONE_URL_RE` to match multi-label domains, compound/country TLDs, and `[/?#]\S*` paths/queries/fragments, reducing missed single-token original link posts across `state.sqlite3` from **`250` → `0`**, and raising `is_link_or_mention_only = True` on the 10-post `link_only` probe from **`5/10` (`50.0%`) → `10/10` (`100.0%`)**.
2. **Hashtag-Only No-Alt Image Posts Counted as Substantive Prose (`103` real corpus posts missed; `2/10` probe posts partial `4/5` `insufficient`):**
   - **Before:** `_analyze_systemone_text_substance` did not strip `#hashtags` before counting prose words, so `103` hashtag-only original image posts in `state.sqlite3` got `is_media_without_substantive_prose = False`. On the 10-post `image_only_no_alt` probe, `4/10` posts had `is_media_without_substantive_prose = False`, and Posts `18` (`#Sky #Clouds #Photography`) and `19` (`#YouCanSeeEverything`) only fired `insufficient` on **`4/5` components** (`48/50 = 96.0%` component cells insufficient; `pol_noul` down to `0.682`).
   - **After:** Added `_SYSTEMONE_HASHTAG_RE = re.compile(r"#[^\s#]+")` to `_analyze_systemone_text_substance`. All **`10/10` (`100.0%`)** `image_only_no_alt` probe posts now have `is_media_without_substantive_prose = True` and fire `insufficient` on **all `5/5` components (`50/50 = 100.0%` component cells, `pol_noul = 0.890..0.956`, `missing_context_kind = media_visual`)**.
3. **Unicode `NFKC` Compatibility Font & Non-Latin Script Blindness in `_analyze_systemone_text_substance`:**
   - **Before:** `_SYSTEMONE_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F]{2,}")` without `unicodedata.normalize("NFKC", ...)` treated Mathematical Bold/Italic Unicode prose (Post `30`, `𝐴𝑟𝑒 𝑤𝑒 𝑝𝑟𝑒𝑜𝑟𝑑𝑎𝑖𝑛𝑒𝑑...`) and non-Latin scripts as `0` words (`is_media_without_substantive_prose = True`), inflating Post `30` `pol_noul` to `0.424`.
   - **After:** Added `unicodedata.normalize("NFKC", ...)` and Unicode/CJK word matching (`[^\W\d_]{2,}|[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]`). Post `30` now registers `23` words (`is_media_without_substantive_prose = False`) and `pol_noul` drops from **`0.424` → `0.105`**.
4. **Spurious Non-`"none"` `missing_context_kind` When `context_dependent == False`:**
   - **Before:** `parse_systemone_clef_response` read `missing_context_kind` directly from the `choice` head even when `context_dependent == False` and `any_insufficient == False`, producing `"multi_post_history"` and `"reply_parent"` on self-contained `normal_prose` Posts `21` and `22` (`8/10` `"none"`).
   - **After:** Enforced `if not context_dependent: missing_context_kind = "none"` in `parse_systemone_clef_response`, yielding **`10/10` (`100.0%`)** `missing_context_kind = "none"` on `normal_prose`.

---

## 3. Before vs. After Summary Table (`N = 30` Posts)

| Probe Group (`n = 10` each) | Metric | Before Fix | After Fix |
|---|---|---:|---:|
| **Group 1: `link_only` (`n = 10`)** | Structural `is_link_or_mention_only == True` | `5 / 10` (`50.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 1: `link_only` (`n = 10`)** | Posts with any component `insufficient` | `10 / 10` (`100.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 1: `link_only` (`n = 10`)** | Posts with all `5/5` components `insufficient` | `10 / 10` (`100.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 1: `link_only` (`n = 10`)** | Component cells `insufficient` (`50` total) | `50 / 50` (`100.0%`) | **`50 / 50` (`100.0%`)** |
| **Group 1: `link_only` (`n = 10`)** | `politicalness` `noul` range | `0.972 – 0.983` | **`0.975 – 0.984`** |
| **Group 2: `image_only_no_alt` (`n = 10`)** | Structural `is_media_without_substantive_prose == True` | `6 / 10` (`60.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 2: `image_only_no_alt` (`n = 10`)** | Posts with any component `insufficient` | `10 / 10` (`100.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 2: `image_only_no_alt` (`n = 10`)** | Posts with all `5/5` components `insufficient` | `8 / 10` (`80.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 2: `image_only_no_alt` (`n = 10`)** | Component cells `insufficient` (`50` total) | `48 / 50` (`96.0%`) | **`50 / 50` (`100.0%`)** |
| **Group 2: `image_only_no_alt` (`n = 10`)** | `politicalness` `noul` range | `0.682 – 0.956` | **`0.890 – 0.956`** |
| **Group 3: `normal_prose` (`n = 10`)** | False `is_media_without_substantive_prose == True` | `1 / 10` (`10.0%`) | **`0 / 10` (`0.0%`)** |
| **Group 3: `normal_prose` (`n = 10`)** | Posts with all `5/5` components scorable (`0/5` `insufficient`) | `10 / 10` (`100.0%`) | **`10 / 10` (`100.0%`)** |
| **Group 3: `normal_prose` (`n = 10`)** | `politicalness` `noul` range | `0.022 – 0.424` | **`0.022 – 0.111`** |
| **Group 3: `normal_prose` (`n = 10`)** | `missing_context_kind == "none"` | `8 / 10` (`80.0%`) | **`10 / 10` (`100.0%`)** |

---

## 4. Cross-Teacher Agreement (`Cloudflare/clef-flash` vs. `mlburnham/Political_DEBATE_large_v1.0`)

- **30-Post Probe Scorable Subset (`n = 10` scorable out of `N = 30` total; `20` excluded as `politicalness = insufficient`):**
  - **Binary Agreement at `>= 0.50`:** **`9 / 10` (`90.0%`, `n = 10`)**
  - **Pearson Correlation ($r$):** **`0.8380`** (`n = 10`)
  - **Spearman Rank Correlation ($\rho$):** **`0.8780`** (`n = 10`)
  - **Single Disagreement (`1 / 10`):** `at://did:plc:5fl4f67ahwmcjxzuphkcshqz/app.bsky.feed.post/3mybbvrje5c2x` (`"if you were maga you don’t get a place in the..."`), where `Clef-Flash` scored `continuous_0_1 = 0.655` (`ordinal = 2`) and `Political_DEBATE_large_v1.0` scored `0.000` (missing lowercase `"maga"`).
- **Why Insufficiency Gating Is Essential (`n = 20` non-prose posts in Groups 1 & 2):**
  - Because `Political_DEBATE_large_v1.0` is a 2-class NLI classifier with no `insufficient` head, it hallucinated `political_debate_prob >= 0.98` (`is_political_civic = True`) on **`7 / 10` (`70.0%`)** bare URLs in Group 1 (including `www.threads.com/@alisonspenc... = 0.999` and `ketv.com/article/kaye... = 0.999` about a food recall) and **`2 / 10` (`20.0%`)** no-alt image posts in Group 2 (`👏🏻👏🏻👏🏻 = 0.983` and `#streetphotography #WashingtonDC = 1.000`).
  - Without `Clef-Flash` `insufficient` gating, unmasked binary agreement across all `n = 30` probe posts would drop from **`90.0%` (`9/10`)** to **`66.7%` (`20/30`)**.
- **64-Post Stratified Pilot Reference (`catbird-intel/eval/clef_flash_local_prototype_20261021.json`):**
  - **Explicit Scorable `n`:** **`n = 54`** scorable posts out of `N = 64` total (`10` excluded as `politicalness = insufficient`; `52 / 64 = 81.25%` scorable across all 5 components simultaneously).
  - **Binary Agreement at `>= 0.50`:** **`50 / 54` (`92.59%`, `n = 54`)**, Pearson $r = \mathbf{0.8241}$, Spearman $\rho = \mathbf{0.5726}$.
