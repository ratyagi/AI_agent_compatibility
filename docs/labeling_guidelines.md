# Labeling Guidelines 
**Project:** AI Agent Compatibility Predictor  

---

## 1. Purpose
The goal of this step is to produce a reproducible, interpretable *ground-truth labeling dataset* describing how legally and technically “friendly” a website is toward automated agents (e.g., crawlers, LLMs, research bots).  
Labels are derived from each domain’s public **Terms of Service**, **robots.txt**, and **HTML accessibility metrics** collected in Data Collection stage.

The resulting file `data/processed/labels.csv` is used in later ML phases for feature engineering and compatibility prediction.

---

## 2. Label Schema

| Column | Type | Description |
|:--|:--|:--|
| **domain** | string | Website hostname. |
| **friendly_label** | categorical {0, 1, 2} | Overall friendliness toward AI agents:<br>• 0 = Hostile (explicitly forbids)<br>• 1 = Neutral (conditional / unclear)<br>• 2 = Friendly (explicitly allows). |
| **compat_score** | float [0 – 1] | Continuous version of friendliness. Derived from weighted mix of legal + technical signals. |
| **scraping_perm** | {0, 1, 2} | Permission to scrape / crawl data.<br>0 = Prohibited  1 = Restricted  2 = Allowed |
| **ai_training_perm** | {0, 1, 2} | Permission to use data for AI / ML training.<br>0 = Prohibited  1 = Unclear  2 = Allowed |
| **derivative_perm** | {0, 1, 2} | Permission to make derivative works / adaptations.<br>0 = Prohibited  1 = Unclear  2 = Allowed |
| **notes** | text | Explanations, matched phrases, or reviewer comments. |
| **data_quality** | text | Automatically generated flags for missing or weak inputs (`complete`, `missing_tos_file`, `tiny_tos`, etc.). |

---

## 3. Labeling Approach

### 3.1 Semi-Automatic Labeling
Labels were first generated using the scripts:
- `src/labeling_utils.py` – pattern matching and scoring logic  
- `src/auto_label.py` – merges TOS, robots, and HTML data to produce initial labels  

The pipeline scans each TOS text and robots.txt for indicative phrases:

| Category | Representative Phrases |
|:--|:--|
| **Prohibitive (Hostile)** | “no automated access”, “bots prohibited”, “you may not scrape”, “machine learning use forbidden” |
| **Neutral / Conditional** | “with prior consent”, “subject to approval”, “must comply with robots.txt” |
| **Friendly / Permissive** | “automated access allowed”, “for research purposes”, “public API available”, “you may crawl” |

The counts of these phrases are weighted with robots.txt signals (`Disallow: /`, `Allow:`) and simple HTML availability indicators (`status_code < 400`).

A compatibility score is computed as:
compat_score = (friendly_label, scraping_perm, ai_training_perm, derivative_perm, html_status)

where friendly_label ≈ 0.2 / 0.5 / 0.8 baseline, adjusted by permissions and down-weighted if the site is technically broken.

---

## 4. Manual Review Procedure

After automatic labeling:

1. **Load summary** from the script output (class distribution, data-quality counts).  
2. **Review high-risk rows** where  
   - `data_quality != "complete"` or  
   - extreme `friendly_label ∈ {0, 2}` with missing data.  
3. **Open underlying TOS / robots files** to confirm interpretation.  
4. **Correct obvious mislabels** and append reason to `notes` (e.g., “manual_override_neutral – vague wording”).  
5. **Optionally mark** reviewed rows with `review_status = human_checked`.

Roughly 20–30 % of domains (all low-quality cases + random sample per class) were manually validated.

---

## 5. Handling Missing or Inconsistent Data

| Case | Auto behavior | Rationale |
|:--|:--|:--|
| Missing TOS file | Defaults to Neutral (1) / Restricted (1) | Cannot infer legal intent. |
| Missing robots.txt | Defaults to Neutral (1) / Restricted (1) | Unclear technical policy. |
| Missing HTML stats | Score unchanged; flagged as `missing_html_stats`. | Not essential for legality. |
| All three missing | Forced Neutral (1) with `all_sources_missing_default_neutral`. | Avoid false extremes. |
| Tiny TOS (< 500 chars) | Flagged `tiny_tos` and down-weighted implicitly. | Usually not real policy text. |
| Conflicting TOS vs robots | Combined score averaged; flagged `tos_robots_conflict` in notes. | Shows inconsistency. |

---

## 6. Data Quality Flags

The pipeline records cumulative flags in `data_quality` (comma-separated):

- `missing_tos_file`, `missing_robots_file`, `missing_html_stats`  
- `tiny_tos`  
- `tos_http_4xx`, `robots_http_4xx`, `html_status_404`  
- `tos_status_error`, `robots_status_error`  
- `complete` (no issues)

These flags allow later phases to filter or down-weight uncertain rows.

---

## 7. Ethical & Reproducibility Statement

- All inputs are **publicly accessible documents** (no private user data).  
- Labeling rules are **explicit, reproducible, and auditable** via source code.  
- Automatic labels serve as *weak supervision*; human review provides correction and bias control.  
- The dataset reflects website policy signals, **not subjective opinions** of the author.

---

## 8. Example Rows

| domain | friendly_label | compat_score | scraping_perm | ai_training_perm | derivative_perm | notes | data_quality |
|:--|:--:|:--:|:--:|:--:|:--:|:--|:--|
| example.com | 2 | 0.87 | 2 | 2 | 2 | “explicitly allows crawling; public API” | complete |
| privacywall.io | 0 | 0.18 | 0 | 0 | 0 | “no automated access; machine learning forbidden” | complete |
| obscure-site.net | 1 | 0.52 | 1 | 1 | 1 | “default neutral – missing TOS and robots” | missing_tos_file,missing_robots_file |

---

## 9. Version History
| Date | Version | Notes |
|:--|:--|:--|
| 2025-12-13 | v1.0 | Initial guidelines for Phase 2 labeling. |

---

## 10. Next Phase
Phase 3 will convert `labels.csv` into numerical features (TF-IDF, embeddings, structural stats) for model training.

---

