"""
Counterfactual analysis: withdrawn interconnection queue projects (>50 MW) that
may represent planned data centers that were never built.

Matching logic (scored 0-3):
  +1  Same US state as a known data center in the epoch CSV
  +1  Hyperscaler keyword found in project/entity name
  +1  Withdrawal after 2018 (AI-era) and MW >= 100
"""

import warnings, re
import pandas as pd
import numpy as np
import gridstatus

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load epoch CSV and normalise
# ---------------------------------------------------------------------------
dc = pd.read_csv("/Users/murraybone/Downloads/PUBLIC Website view.csv")

# Clean owner: strip confidence tag, lower
dc["owner_clean"] = (
    dc["Owner"]
    .fillna("")
    .str.replace(r"\s*#\w+", "", regex=True)
    .str.strip()
    .str.lower()
)

# Extract 2-letter US state from address
def _state(addr):
    if pd.isna(addr): return None
    m = re.search(r',\s*([A-Z]{2})\s+\d{5}', str(addr))
    if m: return m.group(1)
    # fallback: look for state abbreviation before a 5-digit zip anywhere
    m2 = re.search(r'\b([A-Z]{2})\b', str(addr))
    return None  # don't guess without zip anchor

dc["dc_state"] = dc["Address"].apply(_state)

# Also handle addresses like "Kuna ID 83634" (no comma before state)
def _state2(addr):
    if pd.isna(addr): return None
    m = re.search(r'\b([A-Z]{2})\s+\d{5}', str(addr))
    return m.group(1) if m else None

dc["dc_state"] = dc["dc_state"].fillna(dc["Address"].apply(_state2))

dc_us = dc[dc["Country"].fillna("").str.contains("United States")].copy()

# Extract earliest year mentioned in Notes (proxy for first construction epoch)
def _first_year(notes):
    if pd.isna(notes): return None
    years = [int(y) for y in re.findall(r'\b(20[12]\d)\b', str(notes))]
    return min(years) if years else None

dc_us["first_epoch_year"] = dc_us["Notes"].apply(_first_year)

# States with known data centers
dc_states = set(dc_us["dc_state"].dropna())

# Hyperscaler keyword map: CSV owner → list of search terms
HYPERSCALER_KEYWORDS = {
    "microsoft":  ["microsoft", "msft"],
    "google":     ["google", "alphabet", "goodnight"],
    "amazon":     ["amazon", "aws"],
    "meta":       ["meta", "facebook"],
    "oracle":     ["oracle"],
    "coreweave":  ["coreweave", "core weave"],
    "spacexai":   ["xai", "spacex", "colossus"],
    "openai":     ["openai", "open ai", "stargate"],
    "softbank":   ["softbank", "soft bank"],
    "apple":      ["apple"],
    "nscale":     ["nscale"],
    "fluidstack": ["fluidstack"],
    "qts":        ["qts"],
}

ALL_KEYWORDS = [kw for kws in HYPERSCALER_KEYWORDS.values() for kw in kws]

# Terms that appear in project names when a generator is built for data center load
DC_ADJACENT_KEYWORDS = [
    "data center", "datacenter", "hyperscale", "colocation",
    "compute", "ai campus", "cloud", "server farm",
    "hpc", "gpu", "colossus", "stargate", "rainier",
]

def match_hyperscaler(text):
    """Return matched owner name or None."""
    t = str(text).lower()
    for owner, kws in HYPERSCALER_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return owner
    return None

def match_dc_adjacent(text):
    t = str(text).lower()
    return any(kw in t for kw in DC_ADJACENT_KEYWORDS)

# ---------------------------------------------------------------------------
# 2. Pull withdrawal data with per-ISO entity fields
# ---------------------------------------------------------------------------
ISO_CONFIGS = {
    "CAISO": {
        "cls": gridstatus.CAISO, "withdrawn": "WITHDRAWN",
        "entity_cols": ["Project Name"],           # developer name mostly here
    },
    "ISONE": {
        "cls": gridstatus.ISONE, "withdrawn": "Withdrawn",
        "entity_cols": ["Project Name", "Dev"],
    },
    "MISO": {
        "cls": gridstatus.MISO,  "withdrawn": "Withdrawn",
        "entity_cols": ["Project Name"],            # mostly blank but try
    },
    "NYISO": {
        "cls": gridstatus.NYISO, "withdrawn": "Withdrawn",
        "entity_cols": ["Interconnecting Entity", "Project Name"],
    },
}

frames = []
for iso_name, cfg in ISO_CONFIGS.items():
    try:
        raw = cfg["cls"]().get_interconnection_queue()
        w = raw[
            (raw["Status"] == cfg["withdrawn"]) &
            (raw["Capacity (MW)"] > 50)
        ].copy()
        w["ISO"] = iso_name

        # Build a combined search text from available entity columns
        parts = [w[col].fillna("").astype(str) for col in cfg["entity_cols"] if col in w.columns]
        if parts:
            w["search_text"] = parts[0] if len(parts) == 1 else parts[0].str.cat(parts[1:], sep=" ")
        else:
            w["search_text"] = ""

        keep = ["ISO", "Queue ID", "Project Name", "State",
                "County", "Capacity (MW)", "Queue Date", "Withdrawn Date",
                "Generation Type", "search_text"]
        keep = [c for c in keep if c in w.columns]
        frames.append(w[keep])
        print(f"{iso_name:6}  {len(w):4} withdrawn >50 MW loaded")
    except Exception as e:
        print(f"{iso_name:6}  ERROR: {e}")

combined = pd.concat(frames, ignore_index=True)

# Parse dates
combined["Withdrawn Date"] = pd.to_datetime(combined["Withdrawn Date"], utc=True, errors="coerce")
combined["Queue Date"]     = pd.to_datetime(combined["Queue Date"],     utc=True, errors="coerce")
combined["withdraw_year"]  = combined["Withdrawn Date"].dt.year.fillna(
                             combined["Queue Date"].dt.year)

# ---------------------------------------------------------------------------
# 3. Score each withdrawn project
# ---------------------------------------------------------------------------
combined["score"] = 0

# +1 same state as a known DC
combined["in_dc_state"] = combined["State"].isin(dc_states)
combined["score"] += combined["in_dc_state"].astype(int)

# +1 hyperscaler keyword OR data-center-adjacent keyword in project/entity name
combined["matched_owner"]   = combined["search_text"].apply(match_hyperscaler)
combined["is_dc_adjacent"]  = combined["search_text"].apply(match_dc_adjacent)
combined["score"] += (combined["matched_owner"].notna() | combined["is_dc_adjacent"]).astype(int)

# +1 AI-era (post-2018) and large scale (>=100 MW)
combined["ai_era_large"] = (
    (combined["withdraw_year"] >= 2018) &
    (combined["Capacity (MW)"] >= 100)
)
combined["score"] += combined["ai_era_large"].astype(int)

# ---------------------------------------------------------------------------
# 4. Counterfactual candidates (score >= 2)
# ---------------------------------------------------------------------------
candidates = combined[combined["score"] >= 2].copy().sort_values(
    ["score", "Capacity (MW)"], ascending=[False, False]
)

print(f"\nTotal withdrawn >50 MW: {len(combined)}")
print(f"Score breakdown:")
print(combined["score"].value_counts().sort_index(ascending=False).to_string())
print(f"\nCounterfactual candidates (score >= 2): {len(candidates)}")

# ---------------------------------------------------------------------------
# 5. Match each candidate to the nearest DC entry by state + owner
# ---------------------------------------------------------------------------
def find_dc_match(row):
    """Return (dc_name, match_type, dc_epoch_year) for best-matching DC."""
    matches = dc_us[dc_us["dc_state"] == row["State"]]
    if row["matched_owner"]:
        owner_matches = matches[matches["owner_clean"].str.contains(
            row["matched_owner"], na=False)]
        if not owner_matches.empty:
            best = owner_matches.nlargest(1, "Current power (MW)").iloc[0]
            return best["Name"], "owner+state", best["first_epoch_year"]
    if not matches.empty:
        best = matches.nlargest(1, "Current power (MW)").iloc[0]
        return best["Name"] + " (state only)", "state only", best["first_epoch_year"]
    return None, None, None

candidates[["matched_dc", "match_type", "dc_epoch_year"]] = candidates.apply(
    lambda r: pd.Series(find_dc_match(r)), axis=1
)

# Years before DC first epoch that the generation project was withdrawn
# (negative = withdrew after DC was underway; positive = withdrew before)
candidates["years_before_dc"] = candidates["dc_epoch_year"] - candidates["withdraw_year"]

# ---------------------------------------------------------------------------
# 6. Output
# ---------------------------------------------------------------------------
display_cols = ["ISO", "Score", "Queue ID", "Project Name", "State",
                "Capacity (MW)", "Generation Type", "withdraw_year",
                "matched_owner", "is_dc_adjacent", "match_type",
                "matched_dc", "dc_epoch_year", "years_before_dc"]

candidates = candidates.rename(columns={"score": "Score"})

print("\n=== Score-3 Candidates (state + keyword/DC-adjacent + AI-era large) ===")
top = candidates[candidates["Score"] == 3][
    [c for c in display_cols if c in candidates.columns]
].head(30)
print(top.to_string(index=False))

print("\n=== Score-2 Candidates (state + AI-era-scale) ===")
s2 = candidates[candidates["Score"] == 2][
    [c for c in display_cols if c in candidates.columns]
].head(20)
print(s2.to_string(index=False))

# Summary by matched DC
print("\n=== Withdrawn MW by Matched Data Center ===")
summary = (
    candidates.dropna(subset=["matched_dc"])
    .groupby(["matched_dc", "match_type"])
    .agg(
        withdrawn_projects=("Queue ID", "count"),
        total_withdrawn_mw=("Capacity (MW)", "sum"),
        earliest_withdrawal=("withdraw_year", "min"),
        latest_withdrawal=("withdraw_year", "max"),
        dc_epoch_year=("dc_epoch_year", "first"),
    )
    .sort_values("total_withdrawn_mw", ascending=False)
    .head(20)
)
print(summary.to_string())

# Temporal alignment: withdrawals that happened BEFORE the DC was built
print("\n=== Withdrawals Preceding DC Construction (potential power shortfall) ===")
precede = candidates[
    candidates["years_before_dc"].fillna(-999) > 0
].sort_values("years_before_dc", ascending=False)
print(f"Count: {len(precede)}")
print(precede[[c for c in display_cols if c in precede.columns]].head(20).to_string(index=False))

# Save
candidates.to_csv("/Users/murraybone/counterfactual_candidates.csv", index=False)
print("\nSaved: counterfactual_candidates.csv")
