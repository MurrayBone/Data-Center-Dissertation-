"""
Cluster withdrawn interconnection queue projects (>50 MW) by owner entity and
withdrawal date across CAISO, ISONE, MISO, and NYISO.

Approach
--------
1. Pull and label each ISO's withdrawn >50 MW projects.
2. Normalise the entity column (each ISO uses a different field name).
3. Aggregate to entity level: count, total MW, mean MW, withdrawal year span.
4. K-means cluster the entity aggregates to find behavioural groups.
5. Print per-cluster summaries and the top entities in each cluster.
"""

import warnings
import pandas as pd
import numpy as np
import gridstatus
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Pull data
# ---------------------------------------------------------------------------
ISO_CONFIGS = {
    "CAISO": {
        "cls":       gridstatus.CAISO,
        "withdrawn": "WITHDRAWN",
        "entity":    "Transmission Owner",   # developer field is empty for CAISO
    },
    "ISONE": {
        "cls":       gridstatus.ISONE,
        "withdrawn": "Withdrawn",
        "entity":    "Dev",                  # 80 % filled; best available
    },
    "MISO": {
        "cls":       gridstatus.MISO,
        "withdrawn": "Withdrawn",
        "entity":    "Transmission Owner",   # interconnecting entity is empty
    },
    "NYISO": {
        "cls":       gridstatus.NYISO,
        "withdrawn": "Withdrawn",
        "entity":    "Interconnecting Entity",
    },
}

frames = []
for iso_name, cfg in ISO_CONFIGS.items():
    try:
        df = cfg["cls"]().get_interconnection_queue()
        w = df[
            (df["Status"] == cfg["withdrawn"]) &
            (df["Capacity (MW)"] > 50)
        ].copy()
        w["ISO"] = iso_name
        # Normalise to a single "Entity" column
        w["Entity"] = w[cfg["entity"]] if cfg["entity"] in w.columns else pd.NA
        frames.append(w[["ISO", "Entity", "Capacity (MW)", "Queue Date", "Withdrawn Date"]])
        print(f"{iso_name:6}  {len(w):4} withdrawn >50 MW rows loaded")
    except Exception as e:
        print(f"{iso_name:6}  ERROR: {e}")

combined = pd.concat(frames, ignore_index=True)

# Parse dates → withdrawal year (fall back to queue year)
combined["Withdrawn Date"] = pd.to_datetime(combined["Withdrawn Date"], utc=True, errors="coerce")
combined["Queue Date"]     = pd.to_datetime(combined["Queue Date"],     utc=True, errors="coerce")
combined["WithdrawYear"]   = combined["Withdrawn Date"].dt.year.fillna(
                                combined["Queue Date"].dt.year)

combined = combined.dropna(subset=["Entity", "WithdrawYear"])
combined["Entity"] = combined["Entity"].str.strip()

print(f"\nRows after dropping null entity/date: {len(combined)}")

# ---------------------------------------------------------------------------
# 2. Aggregate to entity level
# ---------------------------------------------------------------------------
agg = (
    combined
    .groupby("Entity")
    .agg(
        num_withdrawals  = ("Capacity (MW)", "count"),
        total_mw         = ("Capacity (MW)", "sum"),
        mean_mw          = ("Capacity (MW)", "mean"),
        first_withdrawal = ("WithdrawYear", "min"),
        last_withdrawal  = ("WithdrawYear", "max"),
    )
    .reset_index()
)
agg["year_span"] = agg["last_withdrawal"] - agg["first_withdrawal"]

# Keep entities with at least 1 withdrawal (all of them) — filter noise
agg = agg[agg["num_withdrawals"] >= 1].copy()
print(f"Unique entities: {len(agg)}")

# ---------------------------------------------------------------------------
# 3. K-means clustering — pick k via silhouette score
# ---------------------------------------------------------------------------
features = ["num_withdrawals", "total_mw", "mean_mw", "year_span", "first_withdrawal"]
X = agg[features].fillna(0).values
X_scaled = StandardScaler().fit_transform(X)

best_k, best_score = 3, -1
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    if score > best_score:
        best_k, best_score = k, score

print(f"\nBest k={best_k}  (silhouette={best_score:.3f})")

km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
agg["Cluster"] = km_final.fit_predict(X_scaled)

# ---------------------------------------------------------------------------
# 4. Summarise clusters
# ---------------------------------------------------------------------------
cluster_summary = (
    agg
    .groupby("Cluster")
    .agg(
        entities        = ("Entity",         "count"),
        total_withdrawals = ("num_withdrawals", "sum"),
        total_mw        = ("total_mw",        "sum"),
        avg_mw_per_proj = ("mean_mw",         "mean"),
        typical_year    = ("first_withdrawal", "median"),
        year_span_avg   = ("year_span",        "mean"),
    )
    .round(1)
    .reset_index()
)

print("\n--- Cluster Summary ---")
print(cluster_summary.to_string(index=False))

print("\n--- Top 5 Entities per Cluster ---")
for c in sorted(agg["Cluster"].unique()):
    top = (
        agg[agg["Cluster"] == c]
        .nlargest(5, "total_mw")[["Entity", "num_withdrawals", "total_mw", "first_withdrawal", "last_withdrawal"]]
    )
    print(f"\nCluster {c}:")
    print(top.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Year-over-year withdrawal volume by ISO (temporal view)
# ---------------------------------------------------------------------------
print("\n--- Withdrawal Volume by Year and ISO (MW) ---")
yoy = (
    combined
    .groupby(["WithdrawYear", "ISO"])["Capacity (MW)"]
    .sum()
    .unstack(fill_value=0)
    .astype(int)
    .sort_index()
)
print(yoy.to_string())
