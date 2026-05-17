import warnings
import pandas as pd
import gridstatus

warnings.filterwarnings("ignore")

ISO_CONFIGS = {
    "CAISO": {"cls": gridstatus.CAISO,  "withdrawn": "WITHDRAWN"},
    "ISONE": {"cls": gridstatus.ISONE,  "withdrawn": "Withdrawn"},
    "MISO":  {"cls": gridstatus.MISO,   "withdrawn": "Withdrawn"},
    "NYISO": {"cls": gridstatus.NYISO,  "withdrawn": "Withdrawn"},
    # ERCOT: blocked (403)
    # PJM:   requires PJM_API_KEY env var
    # SPP:   connection timeout
}

results = []

for name, cfg in ISO_CONFIGS.items():
    try:
        iso = cfg["cls"]()
        df = iso.get_interconnection_queue()
        filtered = df[
            (df["Status"] == cfg["withdrawn"]) &
            (df["Capacity (MW)"] > 50)
        ].copy()
        filtered.insert(0, "ISO", name)
        results.append(filtered)
        print(f"{name:6}  total={len(df):5}  withdrawn >50MW={len(filtered):4}")
    except Exception as e:
        print(f"{name:6}  ERROR: {e}")

combined = pd.concat(results, ignore_index=True)

print(f"\nCombined shape: {combined.shape}")
print(f"\nWithdrawn >50MW by ISO:")
print(combined.groupby("ISO")["Capacity (MW)"].agg(count="count", total_MW="sum").to_string())

cols = ["ISO", "Queue ID", "Project Name", "Generation Type",
        "Capacity (MW)", "State", "Queue Date", "Withdrawn Date", "Withdrawal Comment"]
display_cols = [c for c in cols if c in combined.columns]

print(f"\nFirst 10 rows:")
print(combined[display_cols].head(10).to_string(index=False))
