import warnings
import gridstatus

warnings.filterwarnings("ignore")

miso = gridstatus.MISO()
df = miso.get_interconnection_queue()

print(f"Shape: {df.shape}")
print(f"\nColumns:\n{list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
