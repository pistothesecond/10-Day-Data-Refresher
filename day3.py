import numpy as np
import pandas as pd

# ==============================================================================
# 0. LOAD RAW DATA
# ==============================================================================
path = r"D:\10-Day-Data-Refresher\Data\Dryer plant data.xlsx"
df_raw = pd.read_excel(path, sheet_name="dryer_plant_data")

# Clean whitespace from column headers and ensure timestamp is datetime
df_raw.columns = df_raw.columns.astype(str).str.strip()
df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])

# Work on a copy to preserve raw integrity
df = df_raw.copy()

print(f"Raw data loaded: {len(df)} rows.")

# ==============================================================================
# 1. DROP EXACT DUPLICATES (10 Simple Logger Double-Writes)
# ==============================================================================
# Drop identical row copies (same timestamp and identical process readings)
rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
dropped_exact = rows_before - len(df)
print(f"1. Dropped exact duplicates: {dropped_exact} rows.")

# ==============================================================================
# 2. RESOLVE 60 MISMATCHED DUPLICATES & 61-MIN GAP (Shift Block)
# ==============================================================================
# Find colliding timestamps that still exist after dropping exact duplicates
collision_mask = df["timestamp"].duplicated(keep=False)
colliding_rows = df[collision_mask].sort_values("timestamp")

if not colliding_rows.empty:
    # Identify the gap position
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df_sorted["timestamp"].diff()
    gap_idx = df_sorted[time_diffs > pd.Timedelta(minutes=1)].index

    # Shift the second occurrence of each colliding timestamp forward by +60 min
    # keep='first' marks the second pass of colliding timestamps as True
    second_collision_mask = df.duplicated(subset=["timestamp"], keep="first")
    shift_count = second_collision_mask.sum()
    df.loc[second_collision_mask, "timestamp"] += pd.Timedelta(minutes=60)
    print(
        f"2. Shifted mislabeled block: {shift_count} rows adjusted forward by 60 mins."
    )

# Re-sort chronologically after repairing timestamps
df = df.sort_values("timestamp").reset_index(drop=True)

# ==============================================================================
# 3. HANDLE IMPOSSIBLE MOISTURE READING (-5.0% -> NaN)
# ==============================================================================
# Replace physically impossible moisture (< 0%) with NaN
bad_moisture_count = (df["product_moisture_pct"] < 0).sum()
df.loc[df["product_moisture_pct"] < 0, "product_moisture_pct"] = np.nan
print(f"3. Nullified impossible moisture readings: {bad_moisture_count} rows.")

# ==============================================================================
# 4. MOTOR CURRENT: FLAG & CLEAN SPIKES, NULLIFY FLAT-LINE (371 MIN)
# ==============================================================================
# Create clean working column while preserving raw motor_current_A
df["motor_current_A_clean"] = df["motor_current_A"].copy()

# A. Detect and nullify flat-lines (rolling std == 0 over a 30-min window)
window_size = 30
roll_std = (
    df["motor_current_A"]
    .rolling(window=window_size, min_periods=window_size)
    .std()
)
flatline_mask = (roll_std == 0.0) & roll_std.notna()

# If the rolling tail is flat, expand mask across the full constant run length
run_groups = (df["motor_current_A"].diff() != 0).cumsum()
long_run_mask = (
    df.groupby(run_groups)["motor_current_A"].transform("count") >= window_size
)
flatline_full_mask = long_run_mask

flatline_count = flatline_full_mask.sum()
df.loc[flatline_full_mask, "motor_current_A_clean"] = np.nan
df.loc[flatline_full_mask, "motor_current_A"] = np.nan   # <-- add here, not at the end of the file
print(f"4a. Nullified flat-line motor current: {flatline_count} rows set to NaN (no fill).")

# B. Detect spikes using local rolling Z-score on the non-flatlined points
roll_mean = df["motor_current_A_clean"].rolling(
    window=window_size, min_periods=5, center=True
).mean()
roll_std_local = df["motor_current_A_clean"].rolling(
    window=window_size, min_periods=5, center=True
).std()
local_z = (
    (df["motor_current_A_clean"] - roll_mean).abs() / roll_std_local
).replace(np.inf, np.nan)

spike_mask = (local_z > 3.0) & df["motor_current_A_clean"].notna()
spike_count = spike_mask.sum()

# Flag spikes with a dedicated audit column
df["motor_current_is_spike"] = spike_mask

# Nullify spike positions in the clean column and linearly interpolate across the 1-3 min gap
df.loc[spike_mask, "motor_current_A_clean"] = np.nan
df["motor_current_A_clean"] = df["motor_current_A_clean"].interpolate(
    method="linear", limit=5
)
print(
    f"4b. Interpolated motor current spikes: {spike_count} needle spikes bridged."
)

# ==============================================================================
# 5. FLAG INSTRUMENT FAULT MISSING BLOCK (91 Rows on Jan 4)
# ==============================================================================
# Flag rows where flow and temp sensors are missing while motor current proves operation
instrument_fault_mask = (
    df["feed_flow_tph"].isna()
    & df["steam_flow_tph"].isna()
    & df["drum_temp_C"].isna()
    & df["motor_current_A"].notna()
)
df["instrument_fault_flag"] = instrument_fault_mask
missing_block_count = instrument_fault_mask.sum()
print(
    f"5. Flagged missing instrument block: {missing_block_count} rows flagged (kept as NaN)."
)

# ==============================================================================
# FINAL SUMMARY & SANITY CHECK
# ==============================================================================
print("\n" + "=" * 65)
print("=== CLEANING AUDIT COMPLETE ===")
print("=" * 65)
print(f"Final Row Count:        {len(df)}")
print(f"Strictly Monotonic:     {df['timestamp'].is_monotonic_increasing}")
print(f"Duplicate Timestamps:   {df['timestamp'].duplicated().sum()}")
print(
    f"Remaining Time Gaps >1m: {(df['timestamp'].diff() > pd.Timedelta(minutes=1)).sum()}"
)
print(
    "\nNull Value Summary:\n",
    df[
        [
            "feed_flow_tph",
            "steam_flow_tph",
            "drum_temp_C",
            "motor_current_A",
            "motor_current_A_clean",
            "product_moisture_pct",
        ]
    ].isna().sum(),
)
import numpy as np
import pandas as pd

# Set timestamp as index for explicit time-slice indexing
df_indexed = df.set_index("timestamp")

print("=== 1. MOTOR CURRENT RAW vs. FLAT-LINE (Jan 10) ===")
# Check raw vs clean inside the exact 371-minute window
flatline_window_raw = df_indexed.loc[
    "2026-01-10 01:09:00":"2026-01-10 07:19:00", "motor_current_A"
]
flatline_window_clean = df_indexed.loc[
    "2026-01-10 01:09:00":"2026-01-10 07:19:00", "motor_current_A_clean"
]

print(f"Total minutes in Jan 10 flatline window: {len(flatline_window_raw)}")
print(f"NaNs in raw 'motor_current_A':           {flatline_window_raw.isna().sum()}")
print(f"NaNs in 'motor_current_A_clean':         {flatline_window_clean.isna().sum()}")
print(f"Unique values in raw window:            {flatline_window_raw.unique()}")

print("\n=== 2. AUDITING THE 411 NaNs IN 'motor_current_A_clean' ===")
# Breakdown of where the 411 NaNs in the clean column are coming from
nan_clean_mask = df["motor_current_A_clean"].isna()
nan_summary = pd.DataFrame({
    "Total NaNs in clean": nan_clean_mask.sum(),
    "NaNs from Flatline Window": df_indexed.loc["2026-01-10 01:09:00":"2026-01-10 07:19:00", "motor_current_A_clean"].isna().sum(),
    "NaNs from Original Raw Nulls": df["motor_current_A"].isna().sum(),
    "NaNs from Unbridged Spikes": (nan_clean_mask & df["motor_current_is_spike"]).sum()
}, index=["Count"]).T
print(nan_summary.to_string())

# Inspect any NaNs that fall outside the flat-line period and original nulls
outside_flatline_nans = df[
    df["motor_current_A_clean"].isna() &
    ~df["timestamp"].between("2026-01-10 01:09:00", "2026-01-10 07:19:00") &
    df["motor_current_A"].notna()
]
print(f"Unfilled NaNs outside flat-line window: {len(outside_flatline_nans)}")
if not outside_flatline_nans.empty:
    print(outside_flatline_nans[["timestamp", "motor_current_A", "motor_current_A_clean", "motor_current_is_spike"]].head(10))

print("\n=== 3. DRUM TEMPERATURE NULL BREAKDOWN (Target: 240) ===")
# 1. The 91-row instrument fault block on Jan 4
jan4_block_nulls = df_indexed.loc["2026-01-04 11:20:00":"2026-01-04 12:50:00", "drum_temp_C"].isna().sum()
# 2. Total nulls overall
total_drum_nulls = df["drum_temp_C"].isna().sum()
# 3. Scattered nulls outside the known instrument block
scattered_nulls = total_drum_nulls - jan4_block_nulls

print(f"Total 'drum_temp_C' NaNs:                {total_drum_nulls}")
print(f"Jan 4 (11:20-12:50) Instrument Block:    {jan4_block_nulls} (Expected: 91)")
print(f"Scattered Background NaNs:              {scattered_nulls} (Expected: ~149)")


df.loc[df["timestamp"].between("2026-01-10 01:09:00","2026-01-10 07:19:00"), "motor_current_A"].isna().sum()



unexplained = df[
    df["motor_current_A_clean"].isna()
    & df["motor_current_A"].notna()
    & ~df["timestamp"].between("2026-01-10 01:09:00", "2026-01-10 07:19:00")
]
print(len(unexplained))
print(unexplained[["timestamp","motor_current_A","motor_current_A_clean","motor_current_is_spike"]])


clean_path = r"D:\10-Day-Data-Refresher\Data\Dryer_plant_data_clean.csv"
df.to_csv(clean_path, index=False)
print("Cleaned data successfully saved to CSV!")