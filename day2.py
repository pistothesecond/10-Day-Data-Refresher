# Import the pandas library (a powerful toolkit for working with tables of data)
# 'as pd' gives it a short nickname so we only have to type pd instead of pandas
import pandas as pd

# Store the exact path to our Excel file as a string of text inside standard quotes ""
# The small letter 'r' in front means 'raw string', telling Python to treat backslashes (\) literally
path = r"D:\10-Day-Data-Refresher\Data\Dryer plant data.xlsx"

# Read the specific sheet named 'dryer_plant_data' from the Excel file and store it in a DataFrame variable called 'df'
# A DataFrame is simply a 2-dimensional table made of rows and columns (like an Excel sheet in memory)
df = pd.read_excel(path, sheet_name="dryer_plant_data")

# Clean column headers by turning them into text with .astype(str) and removing any accidental leading or trailing spaces with .str.strip()
df.columns = df.columns.astype(str).str.strip()

# Convert the 'timestamp' column into actual date-time objects so Python can do calendar math and time calculations on it
# Square brackets [] are index brackets used here to select and modify a single column from our DataFrame table
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ==============================================================================
# SECTION 1: INVESTIGATING DUPLICATE TIMESTAMPS
# ==============================================================================

# Print a clear section header to the screen
print("=== 1. INVESTIGATING DUPLICATE TIMESTAMPS ===")

# .duplicated(keep=False) checks every row in the 'timestamp' column
# It returns True for every single row involved in a collision (both the first copy and any later copies) and False for unique rows
# Round brackets () are used to call functions and pass configuration settings to them
dupe_mask = df["timestamp"].duplicated(keep=False)

# Filter the table using the True/False mask inside square brackets [] to keep only the colliding rows,
# then sort them chronologically so matching timestamps sit side by side
dupes = df[dupe_mask].sort_values("timestamp")

# len(dupes) counts the total number of colliding rows found
print(f"Total rows involved in timestamp collisions: {len(dupes)}")

# Create a standard Python list of column names using square brackets [].
# A list is an ordered collection of items separated by commas.
# We will use this list to pick which specific measurement columns we want to compare.
cols_to_check = [
    "feed_flow_tph",
    "steam_flow_tph",
    "drum_temp_C",
    "motor_current_A",
    "ambient_temp_C",
]

# Pull the top 10 colliding rows containing 'timestamp' plus our measurement columns list using the plus (+) symbol to join them
# .head(10) takes only the first 10 rows
sample_dupes = dupes[["timestamp"] + cols_to_check].head(10)

# Display those 10 rows on screen cleanly using .to_string()
print("\nSample of Colliding Rows (Values Side-by-Side):")
print(sample_dupes.to_string())

# Group all colliding rows by their shared timestamp using .groupby()
# .apply() runs a custom mini-function (lambda) on each timestamp group:
# It checks if the numbers in row 0 match the numbers in row 1 for that exact same timestamp
# Curly braces {} are not needed here because pandas builds the result series directly
dupe_comparison = dupes.groupby("timestamp")[cols_to_check].apply(
    lambda g: (
        (g.iloc[0] == g.iloc[1]).all()
        if len(g) == 2
        else "More than 2 collisions"
    )
)

# .value_counts() counts how many timestamp pairs were identical copies (True) vs how many had different sensor numbers (False)
print("\nDo process variables match identically across duplicates?")
print(dupe_comparison.value_counts())

# ==============================================================================
# SECTION 2: INVESTIGATING THE 91 MISSING (NaN) FEED FLOW ROWS
# ==============================================================================

# Print a section header for the missing data check
print("\n=== 2. INVESTIGATING THE 91 MISSING (NaN) FEED FLOW ROWS ===")

# .isna() checks every row in the 'feed_flow_tph' column and returns True if the cell is completely empty or blank (NaN)
# The square brackets df[...] filter the table so we only keep rows where feed flow is missing
missing_feed = df[df["feed_flow_tph"].isna()]

# Count and display how many blank feed flow rows were collected
print(f"Total NaN rows in feed_flow_tph: {len(missing_feed)}")

# .min() finds the earliest timestamp where a blank feed flow appears
print("Earliest missing feed timestamp:", missing_feed["timestamp"].min())

# .max() finds the latest timestamp where a blank feed flow appears
print("Latest missing feed timestamp:  ", missing_feed["timestamp"].max())

# .diff() calculates the time step between consecutive missing rows (Row 2 minus Row 1)
# This shows whether the missing rows happened all in one unbroken block or were scattered randomly across the 15 days
missing_feed_steps = missing_feed["timestamp"].diff()

# Count and print the step sizes between missing rows
print("\nTime steps between consecutive missing feed rows:")
print(missing_feed_steps.value_counts())

# .loc[] is a label-based indexer using square brackets [] to slice specific rows by index number and select specific columns
# missing_feed.index[:10] grabs the index position numbers of the first 10 missing rows
print("\nSample of rows with missing feed flow:")
print(
    df.loc[
        missing_feed.index[:10],
        [
            "timestamp",
            "feed_flow_tph",
            "steam_flow_tph",
            "drum_temp_C",
            "motor_current_A",
        ],
    ].to_string()
)

# ==============================================================================
# SECTION 3: INVESTIGATING THE 61-MINUTE TIME GAP
# ==============================================================================

# Print a section header for the time jump check
print("\n=== 3. INVESTIGATING THE 61-MINUTE TIME GAP ===")

# .diff() calculates the time difference between every row's timestamp and the row immediately above it
time_diffs = df["timestamp"].diff()

# Find row positions where the time gap is greater than 1 minute (pd.Timedelta creates a 1-minute time duration object)
# .index grabs the row numbers where this condition is True
gap_indices = df[time_diffs > pd.Timedelta(minutes=1)].index

# A 'for' loop iterates through every gap row index found (if there are multiple)
for idx in gap_indices:
    print(f"\nGap detected at index {idx}:")

    # .loc[] retrieves the timestamp on the row immediately before the jump (idx - 1)
    print(f"Row before gap (index {idx-1}):", df.loc[idx - 1, "timestamp"])

    # .loc[] retrieves the timestamp on the row where the jump landed (idx)
    print(f"Row after gap  (index {idx}):  ", df.loc[idx, "timestamp"])

    # Look up the exact duration of the jump from our time_diffs series
    print(f"Total jump duration:           {time_diffs.loc[idx]}")

    # Inspect the surrounding context: print 5 rows before the gap up to 5 rows after the gap (idx - 5 : idx + 5)
    # The colon (:) creates a range slice from start to end
    print("\nSurrounding rows context:")
    print(
        df.loc[
            idx - 5 : idx + 5,
            ["timestamp", "feed_flow_tph", "steam_flow_tph", "drum_temp_C"],
        ].to_string()
    )

import numpy as np
import pandas as pd

# ==============================================================================
# SECTION 4: MULTI-SENSOR ROLLING DISPERSION & ARTIFACT ADJACENCY AUDIT
# ==============================================================================

window_size = 30  # 30-minute rolling window
min_noise = 1e-4  # Zero/near-zero variance threshold
step_threshold_sigma = (
    3.5  # Rate-of-change threshold in standard deviations for step jumps
)
adjacency_buffer_min = 15  # Minutes before/after a flat-line to test for step jumps

variables = ["motor_current_A", "drum_temp_C", "ambient_temp_C"]

print("=== 4. MULTI-VARIABLE ROLLING DISPERSION & SENSOR RECOVERY AUDIT ===")

for col in variables:
    if col not in df.columns:
        continue

    print(f"\n" + "=" * 60)
    print(f"ANALYSIS FOR: {col}")
    print("=" * 60)

    # 1. Rolling local statistics
    roll_mean = (
        df[col].rolling(window=window_size, min_periods=window_size).mean()
    )
    roll_std = (
        df[col].rolling(window=window_size, min_periods=window_size).std()
    )

    # 2. Minute-to-minute delta (Rate of Change)
    delta = df[col].diff()
    delta_std = delta.std()
    is_step_jump = delta.abs() > (step_threshold_sigma * delta_std)

    # 3. Detect flat-lines
    is_flatline = (roll_std < min_noise) & roll_std.notna()

    print(f"Total Flat-line Records:  {is_flatline.sum()}")
    print(f"Total Step-Change Jumps:  {is_step_jump.sum()}")

    # Extract distinct Flat-line Events
    flatline_events = []
    if is_flatline.any():
        flat_df = df[is_flatline].copy()
        flat_df["event_id"] = (
            flat_df["timestamp"].diff() > pd.Timedelta(minutes=1)
        ).cumsum()

        flatline_events = (
            flat_df.groupby("event_id")
            .agg(
                start_time=("timestamp", "min"),
                end_time=("timestamp", "max"),
                duration_min=("timestamp", "count"),
                frozen_value=(col, "first"),
            )
            .reset_index(drop=True)
        )
        print("\n--- Detected Flat-line Periods ---")
        print(flatline_events.to_string(index=False))

    # Extract distinct Step-Change Jumps
    step_events = df[is_step_jump][["timestamp", col]].copy()
    step_events["jump_magnitude"] = delta[is_step_jump]

    if not step_events.empty:
        print("\n--- Detected Step-Change Transition Points ---")
        print(step_events.head(10).to_string(index=False))

    # 4. Check Temporal Adjacency / Overlap: Flat-line Release vs Step Change
    if len(flatline_events) > 0 and not step_events.empty:
        print(
            "\n--- ADJACENCY AUDIT: Did Step Changes Coincide with Sensor Recovery? ---"
        )
        suspicious_jumps = []

        for _, f_row in flatline_events.iterrows():
            f_start = f_row["start_time"]
            f_end = f_row["end_time"]

            # Define time window around the entry and exit of the flat-line
            near_entry = step_events[
                (
                    step_events["timestamp"]
                    >= f_start - pd.Timedelta(minutes=adjacency_buffer_min)
                )
                & (
                    step_events["timestamp"]
                    <= f_start + pd.Timedelta(minutes=adjacency_buffer_min)
                )
            ]
            near_exit = step_events[
                (
                    step_events["timestamp"]
                    >= f_end - pd.Timedelta(minutes=adjacency_buffer_min)
                )
                & (
                    step_events["timestamp"]
                    <= f_end + pd.Timedelta(minutes=adjacency_buffer_min)
                )
            ]

            if not near_exit.empty:
                for _, s_row in near_exit.iterrows():
                    suspicious_jumps.append(
                        {
                            "flatline_end": f_end,
                            "step_time": s_row["timestamp"],
                            "time_delta_min": (
                                s_row["timestamp"] - f_end
                            ).total_seconds()
                            / 60.0,
                            "jump_mag": s_row["jump_magnitude"],
                            "frozen_val": f_row["frozen_value"],
                            "verdict": (
                                "Artifact (Sensor Unfreeze / Recovery)"
                            ),
                        }
                    )

        if suspicious_jumps:
            suspicious_df = pd.DataFrame(suspicious_jumps)
            print(suspicious_df.to_string(index=False))
        else:
            print(
                "No step-changes were adjacent to flat-line exit points. Step changes appear process-driven."
            )
    else:
        print(
            "\nNo overlap check required (no flat-lines or no step-changes detected)."
        )


# ==============================================================================
# SECTION 5: PHYSICAL BOUNDS & LAB ENTRY QUIRKS AUDIT
# ==============================================================================

print("=== 5A. PHYSICAL BOUNDS AUDIT (IMPOSSIBLE SENSOR READINGS) ===")

# Define physical boundary rules based on dryer plant thermodynamics and physics:
# - Flows and electrical current cannot physically be negative.
# - Drum temperature cannot be below ambient temperature during operation.
# - Temperatures exceeding reasonable industrial design limits (e.g., drum > 200°C or ambient < -30°C / > 50°C).

bound_checks = {
    "Negative Feed Flow": df["feed_flow_tph"] < 0,
    "Negative Steam Flow": df["steam_flow_tph"] < 0,
    "Negative Motor Current": df["motor_current_A"] < 0,
    "Negative Ambient Temp": df["ambient_temp_C"] < -20.0,
    "Absurd High Drum Temp (>200°C)": df["drum_temp_C"] > 200.0,
    "Drum Temp Below Ambient (Cold Drum Violation)": (
        df["drum_temp_C"] < df["ambient_temp_C"]
    )
    & df["drum_temp_C"].notna()
    & df["ambient_temp_C"].notna(),
    "Negative Moisture Reading": df["product_moisture_pct"] < 0,
    "Moisture Exceeding 100%": df["product_moisture_pct"] > 100.0,
}

bounds_summary = []
for test_name, condition in bound_checks.items():
    violations = df[condition]
    bounds_summary.append(
        {
            "Physical Boundary Check": test_name,
            "Violations Count": len(violations),
            "Min Violation Val": (
                violations.select_dtypes(include="number")
                .min()
                .dropna()
                .to_dict()
                if not violations.empty
                else "None"
            ),
        }
    )

bounds_df = pd.DataFrame(bounds_summary)
print(bounds_df.to_string(index=False))

# Show exact timestamps and values for any physical violations found
all_physical_violations_mask = pd.concat(bound_checks.values(), axis=1).any(
    axis=1
)
if all_physical_violations_mask.any():
    print("\n--- Detected Physical Bound Violations (First 10 Rows) ---")
    print(
        df.loc[
            all_physical_violations_mask,
            [
                "timestamp",
                "feed_flow_tph",
                "steam_flow_tph",
                "drum_temp_C",
                "ambient_temp_C",
                "product_moisture_pct",
            ],
        ]
        .head(10)
        .to_string()
    )


print(
    "\n========================================================================"
)
print("=== 5B. LAB MOISTURE ENTRY QUIRKS AUDIT ===")
print(
    "========================================================================"
)

# 1. Isolate the 90 non-null lab moisture rows
lab_df = (
    df.dropna(subset=["product_moisture_pct"])
    .sort_values("timestamp")
    .copy()
    .reset_index()
)
print(f"Total valid lab moisture records: {len(lab_df)}")

# 2. Check for duplicate timestamps specifically within the lab entries
lab_time_dupes = lab_df["timestamp"].duplicated(keep=False)
print(f"Duplicate timestamps in lab data: {lab_time_dupes.sum()}")

if lab_time_dupes.any():
    print("\n--- Lab Entries Sharing Duplicate Timestamps ---")
    print(
        lab_df.loc[
            lab_time_dupes,
            ["index", "timestamp", "product_moisture_pct", "drum_temp_C"],
        ].to_string(index=False)
    )

# 3. Check for suspiciously repeated lab values (e.g., default placeholders, operator bias)
moisture_counts = lab_df["product_moisture_pct"].value_counts()
print("\n--- Top 10 Most Frequent Lab Moisture Values ---")
print(
    pd.DataFrame(
        {
            "Moisture (%)": moisture_counts.index,
            "Occurrences": moisture_counts.values,
            "Pct of All Lab Tests (%)": (
                moisture_counts.values / len(lab_df) * 100
            ).round(2),
        }
    )
    .head(10)
    .to_string(index=False)
)

# 4. Check sampling rhythm / time delta between consecutive lab samples
lab_deltas = lab_df["timestamp"].diff()
print("\n--- Lab Sample Interval Breakdown ---")
print(lab_deltas.value_counts(dropna=False).to_string())

# 5. Check consecutive identical value runs (Operator copying previous shift's value)
lab_identical_consecutive = (
    lab_df["product_moisture_pct"].diff() == 0
).sum()
print(
    f"\nConsecutive tests with exact identical numbers to previous test: {lab_identical_consecutive}"
)










# 1. Missing values (NaNs) — you know at least one block exists (the ~91 missing feed_flow_tph rows). Are they scattered randomly, or clustered in blocks? Does the pattern differ by column?
# 2. Duplicate timestamps — you found 70 of them yesterday but didn't yet resolve whether the values on those rows match or differ. Finish that.
# 3. Timestamp irregularities — the single 61-minute gap. Is it really isolated, or connected to the duplicate block?
# 4. Outliers / spikes — motor current is your prime suspect. What counts as a spike vs. real variation?
# 5. Flat-lined sensor — does any variable hold an identical value for an implausibly long stretch?
# 6. Impossible/physically unrealistic values — you already spotted -5.0% moisture. Are there others, in other columns, using physical bounds you can define from process knowledge (not statistics)?
# 7. Manual/lab entry issues — you noted product_moisture_pct is sparse. Check it for its own quirks (duplicate lab entries, suspicious repeats).