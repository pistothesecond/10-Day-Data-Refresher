import matplotlib.pyplot as plt
import pandas as pd

path = r"D:\10-Day-Data-Refresher\Data\Dryer plant data.xlsx"
xl = pd.ExcelFile(path)
print("SHEETS:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = pd.read_excel(path, sheet_name=sheet)

    # Clean up column names
    df.columns = df.columns.astype(str).str.strip()

    print(
        "\n=================================================================="
    )
    print(f"--- SHEET: {sheet} (rows={len(df)}, cols={len(df.columns)})")
    print("COLUMNS:", list(df.columns))

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. Exact Time Period
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        print("\n--- 1. TIME RANGE ---")
        print("Earliest Timestamp:", t_min)
        print("Latest Timestamp:  ", t_max)
        print("Total Duration:    ", t_max - t_min)

        # 2. Timestamp Continuity and Monotonicity Checks
        print("\n--- 2. TIMESTAMP CONTINUITY & SAMPLING ---")
        print("Strictly monotonic:   ", df["timestamp"].is_monotonic_increasing)
        duplicates = df["timestamp"].duplicated().sum()
        print("Duplicate timestamps: ", duplicates)

        time_diffs = df["timestamp"].diff()
        print("\nTime step distribution:")
        print(time_diffs.value_counts(dropna=False))

        # Highlight intervals that deviate from exactly 1 minute (ignoring index 0 NaT)
        gap_mask = time_diffs != pd.Timedelta(minutes=1)
        gaps = df.loc[gap_mask, ["timestamp"]].copy()
        gaps["gap_size"] = time_diffs[gap_mask]
        real_gaps = gaps.iloc[1:]
        # Print the exact rows where the 61-minute gap occurs (before and after the jump)
        gap_index = df[df["timestamp"].diff() > pd.Timedelta(minutes=1)].index[0]
        print(df.loc[gap_index - 1 : gap_index, ["timestamp"]])
        print(f"\nTotal interval anomalies (gaps != 1 min): {len(real_gaps)}")

        if not real_gaps.empty:
            print("\nFirst 10 timestamp irregularities:")
            print(real_gaps.head(10).to_string())

        # 3. Campaign Start/Stop (Downtime periods)
        if "feed_flow_tph" in df.columns:
            downtime = df[df["feed_flow_tph"] <= 0]
            running = df[df["feed_flow_tph"] > 0]
            print("\n--- 3. OPERATIONAL CONTINUITY ---")
            print(f"Total records:      {len(df)}")
            print(f"Running records:    {len(running)}")
            print(f"Downtime records:   {len(downtime)}")

        # 4. Process Summary Statistics
        print("\n--- 4. PROCESS SUMMARY STATISTICS ---")
        print(df.describe().to_string())

        # 5. Daily Aggregations (to spot product changes & seasons)
        numeric_cols = df.select_dtypes(include="number").columns
        daily = df.set_index("timestamp")[numeric_cols].resample("D").mean()
        print("\n--- 5. DAILY AVERAGES (FIRST 10 DAYS) ---")
        print(daily.head(10).to_string())
        print("\nCorrelation between ambient temperature and feed flow:")
        print(daily[["ambient_temp_C", "feed_flow_tph"]].corr())

        # 6. Timeline Plots
        plot_cols = [
            c
            for c in [
                "feed_flow_tph",
                "steam_flow_tph",
                "drum_temp_C",
                "ambient_temp_C",
                "product_moisture_pct",
            ]
            if c in df.columns
        ]

        if plot_cols:
            df.plot(x="timestamp", y=plot_cols, subplots=True, figsize=(12, 10))
            plt.suptitle(
                f"Process Trends - Sheet: {sheet}", fontsize=14, y=1.00
            )
            plt.tight_layout()
            plt.show()

    else:
        print("\n(Non-timeseries sheet - skipping trend analysis)")

# DAY 1 QUESTIONS:
# 1. What does each column physically represent, and in what units?
# 2. What's the sampling frequency — and is it the same for every column?
# 3. What time period does it cover, and does anything change over that period (campaign start/stop, product change, season)?
# 4. Which variables are measurements (sensors), which are setpoints (targets/manipulated), which are calculated (derived from other tags), and which are lab/manual entries?
# 5. What's the actual question you're trying to answer — even if it's vague right now?
# 6. What assumptions would you be making if you started analysing right now?

# DAY 1 ANSWERS:
#1. The tph in steam flow and feed flow is tons per hour, the temperature is in celcius, motor current is in amps, and the product moisture is in percent.
#2. The sampling frequency is 1 minute for all columns. However the product moisture is measured separately perhaps in a lab. It has a count of 90 within the 14 days.  
#3. The period covered is Earliest Timestamp: 2026-01-01 00:00:00Latest Timestamp:   2026-01-15 23:59:00 Total Duration:     14 days 23:59:004. There are timestamp irregularities but no distinct campaign start or stop. 
#4. The measurements are feed flow, sleam flow, drum temp, mtor current, ambient temp. the measured probably in a lab is the moisture percentage as it appears at sinle data points and not a continous data set that is measured at every timestamp.
#5. I want to show the correlations between this data and dashboard them to bring actionable insights.
#6. I would say steam flow and drum temperature are directly correlated as they appear to have a similar trendline. Feed flow seems to affect steam flow because when feed changes there is a visible change with steam flow. The ambient temperature seems to oscillate the whole time.