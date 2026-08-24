import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 0. LOAD CLEANED DATA (FROM DAY 3)
# ==============================================================================
path = r"D:\10-Day-Data-Refresher\Data\Dryer_plant_data_clean.csv"
df = pd.read_csv(path, parse_dates=["timestamp"])

eda_cols = [
    "feed_flow_tph",
    "steam_flow_tph",
    "drum_temp_C",
    "motor_current_A_clean",
    "ambient_temp_C",
    "product_moisture_pct",
]

# Continuous 1-minute time series subset for multi-panel plotting
ts_cols = [
    "feed_flow_tph",
    "steam_flow_tph",
    "drum_temp_C",
    "motor_current_A_clean",
    "ambient_temp_C",
]

# ==============================================================================
# 1. DESCRIPTIVE STATISTICS & SHAPE PREDICTIONS
# ==============================================================================
print("=" * 80)
print("=== 1. DESCRIPTIVE STATISTICS & DISTRIBUTION DIAGNOSTICS ===")
print("=" * 80)

desc = df[eda_cols].describe().T
desc["median"] = df[eda_cols].median()
desc["mean_minus_median"] = desc["mean"] - desc["median"]
desc["IQR"] = desc["75%"] - desc["25%"]
# Theoretical normal distribution has std ≈ IQR / 1.349 (Ratio ≈ 1.0)
desc["std_to_robust_std_ratio"] = desc["std"] / (desc["IQR"] / 1.34898)

# Classify predicted shape mathematically before plotting
shape_predictions = []
for col in eda_cols:
    diff = desc.loc[col, "mean_minus_median"]
    ratio = desc.loc[col, "std_to_robust_std_ratio"]
    rel_skew = abs(diff) / (desc.loc[col, "std"] + 1e-6)

    if rel_skew < 0.05:
        skew_pred = "Roughly Symmetric (Mean ≈ Median)"
    elif diff > 0:
        skew_pred = "Right-Skewed / Upper Tail (Mean > Median)"
    else:
        skew_pred = "Left-Skewed / Lower Tail (Mean < Median)"

    dispersion_pred = (
        "Heavy Tails / Multimodal Regimes"
        if ratio > 1.2
        else "Standard Dispersion"
    )
    shape_predictions.append(f"{skew_pred} | {dispersion_pred}")

desc["Pre-Plot Prediction"] = shape_predictions

print(
    desc[
        [
            "count",
            "mean",
            "median",
            "mean_minus_median",
            "std",
            "IQR",
            "std_to_robust_std_ratio",
            "Pre-Plot Prediction",
        ]
    ]
    .round(3)
    .to_string()
)

print("\n" + "-" * 80)
print("SUMMARY OF WRITTEN PREDICTIONS:")
for col in eda_cols:
    print(f" - {col:22s}: {desc.loc[col, 'Pre-Plot Prediction']}")
print("-" * 80)

# ==============================================================================
# 2. HISTOGRAMS (FIGURE 1)
# ==============================================================================
fig_hist, axes_hist = plt.subplots(
    nrows=2, ncols=3, figsize=(16, 9), num="Histograms"
)
axes_hist = axes_hist.flatten()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for i, col in enumerate(ts_cols):
    ax = axes_hist[i]
    series = df[col].dropna()

    mean_val = series.mean()
    median_val = series.median()
    q25, q75 = series.quantile(0.25), series.quantile(0.75)

    ax.hist(
        series,
        bins=50,
        density=True,
        alpha=0.6,
        color=colors[i],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_val:.2f}",
    )
    ax.axvline(
        median_val,
        color="black",
        linestyle="-",
        linewidth=2,
        label=f"Median: {median_val:.2f}",
    )
    ax.axvspan(
        q25, q75, color="gray", alpha=0.18, label=f"IQR ({q25:.1f}–{q75:.1f})"
    )

    ax.set_title(f"Histogram: {col}", fontsize=11, fontweight="bold")
    ax.set_xlabel(col, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

# Remove unused 6th subplot
fig_hist.delaxes(axes_hist[5])

plt.suptitle(
    "Day 4 EDA: Process Variable Histograms & Shape Verification",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
fig_hist.tight_layout()

# ==============================================================================
# 3. BOX PLOTS (FIGURE 2)
# ==============================================================================
fig_box, axes_box = plt.subplots(
    nrows=1, ncols=len(ts_cols), figsize=(18, 6), num="Box Plots"
)

print("\n" + "=" * 80)
print("=== 3. BOX PLOT OUTLIER AUDIT (WHISKER RANGE: 1.5 * IQR) ===")
print("=" * 80)

for i, col in enumerate(ts_cols):
    ax = axes_box[i]
    series = df[col].dropna()

    # Calculate IQR fences (1.5 * IQR standard)
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = series[(series < lower_fence) | (series > upper_fence)]
    print(
        f"{col:22s} | IQR: {iqr:6.2f} | Bounds: [{lower_fence:7.2f}, {upper_fence:7.2f}] | Outlier Points: {len(outliers):4d}"
    )

    # Box plot styling
    ax.boxplot(
        series,
        patch_artist=True,
        boxprops=dict(facecolor=colors[i], alpha=0.5),
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        flierprops=dict(
            marker="o",
            markerfacecolor="red",
            markersize=3,
            alpha=0.4,
            linestyle="none",
        ),
    )

    ax.set_title(col, fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_xticks([])

plt.suptitle(
    "Day 4 EDA: Process Variable Box Plots & Dispersion Fences",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
fig_box.tight_layout()

# ==============================================================================
# 4. TIME-SERIES OVERVIEW PLOTS (FIGURE 3: 15-DAY OPERATIONAL REGIMES)
# ==============================================================================
# Plot all cleaned operational signals aligned along a shared 15-day time axis
fig_ts, axes_ts = plt.subplots(
    nrows=len(ts_cols),
    ncols=1,
    figsize=(18, 12),
    sharex=True,
    num="Time Series Overview",
)

print("\n" + "=" * 80)
print("=== 4. TIME-SERIES OVERVIEW AUDIT (15-DAY CAMPAIGN REGIMES) ===")
print("=" * 80)

for i, col in enumerate(ts_cols):
    ax = axes_ts[i]

    # Plot cleaned 1-minute time series trajectory
    ax.plot(
        df["timestamp"],
        df[col],
        color=colors[i],
        linewidth=0.8,
        label="Cleaned Signal (1-min)",
    )

    # Overlay 24-hour moving average to highlight true underlying macro regime shifts
    rolling_24h = df[col].rolling(window=1440, min_periods=60).mean()
    ax.plot(
        df["timestamp"],
        rolling_24h,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="24h Rolling Trend",
    )

    ax.set_ylabel(col, fontsize=9, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)

# Format time axis on bottom subplot
axes_ts[-1].set_xlabel("Date & Time", fontsize=10, fontweight="bold")
axes_ts[-1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
axes_ts[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig_ts.autofmt_xdate()

plt.suptitle(
    "Day 4 EDA: 15-Day Cleaned Process Trends & Campaign Regime Tracking",
    fontsize=14,
    fontweight="bold",
    y=0.99,
)
fig_ts.tight_layout()

# Single show call at the very end renders all three figures together without blocking
plt.show()
