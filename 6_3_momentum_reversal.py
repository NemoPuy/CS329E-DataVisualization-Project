import altair as alt
import pandas as pd
import numpy as np

# --- Load dataset ---
df = pd.read_csv("master_df.csv")

# Ensure Date is parsed correctly if needed
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

# --- Use df directly because your CSV already has flat column names ---
plot_data = df.copy()

# --- Calculate daily returns for each ETF ---
# Your actual column names are IWM_Close, QQQ_Close, SPY_Close
for etf in ["IWM", "QQQ", "SPY"]:
    col_name = f"{etf}_Close"  # correct ordering based on your header
    if col_name in plot_data.columns:
        plot_data[f"{etf}_Daily_Return"] = plot_data[col_name].pct_change()

# --- Use SPY as the primary momentum indicator ---
plot_data["Momentum"] = plot_data["SPY_Daily_Return"]

# Drop rows where crucial fields are missing
plot_data = plot_data.dropna(subset=["Momentum", "VIX"])

# --- Volatility regime thresholds ---
vix_median = plot_data["VIX"].median()
vix_75th = plot_data["VIX"].quantile(0.75)

def get_vol_regime(vix_value):
    if vix_value <= vix_median:
        return "Low Volatility"
    elif vix_value <= vix_75th:
        return "Medium Volatility"
    return "High Volatility"

plot_data["Vol_Regime"] = plot_data["VIX"].apply(get_vol_regime)

# --- Momentum regimes ---
def get_momentum_regime(return_value):
    if return_value > 0.01:
        return "Strong Bullish"
    elif return_value > 0:
        return "Mild Bullish"
    elif return_value > -0.01:
        return "Mild Bearish"
    else:
        return "Strong Bearish"

plot_data["Momentum_Regime"] = plot_data["Momentum"].apply(get_momentum_regime)

# Create selection filters
vol_regime_selection = alt.selection_point(
    fields=["Vol_Regime"], bind="legend", name="Vol_Filter"
)

momentum_regime_selection = alt.selection_point(
    fields=["Momentum_Regime"], bind="legend", name="Momentum_Filter"
)

# Main scatter plot with SPY volume as color
scatter = (
    alt.Chart(plot_data)
    .mark_circle(opacity=0.6, size=60)
    .encode(
        x=alt.X(
            "Momentum:Q",
            title="SPY Daily Return (Momentum)",
            scale=alt.Scale(domain=[plot_data["Momentum"].min(), plot_data["Momentum"].max()])
        ),
        y=alt.Y("VIX:Q", title="VIX Level", scale=alt.Scale(zero=False)),
        color=alt.Color(
            "SPY_Volume:Q",
            scale=alt.Scale(
                scheme="viridis",
                domain=[plot_data["SPY_Volume"].min(), plot_data["SPY_Volume"].max()]  # fixed scale
            ),
            title="SPY Daily Volume"
        ),
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Momentum:Q", format=".3%", title="SPY Daily Return"),
            alt.Tooltip("VIX:Q", format=".2f", title="VIX"),
            alt.Tooltip("Vol_Regime:N", title="Volatility Regime"),
            alt.Tooltip("SPY_Volume:Q", title="SPY Volume"),
            alt.Tooltip("Close_SPY:Q", format=".2f", title="SPY Close"),
            alt.Tooltip("Close_QQQ:Q", format=".2f", title="QQQ Close"),
            alt.Tooltip("Close_IWM:Q", format=".2f", title="IWM Close"),
        ],
    )
    .add_params(vol_regime_selection, momentum_regime_selection)
    .transform_filter(vol_regime_selection)
    .transform_filter(momentum_regime_selection)
    .properties(
        width=800,
        height=500,
        title="Momentum Reversal During Volatility Spikes: SPY Daily Return vs VIX (Volume Colored)"
    )
)

# Combine all bands into one DataFrame for color encoding
bands_df_long = pd.DataFrame({
    "Vol_Regime": ["Low Volatility", "Medium Volatility", "High Volatility"],
    "y0": [plot_data["VIX"].min(), plot_data["VIX"].quantile(0.33), plot_data["VIX"].quantile(0.66)],
    "y1": [plot_data["VIX"].quantile(0.33), plot_data["VIX"].quantile(0.66), plot_data["VIX"].max()]
})

# Create bands with color encoding
vol_bands = alt.Chart(bands_df_long).mark_rect(opacity=0.12).encode(
    y="y0:Q",
    y2="y1:Q",
    color=alt.Color(
        "Vol_Regime:N",
        scale=alt.Scale(
            domain=["Low Volatility", "Medium Volatility", "High Volatility"],
            range=["#2E8B57", "#FFA500", "#DC143C"]
        ),
        legend=alt.Legend(title="Volatility Regime")    # Add legend
    )
).properties(width=800)

# Combine with scatter plot
main_chart = vol_bands + scatter

# --- Interval selection for SPY Volume (brushable) ---
volume_brush = alt.selection_interval(
    encodings=["x"],  # horizontal brushing
    name="VolumeSelector"
)

# --- Horizontal color bar with brush ---
volume_legend = (
    alt.Chart(plot_data)
    .mark_rect()
    .encode(
        x=alt.X(
            "SPY_Volume:Q",
            axis=alt.Axis(
                title="SPY Daily Volume",
                labels=True,
                ticks=True,
                domain=True
            )
        ),
        y=alt.value(0),    # start of the bar
        y2=alt.value(20),  # end of the bar
        color=alt.Color(
            "SPY_Volume:Q",
            scale=alt.Scale(scheme="viridis"),
            legend=None
        ),
    )
    .properties(
        width=800,
        height=20
    )
    .add_params(volume_brush)  # attach brush selection
)

# --- Main chart filtered by the volume brush ---
main_chart_filtered = vol_bands + scatter.transform_filter(volume_brush)

# --- Combine main chart and color bar ---
final_chart = alt.vconcat(
    main_chart_filtered,
    volume_legend
).resolve_scale(
    color="independent"  # ensure vol bands and legend have independent color scales
)

# Save chart
final_chart.save("momentum_reversal_during_volatility_spikes.html")
