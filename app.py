# app_nobeyama_vs_chico.py
"""
Nobeyama vs Chico — Climate Comparison Demo
信州大学野辺山農場 × CSU Chico 短期滞在プログラム用

2地点の気温データを読み込み、温暖化傾向を比較するStreamlitアプリ。
Streamlit Cloud対応版 — データセットは dataset/ フォルダから読み込み
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# =====================================================
# ページ設定
# =====================================================
st.set_page_config(
    page_title="Nobeyama vs Chico Climate Comparison",
    layout="wide",
    page_icon="🌡️",
)

st.title("🌡️ Nobeyama vs Chico — Climate Comparison")
st.markdown(
    """
Compare daily temperature data between **Nobeyama, Japan** (AMeDAS)
and **Chico, California** (NOAA / Meteostat) to explore warming trends
at two contrasting cold-region agricultural sites.

| | Nobeyama 野辺山 | Chico チコ |
|---|---|---|
| **Elevation** | 1,350 m | 73 m |
| **Latitude** | 35.94°N | 39.73°N |
| **Climate** | Highland (Cfa/Dfb border) | Mediterranean (Csa) |
| **Agriculture** | Lettuce 🥬 | Almonds, rice, walnuts 🌾 |
"""
)

# =====================================================
# データセットのパス定義
# =====================================================
DATASET_DIR = Path(__file__).parent / "dataset"

AVAILABLE_DATASETS = {
    "Nobeyama": DATASET_DIR / "dataset_nobeyama.csv",
    "Chico": DATASET_DIR / "dataset_chico.csv",
}


# =====================================================
# CSV reader
# =====================================================
@st.cache_data
def read_csv_auto(filepath: Path) -> pd.DataFrame:
    """Read CSV with auto-detected encoding."""
    raw = filepath.read_bytes()
    for enc in ["utf-8-sig", "utf-8", "cp932", "shift_jis"]:
        try:
            text = raw.decode(enc, errors="replace")
            from io import StringIO
            return pd.read_csv(StringIO(text))
        except Exception:
            continue
    raise ValueError(f"Could not read {filepath}. Please check the encoding.")


def validate_and_prepare(df, label):
    """Validate required columns and preprocess."""
    required = ["date", "mea_temp", "max_temp", "min_temp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"**{label}**: Missing columns: {missing}")
        st.write("Detected columns:", list(df.columns))
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["mea_temp", "max_temp", "min_temp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "mea_temp_norm" in df.columns:
        df["mea_temp_norm"] = pd.to_numeric(df["mea_temp_norm"], errors="coerce")

    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    return df


# =====================================================
# データセット選択
# =====================================================
st.subheader("1️⃣ Select datasets / データセットの選択")

# 利用可能なデータセットを確認
available = {}
for name, path in AVAILABLE_DATASETS.items():
    if path.exists():
        available[name] = path

if not available:
    st.error(
        f"No dataset files found in `{DATASET_DIR}/`. "
        "Please add `dataset_nobeyama.csv` and/or `dataset_chico.csv`."
    )
    st.stop()

col_sel1, col_sel2 = st.columns(2)

with col_sel1:
    use_nobeyama = st.checkbox(
        "🇯🇵 **Nobeyama 野辺山**",
        value="Nobeyama" in available,
        disabled="Nobeyama" not in available,
    )
    if "Nobeyama" not in available:
        st.caption("dataset_nobeyama.csv not found")

with col_sel2:
    use_chico = st.checkbox(
        "🇺🇸 **Chico チコ**",
        value="Chico" in available,
        disabled="Chico" not in available,
    )
    if "Chico" not in available:
        st.caption("dataset_chico.csv not found")

if not use_nobeyama and not use_chico:
    st.info("Select at least one dataset to start.  /  少なくとも1つのデータセットを選択してください。")
    st.stop()

# =====================================================
# Load & validate
# =====================================================
datasets = {}

if use_nobeyama and "Nobeyama" in available:
    try:
        df_n = validate_and_prepare(read_csv_auto(available["Nobeyama"]), "Nobeyama")
        if df_n is not None:
            datasets["Nobeyama"] = df_n
    except Exception as e:
        st.error(f"Nobeyama CSV read error: {e}")

if use_chico and "Chico" in available:
    try:
        df_c = validate_and_prepare(read_csv_auto(available["Chico"]), "Chico")
        if df_c is not None:
            datasets["Chico"] = df_c
    except Exception as e:
        st.error(f"Chico CSV read error: {e}")

if not datasets:
    st.error("No valid dataset loaded. Please check your CSV files.")
    st.stop()

# =====================================================
# Sidebar settings
# =====================================================
st.sidebar.header("⚙️ Settings / 設定")

temp_options = {
    "Mean temperature / 平均気温": "mea_temp",
    "Maximum temperature / 最高気温": "max_temp",
    "Minimum temperature / 最低気温": "min_temp",
}
selected_label = st.sidebar.selectbox("Temperature variable", list(temp_options.keys()))
temp_col = temp_options[selected_label]

use_season = st.sidebar.checkbox(
    "Growing season only / 生育期間のみ (May–Oct)", value=False
)
start_month, end_month = 5, 10

high_temp_threshold = st.sidebar.radio(
    "High-temp threshold (max temp) / 高温日しきい値（最高気温）",
    options=[25.0, 30.0, 35.0],
    format_func=lambda x: f"{x:.0f} °C",
    index=1,  # default 30°C
    horizontal=True,
)

future_years_n = st.sidebar.slider(
    "Future simulation years / 将来予測年数", 5, 30, 20
)

# =====================================================
# Determine overlapping period
# =====================================================
if len(datasets) == 2:
    all_years = [set(d["year"].unique()) for d in datasets.values()]
    common_years = sorted(all_years[0] & all_years[1])
    use_common = st.sidebar.checkbox(
        f"Use common period only / 共通期間のみ ({min(common_years)}–{max(common_years)})",
        value=True,
    )
else:
    use_common = False
    common_years = None

# =====================================================
# Colors for each site
# =====================================================
COLORS = {
    "Nobeyama": {"line": "#2563eb", "fill": "#93bbfd"},
    "Chico": {"line": "#dc2626", "fill": "#fca5a5"},
}

# =====================================================
# Analysis per site
# =====================================================
def analyze_site(df, site_name, temp_col, use_season, start_month, end_month,
                 high_temp_threshold, common_years, use_common):
    """Return yearly aggregated DataFrame with trend info."""
    if use_season:
        df = df[(df["month"] >= start_month) & (df["month"] <= end_month)].copy()

    if use_common and common_years:
        df = df[df["year"].isin(common_years)].copy()

    has_norm = "mea_temp_norm" in df.columns and temp_col == "mea_temp"

    agg_dict = {
        "observed_temp": (temp_col, "mean"),
        "high_temp_days": ("max_temp", lambda x: (x >= high_temp_threshold).sum()),
        "valid_days": (temp_col, "count"),
    }
    if has_norm:
        agg_dict["normal_temp"] = ("mea_temp_norm", "mean")

    yearly = df.groupby("year").agg(**agg_dict).reset_index()

    if has_norm:
        yearly["anomaly"] = yearly["observed_temp"] - yearly["normal_temp"]

    # Trend
    if len(yearly) >= 3:
        sl, ic, r, p, se = linregress(yearly["year"], yearly["observed_temp"])
        yearly["trend"] = ic + sl * yearly["year"]
        trend_info = {"slope": sl, "intercept": ic, "r2": r**2, "p": p, "per_decade": sl * 10}
    else:
        trend_info = None

    return yearly, trend_info


results = {}
for name, df in datasets.items():
    yearly, trend_info = analyze_site(
        df, name, temp_col, use_season, start_month, end_month,
        high_temp_threshold, common_years, use_common,
    )
    results[name] = {"yearly": yearly, "trend": trend_info, "df": df}

season_text = "Growing season (May–Oct)" if use_season else "Full year"

# =====================================================
# Dataset overview
# =====================================================
st.subheader("2️⃣ Dataset overview / データ概要")

for name, r in results.items():
    df = r["df"]
    has_norm = "mea_temp_norm" in df.columns
    cols = st.columns(5)
    cols[0].metric(f"📍 {name}", "")
    cols[1].metric("Start", df["date"].min().strftime("%Y-%m-%d"))
    cols[2].metric("End", df["date"].max().strftime("%Y-%m-%d"))
    cols[3].metric("Records", f"{len(df):,}")
    cols[4].metric("Normal values", "✅" if has_norm else "—")

# =====================================================
# Daily mean temperature time series (both sites)
# =====================================================
if len(datasets) >= 1:
    st.subheader("📈 Daily mean temperature / 日平均気温の推移")

    fig_daily, ax_daily = plt.subplots(figsize=(12, 4))
    for name, r in results.items():
        df_site = r["df"].copy()
        c = COLORS[name]
        ax_daily.plot(df_site["date"], df_site["mea_temp"],
                      linewidth=0.3, color=c["line"], alpha=0.7, label=name)

    ax_daily.set_xlabel("Year")
    ax_daily.set_ylabel("Daily mean temperature (°C)")
    ax_daily.set_title("Daily Mean Temperature 2005–2025")
    ax_daily.legend(fontsize=9)
    ax_daily.grid(axis="y", alpha=0.3)
    fig_daily.tight_layout()
    st.pyplot(fig_daily)

# =====================================================
# Temperature trend comparison
# =====================================================
st.subheader(f"3️⃣ Temperature trend / 気温のトレンド")
st.markdown(f"**{selected_label}** — {season_text}")

fig, ax = plt.subplots(figsize=(11, 5))

for name, r in results.items():
    y = r["yearly"]
    t = r["trend"]
    c = COLORS[name]
    ax.plot(y["year"], y["observed_temp"], marker="o", markersize=4,
            color=c["line"], label=f"{name} observed")
    if t:
        ax.plot(y["year"], y["trend"], linestyle="--", color=c["line"], alpha=0.6,
                label=f"{name} trend ({t['per_decade']:+.2f} °C/decade)")

    if "normal_temp" in y.columns:
        ax.plot(y["year"], y["normal_temp"], marker="s", markersize=3,
                color=c["fill"], label=f"{name} climatological normal")

ax.set_xlabel("Year")
ax.set_ylabel("Temperature (°C)")
ax.set_title(f"Annual {selected_label.split('/')[0].strip().lower()} — {season_text}")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
st.pyplot(fig)

# Trend metrics side by side
if len(results) == 2:
    col_t1, col_t2 = st.columns(2)
    for col_widget, (name, r) in zip([col_t1, col_t2], results.items()):
        t = r["trend"]
        if t:
            col_widget.markdown(f"**{name}**")
            mc1, mc2, mc3 = col_widget.columns(3)
            mc1.metric("°C / decade", f"{t['per_decade']:+.2f}")
            mc2.metric("p-value", f"{t['p']:.4f}")
            mc3.metric("R²", f"{t['r2']:.3f}")

            if t["slope"] > 0 and t["p"] < 0.05:
                col_widget.success("Significant warming trend detected ✅")
            elif t["slope"] > 0:
                col_widget.warning("Warming tendency (not significant at 5%)")
            else:
                col_widget.info("No clear warming trend")
else:
    name, r = list(results.items())[0]
    t = r["trend"]
    if t:
        m1, m2, m3 = st.columns(3)
        m1.metric("°C / decade", f"{t['per_decade']:+.2f}")
        m2.metric("p-value", f"{t['p']:.4f}")
        m3.metric("R²", f"{t['r2']:.3f}")

st.caption(
    "Note: This is an educational demonstration. "
    "Climate trend interpretation depends on period length, data quality, and statistical assumptions."
)

# =====================================================
# Anomaly (Nobeyama only — has normal values)
# =====================================================
if any("anomaly" in r["yearly"].columns for r in results.values()):
    st.subheader("4️⃣ Anomaly from climatological normal / 平年値からの偏差")
    st.markdown("Observed mean temperature minus climatological normal (Nobeyama only).")

    for name, r in results.items():
        y = r["yearly"]
        if "anomaly" not in y.columns:
            continue
        c = COLORS[name]

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        colors_bar = [c["line"] if v >= 0 else c["fill"] for v in y["anomaly"]]
        ax2.bar(y["year"], y["anomaly"], color=colors_bar)
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Year")
        ax2.set_ylabel("Anomaly (°C)")
        ax2.set_title(f"{name} — Mean temperature anomaly during {season_text}")
        ax2.grid(axis="y", alpha=0.3)
        st.pyplot(fig2)

        avg_a = y["anomaly"].mean()
        recent_a = y.tail(5)["anomaly"].mean()
        ca1, ca2 = st.columns(2)
        ca1.metric("Overall average anomaly", f"{avg_a:+.2f} °C")
        ca2.metric("Latest 5-year average", f"{recent_a:+.2f} °C")
else:
    if temp_col != "mea_temp":
        st.info("Select 'Mean temperature' to view anomaly charts (normal values available for Nobeyama only).")

# =====================================================
# High-temperature days
# =====================================================
st.subheader("5️⃣ High-temperature days / 高温日数")
st.markdown(f"Days with **daily maximum temperature ≥ {high_temp_threshold:.0f} °C** / 日最高気温 ≥ {high_temp_threshold:.0f} °C の日数")

if len(results) == 2:
    # Y軸の上限を揃える
    y_max = max(r["yearly"]["high_temp_days"].max() for r in results.values()) * 1.1

    col_h1, col_h2 = st.columns(2)
    for col_widget, (name, r) in zip([col_h1, col_h2], results.items()):
        y = r["yearly"]
        c = COLORS[name]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(y["year"], y["high_temp_days"], color=c["line"], alpha=0.8)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of days")
        ax3.set_title(f"{name}")
        ax3.set_ylim(0, y_max)
        ax3.grid(axis="y", alpha=0.3)
        col_widget.pyplot(fig3)
else:
    for name, r in results.items():
        y = r["yearly"]
        c = COLORS[name]
        fig3, ax3 = plt.subplots(figsize=(11, 4))
        ax3.bar(y["year"], y["high_temp_days"], color=c["line"], alpha=0.8)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Number of days")
        ax3.set_title(f"{name} — High-temperature days — {season_text}")
        ax3.grid(axis="y", alpha=0.3)
        st.pyplot(fig3)

# =====================================================
# Monthly comparison (side by side)
# =====================================================
if len(results) == 2:
    st.subheader("6️⃣ Monthly temperature comparison / 月別気温比較")

    # Compute monthly mean across all years
    fig_m, axes_m = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax_m, (name, r) in zip(axes_m, results.items()):
        df_site = r["df"].copy()
        if use_season:
            df_site = df_site[(df_site["month"] >= start_month) & (df_site["month"] <= end_month)]

        monthly = df_site.groupby("month")[temp_col].mean()
        c = COLORS[name]
        ax_m.bar(monthly.index, monthly.values, color=c["line"], alpha=0.8)
        ax_m.set_xlabel("Month")
        ax_m.set_ylabel("Temperature (°C)")
        ax_m.set_title(name)
        ax_m.set_xticks(monthly.index)
        ax_m.grid(axis="y", alpha=0.3)

    fig_m.suptitle(f"Monthly average {selected_label.split('/')[0].strip().lower()}", y=1.02)
    fig_m.tight_layout()
    st.pyplot(fig_m)

# =====================================================
# Future simulation
# =====================================================
st.subheader("7️⃣ Future simulation / 将来シミュレーション")
st.markdown(
    "Simple linear‐trend extrapolation for classroom demonstration — "
    "**not** an operational climate projection."
)

if st.button(f"▶ Run {future_years_n}-year simulation"):
    fig4, ax4 = plt.subplots(figsize=(11, 5))

    future_all = {}
    for name, r in results.items():
        y = r["yearly"]
        t = r["trend"]
        c = COLORS[name]
        if t is None:
            continue

        last_yr = int(y["year"].max())
        fut_yrs = np.arange(last_yr + 1, last_yr + future_years_n + 1)
        fut_pred = t["intercept"] + t["slope"] * fut_yrs

        ax4.plot(y["year"], y["observed_temp"], marker="o", markersize=4,
                 color=c["line"], label=f"{name} observed")
        ax4.plot(y["year"], y["trend"], linestyle="--", color=c["line"], alpha=0.5)
        ax4.plot(fut_yrs, fut_pred, marker="o", markersize=3, linestyle="--",
                 color=c["fill"], label=f"{name} projection")

        future_all[name] = pd.DataFrame({"year": fut_yrs, f"{name}_predicted": fut_pred})

    ax4.set_xlabel("Year")
    ax4.set_ylabel("Temperature (°C)")
    ax4.set_title(f"{future_years_n}-year simulation — {season_text}")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    st.pyplot(fig4)

    if future_all:
        keys = list(future_all.keys())
        merged = future_all[keys[0]]
        for k in keys[1:]:
            merged = merged.merge(future_all[k], on="year", how="outer")
        st.dataframe(merged)

        st.download_button(
            "📥 Download simulation CSV",
            data=merged.to_csv(index=False).encode("utf-8-sig"),
            file_name="future_simulation.csv",
            mime="text/csv",
        )

# =====================================================
# Annual summary download
# =====================================================
st.subheader("8️⃣ Annual summary / 年次サマリー")

for name, r in results.items():
    y = r["yearly"].copy()
    y.insert(0, "site", name)
    st.markdown(f"**{name}**")
    st.dataframe(y)

# Combined download
all_yearly = []
for name, r in results.items():
    tmp = r["yearly"].copy()
    tmp.insert(0, "site", name)
    all_yearly.append(tmp)

combined = pd.concat(all_yearly, ignore_index=True)
st.download_button(
    "📥 Download combined annual summary CSV",
    data=combined.to_csv(index=False).encode("utf-8-sig"),
    file_name="annual_summary_combined.csv",
    mime="text/csv",
)
