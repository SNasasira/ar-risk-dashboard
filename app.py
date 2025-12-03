# app.py
# Accounts Receivable Risk & Collections Dashboard

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ===========================
# PAGE CONFIG & COLORS
# ===========================
st.set_page_config(
    page_title="AR Risk & Collections Dashboard",
    layout="wide",
)

# Color palette
PRIMARY_BLUE = "#1e88e5"
MID_GREEN = "#43a047"
LIGHT_GREEN = "#a5d6a7"
LATE_RED = "#e53935"
LIGHT_RED = "#ffcdd2"
ONTIME_GREEN = "#2ecc71"
DSO_BLUE = "#636EFA" 
BACKGROUND_GRAY = "#f5f5f5"
AMOUNT_BLUE = "#1f77b4"
TERMS_BLUE = "#1f77b4"


# ===========================
# DATA LOADING
# ===========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["InvoiceDate", "DueDate", "PaymentDate"],
    )

    # Basic flags and derived fields
    df["LateFlag"] = (df["DaysLate"] > 0).astype(int)
    df["Status"] = np.where(df["LateFlag"] == 1, "Late", "On-time")

    df["DaysToCollect"] = (df["PaymentDate"] - df["InvoiceDate"]).dt.days

    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["InvoiceYear"] = df["InvoiceDate"].dt.year
    df["InvoiceMonthNum"] = df["InvoiceDate"].dt.month

    # Simple aging bucket (can be overridden later)
    def aging_bucket(d):
        if d <= 0:
            return "Current/On-time"
        elif d <= 30:
            return "1–30"
        elif d <= 60:
            return "31–60"
        elif d <= 90:
            return "61–90"
        else:
            return "90+"

    df["AgingBucket"] = df["DaysLate"].apply(aging_bucket)

    return df


# 🔁 EDIT THIS if your file name/path is different
DATA_PATH = "ar_ledger_50000_wb_calibrated.csv"
df = load_data(DATA_PATH)

# ===========================
# FILTERS (with Reset)
# ===========================
st.sidebar.header("Filters")

min_date = df["InvoiceDate"].min().date()
max_date = df["InvoiceDate"].max().date()

wb_options = ["All"] + sorted(df["WB_Category"].dropna().unique().tolist())
size_options = ["All"] + sorted(df["FirmSizeProxy"].dropna().unique().tolist())
status_options = ["All", "Late only", "On-time only"]

# Initialize session_state defaults once
if "filters_initialized" not in st.session_state:
    st.session_state["date_range"] = (min_date, max_date)
    st.session_state["wb_category"] = "All"
    st.session_state["firm_size"] = "All"
    st.session_state["status_filter"] = "All"
    st.session_state["filters_initialized"] = True


def reset_filters():
    st.session_state["date_range"] = (min_date, max_date)
    st.session_state["wb_category"] = "All"
    st.session_state["firm_size"] = "All"
    st.session_state["status_filter"] = "All"


# Reset button
st.sidebar.button("🔄 Reset filters", on_click=reset_filters)

# Widgets bound to session_state
date_range = st.sidebar.date_input(
    "Invoice date range",
    value=st.session_state["date_range"],
    min_value=min_date,
    max_value=max_date,
    key="date_range",
)

wb_category = st.sidebar.selectbox(
    "WB Category",
    options=wb_options,
    key="wb_category",
)

firm_size = st.sidebar.selectbox(
    "Firm Size (Proxy)",
    options=size_options,
    key="firm_size",
)

status_choice = st.sidebar.radio(
    "Invoice Status",
    options=status_options,
    key="status_filter",
)

# Apply filters
if isinstance(date_range, (tuple, list)):
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (df["InvoiceDate"].dt.date >= start_date) & (df["InvoiceDate"].dt.date <= end_date)

if wb_category != "All":
    mask &= df["WB_Category"] == wb_category

if firm_size != "All":
    mask &= df["FirmSizeProxy"] == firm_size

if status_choice == "Late only":
    mask &= df["LateFlag"] == 1
elif status_choice == "On-time only":
    mask &= df["LateFlag"] == 0

df_filtered = df[mask].copy()

st.sidebar.markdown(f"**Filtered invoices:** {len(df_filtered):,}")

# Download filtered data
csv_download = df_filtered.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="⬇️ Download filtered data (CSV)",
    data=csv_download,
    file_name="filtered_ar_ledger.csv",
    mime="text/csv",
)

# ===========================
# KPIs (use filtered data)
# ===========================
def compute_kpis(data: pd.DataFrame):
    total_amt = data["InvoiceAmount"].sum()
    late_rate = data["LateFlag"].mean() * 100 if len(data) > 0 else 0.0
    avg_days_late = data.loc[data["LateFlag"] == 1, "DaysLate"].mean()
    dso_proxy = data["DaysToCollect"].mean()
    return total_amt, late_rate, avg_days_late, dso_proxy


total_amt, late_rate, avg_days_late, dso_proxy = compute_kpis(df_filtered)

st.title("📊 Accounts Receivable Risk & Cash Collections Dashboard")
st.caption(
    "Synthetic 50,000-invoice AR ledger calibrated to World Bank Enterprise Survey benchmarks. "
    "Use filters and tabs to explore late payment risk, trends, and exposure by segment."
)

kpi_cols = st.columns(4)
kpi_cols[0].metric("Total Invoice Amount", f"${total_amt:,.0f}")
kpi_cols[1].metric("Late Rate", f"{late_rate:.1f}%")
kpi_cols[2].metric("Avg Days Late (late only)", f"{avg_days_late:.1f} days" if not np.isnan(avg_days_late) else "N/A")
kpi_cols[3].metric("DSO Proxy", f"{dso_proxy:.1f} days" if not np.isnan(dso_proxy) else "N/A")

# ===========================
# COMMON AGGREGATES (filtered)
# ===========================
# Segment-level summary
seg = (
    df_filtered.groupby("WB_Category")
    .agg(
        Invoices=("InvoiceNumber", "nunique"),
        TotalAmount=("InvoiceAmount", "sum"),
        LateRate=("LateFlag", "mean"),
        AvgDaysLate=("DaysLate", lambda x: x[x > 0].mean()),
    )
    .reset_index()
)
seg["LateRatePct"] = seg["LateRate"] * 100
seg["AvgDaysLate"] = seg["AvgDaysLate"].fillna(0)
seg_top = seg.sort_values("LateRate", ascending=False)

# ===========================
# TABS
# ===========================
tabs = st.tabs(
    [
        "Overview",
        "Risk by Segment",
        "Time Trends",
        "Drivers & Correlations",
        "Distributions & Aging",
        "Predictive Insights",
    ]
)

# --------------------------------------------------------
# TAB 0: OVERVIEW
# --------------------------------------------------------
with tabs[0]:
    st.subheader("Portfolio Overview")

    # Work off the currently filtered data
    df_overview = df.copy()

    # ------------------------------------------------------------------
    # 1. Late vs On-time (by Amount) – Donut chart
    # ------------------------------------------------------------------
    # Define late vs on-time based on DaysLate (>0 -> Late)
    df_overview["StatusBucket"] = np.where(df_overview["DaysLate"] > 0,
                                           "Late",
                                           "On-time")

    status_agg = (
        df_overview
        .groupby("StatusBucket", as_index=False)["InvoiceAmount"]
        .sum()
    )

    # Ensure consistent order: Late first, On-time second
    status_agg["StatusBucket"] = pd.Categorical(
        status_agg["StatusBucket"],
        categories=["Late", "On-time"],
        ordered=True
    )
    status_agg = status_agg.sort_values("StatusBucket")

    donut_colors = {
        "Late": "#E74C3C",      # strong red
        "On-time": "#2ECC71"    # bright green
    }

    fig_donut = px.pie(
        status_agg,
        names="StatusBucket",
        values="InvoiceAmount",
        hole=0.6,
        color="StatusBucket",
        color_discrete_map=donut_colors
    )

    fig_donut.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{percent:.1%}",
        pull=[0.04 if s == "Late" else 0 for s in status_agg["StatusBucket"]],
        marker=dict(line=dict(color="white", width=2))
    )

    fig_donut.update_layout(
        title=dict(
            text="Late vs On-time (by Amount)",
            x=0.5,
            xanchor="center"
        ),
        showlegend=False,
        margin=dict(t=60, l=0, r=0, b=0)
    )

    # ------------------------------------------------------------------
    # 2. Top Risk Categories table (by Late Rate)
    # ------------------------------------------------------------------
    # Per-category stats
    cat_grp = df_overview.groupby("WB_Category")

    cat_summary = pd.DataFrame({
        "WB_Category": cat_grp.size().index,
        "Invoices": cat_grp.size().values,
        "TotalAmount": cat_grp["InvoiceAmount"].sum().values,
        "LateInvoices": cat_grp.apply(lambda g: (g["DaysLate"] > 0).sum()).values,
        "AvgDaysLate": cat_grp.apply(lambda g: g.loc[g["DaysLate"] > 0, "DaysLate"].mean()).values
    })

    cat_summary["LateRate"] = cat_summary["LateInvoices"] / cat_summary["Invoices"] * 100
    cat_summary["AvgDaysLate"] = cat_summary["AvgDaysLate"].fillna(0)

    # Sort by Late Rate descending
    cat_summary = cat_summary.sort_values("LateRate", ascending=False)

    # Nicely formatted copy for display
    display_cols = ["WB_Category", "Invoices", "TotalAmount", "LateRate", "AvgDaysLate"]
    cat_display = cat_summary[display_cols].copy()
    cat_display["TotalAmount"] = cat_display["TotalAmount"].map("${:,.0f}".format)
    cat_display["LateRate"] = cat_display["LateRate"].map("{:.1f}%".format)
    cat_display["AvgDaysLate"] = cat_display["AvgDaysLate"].map("{:.1f}".format)

    # ------------------------------------------------------------------
    # 3. Exposure by Category (Color = Late Rate)
    # ------------------------------------------------------------------
    fig_exposure = px.bar(
        cat_summary,
        x="WB_Category",
        y="TotalAmount",
        color="LateRate",
        color_continuous_scale="Reds",
        labels={
            "WB_Category": "WB Category",
            "TotalAmount": "Total Invoice Amount",
            "LateRate": "Late Rate (%)"
        },
        title="Exposure by Category (Colored by Late Rate)"
    )

    fig_exposure.update_layout(
        xaxis_tickangle=-35,
        margin=dict(t=60, l=40, r=40, b=80),
        coloraxis_colorbar=dict(title="Late Rate (%)")
    )

    # ------------------------------------------------------------------
    # 4. OPTIONAL: Risk Bubble Plot (Exposure × Late Rate × Avg Days Late)
    # ------------------------------------------------------------------
    show_bubbles = st.checkbox(
        "Show Risk Bubble Plot (Exposure × Late Rate × Avg Days Late)",
        value=False
    )

    if show_bubbles:
        fig_bubble = px.scatter(
            cat_summary,
            x="LateRate",
            y="AvgDaysLate",
            size="TotalAmount",
            color="LateRate",
            color_continuous_scale="Reds",
            hover_name="WB_Category",
            labels={
                "LateRate": "Late Rate (%)",
                "AvgDaysLate": "Avg Days Late (Late invoices)",
                "TotalAmount": "Total Invoice Amount"
            },
            title="Category Risk Bubble Plot"
        )

        fig_bubble.update_layout(
            margin=dict(t=60, l=40, r=40, b=40),
            coloraxis_colorbar=dict(title="Late Rate (%)")
        )

    # ------------------------------------------------------------------
    # Layout on the page
    # ------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("**Top Risk Categories (by Late Rate)**")
        st.dataframe(
            cat_display.reset_index(drop=True),
            use_container_width=True,
            height=260
        )

    st.plotly_chart(fig_exposure, use_container_width=True)

    if show_bubbles:
        st.plotly_chart(fig_bubble, use_container_width=True)


# --------------------------------------------------------
# TAB 1: RISK BY SEGMENT (DEEP)
# --------------------------------------------------------
with tabs[1]:
    st.subheader("Risk by Segment")

    # 1. Heatmap Late Rate by Category & Terms
    st.markdown("#### Late Rate (%) by Category & Payment Terms")
    if len(df_filtered) > 0:
        heatmap_df = (
            df_filtered.groupby(["WB_Category", "TermsDays"])["LateFlag"]
            .mean()
            .reset_index()
        )
        heatmap_df["LateRatePct"] = heatmap_df["LateFlag"] * 100

        pivot = heatmap_df.pivot(
            index="WB_Category", columns="TermsDays", values="LateRatePct"
        )
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            aspect="auto",
            labels=dict(x="Payment Terms (Days)", y="WB Category", color="Late Rate (%)"),
        )
        fig_heat.update_layout(margin=dict(l=60, r=40, t=40, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Darker cells highlight category–terms combinations where late payment is structurally high."
        )
    else:
        st.info("No data to show heatmap for current filters.")

    st.markdown("---")

    # 2. Segment Risk Profile (LateRate x AvgDaysLate x Exposure)
    st.markdown("#### Segment Risk Profile (Late Rate × Avg Days Late × Exposure)")
    if len(seg_top) > 0:
        seg_profile = (
            df_filtered.groupby("WB_Category")
            .agg(
                LateRate=("LateFlag", "mean"),
                AvgDaysLate=("DaysLate", "mean"),
                Exposure=("InvoiceAmount", "sum"),
            )
            .reset_index()
        )
        seg_profile["LateRatePct"] = seg_profile["LateRate"] * 100

        fig_profile = px.scatter(
            seg_profile,
            x="LateRatePct",
            y="AvgDaysLate",
            size="Exposure",
            color="WB_Category",
            hover_data=["Exposure"],
            text="WB_Category",
            labels={
                "LateRatePct": "Late Rate (%)",
                "AvgDaysLate": "Average Days Late",
            },
        )
        fig_profile.update_traces(textposition="top center")
        fig_profile.update_layout(margin=dict(l=40, r=40, t=40, b=40))
        st.plotly_chart(fig_profile, use_container_width=True)
        st.caption(
            "Segments in the upper-right with larger bubbles combine high late rate, long delays, and large balances."
        )
    else:
        st.info("No segment risk profile to display for current filters.")

    st.markdown("---")

    # 3. Aging Structure by Category
    st.markdown("#### AR Aging Structure by Category")
    if len(df_filtered) > 0:
        aging_cat_df = df_filtered.copy()
        aging_cat_df["AgingBucketAdv"] = pd.cut(
            aging_cat_df["DaysLate"],
            bins=[-9999, 0, 30, 60, 90, 9999],
            labels=["On-time/Current", "1–30", "31–60", "61–90", "90+"],
        )
        aging_summary = (
            aging_cat_df.groupby(["WB_Category", "AgingBucketAdv"])["InvoiceAmount"]
            .sum()
            .reset_index()
        )

        fig_aging_cat = px.bar(
            aging_summary,
            x="WB_Category",
            y="InvoiceAmount",
            color="AgingBucketAdv",
            barmode="stack",
            labels={"InvoiceAmount": "Total Amount", "AgingBucketAdv": "Aging Bucket"},
        )
        fig_aging_cat.update_layout(
            margin=dict(l=40, r=40, t=40, b=80),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_aging_cat, use_container_width=True)
        st.caption(
            "Allows you to see which categories have chronic late balances sitting in 61–90 or 90+ days."
        )
    else:
        st.info("No aging data to display for current filters.")

    st.markdown("---")

    # 4. Exposure by Category (Colored by risk)
    st.markdown("#### Exposure by Category (Colored by Late Rate)")
    if len(seg_top) > 0:
        seg_expo = (
            df_filtered.groupby("WB_Category")
            .agg(
                Exposure=("InvoiceAmount", "sum"),
                LateRate=("LateFlag", "mean"),
            )
            .reset_index()
        )
        seg_expo["LateRatePct"] = seg_expo["LateRate"] * 100

        fig_expo = px.bar(
            seg_expo.sort_values("Exposure", ascending=False),
            x="WB_Category",
            y="Exposure",
            color="LateRatePct",
            color_continuous_scale="Reds",
            labels={"Exposure": "Total Invoice Amount", "LateRatePct": "Late Rate (%)"},
        )
        fig_expo.update_layout(
            xaxis_tickangle=-30,
            margin=dict(l=40, r=40, t=40, b=80),
            coloraxis_colorbar=dict(title="Late Rate (%)"),
        )
        st.plotly_chart(fig_expo, use_container_width=True)
    else:
        st.info("No exposure data to display for current filters.")

    st.markdown("---")

    # 5. Payment Behavior Curves (CDF)
    st.markdown("#### Payment Behavior Curves (Cumulative % of Invoices Paid vs Days)")
    if len(df_filtered) > 0:
        cdf_df = df_filtered.dropna(subset=["DaysToCollect"]).copy()
        cdf_df["DaysToCollect"] = cdf_df["DaysToCollect"].clip(lower=0)

        fig_cdf = go.Figure()
        for cat in cdf_df["WB_Category"].unique():
            temp = cdf_df[cdf_df["WB_Category"] == cat]["DaysToCollect"].sort_values()
            if len(temp) == 0:
                continue
            pct = np.arange(1, len(temp) + 1) / len(temp) * 100
            fig_cdf.add_trace(
                go.Scatter(
                    x=temp,
                    y=pct,
                    mode="lines",
                    name=cat,
                )
            )

        fig_cdf.update_layout(
            xaxis_title="Days to Collect",
            yaxis_title="% of Invoices Paid",
            yaxis=dict(range=[0, 100]),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        st.plotly_chart(fig_cdf, use_container_width=True)
        st.caption(
            "Categories with slower CDF curves have long-tail delays that drag on liquidity."
        )
    else:
        st.info("No payment behavior data for current filters.")

# --------------------------------------------------------
# TAB 2: TIME TRENDS
# --------------------------------------------------------
with tabs[2]:
    st.header("Historical Risk & Cash Cycle")

    # --------------------------------------------------
    # 0. Data for this tab
    # --------------------------------------------------
    if "filtered_df" in locals():
        df_trend = filtered_df.copy()
    else:
        df_trend = df.copy()

    df_trend["InvoiceDate"] = pd.to_datetime(df_trend["InvoiceDate"])
    df_trend["InvoiceMonth"] = df_trend["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    # Late flag & simple DSO proxy
    df_trend["LateFlag"] = (df_trend["Status"] == "Late").astype(int)
    df_trend["DaysToCollect"] = df_trend["TermsDays"] + df_trend["DaysLate"]

    # --------------------------------------------------
    # 1. Monthly Late Rate & DSO (3-month rolling)
    # --------------------------------------------------
    monthly = (
        df_trend
        .groupby("InvoiceMonth")
        .agg(
            LateRate_pct=("LateFlag", lambda x: 100 * x.mean()),
            DSO_days=("DaysToCollect", "mean"),
        )
        .sort_index()
    )

    monthly["LateRate_3m"] = monthly["LateRate_pct"].rolling(3, min_periods=1).mean()
    monthly["DSO_3m"] = monthly["DSO_days"].rolling(3, min_periods=1).mean()

    fig_trend = go.Figure()

    fig_trend.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["LateRate_3m"],
            name="Late Rate (3-month avg)",
            mode="lines",
            line=dict(color=LATE_RED, width=2),
        )
    )

    fig_trend.add_trace(
        go.Scatter(
            x=monthly.index,
            y=monthly["DSO_3m"],
            name="DSO (3-month avg)",
            mode="lines",
            yaxis="y2",
            line=dict(color=DSO_BLUE, width=2),
        )
    )

    fig_trend.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(title="Invoice Month"),
        yaxis=dict(title="Late Rate (%)"),
        yaxis2=dict(
            title="DSO (days)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig_trend, use_container_width=True, key="trend_late_dso")

    # --------------------------------------------------
    # 2. Volume share of Late vs On-time over time
    # --------------------------------------------------
    by_month_share = (
        df_trend
        .groupby("InvoiceMonth")
        .agg(LateShare=("LateFlag", "mean"))
        .sort_index()
    )
    by_month_share["OnTimeShare"] = 1 - by_month_share["LateShare"]

    fig_stack = go.Figure()

    fig_stack.add_trace(
        go.Scatter(
            x=by_month_share.index,
            y=by_month_share["LateShare"] * 100,
            stackgroup="one",
            name="Late",
            mode="lines",
            line=dict(color=LATE_RED),
        )
    )

    fig_stack.add_trace(
        go.Scatter(
            x=by_month_share.index,
            y=by_month_share["OnTimeShare"] * 100,
            stackgroup="one",
            name="On-time",
            mode="lines",
            line=dict(color=ONTIME_GREEN),
        )
    )

    fig_stack.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="Invoice Month"),
        yaxis=dict(title="Share of Invoices", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        showlegend=True,
    )

    st.subheader("Volume Share of Late vs On-time Invoices Over Time")
    st.plotly_chart(fig_stack, use_container_width=True, key="trend_volume_share")

    # --------------------------------------------------
    # 3. Seasonality – average Late Rate by calendar month
    # --------------------------------------------------
    df_trend["MonthName"] = df_trend["InvoiceDate"].dt.month_name().str.slice(0, 3)
    df_trend["MonthNo"] = df_trend["InvoiceDate"].dt.month

    seasonality = (
        df_trend
        .groupby(["MonthNo", "MonthName"])
        .agg(LateRate_pct=("LateFlag", lambda x: 100 * x.mean()))
        .reset_index()
        .sort_values("MonthNo")
    )

    fig_season = px.bar(
        seasonality,
        x="MonthName",
        y="LateRate_pct",
        labels={"MonthName": "Calendar Month", "LateRate_pct": "Late Rate (%)"},
        text=seasonality["LateRate_pct"].round(1).astype(str) + "%",
    )
    fig_season.update_traces(marker_color=LATE_RED, textposition="outside")
    fig_season.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(range=[0, max(65, seasonality["LateRate_pct"].max() + 5)]),
    )

    st.subheader("Seasonality: Average Late Rate by Calendar Month")
    st.plotly_chart(fig_season, use_container_width=True, key="trend_seasonality")

    # --------------------------------------------------
    # 4. Year-over-year Late Rate
    # --------------------------------------------------
    df_trend["Year"] = df_trend["InvoiceDate"].dt.year

    yoy = (
        df_trend
        .groupby("Year")
        .agg(LateRate_pct=("LateFlag", lambda x: 100 * x.mean()))
        .reset_index()
        .sort_values("Year")
    )

    fig_yoy = px.bar(
        yoy,
        x="Year",
        y="LateRate_pct",
        labels={"LateRate_pct": "Late Rate (%)"},
        text=yoy["LateRate_pct"].round(1).astype(str) + "%",
    )
    fig_yoy.update_traces(marker_color=DSO_BLUE, textposition="outside")
    fig_yoy.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(range=[0, max(65, yoy["LateRate_pct"].max() + 5)]),
    )

    st.subheader("Year-over-year Late Rate Trend")
    st.plotly_chart(fig_yoy, use_container_width=True, key="trend_yoy")

# --------------------------------------------------------
# TAB 3: DRIVERS & CORRELATIONS
# --------------------------------------------------------
with tabs[3]:
    st.header("Drivers of Delay & Correlations")
    df_drivers = df.copy()

    # ============================================================
    # 1. CORRELATION HEATMAP + SCATTER
    # ============================================================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Correlation Heatmap (Numeric Features)")

        num_cols = ["InvoiceAmount", "TermsDays", "DaysLate", "DaysToCollect"]

        # Avoid crashing if missing columns
        num_cols = [c for c in num_cols if c in df_drivers.columns]

        corr = df_drivers[num_cols].corr()

        fig_corr = px.imshow(
            corr,
            x=num_cols,
            y=num_cols,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            text_auto=".2f",
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")

    with col2:
        st.subheader("Days to Collect vs Invoice Amount (Colored by Category)")

        fig_scatter = px.scatter(
            df_drivers,
            x="DaysToCollect",
            y="InvoiceAmount",
            color="WB_Category",
            hover_data=["InvoiceAmount", "DaysToCollect", "WB_Category"],
        )
        fig_scatter.update_yaxes(type="log")
        st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_days_vs_amt")

    # ============================================================
    # 2. LATE RATE + EXPOSURE BY AMOUNT BUCKET
    # ============================================================
    st.subheader("Late Rate and Exposure by Invoice Amount Bucket")

    if "InvoiceAmountBucket" not in df_drivers.columns:
        df_drivers["InvoiceAmountBucket"] = pd.cut(
            df_drivers["InvoiceAmount"],
            bins=[0, 1000, 10000, 50000, df_drivers["InvoiceAmount"].max()],
            labels=["< $1K", "$1K–$10K", "$10K–$50K", ">$50K"],
            include_lowest=True,
        )

    bucket_grp = (
        df_drivers.groupby("InvoiceAmountBucket")
        .agg(
            LateRate=("LateFlag", "mean"),
            TotalAmount=("InvoiceAmount", "sum")
        )
        .reset_index()
    )

    bucket_grp["LateRatePct"] = bucket_grp["LateRate"] * 100
    bucket_grp["TotalAmountBn"] = bucket_grp["TotalAmount"] / 1e9

    fig_bucket = go.Figure()

    fig_bucket.add_trace(go.Bar(
        x=bucket_grp["InvoiceAmountBucket"],
        y=bucket_grp["LateRatePct"],
        name="Late Rate (%)",
        marker_color="#EF553B",
        text=bucket_grp["LateRatePct"].round(1).astype(str) + "%",
        textposition="outside",
    ))

    fig_bucket.add_trace(go.Bar(
        x=bucket_grp["InvoiceAmountBucket"],
        y=bucket_grp["TotalAmountBn"],
        name="Total Amount (B)",
        marker_color="#1f77b4",
        opacity=0.5,
    ))

    fig_bucket.update_layout(barmode="group")
    st.plotly_chart(fig_bucket, use_container_width=True, key="bucket_late_exposure")

    # ============================================================
    # 3. LATE RATE BY PAYMENT TERMS × FIRM SIZE
    # ============================================================
    st.subheader("Late Rate by Payment Terms and Firm Size")

    terms_size = (
        df_drivers.groupby(["TermsDays", "FirmSizeProxy"])
        .agg(LateRate=("LateFlag", "mean"))
        .reset_index()
    )
    terms_size["LateRatePct"] = terms_size["LateRate"] * 100

    fig_terms = px.bar(
        terms_size,
        x="TermsDays",
        y="LateRatePct",
        color="FirmSizeProxy",
        barmode="group",
        text=terms_size["LateRatePct"].round(1).astype(str) + "%",
    )
    fig_terms.update_traces(textposition="outside")
    st.plotly_chart(fig_terms, use_container_width=True, key="terms_firm_late")

    # ============================================================
    # 4. AVERAGE DAYS TO COLLECT BY WB CATEGORY
    # ============================================================
    st.subheader("Average Days to Collect by WB Category")

    avg_days = (
        df_drivers.groupby("WB_Category")["DaysToCollect"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig_days = px.bar(
        avg_days,
        x="WB_Category",
        y="DaysToCollect",
        text=avg_days["DaysToCollect"].round(1).astype(str) + " days",
        color_discrete_sequence=["#1f77b4"],
    )

    fig_days.update_traces(textposition="outside")
    st.plotly_chart(fig_days, use_container_width=True, key="avg_days_collect")




# --------------------------------------------------------
# TAB 4: DISTRIBUTIONS & AGING
# --------------------------------------------------------
with tabs[4]:
    st.subheader("Payment Delay Distribution & Aging")

    if len(df_filtered) == 0:
        st.info("No data to display for current filters.")
    else:
        # Top row: Violin + Advanced Aging
        top_row = st.columns(2)

        # Violin by Firm Size
        with top_row[0]:
            st.markdown("#### Distribution of Days Late by Firm Size (Late Invoices Only)")
            late_only = df_filtered[df_filtered["LateFlag"] == 1].copy()
            if len(late_only) > 0:
                fig_violin = px.violin(
                    late_only,
                    x="FirmSizeProxy",
                    y="DaysLate",
                    box=True,
                    points="all",
                    color="FirmSizeProxy",
                    labels={"FirmSizeProxy": "Firm Size", "DaysLate": "Days Late"},
                )
                fig_violin.update_layout(
                    showlegend=False,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_violin, use_container_width=True)
            else:
                st.info("No late invoices for current filters.")

        # Advanced Aging by Amount
        with top_row[1]:
            st.markdown("#### AR Aging by Amount – On-time vs Overdue")

            aging_df = df_filtered.copy()
            aging_df["AgingBucketAdv"] = pd.cut(
                aging_df["DaysLate"],
                bins=[-9999, 0, 30, 60, 90, 9999],
                labels=[
                    "Current/On-time",
                    "1–30",
                    "31–60",
                    "61–90",
                    "90+",
                ],
                right=True,
                include_lowest=True,
            )
            aging_df = aging_df.dropna(subset=["AgingBucketAdv"])

            aging_agg = (
                aging_df.groupby("AgingBucketAdv", observed=True)["InvoiceAmount"]
                .sum()
                .reindex(["Current/On-time", "1–30", "31–60", "61–90", "90+"])
                .reset_index()
            )

            total_amt_aging = aging_agg["InvoiceAmount"].sum()
            aging_agg["PctOfTotal"] = np.where(
                total_amt_aging > 0,
                aging_agg["InvoiceAmount"] / total_amt_aging * 100,
                0,
            )

            aging_agg["LateStatus"] = np.where(
                aging_agg["AgingBucketAdv"] == "Current/On-time",
                "On-time",
                "Overdue",
            )

            # Colors
            LATE_ORANGE = "#f39c12"
            LATE_ORANGE_DARK = "#e67e22"
            LATE_RED_DARK = "#c0392b"

            bucket_colors = {
                "Current/On-time": ONTIME_GREEN,
                "1–30": LATE_ORANGE,
                "31–60": LATE_ORANGE_DARK,
                "61–90": LATE_RED,
                "90+": LATE_RED_DARK,
            }
            aging_agg["Color"] = aging_agg["AgingBucketAdv"].map(bucket_colors)

            fig_aging = go.Figure()
            fig_aging.add_trace(
                go.Bar(
                    x=aging_agg["AgingBucketAdv"],
                    y=aging_agg["InvoiceAmount"],
                    marker_color=aging_agg["Color"],
                    text=[f"{p:,.1f}%" for p in aging_agg["PctOfTotal"]],
                    textposition="outside",
                    customdata=np.stack(
                        [aging_agg["LateStatus"], aging_agg["PctOfTotal"]], axis=-1
                    ),
                    hovertemplate=(
                        "<b>Aging Bucket:</b> %{x}<br>"
                        "<b>Status:</b> %{customdata[0]}<br>"
                        "<b>Amount:</b> $%{y:,.0f}<br>"
                        "<b>% of Total AR:</b> %{customdata[1]:.1f}%<extra></extra>"
                    ),
                )
            )
            fig_aging.update_layout(
                yaxis_title="Invoice Amount",
                xaxis_title="Aging Bucket",
                plot_bgcolor="white",
                bargap=0.25,
                margin=dict(l=40, r=20, t=40, b=80),
                showlegend=False,
                height=450,
                annotations=[
                    dict(
                        x=0,
                        y=1.12,
                        xref="paper",
                        yref="paper",
                        xanchor="left",
                        showarrow=False,
                        text="Green = On-time · Orange/Red = Overdue",
                        font=dict(size=12, color="#555555"),
                    )
                ],
            )
            st.plotly_chart(fig_aging, use_container_width=True)

        # Second row: Boxplot & Histogram
        bottom_row = st.columns(2)

        with bottom_row[0]:
            st.markdown("#### Boxplot – Days Late by Payment Terms (Late Invoices)")
            late_box = df_filtered[df_filtered["LateFlag"] == 1].copy()
            if len(late_box) > 0:
                fig_box = px.box(
                    late_box,
                    x="TermsDays",
                    y="DaysLate",
                    labels={"TermsDays": "Payment Terms (Days)", "DaysLate": "Days Late"},
                )
                fig_box.update_traces(marker_color=PRIMARY_BLUE)
                fig_box.update_layout(margin=dict(l=40, r=40, t=40, b=40))
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No late invoices for current filters.")

        with bottom_row[1]:
            st.markdown("#### Histogram – Overall Days Late")
            fig_hist = px.histogram(
                df_filtered,
                x="DaysLate",
                nbins=40,
                labels={"DaysLate": "Days Late"},
            )
            fig_hist.update_traces(marker_color=PRIMARY_BLUE)
            fig_hist.update_layout(margin=dict(l=40, r=40, t=40, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)

       
# --------------------------------------------------------
# TAB 5: PREDICTIVE INSIGHTS (LOGISTIC REGRESSION)
# --------------------------------------------------------
@st.cache_resource
def train_logistic_model(full_df: pd.DataFrame):
    model_df = full_df.copy()
    y = model_df["LateFlag"]

    num_features = ["InvoiceAmount", "TermsDays", "InvoiceYear", "InvoiceMonthNum"]
    cat_features = ["WB_Category", "FirmSizeProxy"]

    X = model_df[num_features + cat_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    log_reg = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs",
    )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", log_reg)])
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    feature_names = clf.named_steps["preprocess"].get_feature_names_out()
    coefs = clf.named_steps["model"].coef_[0]
    feat_imp = pd.DataFrame(
        {"Feature": feature_names, "Coefficient": coefs, "AbsCoef": np.abs(coefs)}
    ).sort_values("AbsCoef", ascending=False)

    proba_all = clf.predict_proba(X)[:, 1]
    model_df["PredLateProb"] = proba_all

    cat_risk = (
        model_df.groupby("WB_Category")
        .agg(
            AvgPredProb=("PredLateProb", "mean"),
            LateRate=("LateFlag", "mean"),
            Invoices=("InvoiceNumber", "nunique"),
        )
        .reset_index()
    )
    cat_risk["AvgPredProbPct"] = cat_risk["AvgPredProb"] * 100
    cat_risk["LateRatePct"] = cat_risk["LateRate"] * 100

    return clf, acc, f1, auc, feat_imp, cat_risk


with tabs[5]:
    st.subheader("Predictive Insights & Root Cause Analysis")

    if len(df) == 0:
        st.info("No data available to train model.")
    else:
        with st.spinner("Training logistic regression model on full portfolio..."):
            clf, acc, f1, auc, feat_imp, cat_risk = train_logistic_model(df)

        st.markdown("#### Model Performance (Baseline Features)")
        perf_cols = st.columns(3)
        perf_cols[0].metric("Accuracy", f"{acc:.3f}")
        perf_cols[1].metric("F1-score", f"{f1:.3f}")
        perf_cols[2].metric("ROC-AUC", f"{auc:.3f}")

        st.caption(
            "Performance is measured on a 20% holdout sample. ROC-AUC around 0.74 indicates the model "
            "can rank invoices by lateness risk reasonably well, given invoice-level features."
        )

        st.markdown("#### Feature Importance (Logistic Regression Coefficients)")
        top_n = st.slider("Number of features to display", 5, 20, 10, key="feat_topn")

        feat_display = feat_imp.head(top_n).sort_values("Coefficient")

        fig_feat = px.bar(
            feat_display,
            x="Coefficient",
            y="Feature",
            orientation="h",
            color="Coefficient",
            color_continuous_scale="RdBu",
        )
        fig_feat.update_layout(
            margin=dict(l=60, r=40, t=40, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

        st.caption(
            "Positive coefficients push an invoice towards 'Late', while negative coefficients make lateness less likely."
        )

        st.markdown("#### Predicted Probability of Lateness by WB Category")
        cat_risk_disp = cat_risk.sort_values("AvgPredProbPct", ascending=False)
        fig_cat = px.bar(
            cat_risk_disp,
            x="WB_Category",
            y="AvgPredProbPct",
            text=cat_risk_disp["AvgPredProbPct"].map(lambda x: f"{x:.1f}%"),
            labels={
                "WB_Category": "WB Category",
                "AvgPredProbPct": "Predicted Late Probability (%)",
            },
        )
        fig_cat.update_traces(marker_color=LATE_RED)
        fig_cat.update_layout(
            xaxis_tickangle=-30,
            margin=dict(l=40, r=40, t=40, b=80),
        )
        st.plotly_chart(fig_cat, use_container_width=True)

        st.caption(
            "The model highlights structurally riskier sectors, which can inform credit limits and collection prioritization."
        )

