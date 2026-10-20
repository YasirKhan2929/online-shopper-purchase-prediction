
from pathlib import Path

import pandas as pd
import streamlit as st


# -----------------------------
# 1. Page configuration
# -----------------------------
st.set_page_config(
    page_title="Online Shopper Analytics",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "online_shoppers_intention.csv"


# -----------------------------
# 2. Load the historical data
# -----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Convert the target into consistent True/False values.
    if df["Revenue"].dtype != bool:
        df["Revenue"] = (
            df["Revenue"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    df = df.dropna(subset=["Revenue"]).copy()
    df["Revenue"] = df["Revenue"].astype(bool)

    return df


# -----------------------------
# 3. Dashboard heading
# -----------------------------
st.title("📊 Online Shopper Business Analytics")
st.write(
    "Explore historical shopping sessions, purchase behaviour, "
    "and browsing patterns."
)

st.info(
    "This dashboard analyses the historical dataset. "
    "It does not display live visitors from the demo store."
)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

try:
    df = load_data(DATA_PATH)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# -----------------------------
# 4. Sidebar filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

months = sorted(df["Month"].dropna().unique().tolist())
visitor_types = sorted(df["VisitorType"].dropna().unique().tolist())

selected_months = st.sidebar.multiselect(
    "Select month(s)",
    options=months,
    default=months,
)

selected_visitors = st.sidebar.multiselect(
    "Select visitor type(s)",
    options=visitor_types,
    default=visitor_types,
)

weekend_filter = st.sidebar.selectbox(
    "Visit day",
    ["All days", "Weekend only", "Weekdays only"],
)

filtered_df = df[
    df["Month"].isin(selected_months)
    & df["VisitorType"].isin(selected_visitors)
].copy()

if weekend_filter == "Weekend only":
    filtered_df = filtered_df[filtered_df["Weekend"] == True]
elif weekend_filter == "Weekdays only":
    filtered_df = filtered_df[filtered_df["Weekend"] == False]

if filtered_df.empty:
    st.warning("No sessions match these filters. Change the filters in the sidebar.")
    st.stop()


# -----------------------------
# 5. Key business metrics
# -----------------------------
total_sessions = len(filtered_df)
purchases = int(filtered_df["Revenue"].sum())
non_purchases = total_sessions - purchases
purchase_rate = purchases / total_sessions * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sessions", f"{total_sessions:,}")
col2.metric("Purchases", f"{purchases:,}")
col3.metric("Non-purchases", f"{non_purchases:,}")
col4.metric("Purchase Rate", f"{purchase_rate:.2f}%")


st.divider()


# -----------------------------
# 6. Purchase rate by visitor type
# -----------------------------
st.subheader("1. Which visitor types purchase more often?")

visitor_summary = (
    filtered_df.groupby("VisitorType", observed=False)
    .agg(
        Sessions=("Revenue", "size"),
        Purchases=("Revenue", "sum"),
        Purchase_Rate=("Revenue", "mean"),
    )
    .reset_index()
)

visitor_summary["Purchase_Rate"] *= 100
visitor_summary["Purchases"] = visitor_summary["Purchases"].astype(int)

left, right = st.columns(2)

with left:
    st.write("**Purchase rate by visitor type (%)**")
    st.bar_chart(
        visitor_summary.set_index("VisitorType")[["Purchase_Rate"]]
    )

with right:
    st.write("**Visitor type summary**")
    st.dataframe(
        visitor_summary.rename(
            columns={
                "VisitorType": "Visitor Type",
                "Sessions": "Sessions",
                "Purchases": "Purchases",
                "Purchase_Rate": "Purchase Rate (%)",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------
# 7. Monthly purchase trends
# -----------------------------
st.subheader("2. How does purchase rate vary by month?")

month_order = [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

monthly = (
    filtered_df.groupby("Month", observed=False)
    .agg(
        Sessions=("Revenue", "size"),
        Purchases=("Revenue", "sum"),
        Purchase_Rate=("Revenue", "mean"),
    )
    .reset_index()
)

monthly["Month"] = pd.Categorical(
    monthly["Month"],
    categories=month_order,
    ordered=True,
)

monthly = monthly.sort_values("Month")
monthly["Purchase_Rate"] *= 100
monthly["Month"] = monthly["Month"].astype(str)

left, right = st.columns(2)

with left:
    st.write("**Purchase rate by month (%)**")
    st.line_chart(
        monthly.set_index("Month")[["Purchase_Rate"]]
    )

with right:
    st.write("**Number of shopping sessions by month**")
    st.bar_chart(
        monthly.set_index("Month")[["Sessions"]]
    )


# -----------------------------
# 8. Browsing behaviour
# -----------------------------
st.subheader("3. How does browsing behaviour differ?")

behaviour_columns = [
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
]

available_columns = [
    column for column in behaviour_columns
    if column in filtered_df.columns
]

behaviour = (
    filtered_df.groupby("Revenue")[available_columns]
    .mean()
    .rename(index={
        False: "No purchase",
        True: "Purchase",
    })
)

behaviour = behaviour.rename(
    columns={
        "ProductRelated": "Average product pages",
        "ProductRelated_Duration": "Average time on product pages (seconds)",
        "BounceRates": "Average bounce rate",
        "ExitRates": "Average exit rate",
    }
)

left, right = st.columns(2)

with left:
    st.write("**Average product pages viewed**")
    st.bar_chart(behaviour[["Average product pages"]])

with right:
    st.write("**Average time on product pages**")
    st.bar_chart(
        behaviour[["Average time on product pages (seconds)"]]
    )

st.write("**Other browsing metrics**")
other_metrics = [
    column for column in [
        "Average bounce rate",
        "Average exit rate",
    ]
    if column in behaviour.columns
]

if other_metrics:
    st.dataframe(
        behaviour[other_metrics].round(4),
        use_container_width=True,
    )


# -----------------------------
# 9. Show the filtered data
# -----------------------------
with st.expander("Inspect filtered dataset"):
    st.write(f"Showing {len(filtered_df):,} sessions.")
    st.dataframe(filtered_df, use_container_width=True)

with st.expander("How to interpret this dashboard"):
    st.markdown(
        """
        - **Purchase rate:** percentage of sessions that ended in a purchase.
        - **Visitor type:** compares new, returning, and other visitors.
        - **Monthly trends:** shows differences across the months in the dataset.
        - **Browsing behaviour:** compares average browsing measurements
          for sessions with and without a purchase.
        
        These are descriptive patterns, not proof that a behaviour causes a purchase.
        Results change when you change the sidebar filters.
        """
    )

st.caption(
    "Data source: Online Shoppers Purchasing Intention dataset. "
    "Historical results may not represent current store visitors."
)
