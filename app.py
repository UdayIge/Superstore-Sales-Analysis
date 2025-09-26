import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Superstore Sales Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Superstore Sales Analysis Dashboard")
st.markdown(
    "Interactive analytics for Superstore sales and profitability. "
    "Use the sidebar filters to drill down into the data."
)

# ---------------- Load Data ----------------
DATA_PATH = Path("Superstore.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, encoding="ISO-8859-1")
    # Drop unused columns if they exist
    for col in ["Row ID", "Country", "Postal Code"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    # Robust date parsing
    df["Order Date"] = pd.to_datetime(
        df["Order Date"], errors="coerce"
    )
    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"], errors="coerce"
    )
    df.dropna(subset=["Order Date", "Ship Date"], inplace=True)
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df

if not DATA_PATH.exists():
    st.error(f"❌ Data file not found at {DATA_PATH}.")
    st.stop()

df = load_data(DATA_PATH)

# ---------------- Sidebar Filters ----------------
st.sidebar.header("🔎 Filters")
years = sorted(df["Order Year"].unique())
regions = sorted(df["Region"].unique())
categories = sorted(df["Category"].unique())

year_filter = st.sidebar.multiselect("Select Year(s)", years, default=years)
region_filter = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
category_filter = st.sidebar.multiselect("Select Category(ies)", categories, default=categories)

filtered_df = df[
    df["Order Year"].isin(year_filter)
    & df["Region"].isin(region_filter)
    & df["Category"].isin(category_filter)
]

# ---------------- KPIs ----------------
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
avg_discount = filtered_df["Discount"].mean()
order_count = len(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${total_sales:,.2f}")
col2.metric("Total Profit", f"${total_profit:,.2f}")
col3.metric("Avg. Discount", f"{avg_discount:.2%}")
col4.metric("Total Orders", f"{order_count:,}")

st.markdown("---")

# ---------------- Charts ----------------
st.subheader("📈 Sales and Profit Over Time")
time_data = (
    filtered_df.groupby("Order Month")[["Sales", "Profit"]]
    .sum()
    .reset_index()
    .sort_values("Order Month")
)
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=time_data["Order Month"], y=time_data["Sales"],
                              mode="lines+markers", name="Sales"))
fig_line.add_trace(go.Scatter(x=time_data["Order Month"], y=time_data["Profit"],
                              mode="lines+markers", name="Profit"))
fig_line.update_layout(xaxis_title="Month", yaxis_title="Amount", hovermode="x unified")
st.plotly_chart(fig_line, use_container_width=True)

st.subheader("🏷️ Sales by Region and Category")
region_cat_sales = (
    filtered_df.groupby(["Region", "Category"])["Sales"].sum().reset_index()
)
fig_bar = px.bar(
    region_cat_sales,
    x="Region",
    y="Sales",
    color="Category",
    barmode="group",
    title="Sales by Region & Category"
)
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("💰 Top 10 Profitable Products")
top_products = (
    filtered_df.groupby("Product Name")["Profit"]
    .sum()
    .nlargest(10)
    .reset_index()
)
fig_top = px.bar(
    top_products,
    x="Profit",
    y="Product Name",
    orientation="h",
    title="Top 10 Products by Total Profit",
    text="Profit"
)
fig_top.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_top, use_container_width=True)

# ---------------- Data Preview ----------------
with st.expander("🔍 View Filtered Data"):
    st.dataframe(filtered_df)

st.caption(
    "Data Source: Superstore dataset (Kaggle). Dashboard powered by Streamlit & Plotly."
)
