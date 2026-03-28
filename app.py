import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/final_data.csv")

df = load_data()

st.title("📊 Telangana PDS Analytics Dashboard")

# ================= FILTERS =================
district = st.sidebar.selectbox("District", ["All"] + list(df['distcode'].unique()))
year = st.sidebar.selectbox("Year", ["All"] + sorted(df['year'].dropna().unique()))

if district != "All":
    df = df[df['distcode'] == district]

if year != "All":
    df = df[df['year'] == year]

# ================= KPIs =================
col1, col2, col3 = st.columns(3)

col1.metric("Total Shops", df['shopno'].nunique())
col2.metric("Avg Utilization", round(df['utilization_ratio'].mean(), 2))
col3.metric("Anomalies", len(df[df['anomaly'] == -1]))

# ================= TIME SERIES =================
st.subheader("📅 Monthly Trend")

trend = df.groupby('month')['nooftrans'].sum().reset_index()
fig = px.line(trend, x='month', y='nooftrans')
st.plotly_chart(fig)

# ================= CORRELATION =================
st.subheader("📊 Correlation (Ration vs Transactions)")

fig, ax = plt.subplots()
sns.scatterplot(x=df['totalrcs'], y=df['nooftrans'], ax=ax)
st.pyplot(fig)

# ================= CLUSTER =================
st.subheader("📍 Cluster Visualization")

fig = px.scatter(df, x='pca1', y='pca2', color='cluster')
st.plotly_chart(fig)

# ================= MAP =================
if 'latitude' in df.columns:
    st.subheader("🗺️ Shop Map")
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="cluster",
        zoom=5,
        mapbox_style="carto-positron"
    )
    st.plotly_chart(fig)

# ================= COMMODITY =================
if 'rice' in df.columns:
    st.subheader("🍚 Rice Distribution")
    st.bar_chart(df.groupby('distcode')['rice'].mean())

# ================= SEARCH =================
st.subheader("🔍 Shop Comparison")

shop = st.text_input("Enter Shop Number")

if shop:
    shop_data = df[df['shopno'].astype(str) == shop]

    if not shop_data.empty:
        cluster_id = shop_data['cluster'].values[0]
        cluster_avg = df[df['cluster'] == cluster_id]['utilization_ratio'].mean()

        st.write("Shop Data:")
        st.dataframe(shop_data)

        st.write(f"Cluster Avg Utilization: {cluster_avg}")