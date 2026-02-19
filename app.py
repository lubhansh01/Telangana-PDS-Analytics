import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_transactions, load_card_status, load_locations, create_master_dataset
from src.feature_engineering import create_features
from src.clustering import perform_clustering

st.set_page_config(layout="wide")

st.title("Telangana PDS Analytics Dashboard")

@st.cache_data
def load_data():

    transactions = load_transactions("data/raw")
    cards = load_card_status("data/raw/card_status.csv")
    locations = load_locations("data/raw/fps_locations.csv")

    master = create_master_dataset(transactions, cards, locations)
    master = create_features(master)
    master, score = perform_clustering(master)

    return master, score


df, sil_score = load_data()

st.sidebar.header("Filters")

district = st.sidebar.selectbox("Select District", df["district"].unique())
year = st.sidebar.selectbox("Select Year", df["year"].unique())

filtered_df = df[(df["district"] == district) & (df["year"] == year)]

st.metric("Silhouette Score", round(sil_score, 3))

col1, col2 = st.columns(2)

with col1:
    fig = px.scatter(
        filtered_df,
        x="PC1",
        y="PC2",
        color="kmeans_cluster",
        title="Cluster Visualization (PCA)"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = px.histogram(
        filtered_df,
        x="utilization_ratio",
        color="kmeans_cluster",
        title="Utilization Distribution"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Search Shop Performance")

shop_id = st.text_input("Enter Shop Number")

if shop_id:
    shop_data = df[df["shopNo"] == int(shop_id)]
    if not shop_data.empty:
        st.write(shop_data)
    else:
        st.warning("Shop not found")
