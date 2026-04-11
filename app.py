import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import os
import warnings
warnings.filterwarnings('ignore')

# Get the directory where the app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "final_data.csv")

# ================= CONFIG =================
st.set_page_config(
    page_title="Telangana PDS Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    try:
        # Check if file exists
        if not os.path.exists(DATA_PATH):
            return generate_sample_data()
        
        # Check file size
        file_size = os.path.getsize(DATA_PATH)
        if file_size == 0:
            return generate_sample_data()
        
        df = pd.read_csv(DATA_PATH)
        
        if df.empty:
            return generate_sample_data()
        
        # Ensure proper data types
        df['shopno'] = df['shopno'].astype(str)
        df['distcode'] = df['distcode'].astype(str)
        return df
    except Exception as e:
        return generate_sample_data()


def generate_sample_data():
    """
    Generate sample data for demo purposes when actual data is not available.
    This allows the dashboard to be deployed without the large data file.
    """
    np.random.seed(42)
    
    # Sample districts
    districts = ['Adilabad', 'Hyderabad', 'Karimnagar', 'Khammam', 'Medak', 
                 'Nalgonda', 'Nizamabad', 'Rangareddy', 'Warangal']
    district_codes = ['532', '540', '541', '542', '543', '544', '545', '546', '547']
    
    # Generate sample shops
    n_records = 5000
    data = {
        'shopno': [f"1901{i:04d}" for i in range(n_records)],
        'distcode': np.random.choice(district_codes, n_records),
        'distname': np.random.choice(districts, n_records),
        'officecode': np.random.choice(['532001', '540001', '541001'], n_records),
        'officename': np.random.choice(['Talamadugu', 'Hyderabad Central', 'Karimnagar East'], n_records),
        'nooftrans': np.random.randint(50, 2000, n_records),
        'totalrcs': np.random.randint(100, 1500, n_records),
        'totalunits': np.random.randint(500, 5000, n_records),
        'month': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], n_records),
        'year': np.random.choice([2023, 2024, 2025], n_records),
        'othershoptranscnt': np.random.randint(0, 200, n_records),
        'totalamount': np.random.uniform(1000, 50000, n_records),
        'latitude': np.random.uniform(16.0, 19.5, n_records),
        'longitude': np.random.uniform(77.5, 81.0, n_records),
        'fpsstatus': np.random.choice(['Active', 'Active', 'Active', 'Inactive'], n_records),
        'fpstype': np.random.choice(['Normal Shop', 'Normal Shop', 'Portability Hub'], n_records),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['utilization_ratio'] = df['nooftrans'] / (df['totalrcs'] + 1)
    df['portability_ratio'] = df['othershoptranscnt'] / (df['nooftrans'] + 1)
    df['amount_per_card'] = df['totalamount'] / (df['totalrcs'] + 1)
    df['units_per_card'] = df['totalunits'] / (df['totalrcs'] + 1)
    df['efficiency_score'] = df['nooftrans'] / (df['totalunits'] + 1)
    df['commodity_count'] = np.random.randint(1, 6, n_records)
    df['nfsa_ratio'] = np.random.uniform(0.3, 0.9, n_records)
    df['trans_volatility'] = np.random.uniform(0, 50, n_records)
    df['trans_mean'] = df['nooftrans']
    df['trans_cv'] = np.random.uniform(0, 0.5, n_records)
    df['is_festival_season'] = df['month'].isin([10, 11, 12]).astype(int)
    df['is_harvest_season'] = df['month'].isin([4, 5]).astype(int)
    df['quarter'] = ((df['month'] - 1) // 3 + 1).astype(int)
    df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    df['zero_transactions'] = (df['nooftrans'] == 0).astype(int)
    df['high_utilization'] = (df['utilization_ratio'] > df['utilization_ratio'].quantile(0.95)).astype(int)
    df['low_utilization'] = (df['utilization_ratio'] < df['utilization_ratio'].quantile(0.05)).astype(int)
    
    # Add clustering results
    df['cluster'] = np.random.choice([0, 1, 2, 3], n_records)
    df['anomaly'] = np.random.choice([0, 0, 0, 0, -1], n_records)  # 20% anomalies
    df['pca1'] = np.random.normal(0, 1, n_records)
    df['pca2'] = np.random.normal(0, 1, n_records)
    
    # Ensure proper data types
    df['shopno'] = df['shopno'].astype(str)
    df['distcode'] = df['distcode'].astype(str)
    
    return df


# Load data
df = load_data()

if df.empty:
    st.error("Failed to load or generate data.")
    st.stop()

# ================= SIDEBAR =================
st.sidebar.title("🔍 Filters & Navigation")

# District filter
districts = ["All"] + sorted(df['distcode'].unique().tolist())
district = st.sidebar.selectbox("District Code", districts)

# Year filter
years = ["All"] + sorted([y for y in df['year'].dropna().unique() if y > 0])
year = st.sidebar.selectbox("Year", years)

# Cluster filter
clusters = ["All"] + sorted([c for c in df['cluster'].dropna().unique() if c >= 0])
cluster_filter = st.sidebar.selectbox("Cluster", clusters)

# Apply filters
df_filtered = df.copy()

if district != "All":
    df_filtered = df_filtered[df_filtered['distcode'] == district]

if year != "All":
    df_filtered = df_filtered[df_filtered['year'] == year]

if cluster_filter != "All":
    df_filtered = df_filtered[df_filtered['cluster'] == int(cluster_filter)]

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Data Summary")
st.sidebar.write(f"Total Records: {len(df_filtered):,}")
st.sidebar.write(f"Unique Shops: {df_filtered['shopno'].nunique():,}")
st.sidebar.write(f"Districts: {df_filtered['distcode'].nunique()}")

# ================= TITLE =================
st.markdown('<p class="main-header">📊 Telangana PDS Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Real-time Insights on Ration Distribution System | One Nation One Ration Card Analysis</p>', unsafe_allow_html=True)

# ================= KPI CARDS =================
st.subheader("📈 Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("🏪 Total Shops", f"{df_filtered['shopno'].nunique():,}")

with col2:
    avg_util = df_filtered['utilization_ratio'].mean() if 'utilization_ratio' in df_filtered.columns else 0
    st.metric("📈 Avg Utilization", f"{avg_util:.2f}")

with col3:
    anomalies = len(df_filtered[df_filtered['anomaly'] == -1]) if 'anomaly' in df_filtered.columns else 0
    st.metric("🚨 Anomalies", f"{anomalies:,}")

with col4:
    total_trans = int(df_filtered['nooftrans'].sum()) if 'nooftrans' in df_filtered.columns else 0
    st.metric("📦 Total Transactions", f"{total_trans:,}")

with col5:
    if 'othershoptranscnt' in df_filtered.columns:
        port_trans = int(df_filtered['othershoptranscnt'].sum())
        port_pct = (port_trans / (total_trans + 1)) * 100
        st.metric("🔄 Portability %", f"{port_pct:.1f}%")
    else:
        st.metric("🔄 Portability %", "N/A")

st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", 
    "🗺️ Geospatial", 
    "🔍 Shop Search",
    "📋 Cluster Profiles", 
    "🚨 Anomalies"
])

# ================= TAB 1: OVERVIEW =================
with tab1:
    st.subheader("📊 Trend Analysis & Distributions")
    
    col1, col2 = st.columns(2)
    
    # Monthly Trend
    with col1:
        st.markdown("**📅 Monthly Transaction Trend**")
        if 'month' in df_filtered.columns and 'nooftrans' in df_filtered.columns:
            trend = df_filtered.groupby('month')['nooftrans'].sum().reset_index()
            trend = trend[trend['month'] > 0].sort_values('month')
            
            fig = px.line(
                trend, 
                x='month', 
                y='nooftrans',
                markers=True,
                title="Transactions by Month"
            )
            fig.update_layout(xaxis_title="Month", yaxis_title="Total Transactions")
            st.plotly_chart(fig, use_container_width=True)
    
    # Portability Trend
    with col2:
        st.markdown("**🔄 Portability Growth (ONORC)**")
        if 'year' in df_filtered.columns and 'othershoptranscnt' in df_filtered.columns:
            port_trend = df_filtered.groupby('year')['othershoptranscnt'].sum().reset_index()
            port_trend = port_trend[port_trend['year'] > 0]
            
            fig = px.bar(
                port_trend,
                x='year',
                y='othershoptranscnt',
                title="Portability Transactions by Year",
                color='othershoptranscnt',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    # Cluster Distribution
    with col3:
        st.markdown("**📊 Cluster Distribution**")
        if 'cluster' in df_filtered.columns:
            cluster_counts = df_filtered['cluster'].value_counts().sort_index()
            cluster_counts = cluster_counts[cluster_counts.index >= 0]
            
            fig = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Shops by Cluster"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Utilization Distribution
    with col4:
        st.markdown("**⚡ Utilization Ratio Distribution**")
        if 'utilization_ratio' in df_filtered.columns:
            fig = px.histogram(
                df_filtered[df_filtered['utilization_ratio'] < 10],  # Filter extreme outliers
                x="utilization_ratio",
                nbins=50,
                title="Utilization Ratio Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("**🔗 Feature Correlation Matrix**")
    corr_features = ['nooftrans', 'totalrcs', 'utilization_ratio', 'totalunits']
    available_corr = [f for f in corr_features if f in df_filtered.columns]
    
    if len(available_corr) >= 2:
        corr_matrix = df_filtered[available_corr].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Correlation Between Key Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)

# ================= TAB 2: GEOSPATIAL =================
with tab2:
    st.subheader("🗺️ Geospatial Analysis")
    
    if 'latitude' in df_filtered.columns and 'longitude' in df_filtered.columns:
        # Filter valid coordinates
        geo_df = df_filtered[
            (df_filtered['latitude'].notna()) & 
            (df_filtered['longitude'].notna()) &
            (df_filtered['latitude'] != 0) &
            (df_filtered['longitude'] != 0)
        ].copy()
        
        if len(geo_df) > 0:
            # Sample for performance if needed
            if len(geo_df) > 3000:
                geo_df = geo_df.sample(3000, random_state=42)
                st.info(f"Showing sample of 3,000 shops out of {len(df_filtered)} total")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create Folium map
                avg_lat = geo_df['latitude'].mean()
                avg_lon = geo_df['longitude'].mean()
                
                m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)
                
                # Color mapping for clusters
                cluster_colors = {
                    0: 'red',
                    1: 'blue',
                    2: 'green',
                    3: 'purple',
                    4: 'orange'
                }
                
                for idx, row in geo_df.iterrows():
                    cluster = int(row.get('cluster', 0))
                    color = cluster_colors.get(cluster, 'gray')
                    
                    popup_text = f"""
                    <b>Shop:</b> {row['shopno']}<br>
                    <b>District:</b> {row.get('distname', row['distcode'])}<br>
                    <b>Cluster:</b> {cluster}<br>
                    <b>Transactions:</b> {row.get('nooftrans', 'N/A')}<br>
                    <b>Utilization:</b> {row.get('utilization_ratio', 'N/A'):.2f}
                    """
                    
                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=5,
                        popup=folium.Popup(popup_text, max_width=200),
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
                
                # Add legend
                legend_html = '''
                <div style="position: fixed; 
                            bottom: 50px; left: 50px; width: 120px; height: auto; 
                            background-color: white; border:2px solid grey; z-index:9999; 
                            font-size:14px; padding: 10px;">
                <b>Clusters</b><br>
                <i style="background:red;width:10px;height:10px;display:inline-block;"></i> Cluster 0<br>
                <i style="background:blue;width:10px;height:10px;display:inline-block;"></i> Cluster 1<br>
                <i style="background:green;width:10px;height:10px;display:inline-block;"></i> Cluster 2<br>
                <i style="background:purple;width:10px;height:10px;display:inline-block;"></i> Cluster 3<br>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
                st_folium(m, width=700, height=500)
            
            with col2:
                st.markdown("**📍 Cluster Hotspots**")
                
                # Cluster distribution by district
                if 'distname' in geo_df.columns:
                    district_clusters = geo_df.groupby(['distname', 'cluster']).size().reset_index(name='count')
                    top_districts = geo_df['distname'].value_counts().head(5)
                    
                    st.write("Top Districts:")
                    for dist, count in top_districts.items():
                        st.write(f"• {dist}: {count} shops")
                
                # Anomalies on map
                if 'anomaly' in geo_df.columns:
                    anomaly_count = (geo_df['anomaly'] == -1).sum()
                    st.markdown(f"**🚨 Anomalies on Map:** {anomaly_count}")
        else:
            st.warning("No valid geospatial data available for the selected filters.")
    else:
        st.warning("Geospatial data (latitude/longitude) not available in the dataset.")

# ================= TAB 3: SHOP SEARCH =================
with tab3:
    st.subheader("🔍 Shop Performance Search & Comparison")
    
    shop_input = st.text_input("Enter Shop Number", placeholder="e.g., 1901001")
    
    if shop_input:
        shop_data = df[df['shopno'].astype(str) == shop_input]
        
        if not shop_data.empty:
            # Get the latest record for the shop
            shop_latest = shop_data.iloc[0]
            cluster_id = shop_latest.get('cluster', 0)
            
            # Shop details
            st.markdown(f"**🏪 Shop Details: {shop_input}**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("District", shop_latest.get('distname', shop_latest['distcode']))
            
            with col2:
                shop_util = shop_latest.get('utilization_ratio', 0)
                st.metric("Utilization Ratio", f"{shop_util:.2f}")
            
            with col3:
                shop_trans = shop_latest.get('nooftrans', 0)
                st.metric("Transactions", f"{shop_trans:,.0f}")
            
            with col4:
                shop_cluster = int(cluster_id) if cluster_id >= 0 else "Unclustered"
                st.metric("Cluster", shop_cluster)
            
            # Cluster comparison
            st.markdown("**📊 Comparison with Cluster Average**")
            
            if cluster_id >= 0 and 'cluster' in df.columns:
                cluster_data = df[df['cluster'] == cluster_id]
                
                comparison_metrics = ['utilization_ratio', 'nooftrans', 'totalrcs']
                available_metrics = [m for m in comparison_metrics if m in df.columns]
                
                if available_metrics:
                    comparison_data = []
                    
                    for metric in available_metrics:
                        shop_val = shop_latest.get(metric, 0)
                        cluster_avg = cluster_data[metric].mean()
                        cluster_std = cluster_data[metric].std()
                        
                        comparison_data.append({
                            'Metric': metric.replace('_', ' ').title(),
                            'Shop Value': shop_val,
                            'Cluster Avg': cluster_avg,
                            'Difference': shop_val - cluster_avg,
                            'Std Deviations': (shop_val - cluster_avg) / (cluster_std + 0.001)
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df.style.format({
                        'Shop Value': '{:.2f}',
                        'Cluster Avg': '{:.2f}',
                        'Difference': '{:.2f}',
                        'Std Deviations': '{:.2f}'
                    }))
                    
                    # Bar chart comparison
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Shop Value',
                        x=comparison_df['Metric'],
                        y=comparison_df['Shop Value'],
                        marker_color='#1f77b4'
                    ))
                    fig.add_trace(go.Bar(
                        name='Cluster Average',
                        x=comparison_df['Metric'],
                        y=comparison_df['Cluster Avg'],
                        marker_color='#ff7f0e'
                    ))
                    fig.update_layout(
                        title="Shop vs Cluster Average",
                        barmode='group',
                        xaxis_title="Metric",
                        yaxis_title="Value"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Historical data
            st.markdown("**📈 Historical Data**")
            if 'year' in shop_data.columns and 'month' in shop_data.columns:
                hist_cols = ['year', 'month', 'nooftrans', 'utilization_ratio']
                available_hist = [c for c in hist_cols if c in shop_data.columns]
                st.dataframe(shop_data[available_hist].sort_values(['year', 'month']))
            
            # Anomaly flag
            if 'anomaly' in shop_latest and shop_latest['anomaly'] == -1:
                st.error("⚠️ This shop has been flagged as an ANOMALY. Please investigate.")
        else:
            st.error(f"Shop {shop_input} not found in the dataset.")

# ================= TAB 4: CLUSTER PROFILES =================
with tab4:
    st.subheader("📋 Cluster Profiles & Characteristics")
    
    if 'cluster' in df_filtered.columns:
        # Generate cluster profiles
        cluster_profiles = {}
        
        for cluster_id in sorted(df_filtered['cluster'].unique()):
            if cluster_id < 0:
                continue
                
            cluster_data = df_filtered[df_filtered['cluster'] == cluster_id]
            
            profile = {
                'Cluster ID': int(cluster_id),
                'Shop Count': len(cluster_data),
                'Percentage': f"{len(cluster_data) / len(df_filtered) * 100:.1f}%"
            }
            
            # Key metrics
            metrics = ['utilization_ratio', 'nooftrans', 'totalrcs', 'portability_ratio']
            for metric in metrics:
                if metric in cluster_data.columns:
                    profile[metric.replace('_', ' ').title()] = f"{cluster_data[metric].mean():.2f}"
            
            cluster_profiles[cluster_id] = profile
        
        # Display profiles
        profiles_df = pd.DataFrame(cluster_profiles).T
        st.dataframe(profiles_df, use_container_width=True)
        
        # Detailed cluster analysis
        st.markdown("**📊 Cluster Characteristics**")
        
        selected_cluster = st.selectbox(
            "Select Cluster for Detailed Analysis",
            options=[c for c in sorted(df_filtered['cluster'].unique()) if c >= 0],
            format_func=lambda x: f"Cluster {x}"
        )
        
        if selected_cluster is not None:
            cluster_detail = df_filtered[df_filtered['cluster'] == selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Cluster {selected_cluster} - Feature Distribution**")
                
                # Feature distributions
                feature_cols = ['nooftrans', 'utilization_ratio', 'totalrcs']
                available_features = [f for f in feature_cols if f in cluster_detail.columns]
                
                if available_features:
                    selected_feature = st.selectbox("Select Feature", available_features)
                    
                    fig = px.histogram(
                        cluster_detail,
                        x=selected_feature,
                        nbins=30,
                        title=f"{selected_feature.replace('_', ' ').title()} Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**District Distribution**")
                
                if 'distname' in cluster_detail.columns:
                    dist_counts = cluster_detail['distname'].value_counts().head(10)
                else:
                    dist_counts = cluster_detail['distcode'].value_counts().head(10)
                
                fig = px.bar(
                    x=dist_counts.values,
                    y=dist_counts.index,
                    orientation='h',
                    title="Top Districts in Cluster"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster behavior interpretation
            st.markdown("**📝 Cluster Interpretation**")
            
            avg_util = cluster_detail['utilization_ratio'].mean() if 'utilization_ratio' in cluster_detail.columns else 0
            avg_trans = cluster_detail['nooftrans'].mean() if 'nooftrans' in cluster_detail.columns else 0
            avg_port = cluster_detail['portability_ratio'].mean() if 'portability_ratio' in cluster_detail.columns else 0
            
            if avg_port > 0.2:
                interpretation = "🔄 **Portability Hub**: This cluster shows high portability ratios, indicating these shops serve many non-local beneficiaries. Consider increasing stock replenishment frequency."
            elif avg_util > 2 and avg_trans > 500:
                interpretation = "🏙️ **High-Volume Urban**: High transaction volumes with good utilization. These are likely urban shops serving dense populations."
            elif avg_util < 0.5 and avg_trans < 100:
                interpretation = "🌾 **Low-Volume Rural**: Lower transaction volumes typical of rural areas. May need mobile distribution support."
            elif avg_trans > 1000:
                interpretation = "🏪 **Mega Shop**: Very high transaction volumes. These are critical distribution points requiring special attention."
            else:
                interpretation = "✅ **Standard Shop**: Typical performance within expected parameters."
            
            st.info(interpretation)
    else:
        st.warning("Clustering data not available.")

# ================= TAB 5: ANOMALIES =================
with tab5:
    st.subheader("🚨 Anomaly Detection & Fraud Indicators")
    
    if 'anomaly' in df_filtered.columns:
        # Anomaly overview
        anomalies = df_filtered[df_filtered['anomaly'] == -1]
        normal = df_filtered[df_filtered['anomaly'] != -1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Anomalies", f"{len(anomalies):,}")
        
        with col2:
            anomaly_pct = len(anomalies) / len(df_filtered) * 100 if len(df_filtered) > 0 else 0
            st.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")
        
        with col3:
            affected_districts = anomalies['distcode'].nunique() if len(anomalies) > 0 else 0
            st.metric("Affected Districts", affected_districts)
        
        # Anomaly types
        st.markdown("**📊 Anomaly Categories**")
        
        anomaly_types = []
        
        # High utilization anomalies (potential stock diversion)
        if 'utilization_ratio' in anomalies.columns:
            high_util = anomalies[anomalies['utilization_ratio'] > 5]
            anomaly_types.append({
                'Type': 'High Utilization (Stock Diversion Risk)',
                'Count': len(high_util),
                'Description': 'Shops with unusually high transaction-to-card ratios'
            })
        
        # Zero transaction anomalies (ghost beneficiaries)
        if 'nooftrans' in anomalies.columns:
            zero_trans = anomalies[anomalies['nooftrans'] == 0]
            anomaly_types.append({
                'Type': 'Zero Transactions (Ghost Beneficiaries)',
                'Count': len(zero_trans),
                'Description': 'Shops with no transactions despite having registered cards'
            })
        
        # High portability anomalies
        if 'portability_ratio' in anomalies.columns:
            high_port = anomalies[anomalies['portability_ratio'] > 0.5]
            anomaly_types.append({
                'Type': 'High Portability (Unusual Mobility)',
                'Count': len(high_port),
                'Description': 'Shops with abnormally high portability ratios'
            })
        
        if anomaly_types:
            anomaly_df = pd.DataFrame(anomaly_types)
            st.dataframe(anomaly_df, use_container_width=True)
        
        # Anomaly list
        st.markdown("**📋 Anomalous Shops List**")
        
        if len(anomalies) > 0:
            # Select columns to display
            display_cols = ['shopno', 'distcode', 'distname' if 'distname' in anomalies.columns else None, 
                           'nooftrans', 'totalrcs', 'utilization_ratio']
            display_cols = [c for c in display_cols if c and c in anomalies.columns]
            
            st.dataframe(anomalies[display_cols].head(100), use_container_width=True)
            
            if len(anomalies) > 100:
                st.info(f"Showing first 100 of {len(anomalies)} anomalies. Use filters to narrow down.")
        else:
            st.success("✅ No anomalies detected in the current filtered view.")
        
        # Comparison with normal shops
        if len(anomalies) > 0 and len(normal) > 0:
            st.markdown("**📈 Anomaly vs Normal Comparison**")
            
            compare_metrics = ['utilization_ratio', 'nooftrans', 'totalrcs']
            available_compare = [m for m in compare_metrics if m in df_filtered.columns]
            
            if available_compare:
                comparison = pd.DataFrame({
                    'Anomaly Mean': anomalies[available_compare].mean(),
                    'Normal Mean': normal[available_compare].mean()
                })
                comparison['Difference'] = comparison['Anomaly Mean'] - comparison['Normal Mean']
                comparison['Ratio'] = comparison['Anomaly Mean'] / (comparison['Normal Mean'] + 0.001)
                
                st.dataframe(comparison.style.format({
                    'Anomaly Mean': '{:.2f}',
                    'Normal Mean': '{:.2f}',
                    'Difference': '{:.2f}',
                    'Ratio': '{:.2f}'
                }))
    else:
        st.warning("Anomaly detection data not available.")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "Telangana PDS Analytics Dashboard | Built with Streamlit | "
    "Data Source: Telangana Civil Supplies Department"
    "</p>",
    unsafe_allow_html=True
)