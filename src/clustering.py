import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


def get_clustering_features(df):
    """
    Select features for clustering based on availability
    """
    # Priority features for clustering
    priority_features = [
        'utilization_ratio', 'nooftrans', 'totalrcs', 'totalunits',
        'portability_ratio', 'amount_per_card', 'trans_volatility',
        'commodity_count', 'nfsa_ratio', 'efficiency_score'
    ]
    
    # Select available features
    available_features = [f for f in priority_features if f in df.columns]
    
    # Add log-transformed versions if available
    log_features = ['nooftrans_log', 'totalrcs_log', 'totalunits_log']
    available_log = [f for f in log_features if f in df.columns]
    
    all_features = available_features + available_log
    
    print(f"📊 Using features for clustering: {all_features}")
    return all_features


def elbow_curve_analysis(X_scaled, max_k=10):
    """
    Perform Elbow Curve analysis to find optimal number of clusters
    """
    from sklearn.cluster import KMeans
    
    inertias = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    return list(K_range), inertias


def silhouette_analysis(X_scaled, max_k=10):
    """
    Calculate Silhouette Score for different k values
    """
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
    
    return list(K_range), silhouette_scores


def apply_clustering(df, n_clusters=4, use_full_dataset=True):
    """
    Apply clustering with comprehensive evaluation
    
    Parameters:
    -----------
    df : DataFrame
        Input data with engineered features
    n_clusters : int
        Number of clusters for KMeans
    use_full_dataset : bool
        If True, aggregate at shop level and cluster all unique shops
    """
    print("🤖 Starting clustering analysis...")
    
    # Select features
    features = get_clustering_features(df)
    
    if len(features) < 2:
        print("⚠️ Not enough features for clustering. Using basic features.")
        features = ['nooftrans', 'totalrcs', 'utilization_ratio']
        for f in features:
            if f not in df.columns:
                df[f] = 0
    
    # ===============================
    # AGGREGATE AT SHOP LEVEL
    # ===============================
    
    print("📊 Aggregating data at shop level...")
    
    # Define aggregation functions
    agg_funcs = {}
    for col in df.columns:
        if col in ['shopno', 'distcode']:
            continue
        elif col in ['distname', 'fpsstatus', 'fpstype']:
            agg_funcs[col] = 'first'
        elif col in features:
            agg_funcs[col] = 'mean'  # Average over time
        elif col in ['latitude', 'longitude']:
            agg_funcs[col] = 'first'
        elif 'std' in col or 'volatility' in col:
            agg_funcs[col] = 'mean'
    
    # Aggregate by shop
    shop_agg = df.groupby(['shopno', 'distcode']).agg(agg_funcs).reset_index()
    print(f"📊 Aggregated {len(df)} records into {len(shop_agg)} unique shops")
    
    # Prepare data for clustering
    X = shop_agg[features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ===============================
    # CLUSTERING EVALUATION
    # ===============================
    
    print("📈 Running clustering evaluation...")
    
    # Elbow Curve
    k_range, inertias = elbow_curve_analysis(X_scaled, max_k=8)
    
    # Silhouette Analysis
    sil_k_range, sil_scores = silhouette_analysis(X_scaled, max_k=8)
    
    # Find optimal k based on silhouette score
    optimal_k = sil_k_range[np.argmax(sil_scores)]
    best_silhouette = max(sil_scores)
    
    print(f"📊 Optimal clusters (by Silhouette): {optimal_k} (Score: {best_silhouette:.3f})")
    
    # Use specified or optimal clusters
    final_k = n_clusters if n_clusters else optimal_k
    
    # ===============================
    # K-MEANS CLUSTERING
    # ===============================
    
    print(f"🎯 Applying K-Means with {final_k} clusters...")
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=10)
    shop_agg['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate final silhouette score
    final_silhouette = silhouette_score(X_scaled, shop_agg['cluster'])
    print(f"✅ Final Silhouette Score: {final_silhouette:.3f}")
    
    # ===============================
    # PCA FOR VISUALIZATION
    # ===============================
    
    print("📉 Applying PCA for visualization...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    shop_agg['pca1'] = pca_result[:, 0]
    shop_agg['pca2'] = pca_result[:, 1]
    
    # Explained variance
    explained_var = pca.explained_variance_ratio_
    print(f"📊 PCA Explained Variance: {explained_var[0]:.2%}, {explained_var[1]:.2%}")
    
    # ===============================
    # DBSCAN ANOMALY DETECTION
    # ===============================
    
    print("🔍 Applying DBSCAN for anomaly detection...")
    
    # Adjust eps based on data density
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    shop_agg['anomaly'] = dbscan.fit_predict(X_scaled)
    
    # Anomaly statistics
    n_anomalies = (shop_agg['anomaly'] == -1).sum()
    anomaly_pct = n_anomalies / len(shop_agg) * 100
    print(f"🚨 Detected {n_anomalies} anomalies ({anomaly_pct:.2f}%)")
    
    # ===============================
    # CLUSTER PROFILING
    # ===============================
    
    print("📋 Generating cluster profiles...")
    cluster_profiles = generate_cluster_profiles(shop_agg, features)
    
    # ===============================
    # MERGE BACK TO FULL DATASET
    # ===============================
    
    print("🔄 Mapping results to full dataset...")
    
    # Select columns to merge (shop-level only)
    result_cols = ['shopno', 'distcode', 'cluster', 'anomaly', 'pca1', 'pca2']
    merge_df = shop_agg[result_cols].drop_duplicates(subset=['shopno', 'distcode'])
    
    # Merge with original dataframe
    df = df.merge(merge_df, on=['shopno', 'distcode'], how='left')
    
    # Fill missing values
    df['cluster'] = df['cluster'].fillna(-1).astype(int)
    df['anomaly'] = df['anomaly'].fillna(0).astype(int)
    df['pca1'] = df['pca1'].fillna(0)
    df['pca2'] = df['pca2'].fillna(0)
    
    # ===============================
    # STORE EVALUATION METRICS
    # ===============================
    
    df.attrs['clustering_metrics'] = {
        'silhouette_score': final_silhouette,
        'optimal_k': optimal_k,
        'used_k': final_k,
        'explained_variance': explained_var.tolist(),
        'anomaly_count': n_anomalies,
        'anomaly_percentage': anomaly_pct,
        'features_used': features,
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': list(zip(sil_k_range, sil_scores)),
        'total_shops': len(shop_agg)
    }
    
    df.attrs['cluster_profiles'] = cluster_profiles
    
    print("✅ Clustering completed successfully!")
    return df


def generate_cluster_profiles(df, features):
    """
    Generate statistical profiles for each cluster
    """
    profiles = {}
    
    for cluster_id in sorted(df['cluster'].unique()):
        if cluster_id == -1:
            continue
            
        cluster_data = df[df['cluster'] == cluster_id]
        
        profile = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df) * 100,
            'features': {}
        }
        
        # Calculate statistics for each feature
        for feature in features:
            if feature in cluster_data.columns:
                profile['features'][feature] = {
                    'mean': float(cluster_data[feature].mean()),
                    'median': float(cluster_data[feature].median()),
                    'std': float(cluster_data[feature].std())
                }
        
        # Additional metadata
        if 'distname' in cluster_data.columns:
            profile['top_districts'] = cluster_data['distname'].value_counts().head(3).to_dict()
        
        if 'fpsstatus' in cluster_data.columns:
            profile['fps_status'] = cluster_data['fpsstatus'].value_counts().to_dict()
        
        profiles[int(cluster_id)] = profile
    
    return profiles


def get_cluster_labels(cluster_profiles):
    """
    Assign meaningful labels to clusters based on their characteristics
    """
    labels = {}
    
    for cluster_id, profile in cluster_profiles.items():
        features = profile['features']
        
        # Determine cluster characteristics
        utilization = features.get('utilization_ratio', {}).get('mean', 0)
        transactions = features.get('nooftrans', {}).get('mean', 0)
        portability = features.get('portability_ratio', {}).get('mean', 0)
        
        # Assign label based on characteristics
        if portability > 0.2:
            label = "Portability Hub"
        elif utilization > 2 and transactions > 500:
            label = "High-Volume Urban"
        elif utilization < 0.5 and transactions < 100:
            label = "Low-Volume Rural"
        elif transactions > 1000:
            label = "Mega Shop"
        else:
            label = f"Standard Cluster {cluster_id}"
        
        labels[cluster_id] = label
    
    return labels