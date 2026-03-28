from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def apply_clustering(df):

    features = ['utilization_ratio']

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    df['pca1'] = pca_result[:, 0]
    df['pca2'] = pca_result[:, 1]

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['anomaly'] = dbscan.fit_predict(X_scaled)

    return df