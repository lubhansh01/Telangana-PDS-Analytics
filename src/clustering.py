from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd

def perform_clustering(df):

    features = df[[
        "utilization_ratio",
        "rice_wheat_ratio",
        "transaction_volatility",
        "noOfTrans",
        "totalRcs"
    ]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    # PCA
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    df["PC1"] = pca_data[:, 0]
    df["PC2"] = pca_data[:, 1]

    # KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["kmeans_cluster"] = kmeans.fit_predict(scaled_data)

    # DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=10)
    df["dbscan_cluster"] = dbscan.fit_predict(scaled_data)

    score = silhouette_score(scaled_data, df["kmeans_cluster"])

    return df, score
