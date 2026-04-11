import os
from data_integration import merge_data
from preprocessing import clean_data
from feature_engineering import create_features
from clustering import apply_clustering

def run_pipeline():

    print("🔄 Loading & merging data...")
    df = merge_data()

    print("🧹 Cleaning...")
    df = clean_data(df)

    print("⚙️ Feature engineering...")
    df = create_features(df)

    print("🤖 Clustering...")
    df = apply_clustering(df)

    print("💾 Saving...")
    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv("../data/processed/final_data.csv", index=False)

    print("✅ Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()