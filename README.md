# Telangana PDS Analytics

A comprehensive data analytics solution for the Telangana State Civil Supplies Department's Public Distribution System (PDS). This project analyzes Fair Price Shop (FPS) performance, identifies anomalies, and provides actionable insights for policy-making and fraud prevention.

## 🎯 Business Objectives

1. **Policy Impact Analysis**: Understanding the "One Nation One Ration Card" (ONORC) policy effects on distribution patterns
2. **Fraud Prevention**: Flagging shops with unusual transaction-to-card ratios (ghost beneficiaries, stock diversion)
3. **Logistics Optimization**: Identifying "Portability Hubs" requiring higher stock replenishment frequency

## 🏗️ Project Architecture

```
Telangana-PDS-Analytics/
├── data/
│   ├── raw/              # Raw CSV files from official portal
│   │   ├── card_status/  # Monthly card status data
│   │   ├── transactions/ # Monthly transaction data
│   │   └── fps_locations/# Shop location data
│   └── processed/        # Processed datasets
├── notebooks/
│   └── PDS_EDA_Model.ipynb  # Comprehensive EDA & Modeling
├── src/
│   ├── data_integration.py    # Data merging & consolidation
│   ├── preprocessing.py       # Data cleaning & validation
│   ├── feature_engineering.py # Feature creation
│   ├── clustering.py          # ML clustering & anomaly detection
│   └── pipeline.py            # End-to-end pipeline
├── app.py                 # Streamlit Dashboard
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Telangana-PDS-Analytics
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the data pipeline:
```bash
cd src
python pipeline.py
cd ..
```

4. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 📊 Features

### Data Integration
- Triple-join on `shopNo` and `distCode` across transactions, card status, and location datasets
- Automated date extraction from filenames
- Chunk-wise processing for large datasets

### Feature Engineering
- **Utilization Ratio**: Transactions per ration card
- **Portability Ratio**: Other shop transactions / Total transactions
- **Commodity Intensity**: Rice-to-Wheat ratios
- **Volatility Features**: Standard deviation of transactions per shop
- **Category Ratios**: NFSA vs State scheme distribution
- **Anomaly Indicators**: High/low utilization flags

### Clustering & Analysis
- **PCA**: Dimensionality reduction for visualization
- **K-Means**: 4-5 behavioral personas
- **DBSCAN**: Anomaly detection for outlier shops
- **Evaluation Metrics**: Silhouette Score, Elbow Curve

### Streamlit Dashboard
- **Interactive Filters**: District, Year, Cluster
- **KPI Cards**: Real-time metrics
- **Geospatial Map**: Folium-based shop locations color-coded by cluster
- **Trend Analysis**: Portability growth, seasonality
- **Shop Search**: Individual shop performance vs cluster average
- **Cluster Profiles**: Behavioral characteristics and recommendations
- **Anomaly Detection**: Fraud indicators and flagged shops

## 📈 Key Insights

### Portability Analysis (ONORC Impact)
- Track growth of inter-state portability transactions
- Identify portability hubs requiring additional stock

### Cluster Profiles
- **Portability Hubs**: High non-local traffic (>20% portability ratio)
- **High-Volume Urban**: Dense population centers with high throughput
- **Low-Volume Rural**: Rural shops needing mobile distribution support
- **Mega Shops**: Critical distribution points (>1000 transactions)

### Anomaly Detection
- **High Utilization**: Potential stock diversion (>5x normal ratio)
- **Zero Transactions**: Possible ghost beneficiaries
- **Unusual Portability**: Abnormal mobility patterns

## 🛠️ Technical Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-Learn**: Machine learning (K-Means, DBSCAN, PCA)
- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive visualizations
- **Folium**: Geospatial mapping
- **Jupyter**: Exploratory data analysis

## 📋 Project Deliverables

1. **Jupyter Notebook** (`notebooks/PDS_EDA_Model.ipynb`)
   - Comprehensive EDA
   - Trend analysis
   - Correlation studies
   - Clustering visualization
   - Anomaly detection results

2. **Streamlit App** (`app.py`)
   - Real-time interactive dashboard
   - Geospatial visualization
   - Shop performance comparison
   - Cluster profiling

3. **Python Modules** (`src/`)
   - Modular, reusable code
   - Comprehensive feature engineering
   - Advanced clustering with evaluation

## 📊 Evaluation Metrics

- **Silhouette Score**: Cluster quality validation
- **Elbow Curve**: Optimal cluster number determination
- **Cluster Purity**: Alignment with district types (Urban vs Rural)
- **Dashboard Functionality**: Responsiveness and visual clarity

## 🔍 Data Sources

- **Transactions**: Monthly commodity distribution volumes
- **Card Status**: Entitlement and beneficiary information
- **Locations**: Geospatial coordinates and shop status

Data covers the period 2023-2025 from the Telangana Civil Supplies Department portal.

## 📝 Usage Examples

### Run Pipeline
```bash
cd src
python pipeline.py
```

### Launch Dashboard
```bash
streamlit run app.py
```

### Explore in Jupyter
```bash
jupyter notebook notebooks/PDS_EDA_Model.ipynb
```

## 🤝 Contributing

This project is designed for educational and analytical purposes. For improvements or bug fixes, please ensure:
- Code follows PEP 8 standards
- Features are properly documented
- Changes are tested with sample data

## 📄 License

This project is for educational purposes as part of a data analytics capstone.

## 👨‍💻 Author

Developed as a capstone project for Telangana PDS Analytics.

---

**Status**: ✅ Complete and ready for deployment