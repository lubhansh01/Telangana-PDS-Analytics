import pandas as pd
import numpy as np

def clean_data(df):
    """
    Comprehensive data cleaning and preprocessing
    """
    print(f"🔍 Original shape: {df.shape}")
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"🧹 After removing duplicates: {df.shape}")
    
    # Handle missing values strategically
    # Numeric columns - fill with 0 or median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Key columns that shouldn't be 0
    key_cols = ['nooftrans', 'totalrcs', 'totalunits']
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Other numeric columns
    for col in numeric_cols:
        if col not in key_cols and col not in ['month', 'year', 'shopno', 'distcode']:
            df[col] = df[col].fillna(0)
    
    # Categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown')
    
    # Data type conversions
    if 'shopno' in df.columns:
        df['shopno'] = df['shopno'].astype(str)
    if 'distcode' in df.columns:
        df['distcode'] = df['distcode'].astype(str)
    if 'month' in df.columns:
        df['month'] = pd.to_numeric(df['month'], errors='coerce').fillna(0).astype(int)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    # Coordinate validation
    if 'latitude' in df.columns:
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        # Filter valid latitude range (Telangana is roughly 15-20°N)
        df.loc[(df['latitude'] < 15) | (df['latitude'] > 20), 'latitude'] = np.nan
    
    if 'longitude' in df.columns:
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        # Filter valid longitude range (Telangana is roughly 77-81°E)
        df.loc[(df['longitude'] < 77) | (df['longitude'] > 81), 'longitude'] = np.nan
    
    # Remove rows with invalid key identifiers
    if 'shopno' in df.columns:
        df = df[df['shopno'].notna() & (df['shopno'] != '') & (df['shopno'] != '0')]
    if 'distcode' in df.columns:
        df = df[df['distcode'].notna() & (df['distcode'] != '') & (df['distcode'] != '0')]
    
    print(f"✅ Final shape after cleaning: {df.shape}")
    return df


def detect_outliers(df, column, method='iqr', threshold=1.5):
    """
    Detect outliers using IQR or Z-score method
    """
    if column not in df.columns:
        return df
    
    data = df[column].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((df[column] - data.mean()) / data.std())
        outliers = z_scores > threshold
    
    return outliers


def validate_data(df):
    """
    Data validation checks
    """
    validation_report = {}
    
    # Check for required columns
    required_cols = ['shopno', 'distcode']
    missing_cols = [col for col in required_cols if col not in df.columns]
    validation_report['missing_required_columns'] = missing_cols
    
    # Check data ranges
    if 'nooftrans' in df.columns:
        negative_trans = (df['nooftrans'] < 0).sum()
        validation_report['negative_transactions'] = int(negative_trans)
    
    if 'totalrcs' in df.columns:
        negative_cards = (df['totalrcs'] < 0).sum()
        validation_report['negative_cards'] = int(negative_cards)
    
    # Check for data consistency
    if 'nooftrans' in df.columns and 'totalrcs' in df.columns:
        # Transactions should generally not exceed cards by extreme amounts
        high_ratio = (df['nooftrans'] > df['totalrcs'] * 100).sum()
        validation_report['extreme_utilization'] = int(high_ratio)
    
    print("📊 Data Validation Report:")
    for key, value in validation_report.items():
        print(f"  - {key}: {value}")
    
    return validation_report