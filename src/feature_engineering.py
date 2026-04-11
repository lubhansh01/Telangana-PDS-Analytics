import pandas as pd
import numpy as np

def create_features(df):
    """
    Comprehensive feature engineering for PDS analytics
    """
    print("⚙️ Creating features...")
    df = df.copy()
    
    # ===============================
    # 1. CORE UTILIZATION FEATURES
    # ===============================
    
    # Utilization Ratio: Transactions per Ration Card
    if 'nooftrans' in df.columns and 'totalrcs' in df.columns:
        df['utilization_ratio'] = df['nooftrans'] / (df['totalrcs'] + 1)
    else:
        df['utilization_ratio'] = 0
    
    # Transaction Amount per Card
    if 'totalamount' in df.columns and 'totalrcs' in df.columns:
        df['amount_per_card'] = df['totalamount'] / (df['totalrcs'] + 1)
    
    # Units per Card
    if 'totalunits' in df.columns and 'totalrcs' in df.columns:
        df['units_per_card'] = df['totalunits'] / (df['totalrcs'] + 1)
    
    # ===============================
    # 2. PORTABILITY FEATURES (One Nation One Ration Card)
    # ===============================
    
    # Portability Ratio: Other Shop Transactions / Total Transactions
    if 'othershoptranscnt' in df.columns and 'nooftrans' in df.columns:
        df['portability_ratio'] = df['othershoptranscnt'] / (df['nooftrans'] + 1)
        df['is_portability_hub'] = (df['portability_ratio'] > 0.1).astype(int)
    
    # ===============================
    # 3. COMMODITY INTENSITY FEATURES
    # ===============================
    
    # Total Rice (combine all rice types)
    rice_cols = ['riceafsc', 'ricefsc', 'riceaap']
    available_rice = [col for col in rice_cols if col in df.columns]
    if available_rice:
        df['total_rice'] = df[available_rice].sum(axis=1)
    
    # Rice-to-Wheat Ratio
    if 'total_rice' in df.columns and 'wheat' in df.columns:
        df['rice_wheat_ratio'] = df['total_rice'] / (df['wheat'] + 1)
    
    # Commodity Diversity (number of different commodities distributed)
    commodity_cols = ['total_rice', 'wheat', 'sugar', 'rgdal', 'kerosene', 'salt']
    available_commodities = [col for col in commodity_cols if col in df.columns]
    if available_commodities:
        df['commodity_count'] = (df[available_commodities] > 0).sum(axis=1)
    
    # ===============================
    # 4. CARD CATEGORY RATIOS (NFSA vs State)
    # ===============================
    
    # NFSA Ratio
    if 'totalrcnfsa' in df.columns and 'totalrcs' in df.columns:
        df['nfsa_ratio'] = df['totalrcnfsa'] / (df['totalrcs'] + 1)
    
    # State Scheme Ratio
    if 'totalrcstate' in df.columns and 'totalrcs' in df.columns:
        df['state_scheme_ratio'] = df['totalrcstate'] / (df['totalrcs'] + 1)
    
    # AAY vs PHH ratio within NFSA
    if 'rcnfsaay' in df.columns and 'rcnfsaphh' in df.columns:
        df['aay_phh_ratio'] = df['rcnfsaay'] / (df['rcnfsaphh'] + 1)
    
    # ===============================
    # 5. SHOP PERFORMANCE INDICATORS
    # ===============================
    
    # High Activity Flag (top 20% by transactions)
    if 'nooftrans' in df.columns:
        threshold = df['nooftrans'].quantile(0.8)
        df['high_activity'] = (df['nooftrans'] >= threshold).astype(int)
    
    # Efficiency Score (transactions per unit distributed)
    if 'nooftrans' in df.columns and 'totalunits' in df.columns:
        df['efficiency_score'] = df['nooftrans'] / (df['totalunits'] + 1)
    
    # ===============================
    # 6. AGGREGATION FEATURES (by shop)
    # ===============================
    
    # Calculate volatility features grouped by shop
    if 'shopno' in df.columns and 'nooftrans' in df.columns:
        # Transaction volatility (std dev of transactions per shop)
        shop_trans_stats = df.groupby('shopno')['nooftrans'].agg(['std', 'mean']).reset_index()
        shop_trans_stats.columns = ['shopno', 'trans_volatility', 'trans_mean']
        shop_trans_stats['trans_volatility'] = shop_trans_stats['trans_volatility'].fillna(0)
        df = df.merge(shop_trans_stats, on='shopno', how='left')
        
        # Coefficient of variation
        df['trans_cv'] = df['trans_volatility'] / (df['trans_mean'] + 1)
    
    # ===============================
    # 7. TEMPORAL FEATURES
    # ===============================
    
    # Season indicator (festival months: Oct-Nov-Dec, harvest: Apr-May)
    if 'month' in df.columns:
        df['is_festival_season'] = df['month'].isin([10, 11, 12]).astype(int)
        df['is_harvest_season'] = df['month'].isin([4, 5]).astype(int)
        
        # Quarter
        df['quarter'] = df['month'].apply(lambda x: (x-1)//3 + 1 if x > 0 else 0)
    
    # Year-Month combination
    if 'year' in df.columns and 'month' in df.columns:
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    
    # ===============================
    # 8. ANOMALY INDICATORS
    # ===============================
    
    # Zero transaction flag
    if 'nooftrans' in df.columns:
        df['zero_transactions'] = (df['nooftrans'] == 0).astype(int)
    
    # High utilization flag (potential stock diversion indicator)
    if 'utilization_ratio' in df.columns:
        df['high_utilization'] = (df['utilization_ratio'] > df['utilization_ratio'].quantile(0.95)).astype(int)
    
    # Low utilization flag (potential ghost beneficiaries)
    if 'utilization_ratio' in df.columns:
        df['low_utilization'] = (df['utilization_ratio'] < df['utilization_ratio'].quantile(0.05)).astype(int)
    
    # ===============================
    # 9. LOG TRANSFORMATIONS (for skewed data)
    # ===============================
    
    log_cols = ['nooftrans', 'totalrcs', 'totalunits', 'totalamount']
    for col in log_cols:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))
    
    print(f"✅ Created {len([c for c in df.columns if c not in ['shopno', 'distcode', 'month', 'year']])} features")
    return df