def create_features(df):

    # Utilization Ratio
    if 'nooftrans' in df.columns and 'totalrcs' in df.columns:
        df['utilization_ratio'] = df['nooftrans'] / (df['totalrcs'] + 1)
    else:
        df['utilization_ratio'] = 0

    # Commodity ratio
    if 'rice' in df.columns and 'wheat' in df.columns:
        df['rice_wheat_ratio'] = df['rice'] / (df['wheat'] + 1)

    return df