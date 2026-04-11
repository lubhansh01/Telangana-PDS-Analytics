import pandas as pd
import os
import re

# ===============================
# EXTRACT DATE FROM FILENAME
# ===============================
def extract_date(filename):
    match = re.search(r'_(\d{1,2})_(\d{4})', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


# ===============================
# LOAD CARD STATUS (EXPANDED COLUMNS)
# ===============================
def load_cards(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    
    # Expanded columns for comprehensive analysis
    card_columns = [
        'shopno', 'distcode', 'distname', 'officecode', 'officename',
        'totalrcs', 'totalunits', 'totalrcnfsa', 'totalunitsnfsa',
        'rcnfsaay', 'unitsnfsaay', 'rcnfsaphh', 'unitsnfsaphh',
        'rcstateaay', 'unitsstateaay', 'rcstatephh', 'unitsstatephh',
        'rcstateaap', 'unitsstateaap'
    ]

    for file in files:
        print(f"📂 Loading Card: {file}")
        try:
            df = pd.read_csv(os.path.join(folder_path, file))
            df.columns = df.columns.str.lower()
            
            # Keep only columns that exist in the file
            available_cols = [col for col in card_columns if col in df.columns]
            df = df[available_cols]
            
            month, year = extract_date(file)
            df['month'] = month
            df['year'] = year
            
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Error loading {file}: {e}")
            continue

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()


# ===============================
# LOAD LOCATION (EXPANDED)
# ===============================
def load_locations(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not files:
        print("⚠️ No location files found")
        return pd.DataFrame()
    
    file = files[0]
    df = pd.read_csv(os.path.join(folder_path, file))
    df.columns = df.columns.str.lower()
    
    # Select relevant location columns
    location_cols = ['shopno', 'distcode', 'distname', 'officecode', 'officename',
                     'longitude', 'latitude', 'fpsstatus', 'fpstype', 'address']
    available_cols = [col for col in location_cols if col in df.columns]
    
    return df[available_cols]


# ===============================
# PROCESS TRANSACTIONS (EXPANDED COLUMNS)
# ===============================
def process_transactions(folder_path, cards, locations):

    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Expanded transaction columns
    trans_columns = [
        'shopno', 'distcode', 'distname', 'officecode', 'officename',
        'nooftrans', 'noofrcs', 'riceafsc', 'ricefsc', 'riceaap',
        'wheat', 'sugar', 'rgdal', 'kerosene', 'salt', 'totalamount',
        'othershoptranscnt'
    ]

    final_df = pd.DataFrame()

    for file in files:
        print(f"📂 Processing Transaction: {file}")
        
        try:
            df = pd.read_csv(os.path.join(folder_path, file))
            df.columns = df.columns.str.lower()
            
            # Keep available columns
            available_cols = [col for col in trans_columns if col in df.columns]
            df = df[available_cols].copy()

            month, year = extract_date(file)
            df['month'] = month
            df['year'] = year

            # Merge with card data (same month/year)
            if not cards.empty:
                card_merge_cols = ['shopno', 'distcode', 'month', 'year']
                card_cols_to_merge = [col for col in card_merge_cols if col in cards.columns]
                if card_cols_to_merge:
                    df = df.merge(
                        cards,
                        on=card_cols_to_merge,
                        how='left',
                        suffixes=('', '_card')
                    )

            # Merge with location
            if not locations.empty:
                loc_merge_cols = ['shopno', 'distcode']
                loc_cols_to_merge = [col for col in loc_merge_cols if col in locations.columns]
                if loc_cols_to_merge:
                    df = df.merge(
                        locations,
                        on=loc_cols_to_merge,
                        how='left',
                        suffixes=('', '_loc')
                    )

            # Append
            final_df = pd.concat([final_df, df], ignore_index=True)

            # Free memory
            del df
            
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")
            continue

    return final_df


# ===============================
# MAIN FUNCTION
# ===============================
def merge_data():

    print("🔄 Loading card status...")
    cards = load_cards("../data/raw/card_status/")

    print("🔄 Loading locations...")
    locations = load_locations("../data/raw/fps_locations/")

    print("🔄 Processing transactions (chunk-wise)...")
    df = process_transactions(
        "../data/raw/transactions/",
        cards,
        locations
    )

    return df


if __name__ == "__main__":
    df = merge_data()
    print(df.shape)