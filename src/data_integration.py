import pandas as pd
import os
import re

def extract_date(filename):
    match = re.search(r'_(\d{1,2})_(\d{4})', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def load_multiple_csv(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []

    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))
        df.columns = df.columns.str.lower()

        month, year = extract_date(file)
        df['month'] = month
        df['year'] = year

        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def load_single_csv(folder_path):
    file = [f for f in os.listdir(folder_path) if f.endswith('.csv')][0]
    df = pd.read_csv(os.path.join(folder_path, file))
    df.columns = df.columns.str.lower()
    return df


def merge_data():
    transactions = load_multiple_csv("../data/raw/transactions/")
    cards = load_multiple_csv("../data/raw/card_status/")
    locations = load_single_csv("../data/raw/fps_locations/")

    # Merge including time
    df = transactions.merge(
        cards,
        on=['shopno', 'distcode', 'month', 'year'],
        how='left',
        suffixes=('', '_card')
    )

    df = df.merge(
        locations,
        on=['shopno', 'distcode'],
        how='left'
    )

    return df