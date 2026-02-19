import pandas as pd
import os

def load_transactions(data_path):
    files = [f for f in os.listdir(data_path) if f.startswith("transactions")]
    df_list = []

    for file in files:
        df = pd.read_csv(os.path.join(data_path, file))
        df_list.append(df)

    return pd.concat(df_list, ignore_index=True)


def load_card_status(path):
    return pd.read_csv(path)


def load_locations(path):
    return pd.read_csv(path)


def create_master_dataset(transactions, cards, locations):
    df = transactions.merge(cards, on=["shopNo", "distCode"], how="left")
    df = df.merge(locations, on=["shopNo", "distCode"], how="left")
    return df
