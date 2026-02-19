import numpy as np

def create_features(df):

    df["utilization_ratio"] = df["noOfTrans"] / (df["totalRcs"] + 1)

    df["rice_wheat_ratio"] = df["riceQty"] / (df["wheatQty"] + 1)

    volatility = df.groupby("shopNo")["noOfTrans"].std().reset_index()
    volatility.columns = ["shopNo", "transaction_volatility"]

    df = df.merge(volatility, on="shopNo", how="left")

    df.fillna(0, inplace=True)

    return df
