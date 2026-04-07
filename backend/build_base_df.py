import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
M5_DIR = _SCRIPT_DIR.parent / "data" / "raw" / "m5"
SKU_FILE = _SCRIPT_DIR.parent / "selected_products.csv"


def build_df_base(store_ids=("CA_1",)):
    """
    Builds the base dataframe used by all agents.
    Restricted to trained SKU list.
    """

    # Load trained SKU list
    trained_skus = pd.read_csv(SKU_FILE)["item_id"].unique()

    # Load raw data
    sales = pd.read_csv(M5_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(M5_DIR / "calendar.csv")
    prices = pd.read_csv(M5_DIR / "sell_prices.csv")

    # Filter stores
    sales = sales[sales["store_id"].isin(store_ids)].copy()

    # 🔥 NEW: Filter only trained SKUs
    sales = sales[sales["item_id"].isin(trained_skus)].copy()

    # Convert wide → long
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    df = sales[id_cols + d_cols].melt(
        id_vars=id_cols,
        var_name="d",
        value_name="demand"
    )

    # Merge calendar
    calendar["date"] = pd.to_datetime(calendar["date"])
    df = df.merge(
        calendar[["d", "date", "wm_yr_wk"]],
        on="d",
        how="left"
    )

    # Merge prices
    prices = prices.rename(columns={"sell_price": "price"})
    df = df.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    # Clean types
    df["date"] = pd.to_datetime(df["date"])
    df["demand"] = df["demand"].astype(np.float32)
    df["price"] = df["price"].astype(np.float32)

    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    return df