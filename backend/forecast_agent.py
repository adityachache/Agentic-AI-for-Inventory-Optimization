import numpy as np
import pandas as pd
import joblib
from pathlib import Path

class ForecastAgent:
    """
    ForecastAgent
    - Uses a trained global ML model
    - Produces demand forecasts only (NO LLM)
    - Reuses df_base passed at initialization
    """

    def __init__(self, model_dir: str, df_base: pd.DataFrame):
        self.model_dir = Path(model_dir)
        self.df_base = df_base

        # Load trained model artifacts
        self.model = joblib.load(self.model_dir / "model.joblib")
        self.feature_cols = joblib.load(self.model_dir / "feature_cols.joblib")
        self.cat_cols = joblib.load(self.model_dir / "cat_cols.joblib")

    # --------------------------------------------------
    # Feature engineering (must match training)
    # --------------------------------------------------
    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["id", "date"]).copy()

        df["dow"] = df["date"].dt.dayofweek.astype(np.int16)
        df["month"] = df["date"].dt.month.astype(np.int16)

        # Fill missing prices
        df["price"] = df.groupby("id")["price"].ffill().bfill()

        # Promotion flag
        med28 = (
            df.groupby("id")["price"]
              .shift(1)
              .rolling(28)
              .median()
              .reset_index(level=0, drop=True)
        )
        df["promo_flag"] = (df["price"] < med28).astype(np.float32).fillna(0)

        # Demand lags
        for lag in [1, 7, 28]:
            df[f"lag_{lag}"] = df.groupby("id")["demand"].shift(lag)

        # Rolling means
        df["roll_mean_7"] = (
            df.groupby("id")["demand"]
              .shift(1)
              .rolling(7)
              .mean()
              .reset_index(level=0, drop=True)
        )

        df["roll_mean_28"] = (
            df.groupby("id")["demand"]
              .shift(1)
              .rolling(28)
              .mean()
              .reset_index(level=0, drop=True)
        )

        return df

    # --------------------------------------------------
    # Forecast ONE product
    # --------------------------------------------------
    def _forecast_single(self, item_id, store_id, start_date, end_date):
        # Filter base data for this product-store
        mask = (
            (self.df_base["item_id"] == item_id) &
            (self.df_base["store_id"] == store_id)
        )
        hist = self.df_base.loc[mask].copy()

        if hist.empty:
            return {
                "item_id": item_id,
                "store_id": store_id,
                "error": "No historical data available for this product-store."
            }

        hist = hist.sort_values("date")
        series_id = hist["id"].iloc[0]

        # Build continuous date panel
        all_dates = pd.date_range(
            start=hist["date"].min(),
            end=end_date,
            freq="D"
        )

        panel = pd.DataFrame({"date": all_dates})
        panel["id"] = series_id

        # Static categorical fields
        static = hist.iloc[0][
            ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        ].to_dict()

        for k, v in static.items():
            panel[k] = v

        # Merge known demand + price
        panel = panel.merge(
            hist[["date", "demand", "price"]],
            on="date",
            how="left"
        )

        panel["price"] = panel["price"].ffill().bfill()

        # Autoregressive forecasting
        last_known_idx = panel["demand"].last_valid_index()

        for i in range(last_known_idx + 1, len(panel)):
            tmp = self._add_features(panel.copy())
            row = tmp.iloc[[i]]

            required = ["lag_1", "lag_7", "lag_28", "roll_mean_7", "roll_mean_28"]

            if row[required].isna().any(axis=1).iloc[0]:
                pred = float(tmp.iloc[i - 1]["demand"])
            else:
                X = row[self.feature_cols].copy()
                for c in self.cat_cols:
                    X[c] = X[c].astype("category")
                pred = float(self.model.predict(X)[0])

            panel.at[i, "demand"] = max(0.0, pred)

        # Extract requested horizon
        out = panel[
            (panel["date"] >= start_date) &
            (panel["date"] <= end_date)
        ][["date", "demand"]].copy()

        out = out.rename(columns={"demand": "forecast"})

        return {
            "item_id": item_id,
            "store_id": store_id,
            "series_id": series_id,
            "daily_forecast": out,
            "total_units": float(out["forecast"].sum())
        }

    # --------------------------------------------------
    # Public API: forecast multiple products
    # --------------------------------------------------
    def forecast(self, item_ids, store_id, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        results = []
        for item_id in item_ids:
            results.append(
                self._forecast_single(
                    item_id=item_id,
                    store_id=store_id,
                    start_date=start_date,
                    end_date=end_date
                )
            )

        return {
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "results": results
        }