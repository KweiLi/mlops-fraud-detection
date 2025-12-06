import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR


class FraudFeatureEngineering:
    """Feature engineering for fraud detection"""

    def __init__(self):
        self.feature_names = None

    def create_features(self, df):
        """Create all features from transaction data"""
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

        print("Creating features...")
        print(f"  Original columns: {len(df.columns)}")

        df = self._create_time_features(df)
        df = self._create_amount_features(df)
        df = self._create_velocity_features(df)
        df = self._create_customer_features(df)
        df = self._encode_categorical(df)

        # FIXED: Only fill numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        print(f"  Final columns: {len(df.columns)}")
        print(f"  Features added: {len(df.columns) - 10}")

        return df

    def _create_time_features(self, df):
        """Time-based features"""
        print("  - Time features...")

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] <= 17)).astype(int)

        return df

    def _create_amount_features(self, df):
        """Amount-based features"""
        print("  - Amount features...")

        df["amount_log"] = np.log1p(df["amount"])

        # Don't create amount_bin as categorical - we'll one-hot encode it directly
        amount_bins = pd.cut(
            df["amount"], bins=[0, 50, 100, 200, 500, np.inf], labels=["very_low", "low", "medium", "high", "very_high"]
        )
        # Convert to string to avoid categorical issues
        df["amount_bin"] = amount_bins.astype(str)

        df["customer_avg_amount"] = df.groupby("customer_id")["amount"].transform(lambda x: x.expanding().mean().shift(1))

        df["customer_std_amount"] = df.groupby("customer_id")["amount"].transform(lambda x: x.expanding().std().shift(1))

        df["amount_zscore"] = (df["amount"] - df["customer_avg_amount"]) / (df["customer_std_amount"] + 1e-5)

        return df

    def _create_velocity_features(self, df):
        """Transaction velocity features"""
        print("  - Velocity features...")

        df["time_since_last_txn"] = df.groupby("customer_id")["timestamp"].diff()
        df["time_since_last_txn_hours"] = df["time_since_last_txn"].dt.total_seconds() / 3600
        df["time_since_last_txn_hours"] = df["time_since_last_txn_hours"].fillna(24)

        df["txn_count_1h"] = (
            df.groupby("customer_id").rolling(window="1h", on="timestamp")["transaction_id"].count().reset_index(drop=True)
        )

        df["txn_count_24h"] = (
            df.groupby("customer_id").rolling(window="24h", on="timestamp")["transaction_id"].count().reset_index(drop=True)
        )

        df["amount_sum_1h"] = (
            df.groupby("customer_id").rolling(window="1h", on="timestamp")["amount"].sum().reset_index(drop=True)
        )

        df["amount_sum_24h"] = (
            df.groupby("customer_id").rolling(window="24h", on="timestamp")["amount"].sum().reset_index(drop=True)
        )

        return df

    def _create_customer_features(self, df):
        """Customer behavior features"""
        print("  - Customer behavior features...")

        def expanding_nunique(series):
            result = []
            seen = set()
            for val in series:
                seen.add(val)
                result.append(len(seen))
            return pd.Series(result, index=series.index)

        df["customer_unique_merchants"] = df.groupby("customer_id")["merchant_id"].transform(expanding_nunique)

        df["is_new_merchant"] = df.groupby("customer_id")["merchant_id"].transform(lambda x: (~x.duplicated()).astype(int))

        df["customer_txn_count"] = df.groupby("customer_id").cumcount() + 1

        return df

    def _encode_categorical(self, df):
        """Encode categorical variables"""
        print("  - Encoding categorical features...")

        category_dummies = pd.get_dummies(df["merchant_category"], prefix="category")
        df = pd.concat([df, category_dummies], axis=1)

        if "amount_bin" in df.columns:
            bin_dummies = pd.get_dummies(df["amount_bin"], prefix="amount_bin")
            df = pd.concat([df, bin_dummies], axis=1)

        df["is_online"] = df["is_online"].astype(int)

        return df

    def get_feature_columns(self, df):
        """Get list of feature columns for modeling"""
        exclude_cols = [
            "transaction_id",
            "customer_id",
            "timestamp",
            "merchant_id",
            "merchant_category",
            "amount_bin",
            "is_fraud",
            "time_since_last_txn",
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def save_features(self, df, output_path):
        """Save engineered features"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Saved features to: {output_path}")

        feature_cols = self.get_feature_columns(df)
        feature_list_path = Path(output_path).parent / "feature_names.txt"
        with open(feature_list_path, "w") as f:
            f.write("\n".join(feature_cols))
        print(f"✓ Saved feature names to: {feature_list_path}")


def main():
    """Run feature engineering pipeline"""

    print("Loading raw transaction data...")

    transactions_path = RAW_DATA_DIR / "transactions.csv"
    df = pd.read_csv(transactions_path, parse_dates=["timestamp"])

    print(f"  Loaded {len(df):,} transactions")

    fe = FraudFeatureEngineering()
    df_features = fe.create_features(df)

    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    feature_cols = fe.get_feature_columns(df_features)
    print(f"\nTotal features created: {len(feature_cols)}")

    print("\nSample features:")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"  {i}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")

    output_path = PROCESSED_DATA_DIR / "transactions_features.csv"
    fe.save_features(df_features, str(output_path))

    print("\n" + "=" * 60)
    print("FRAUD STATISTICS")
    print("=" * 60)
    print(f"\nFraud transactions: {df_features['is_fraud'].sum():,}")
    print(f"Normal transactions: {(~df_features['is_fraud'].astype(bool)).sum():,}")
    print(f"Fraud rate: {df_features['is_fraud'].mean()*100:.2f}%")

    print("\n✓ Feature engineering complete!")


if __name__ == "__main__":
    main()
