import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from src.utils.paths import RAW_DATA_DIR


class TransactionGenerator:
    """Generate realistic credit card transaction data"""

    def __init__(self, n_customers=1000, n_transactions=100000, fraud_ratio=0.02):
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.fraud_ratio = fraud_ratio

        # Merchant categories
        self.categories = [
            "grocery",
            "restaurant",
            "gas_station",
            "online_shopping",
            "pharmacy",
            "entertainment",
            "travel",
            "utilities",
        ]

        # Generate customer profiles
        self.customers = self._generate_customers()

    def _generate_customers(self):
        """Generate customer profiles with spending patterns"""
        customers = []
        for i in range(self.n_customers):
            customer = {
                "customer_id": f"CUST_{i:06d}",
                "avg_transaction": np.random.uniform(20, 200),
                "std_transaction": np.random.uniform(10, 100),
                "preferred_categories": random.sample(self.categories, k=random.randint(2, 4)),
                "typical_locations": random.sample(range(100), k=random.randint(2, 5)),
                "account_age_days": random.randint(30, 3650),
            }
            customers.append(customer)
        return pd.DataFrame(customers)

    def generate_transactions(self):
        """Generate transaction dataset with REALISTIC fraud AND merchant patterns"""
        transactions = []

        start_date = datetime.now() - timedelta(days=365)

        for _ in range(self.n_transactions):
            customer = self.customers.sample(1).iloc[0]
            is_fraud = random.random() < self.fraud_ratio

            if is_fraud:
                # REALISTIC FRAUD PATTERNS
                fraud_type = random.choice(["test", "medium", "large"])

                if fraud_type == "test":
                    amount = np.random.uniform(10, 50)
                elif fraud_type == "medium":
                    amount = np.random.normal(customer["avg_transaction"] * 1.5, customer["std_transaction"])
                    amount = max(50, min(amount, 1000))
                else:
                    amount = np.random.uniform(500, 3000)

                # FIXED: Fraudsters sometimes know victim's habits (20% of time)
                if random.random() < 0.2:
                    # Sophisticated fraud - use typical merchant
                    location = random.choice(customer["typical_locations"])
                    category = random.choice(customer["preferred_categories"])
                else:
                    # Random merchant/category
                    location = random.randint(0, 100)
                    category = random.choice(self.categories)

                hour = random.randint(0, 23)

            else:
                # NORMAL TRANSACTIONS with realistic exploration behavior
                amount = max(5, np.random.normal(customer["avg_transaction"], customer["std_transaction"]))

                # FIXED: Customers explore new merchants 30% of time
                if random.random() < 0.7:
                    # Usual behavior (70%)
                    location = random.choice(customer["typical_locations"])
                    category = random.choice(customer["preferred_categories"])
                else:
                    # Exploring new places (30%)
                    location = random.randint(0, 100)
                    # Still prefer some categories
                    if random.random() < 0.5:
                        category = random.choice(customer["preferred_categories"])
                    else:
                        category = random.choice(self.categories)

                hour = np.random.choice(range(24), p=self._get_hour_distribution())

            # Generate timestamp
            timestamp = start_date + timedelta(seconds=random.randint(0, 365 * 24 * 60 * 60))

            transaction = {
                "transaction_id": f"TXN_{len(transactions):08d}",
                "customer_id": customer["customer_id"],
                "timestamp": timestamp,
                "amount": round(amount, 2),
                "merchant_category": category,
                "merchant_id": f"MERCH_{location:04d}",
                "transaction_hour": hour,
                "day_of_week": timestamp.weekday(),
                "is_online": random.random() < 0.3,
                "is_fraud": int(is_fraud),
            }
            transactions.append(transaction)

        df = pd.DataFrame(transactions)
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def _get_hour_distribution(self):
        """Realistic hourly transaction distribution"""
        # More transactions during day, less at night
        hours = np.array(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.02,
                0.03,  # 0-5 AM
                0.04,
                0.06,
                0.08,
                0.09,
                0.08,
                0.08,  # 6-11 AM
                0.10,
                0.09,
                0.08,
                0.07,
                0.06,
                0.05,  # 12-5 PM
                0.05,
                0.04,
                0.03,
                0.02,
                0.02,
                0.01,  # 6-11 PM
            ]
        )
        return hours / hours.sum()


if __name__ == "__main__":
    # Generate data
    generator = TransactionGenerator(n_customers=5000, n_transactions=500000, fraud_ratio=0.02)  # 2% fraud rate

    print("Generating transactions...")
    df = generator.generate_transactions()

    # Save to CSV
    output_path = RAW_DATA_DIR / "transactions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\n✓ Generated {len(df):,} transactions")
    print(f"  - Fraud cases: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"  - Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Amount range: ${df['amount'].min():.2f} to ${df['amount'].max():.2f}")
    print(f"  - Unique customers: {df['customer_id'].nunique():,}")
