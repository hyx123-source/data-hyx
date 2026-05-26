"""Generate a synthetic Online Retail dataset for testing and demo.
Source: Student + AI collaboration."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 5000

countries = (
    ["United Kingdom"] * 60
    + ["France"] * 8
    + ["Germany"] * 8
    + ["Netherlands"] * 5
    + ["Belgium"] * 4
    + ["Spain"] * 4
    + ["Switzerland"] * 3
    + ["Portugal"] * 3
    + ["Italy"] * 3
    + ["Sweden"] * 2
)

products = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "REGENCY CAKESTAND 3 TIER",
    "JUMBO BAG RED RETROSPOT",
    "PARTY BUNTING",
    "LUNCH BAG RED RETROSPOT",
    "ASSORTED COLOUR BIRD ORNAMENT",
    "STRAWBERRY CERAMIC TRINKET POT",
    "HEART OF WICKER SMALL",
    "HOME BUILDING BLOCK WORD",
    "LOVE BUILDING BLOCK WORD",
    "CHILLI LIGHTS",
    "HEART SHAPED DOUGHNUT",
    "HANGING METAL LANTERN",
    "BLUE KALMIA CERAMIC MUG",
    "PINK VINTAGE PAISLEY PICNIC BAG",
    "SET OF 3 MONOCHROME HEART MUGS",
    "FAIRY CAKE FLANNEL ASSORTED COLOUR",
    "JUMBO BAG PINK POLKADOT",
    "CHOCOLATE HOT WATER BOTTLE",
    "RETROSPOT TEA SET CERAMIC 11 PC",
]

prices = {p: round(float(np.random.uniform(0.5, 15.0)), 2) for p in products}

rows = []
base_date = datetime(2010, 12, 1)
customer_ids = np.random.randint(12346, 18288, size=500)
invoice_no = 536000

for i in range(n):
    invoice_no += 1 if np.random.random() > 0.3 else 0
    cust = int(np.random.choice(customer_ids))
    country = np.random.choice(countries)
    product = np.random.choice(products)
    qty = int(np.random.exponential(3) + 1)
    price = prices[product]
    day_offset = int(np.random.exponential(30))
    dt = base_date + timedelta(days=int(day_offset))
    is_cancelled = np.random.random() < 0.02
    inv = f"C{invoice_no}" if is_cancelled else str(invoice_no)

    rows.append(
        {
            "InvoiceNo": inv,
            "StockCode": str(21000 + np.random.randint(1, 100)),
            "Description": product,
            "Quantity": min(qty, 50),
            "InvoiceDate": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "UnitPrice": price,
            "CustomerID": cust,
            "Country": country,
        }
    )

df = pd.DataFrame(rows)
path = "data/online_retail.csv"
df.to_csv(path, index=False)
print(f"Generated: {len(df)} rows, {len(df.columns)} columns")
print(f"Countries: {df['Country'].nunique()}")
print(f"Products: {df['Description'].nunique()}")
print(f"Customers: {df['CustomerID'].nunique()}")
print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
