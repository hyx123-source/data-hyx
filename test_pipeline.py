"""Integration test: verify all 5 modules work end-to-end.
Source: Student + AI collaboration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import data_loader, preprocessor, analyzer, visualizer, qa_engine
import pandas as pd

# Load
csv_path = os.path.join("data", "online_retail.csv")
with open(csv_path, "rb") as f:
    df = data_loader.load_file(f.read(), "online_retail.csv")
print(f"1. Data loaded: {df.shape}")

# Preprocess
df_clean = preprocessor.preprocess_online_retail(df)
print(f"2. Preprocessed: {len(df_clean)} valid rows")

# RFM
rfm = preprocessor.get_rfm_table(df_clean)
print(f"3. RFM: {len(rfm)} customers, {rfm['Segment'].nunique()} segments")

# Analysis
top = analyzer.top_n_analysis(df_clean, n=5)
print(f"4. Top products: {len(top)}")

trend = analyzer.monthly_trend(df_clean)
print(f"5. Monthly trend: {len(trend)} months")

# Basket
df_clean_t = preprocessor.get_clean_transactions(df_clean)
rules = analyzer.market_basket_analysis(df_clean_t, min_support=3)
print(f"6. Basket rules: {len(rules)} pairs")

# QA
r1 = qa_engine.parse_query("卖得最好的5个产品", df_clean_t, rfm)
print(f"7. QA: matched={r1['matched']}, intent={r1['intent']}")

r2 = qa_engine.parse_query("月度趋势", df_clean_t, rfm)
print(f"8. QA: matched={r2['matched']}, intent={r2['intent']}")

r3 = qa_engine.parse_query("客户分层", df_clean_t, rfm)
print(f"9. QA: matched={r3['matched']}, intent={r3['intent']}")

# Visualizer
fig = visualizer.plot_top_products(top)
print(f"10. Chart: OK ({type(fig).__name__})")

print("\n=== ALL MODULES WORKING! ===")
