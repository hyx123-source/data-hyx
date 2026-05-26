"""Test auth + QA integration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import auth, qa_engine, preprocessor
import pandas as pd

# 1. Auth
auth.init_db()
ok, msg, user = auth.login_user("admin", "admin123")
assert ok, f"Login failed: {msg}"
assert auth.is_admin(user)
print("1. Auth: admin login OK")

# Registration
ok, msg = auth.register_user("testuser", "test123456", "test@test.com")
if ok:
    ok2, msg2, u2 = auth.login_user("testuser", "test123456")
    role = u2["role"]
    print(f"2. Auth: register+login OK (role={role})")
    auth.delete_user(u2["id"])
else:
    print(f"2. Auth: {msg}")

# 3. QA without LLM
df = pd.read_csv("data/online_retail.csv")
df_c = preprocessor.preprocess_online_retail(df)
df_t = preprocessor.get_clean_transactions(df_c)
rfm = preprocessor.get_rfm_table(df_c)

r = qa_engine.parse_query("卖得最好的5个产品", df_t, rfm)
assert r["matched"]
print(f"3. QA CN: OK (intent={r['intent']}, source={r.get('source', 'rule')})")

r2 = qa_engine.parse_query("What products sell best?", df_t, rfm)
print(f"4. QA EN: intent={r2['intent']}, source={r2.get('source', 'keyword')}")

# 5. Query logging
auth.log_query("admin", "test query", "test answer", "test_intent")
logs = auth.get_user_query_logs("admin")
print(f"5. Log: {len(logs)} entries for admin")

all_logs = auth.get_all_query_logs()
print(f"6. Admin logs: {len(all_logs)} total entries")

print("\n=== ALL TESTS PASSED ===")
