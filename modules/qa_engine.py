"""
Module 5: Intelligent Q&A Engine
Rule-based parser (20+ patterns) + DeepSeek LLM for free-form questions.
Source: Student + AI collaboration.
"""
import re
import json
import os
import hashlib
import pandas as pd

# ---- DeepSeek API configuration ----
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or "sk-4c378ae95b254304aa47b90d9b522122"
_llm_available = None
_llm_last_error = ""

# ---- LLM response cache ----
_llm_cache = {}
CACHE_MAX_SIZE = 200
_cache_hits = 0
_cache_misses = 0


def _check_llm():
    """Lazy check if DeepSeek LLM is available. No live ping — just key check."""
    global _llm_available, _llm_last_error
    if _llm_available is not None:
        return _llm_available
    if not DEEPSEEK_API_KEY:
        _llm_available = False
        _llm_last_error = "未设置 DEEPSEEK_API_KEY"
        return False
    _llm_available = True
    _llm_last_error = ""
    return True


def get_llm_status():
    """Return (available: bool, error_message: str)."""
    _check_llm()
    return _llm_available, _llm_last_error


def get_cache_stats():
    """Return (hits: int, misses: int, size: int)."""
    return _cache_hits, _cache_misses, len(_llm_cache)


def clear_cache():
    """Clear the LLM response cache."""
    global _llm_cache
    _llm_cache.clear()


def _data_fingerprint(df, rfm_df=None):
    """Compute a lightweight fingerprint of the dataset for cache key."""
    h = hashlib.md5()
    h.update(str(len(df)).encode())
    h.update(",".join(sorted(df.columns.tolist())).encode())
    if "TotalPrice" in df.columns:
        h.update(str(int(df["TotalPrice"].sum())).encode())
    if "InvoiceDate" in df.columns:
        h.update(str(df["InvoiceDate"].min()).encode())
        h.update(str(df["InvoiceDate"].max()).encode())
    if rfm_df is not None and "Segment" in rfm_df.columns:
        h.update(str(rfm_df["Segment"].value_counts().to_dict()).encode())
    return h.hexdigest()[:16]


# ---- Rule engine patterns ----
PATTERNS = [
    (r"(卖得?最好|畅销|热销|热门|销售额最高|最好卖|top\s*\d*|best\s*(sell|product))\s*(\d+)?\s*(产品|商品)?",
     "top_products", "bar"),
    (r"(卖得?最差|滞销|最不畅销|销售额最低|worst|worst\s*(sell|product))",
     "bottom_products", "bar"),
    (r"哪些?(国家|地区|市场)\s*(卖得?好|销售额高|畅销)",
     "top_countries", "map"),
    (r"(国家|地区|市场|country|region)\s*(排名|排行|分布|rank)",
     "country_ranking", "map"),
    (r"(\w+国|\w+王国|[一-鿿]{2,4})\s*(的)?\s*(销售|卖得?)",
     "country_detail", "bar"),
    (r"(月度|每月|按月|每个月|每月份|monthly)\s*(趋势|变化|走势|销售|trend)",
     "monthly_trend", "line"),
    (r"(季度|每季)\s*(趋势|变化|走势)",
     "quarterly_trend", "line"),
    (r"(最近|近|过去|last|recent|past)\s*(\d+)\s*(个月?|周|天|日|month|week|day)",
     "recent_period", "line"),
    (r"(什么时候|几点|哪个时间段?|哪个小时)\s*(买|卖|下单|销售)",
     "hourly_pattern", "bar"),
    (r"(星期几|周几|哪天|which\s*day|what\s*day)\s*(买|卖|下单|销售)?",
     "weekday_pattern", "bar"),
    (r"(客户|顾客|用户|customer|user)\s*(分层|分类|分群|细分|画像|segment)",
     "rfm_segments", "pie"),
    (r"(高价值|核心|忠诚|champion|VIP)\s*(客户|顾客|customer)?",
     "rfm_champions", "bar"),
    (r"(流失|流失风险|at\s*risk|lost|churn)\s*(客户|顾客|customer)?",
     "rfm_atrisk", "bar"),
    (r"(客户|顾客|customer)\s*(数量|总数|有多少|count)",
     "customer_count", None),
    (r"(RFM|rfm)", "rfm_overview", "pie"),
    (r"(搜索|查找|找|有没有|search|find)\s*(.+)",
     "search_product", "bar"),
    (r"(.+)\s*(产品|商品)\s*(分析|查询|搜索)",
     "search_product", "bar"),
    (r"(关联|捆绑|搭配|一起买|组合|basket|association|frequent|also\s*buy)",
     "basket_association", "bar"),
    (r"(总共|一共|合计|total)\s*(收入|销售额|营收|订单|revenue|sales)",
     "total_revenue", None),
    (r"(平均|人均|average|avg)\s*(消费|订单金额|客单价|order\s*value)",
     "avg_order_value", None),
    (r"(退货|退款|取消|return|refund|cancel)\s*(率?|情况|分析|rate)?",
     "return_analysis", "bar"),
    (r"(概况|概览|总览|汇总|总结|summary|overview|dashboard)",
     "overview", None),
]


def parse_query(query: str, df: pd.DataFrame, rfm_df: pd.DataFrame = None,
                extra_info: str = "") -> dict:
    """All queries go through DeepSeek LLM. Rule engine is fallback only."""
    global _llm_available, _llm_last_error
    query = query.strip()
    if not query:
        return {"intent": "unknown", "chart_type": None, "data": pd.DataFrame(),
                "answer": "请输入你的数据分析问题。", "matched": False, "source": "none"}

    # 1. DeepSeek LLM (primary)
    if _check_llm():
        try:
            result = _llm_query(query, df, rfm_df, extra_info)
            if result and result.get("matched"):
                result["source"] = "llm"
                return result
        except Exception as e:
            _llm_available = False
            _llm_last_error = str(e)[:200]
            if "402" in str(e) or "Insufficient Balance" in str(e):
                _llm_last_error = "DeepSeek 账户余额不足，请充值后重启应用"

    # 2. Rule engine (fallback if LLM unavailable)
    for pattern, intent, chart_type in PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            result = _execute(intent, chart_type, match, query, df, rfm_df)
            result["source"] = "rule"
            return result

    # 3. Keyword fallback (last resort)
    return _keyword_search(query, df)


# ================================================================
# Rule engine execution
# ================================================================

def _execute(intent, chart_type, match, query, df, rfm_df):
    from modules import analyzer

    try:
        if intent == "top_products":
            n = int(match.group(2)) if match.group(2) else 10
            data = analyzer.top_n_analysis(df, n=n)
            answer = f"以下是销售额最高的 {len(data)} 个产品："

        elif intent == "bottom_products":
            data = analyzer.top_n_analysis(df, n=10).tail(10)
            answer = "以下是销售额最低的 10 个产品："

        elif intent == "top_countries":
            data = analyzer.country_analysis(df).head(10)
            answer = f"销售额最高的国家/地区是 {data.iloc[0]['Country']}。"

        elif intent == "country_ranking":
            data = analyzer.country_analysis(df).head(20)
            answer = "以下是各国/地区销售额排名："

        elif intent == "country_detail":
            country = match.group(1)
            data = analyzer.country_analysis(df)
            data = data[data["Country"].str.contains(country, case=False, na=False)]
            answer = f"{country} 的总销售额为 {data['TotalRevenue'].sum():,.2f}。" if len(data) > 0 else f"未找到与 {country} 相关的数据。"
            chart_type = "bar"

        elif intent == "monthly_trend":
            data = analyzer.monthly_trend(df)
            answer = "以下是月度销售趋势："

        elif intent == "recent_period":
            n = int(match.group(2)) if match.group(2) else 6
            data = analyzer.monthly_trend(df).tail(n)
            answer = f"以下是最近 {n} 个月的销售数据："

        elif intent == "hourly_pattern":
            data = analyzer.hourly_trend(df)
            best = data.loc[data["TotalRevenue"].idxmax(), "Hour"]
            answer = f"下单高峰期在 {int(best)}:00 左右。"

        elif intent == "weekday_pattern":
            data = analyzer.weekday_trend(df)
            best = data.loc[data["TotalRevenue"].idxmax(), "WeekdayName"]
            answer = f"一周中 {best} 的销售额最高。"

        elif intent in ("rfm_segments", "rfm_overview"):
            if rfm_df is not None:
                summary = analyzer.rfm_summary(rfm_df)
                segs = summary["segment_stats"]
                data = pd.DataFrame([
                    {"Segment": k, "Count": v["count"], "AvgSpend": v["avg_monetary"], "Revenue": v["total_revenue"]}
                    for k, v in segs.items()
                ])
                answer = f"共有 {summary['overall_stats']['total_customers']} 名客户，"
                answer += f"总消费 {summary['overall_stats']['total_revenue']:,.2f}。"
            else:
                data, answer = pd.DataFrame(), "请先运行数据预处理。"

        elif intent == "rfm_champions":
            data = rfm_df[rfm_df["Segment"] == "Champions"] if rfm_df is not None else pd.DataFrame()
            answer = f"有 {len(data)} 名核心高价值客户。"

        elif intent == "rfm_atrisk":
            data = rfm_df[rfm_df["Segment"].isin(["At Risk", "Lost"])] if rfm_df is not None else pd.DataFrame()
            answer = f"有 {len(data)} 名流失风险客户需要关注。"

        elif intent == "customer_count":
            cnt = len(rfm_df) if rfm_df is not None else df["CustomerID"].nunique()
            data = pd.DataFrame({"指标": ["客户总数"], "数值": [cnt]})
            answer = f"共有 {cnt} 名独立客户。"

        elif intent == "search_product":
            keyword = match.group(2).strip() if match.lastindex >= 2 else query
            data = analyzer.search_products(df, keyword, n=10)
            answer = f"找到 {len(data)} 个相关产品：" if len(data) > 0 else f"未找到与 '{keyword}' 相关的产品。"

        elif intent == "basket_association":
            rules = analyzer.market_basket_analysis(df, min_support=10)
            data = pd.DataFrame(rules[:20]) if rules else pd.DataFrame()
            answer = f"找到 {len(rules)} 组关联商品。" if rules else "未找到显著的商品关联。"

        elif intent == "total_revenue":
            total = df["TotalPrice"].sum()
            orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else len(df)
            data = pd.DataFrame({"指标": ["总收入", "记录数"], "数值": [f"{total:,.2f}", orders]})
            answer = f"总销售额为 {total:,.2f}。"

        elif intent == "avg_order_value":
            if "InvoiceNo" in df.columns:
                avg = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()
            else:
                avg = df["TotalPrice"].mean()
            data = pd.DataFrame({"指标": ["平均客单价"], "数值": [f"{avg:,.2f}"]})
            answer = f"平均每笔订单金额为 {avg:,.2f}。"

        elif intent == "return_analysis":
            if "IsCancelled" in df.columns:
                cancelled = df[df["IsCancelled"]]
                rate = len(cancelled) / max(len(df), 1) * 100
            else:
                rate = 0
            data = pd.DataFrame({"指标": ["记录总数", "退货率%"], "数值": [len(df), round(rate, 2)]})
            answer = f"共 {len(df):,} 条记录。"

        elif intent == "overview":
            total = f"{df['TotalPrice'].sum():,.2f}"
            orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else len(df)
            cust = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)
            prods = df["StockCode"].nunique() if "StockCode" in df.columns else 0
            data = pd.DataFrame({
                "指标": ["总销售额", "订单/记录数", "客户/实体数", "产品/类别数"],
                "数值": [total, orders, cust, prods]
            })
            answer = f"数据集包含 {orders:,} 条记录。"

        else:
            return _keyword_search(query, df)

        return {"intent": intent, "chart_type": chart_type, "data": data,
                "answer": answer, "matched": True}

    except Exception as e:
        return {"intent": "error", "chart_type": None, "data": pd.DataFrame(),
                "answer": f"处理出错: {str(e)}", "matched": False}


def _keyword_search(query, df):
    from modules import analyzer
    data = analyzer.search_products(df, query, n=10)
    if len(data) > 0:
        return {"intent": "keyword_search", "chart_type": "bar", "data": data,
                "answer": f"搜索 '{query}' 找到 {len(data)} 个相关产品：", "matched": True, "source": "keyword"}
    return {"intent": "unknown", "chart_type": None, "data": pd.DataFrame(),
            "answer": f"我暂时无法理解「{query}」。\n\n你可以尝试：\n- 数据分析类：卖得最好的产品、月度销售趋势、客户分层\n- 电商咨询类：如何提升复购率？RFM模型怎么用？\n- 任意问题：无限制，尽管问！\n\n💡 设置 DEEPSEEK_API_KEY 可启用 AI 智能回答。",
            "matched": False, "source": "fallback"}


# ================================================================
# LLM Integration (DeepSeek)
# ================================================================

def _build_data_context(df, rfm_df=None, extra_info=""):
    lines = []
    if extra_info:
        lines.append(extra_info)
    lines += [f"Rows: {len(df):,}", f"Columns: {', '.join(df.columns.tolist())}"]
    if "InvoiceDate" in df.columns:
        lines.append(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    if "TotalPrice" in df.columns:
        lines.append(f"Total revenue: {df['TotalPrice'].sum():,.2f}")
    if "StockCode" in df.columns:
        lines.append(f"Unique products: {df['StockCode'].nunique()}")
    if "CustomerID" in df.columns:
        lines.append(f"Unique customers: {df['CustomerID'].nunique()}")
    if "Country" in df.columns:
        top_countries = df["Country"].value_counts().head(6).index.tolist()
        lines.append(f"Countries: {', '.join(top_countries)}")
    if "Description" in df.columns:
        top5 = df.groupby("Description")["TotalPrice"].sum().sort_values(ascending=False).head(5)
        lines.append("Top 5 products: " + ", ".join(f"{p} ({v:,.0f})" for p, v in top5.items()))
    if rfm_df is not None and "Segment" in rfm_df.columns:
        segs = rfm_df["Segment"].value_counts().to_dict()
        lines.append("Customer segments: " + ", ".join(f"{k}: {v}" for k, v in segs.items()))
    return "\n".join(lines)


def _llm_query(query, df, rfm_df=None, extra_info=""):
    global _cache_hits, _cache_misses

    # Check cache first
    fp = _data_fingerprint(df, rfm_df)
    cache_key = f"{fp}|{query.strip().lower()}"
    if cache_key in _llm_cache:
        _cache_hits += 1
        return _llm_cache[cache_key]

    _cache_misses += 1

    from openai import OpenAI
    from modules import analyzer

    context = _build_data_context(df, rfm_df, extra_info)
    intents = "top_products, bottom_products, top_countries, country_ranking, country_detail, monthly_trend, recent_period, hourly_pattern, weekday_pattern, rfm_segments, rfm_champions, rfm_atrisk, customer_count, search_product, basket_association, total_revenue, avg_order_value, return_analysis, overview, general_qa"

    system = f"""你是电商数据分析助手。回答任意问题，不仅限于数据查询。

当前数据:
{context}

可用意图: {intents}

仅回复JSON:
{{"intent": "<intent>", "parameters": {{}}, "answer": "<中文回答>", "chart_type": "<bar|line|pie|scatter|map|null>"}}

规则:
- search_product加"keyword"参数
- top_products加"n"参数(默认10)
- general_qa: 用中文回答, chart_type=null
- 始终用中文回答
- 与数据无关的问题使用general_qa"""

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
        temperature=0.3, max_tokens=400,
    )

    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)
    intent = parsed.get("intent", "general_qa")
    params = parsed.get("parameters", {})
    answer = parsed.get("answer", "")
    chart_type = parsed.get("chart_type")

    # Execute intent
    data = pd.DataFrame()
    try:
        if intent == "top_products":
            data = analyzer.top_n_analysis(df, n=params.get("n", 10))
        elif intent == "top_countries":
            data = analyzer.country_analysis(df).head(10); chart_type = chart_type or "map"
        elif intent == "country_ranking":
            data = analyzer.country_analysis(df).head(20)
        elif intent == "monthly_trend":
            data = analyzer.monthly_trend(df)
        elif intent == "recent_period":
            data = analyzer.monthly_trend(df).tail(params.get("n", 6))
        elif intent == "hourly_pattern":
            data = analyzer.hourly_trend(df)
        elif intent == "weekday_pattern":
            data = analyzer.weekday_trend(df)
        elif intent in ("rfm_segments", "rfm_overview") and rfm_df is not None:
            summary = analyzer.rfm_summary(rfm_df)
            segs = summary["segment_stats"]
            data = pd.DataFrame([
                {"Segment": k, "Count": v["count"], "AvgSpend": v["avg_monetary"], "Revenue": v["total_revenue"]}
                for k, v in segs.items()
            ])
        elif intent == "rfm_champions" and rfm_df is not None:
            data = rfm_df[rfm_df["Segment"] == "Champions"]
        elif intent == "rfm_atrisk" and rfm_df is not None:
            data = rfm_df[rfm_df["Segment"].isin(["At Risk", "Lost"])]
        elif intent == "search_product":
            data = analyzer.search_products(df, params.get("keyword", query), n=10)
        elif intent == "basket_association":
            rules = analyzer.market_basket_analysis(df, min_support=10)
            data = pd.DataFrame(rules[:20]) if rules else pd.DataFrame()
        elif intent == "total_revenue":
            total = f"{df['TotalPrice'].sum():,.2f}"
            orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else len(df)
            data = pd.DataFrame({"指标": ["总收入", "记录数"], "数值": [total, orders]})
        elif intent == "avg_order_value":
            if "InvoiceNo" in df.columns:
                avg = f"{df.groupby('InvoiceNo')['TotalPrice'].sum().mean():,.2f}"
            else:
                avg = f"{df['TotalPrice'].mean():,.2f}"
            data = pd.DataFrame({"指标": ["平均每单金额"], "数值": [avg]})
        elif intent == "return_analysis":
            cancelled = df[df["IsCancelled"]] if "IsCancelled" in df.columns else pd.DataFrame()
            data = pd.DataFrame({"类型": ["正常", "退货/取消"], "数量": [len(df), len(cancelled)]})
        elif intent == "overview":
            total = f"{df['TotalPrice'].sum():,.2f}"
            orders = df["InvoiceNo"].nunique() if "InvoiceNo" in df.columns else len(df)
            cust = df["CustomerID"].nunique() if "CustomerID" in df.columns else len(df)
            prods = df["StockCode"].nunique() if "StockCode" in df.columns else 0
            data = pd.DataFrame({
                "指标": ["总销售额", "记录数", "实体数", "类别数"],
                "数值": [total, orders, cust, prods]
            })
        elif intent == "general_qa":
            chart_type = None
    except Exception:
        pass

    result = {"intent": intent, "chart_type": chart_type, "data": data,
              "answer": answer or "分析完成", "matched": True}

    # Store in cache (LRU eviction)
    if len(_llm_cache) >= CACHE_MAX_SIZE:
        _llm_cache.pop(next(iter(_llm_cache)))
    _llm_cache[cache_key] = result.copy()
    return result


def get_example_questions():
    return [
        "卖得最好的15个产品",
        "月度销售趋势",
        "客户分层分析",
        "国家/地区排名",
        "电商数据分析中，RFM模型有什么作用？",
        "购物篮关联分析",
        "如何提高客户复购率？",
        "搜索 WHITE HANGING HEART",
        "帮我分析一下哪个国家的客户最值钱",
        "介绍一下电商常用的数据分析方法",
    ]
