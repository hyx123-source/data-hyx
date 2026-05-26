"""
AI-Data-QA: Intelligent Data Query Web Application
Multi-user with authentication, role-based access, and LLM-powered Q&A.

Run: streamlit run app.py
Source: Student + AI collaboration.
"""
import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import data_loader, preprocessor, analyzer, visualizer, qa_engine, auth

# ---- Page config ----
st.set_page_config(
    page_title="智能问数 - E-commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Init DB (seeds admin/admin123) ----
auth.init_db()

# ---- Session state ----
for key, default in [
    ("authenticated", False), ("user", None), ("page", "login"),
    ("df_raw", None), ("df_clean", None), ("rfm_df", None),
    ("preprocessed", False), ("data_loaded", False), ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ================================================================
# LOGIN PAGE
# ================================================================
def page_login():
    st.title("🔐 登录 — 电商智能分析问答系统")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_pwd")

        if st.button("登 录", use_container_width=True):
            ok, msg, user = auth.login_user(username, password)
            if ok:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.page = "main"
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error(msg)

        st.divider()
        if st.button("还没有账号？点击注册", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()


# ================================================================
# REGISTER PAGE
# ================================================================
def page_register():
    st.title("📝 注册新账号")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("用户名（仅字母数字）", key="reg_user")
        email = st.text_input("邮箱（选填）", key="reg_email")
        password = st.text_input("密码（至少6位）", type="password", key="reg_pwd")
        password2 = st.text_input("确认密码", type="password", key="reg_pwd2")

        if st.button("注 册", use_container_width=True):
            if not username or not password:
                st.error("用户名和密码不能为空")
            elif password != password2:
                st.error("两次密码不一致")
            else:
                ok, msg = auth.register_user(username, password, email)
                if ok:
                    st.success(msg)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error(msg)

        if st.button("← 返回登录", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()


# ================================================================
# MAIN APPLICATION (authenticated)
# ================================================================
def page_main():
    user = st.session_state.user
    is_admin = auth.is_admin(user)

    # ---- Sidebar ----
    with st.sidebar:
        st.title("📊 智能问数")
        role_badge = "🛡️ 管理员" if is_admin else "👤 用户"
        st.markdown(f"**{role_badge}**: `{user['username']}`")
        st.divider()

        # Module 1: Data loading
        st.subheader("📁 数据加载")
        uploaded_file = st.file_uploader(
            "上传数据文件 (CSV/Excel/JSON)",
            type=["csv", "xlsx", "xls", "json"],
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                st.session_state.df_raw = data_loader.load_file(file_bytes, uploaded_file.name)
                st.session_state.data_loaded = True
                st.session_state.preprocessed = False
                st.success(f"✅ 已加载: {uploaded_file.name}")
            except Exception as e:
                st.error(f"加载失败: {e}")
        else:
            default_path = os.path.join(os.path.dirname(__file__), "data", "online_retail.csv")
            if os.path.exists(default_path) and not st.session_state.data_loaded:
                try:
                    st.session_state.df_raw = pd.read_csv(default_path, encoding="utf-8")
                    st.session_state.data_loaded = True
                    st.info("📦 已加载默认数据集")
                except Exception:
                    st.warning("默认数据集不可用，请上传文件。")

        st.divider()

        # Module 2: Preprocessing
        st.subheader("🔧 数据预处理")
        if st.session_state.data_loaded and st.button("运行预处理", width="stretch"):
            with st.spinner("处理中..."):
                df = st.session_state.df_raw.copy()
                df = preprocessor.preprocess_online_retail(df)
                st.session_state.df_clean = df
                st.session_state.rfm_df = preprocessor.get_rfm_table(df)
                st.session_state.preprocessed = True
            st.success(f"✅ 完成: {len(df):,} 条记录")

        if st.session_state.preprocessed:
            df_c = st.session_state.df_clean
            rfm = st.session_state.rfm_df
            st.metric("有效交易", f"{len(df_c):,}")
            st.metric("客户数", f"{rfm['CustomerID'].nunique():,}")
            st.metric("总销售额", f"{df_c['TotalPrice'].sum():,.0f}")

        st.divider()

        # Logout & settings
        with st.expander("⚙️ 设置"):
            if st.button("修改密码", width="stretch"):
                st.session_state.show_change_pwd = True
            if st.button("🚪 退出登录", width="stretch"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

        st.caption(f"© 2025 AI-Data-QA | v2.0 LLM")

    # ---- Change password dialog ----
    if st.session_state.get("show_change_pwd"):
        with st.sidebar:
            with st.form("change_pwd_form"):
                st.subheader("修改密码")
                old = st.text_input("原密码", type="password")
                new = st.text_input("新密码", type="password")
                new2 = st.text_input("确认新密码", type="password")
                if st.form_submit_button("确认修改"):
                    if new != new2:
                        st.error("两次密码不一致")
                    else:
                        ok, msg = auth.change_password(user["username"], old, new)
                        if ok:
                            st.success(msg)
                            st.session_state.show_change_pwd = False
                        else:
                            st.error(msg)

    # ---- Main tabs ----
    tab_names = ["📋 数据概览", "📊 数据分析", "💬 智能问答"]
    if is_admin:
        tab_names.append("🛡️ 管理面板")

    tabs = st.tabs(tab_names)

    # ========== Tab: Data Overview ==========
    with tabs[0]:
        if not st.session_state.data_loaded:
            st.info("👈 请上传数据文件或用默认数据集。")
        else:
            df = st.session_state.df_raw
            c1, c2, c3 = st.columns(3)
            c1.metric("总行数", f"{len(df):,}")
            c2.metric("列数", len(df.columns))
            c3.metric("内存", f"{df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")

            st.subheader("原始数据预览")
            st.dataframe(df.head(20), width="stretch")

            st.subheader("字段信息")
            ic1, ic2 = st.columns(2)
            with ic1:
                st.dataframe(pd.DataFrame({
                    "列名": df.dtypes.index,
                    "类型": df.dtypes.values.astype(str),
                    "缺失": df.isnull().sum().values,
                    "缺失%": (df.isnull().sum()/len(df)*100).round(2).values,
                }), width="stretch")
            with ic2:
                num_cols = df.select_dtypes(include=["number"]).columns.tolist()
                if num_cols:
                    st.dataframe(df[num_cols].describe(), width="stretch")

            if st.session_state.preprocessed:
                st.divider()
                st.subheader("预处理后数据")
                st.dataframe(st.session_state.df_clean.head(20), width="stretch")
                st.subheader("RFM 客户分层表")
                st.dataframe(st.session_state.rfm_df.head(20), width="stretch")

    # ========== Tab: Data Analysis ==========
    with tabs[1]:
        if not st.session_state.preprocessed:
            st.info("请先在左侧运行数据预处理。")
        else:
            df_clean = preprocessor.get_clean_transactions(st.session_state.df_clean)
            rfm_df = st.session_state.rfm_df

            analysis_type = st.selectbox("选择分析方法", [
                "📦 产品销售 Top-N", "🗺️ 国家/地区分析", "📈 月度销售趋势",
                "⏰ 时段分析", "👥 RFM 客户分层", "🔗 购物篮关联", "🔍 产品搜索",
            ])
            st.divider()

            if analysis_type == "📦 产品销售 Top-N":
                n = st.slider("显示前 N 个", 5, 50, 15)
                top = analyzer.top_n_analysis(df_clean, n=n)
                st.plotly_chart(visualizer.plot_top_products(top), width="stretch")
                st.dataframe(top, width="stretch")

            elif analysis_type == "🗺️ 国家/地区分析":
                country = analyzer.country_analysis(df_clean)
                c1, c2 = st.columns(2)
                c1.plotly_chart(visualizer.plot_country_bar(country), width="stretch")
                c2.plotly_chart(visualizer.plot_country_revenue(country), width="stretch")
                st.dataframe(country, width="stretch")

            elif analysis_type == "📈 月度销售趋势":
                trend = analyzer.monthly_trend(df_clean)
                st.plotly_chart(visualizer.plot_monthly_trend(trend), width="stretch")
                st.dataframe(trend, width="stretch")

            elif analysis_type == "⏰ 时段分析":
                c1, c2 = st.columns(2)
                c1.plotly_chart(visualizer.plot_hourly(analyzer.hourly_trend(df_clean)), width="stretch")
                c2.plotly_chart(visualizer.plot_weekday(analyzer.weekday_trend(df_clean)), width="stretch")

            elif analysis_type == "👥 RFM 客户分层":
                summary = analyzer.rfm_summary(rfm_df)
                st.plotly_chart(visualizer.plot_rfm_distribution(rfm_df), width="stretch")
                for seg, stats in summary["segment_stats"].items():
                    with st.expander(f"{seg} ({stats['count']} 人, 营收 {stats['total_revenue']:,.2f})"):
                        cols = st.columns(4)
                        cols[0].metric("客户数", stats["count"])
                        cols[1].metric("平均最近购买(天)", stats["avg_recency"])
                        cols[2].metric("平均频次", stats["avg_frequency"])
                        cols[3].metric("平均消费", f"{stats['avg_monetary']:,.2f}")
                if st.button("运行 KMeans 聚类"):
                    with st.spinner("聚类中..."):
                        clust = analyzer.product_clustering(rfm_df)
                        st.plotly_chart(visualizer.plot_cluster_scatter(rfm_df, clust["cluster_labels"]), width="stretch")

            elif analysis_type == "🔗 购物篮关联":
                ms = st.slider("最小支持度", 5, 100, 20)
                rules = analyzer.market_basket_analysis(df_clean, min_support=ms)
                if rules:
                    st.plotly_chart(visualizer.plot_basket_associations(rules), width="stretch")
                    st.dataframe(pd.DataFrame(rules), width="stretch")
                else:
                    st.warning("未找到关联规则，请降低最小支持度。")

            elif analysis_type == "🔍 产品搜索":
                kw = st.text_input("关键词", placeholder="HEART, CANDLE...")
                if kw:
                    result = analyzer.search_products(df_clean, kw)
                    if len(result) > 0:
                        st.plotly_chart(visualizer.auto_chart(result.head(15), "bar"), width="stretch")
                        st.dataframe(result, width="stretch")

    # ========== Tab: Intelligent Q&A ==========
    with tabs[2]:
        if not st.session_state.preprocessed:
            st.info("请先在左侧运行数据预处理。")
        else:
            st.subheader("💬 智能问答")

            has_llm, llm_error = qa_engine.get_llm_status()
            if has_llm:
                st.success("🤖 DeepSeek AI 已连接 — 支持任意自然语言提问")
            else:
                if "DEEPSEEK_API_KEY" in llm_error:
                    st.info("💡 提示：设置环境变量 `DEEPSEEK_API_KEY` 可接入 AI 智能体")
                else:
                    st.warning(f"⚠️ DeepSeek 连接异常: {llm_error}")

            with st.expander("💡 示例问题"):
                for q in qa_engine.get_example_questions():
                    if st.button(q, key=f"ex_{q}"):
                        st.session_state.current_question = q

            default_q = st.session_state.get("current_question", "")
            query = st.chat_input("输入你的数据问题...") or default_q

            if query:
                df_clean = preprocessor.get_clean_transactions(st.session_state.df_clean)
                rfm = st.session_state.rfm_df

                with st.spinner("分析中..." + (" (DeepSeek AI 思考中...)" if has_llm else "")):
                    result = qa_engine.parse_query(query, df_clean, rfm)

                if not has_llm and result.get("source") == "fallback":
                    st.info("💡 该问题未匹配规则模板，接入 DeepSeek 后可自动理解。设置 `DEEPSEEK_API_KEY` 即可。")

                st.chat_message("user").write(query)
                source_tag = f"`[{result.get('source', 'rule')}]`"
                st.chat_message("assistant").write(f"{result['answer']}  {source_tag}")

                if result["data"] is not None and not result["data"].empty:
                    with st.expander("📋 查看数据"):
                        st.dataframe(result["data"], width="stretch")

                if result.get("chart_type"):
                    with st.spinner("生成图表..."):
                        chart = visualizer.auto_chart(result["data"], result["chart_type"], title=query)
                        st.plotly_chart(chart, width="stretch")

                # Log to DB
                try:
                    auth.log_query(user["username"], query, result["answer"], result.get("intent", ""))
                except Exception:
                    pass

                st.session_state.chat_history.append({
                    "query": query, "answer": result["answer"],
                    "intent": result.get("intent"), "source": result.get("source"),
                })

            if st.session_state.chat_history:
                st.divider()
                st.subheader("📝 问答历史")
                for i, entry in enumerate(reversed(st.session_state.chat_history[-10:])):
                    with st.expander(f"Q{i+1}: {entry['query'][:60]}...", expanded=False):
                        st.write(f"Intent: {entry.get('intent')} | Source: {entry.get('source')}")
                        st.write(entry["answer"])

            if st.button("清空历史", width="stretch"):
                st.session_state.chat_history = []
                st.session_state.pop("current_question", None)
                st.rerun()

    # ========== Tab: Admin Panel ==========
    if is_admin:
        with tabs[3]:
            st.subheader("🛡️ 管理员面板")
            admin_tab = st.radio("管理选项", ["👥 用户管理", "📋 问答日志"], horizontal=True)

            if admin_tab == "👥 用户管理":
                users = auth.get_all_users()
                st.metric("注册用户总数", len(users))
                df_users = pd.DataFrame(users)
                df_users.columns = ["ID", "用户名", "角色", "邮箱", "注册时间", "最后登录"]
                st.dataframe(df_users, width="stretch")

                st.subheader("删除用户")
                col_del = st.columns([3, 1, 2])
                non_admin_users = [u for u in users if u["username"] != "admin"]
                if non_admin_users:
                    del_username = col_del[0].selectbox(
                        "选择用户", [u["username"] for u in non_admin_users], key="del_user"
                    )
                    if col_del[1].button("删除", width="stretch"):
                        target = next((u for u in users if u["username"] == del_username), None)
                        if target:
                            ok, msg = auth.delete_user(target["id"])
                            if ok:
                                st.success(msg); st.rerun()
                            else:
                                st.error(msg)
                else:
                    st.info("没有可删除的普通用户")

            elif admin_tab == "📋 问答日志":
                logs = auth.get_all_query_logs()
                st.metric("总问答次数", len(logs))
                if logs:
                    df_logs = pd.DataFrame(logs)
                    st.dataframe(df_logs[["username", "query", "intent", "timestamp"]].head(50), width="stretch")

                    # Filter by user
                    usernames = list(set(l["username"] for l in logs))
                    filter_user = st.selectbox("按用户筛选", ["全部"] + usernames)
                    if filter_user != "全部":
                        filtered = [l for l in logs if l["username"] == filter_user]
                        st.dataframe(pd.DataFrame(filtered)[["query", "answer", "intent", "timestamp"]].head(20), width="stretch")


# ================================================================
# Router
# ================================================================
if not st.session_state.authenticated:
    if st.session_state.page == "register":
        page_register()
    else:
        page_login()
else:
    page_main()
