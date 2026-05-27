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
    ("_datasets", {}), ("_active_dataset", None), ("_dataset_order", []),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _ds_get(key, default=None):
    """Get a value from the active dataset."""
    name = st.session_state.get("_active_dataset")
    if name and name in st.session_state.get("_datasets", {}):
        return st.session_state._datasets[name].get(key, default)
    return default


def _ds_set(key, value):
    """Set a value on the active dataset."""
    name = st.session_state.get("_active_dataset")
    if name and name in st.session_state.get("_datasets", {}):
        st.session_state._datasets[name][key] = value


def _register_dataset(ds_name: str, df_raw: pd.DataFrame, source: str):
    """Add a dataset to the registry and make it active. Deduplicate name."""
    datasets = st.session_state._datasets
    order = st.session_state._dataset_order
    # Deduplicate name
    base = ds_name
    counter = 1
    while ds_name in datasets:
        counter += 1
        dot_pos = base.rfind(".")
        if dot_pos > 0:
            ds_name = f"{base[:dot_pos]} ({counter}){base[dot_pos:]}"
        else:
            ds_name = f"{base} ({counter})"
    datasets[ds_name] = {
        "df_raw": df_raw,
        "df_clean": None,
        "rfm_df": None,
        "preprocessed": False,
        "data_source": source,
    }
    order.append(ds_name)
    st.session_state._active_dataset = ds_name


def _remove_dataset(ds_name: str):
    """Remove a dataset from the registry."""
    datasets = st.session_state._datasets
    order = st.session_state._dataset_order
    if ds_name in datasets:
        del datasets[ds_name]
    if ds_name in order:
        order.remove(ds_name)
    # Switch active to the next available
    if st.session_state._active_dataset == ds_name:
        if order:
            st.session_state._active_dataset = order[-1]
        else:
            st.session_state._active_dataset = None


# ================================================================
# Helper: save & load uploaded files
# ================================================================
def _get_upload_dir(username: str) -> str:
    upload_dir = os.path.join(os.path.dirname(__file__), "data", "uploads", username)
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def _save_uploaded_file(file_bytes: bytes, filename: str, username: str):
    try:
        upload_dir = _get_upload_dir(username)
        save_path = os.path.join(upload_dir, filename)
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    except Exception:
        pass


def _get_saved_files(username: str) -> list:
    upload_dir = _get_upload_dir(username)
    if not os.path.exists(upload_dir):
        return []
    files = []
    for fname in os.listdir(upload_dir):
        if fname.endswith((".csv", ".xlsx", ".xls", ".json")):
            files.append({
                "name": fname,
                "path": os.path.join(upload_dir, fname),
            })
    return sorted(files, key=lambda x: x["name"])


def _delete_saved_file(file_path: str) -> bool:
    """Delete a saved file from disk. Returns True on success."""
    try:
        os.remove(file_path)
        return True
    except Exception:
        return False


def _get_all_saved_files() -> list:
    """Return all saved files from all users (for admin)."""
    base = os.path.join(os.path.dirname(__file__), "data", "uploads")
    if not os.path.exists(base):
        return []
    all_files = []
    for username in os.listdir(base):
        user_dir = os.path.join(base, username)
        if not os.path.isdir(user_dir):
            continue
        for fname in os.listdir(user_dir):
            if fname.endswith((".csv", ".xlsx", ".xls", ".json")):
                all_files.append({
                    "name": fname,
                    "path": os.path.join(user_dir, fname),
                    "owner": username,
                })
    return sorted(all_files, key=lambda x: (x["name"], x["owner"]))


# ================================================================
# LOGIN PAGE
# ================================================================
def page_login():
    st.title("🔐 登录 — 电商智能分析问答系统")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("用户名", key="login_user", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", key="login_pwd", placeholder="请输入密码")

            submitted = st.form_submit_button("登 录", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("请输入用户名和密码")
                else:
                    with st.spinner("登录中..."):
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
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
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
        with st.form("register_form", clear_on_submit=False):
            username = st.text_input("用户名（仅字母数字）", key="reg_user", placeholder="3-20位字母或数字")
            email = st.text_input("邮箱（选填）", key="reg_email", placeholder="optional@example.com")
            password = st.text_input("密码（至少6位）", type="password", key="reg_pwd", placeholder="至少6位字符")
            password2 = st.text_input("确认密码", type="password", key="reg_pwd2", placeholder="再次输入密码")

            submitted = st.form_submit_button("注 册", use_container_width=True)
            if submitted:
                if not username or not password:
                    st.error("用户名和密码不能为空")
                elif len(username) < 3:
                    st.error("用户名至少3位")
                elif len(password) < 6:
                    st.error("密码至少6位")
                elif password != password2:
                    st.error("两次密码不一致")
                else:
                    with st.spinner("注册中..."):
                        ok, msg = auth.register_user(username, password, email)
                    if ok:
                        st.success(msg)
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error(msg)

        st.divider()
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
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

        # ---- Upload new file ----
        uploaded_file = st.file_uploader(
            "上传数据文件 (CSV/Excel/JSON)",
            type=["csv", "xlsx", "xls", "json"],
            key="file_uploader",
        )

        if uploaded_file is not None:
            prev_name = st.session_state.get("_uploaded_filename", "")
            if prev_name != uploaded_file.name:
                try:
                    file_bytes = uploaded_file.read()
                    df_new = data_loader.load_file(file_bytes, uploaded_file.name)
                    _register_dataset(uploaded_file.name, df_new, f"上传: {uploaded_file.name}")
                    st.session_state._uploaded_filename = uploaded_file.name
                    st.session_state._prev_saved = None
                    st.session_state.pop("saved_file_select", None)
                    _save_uploaded_file(file_bytes, uploaded_file.name, user["username"])
                    file_size_kb = len(file_bytes) / 1024
                    auth.log_upload(user["username"], uploaded_file.name, len(df_new), len(df_new.columns), file_size_kb)
                    st.success(f"✅ 已加载并保存: {uploaded_file.name} ({len(df_new):,} 行)")
                except Exception as e:
                    st.error(f"加载失败: {e}")

        # ---- Load previously saved file ----
        if is_admin:
            saved_files = _get_all_saved_files()
            if saved_files:
                display_names = ["— 不选择 —"]
                name_to_file = {}
                for f in saved_files:
                    label = f"{f['name']}  [{f['owner']}]"
                    display_names.append(label)
                    name_to_file[label] = f
                selected_saved = st.selectbox(
                    "从已保存文件中添加（全部用户）",
                    display_names,
                    key="saved_file_select",
                )
                if selected_saved != "— 不选择 —":
                    prev_selected = st.session_state.get("_prev_saved", "")
                    if prev_selected != selected_saved:
                        try:
                            target = name_to_file[selected_saved]
                            with open(target["path"], "rb") as f:
                                file_bytes = f.read()
                            df_new = data_loader.load_file(file_bytes, target["name"])
                            _register_dataset(target["name"], df_new,
                                              f"已保存: {target['name']} (来自: {target['owner']})")
                            st.session_state._prev_saved = selected_saved
                            st.success(f"✅ 已加载: {target['name']} ({len(df_new):,} 行)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {e}")
                    # Delete from disk button
                    target = name_to_file[selected_saved]
                    if st.button(f"🗑️ 从磁盘删除 {target['name']}", key=f"del_{target['path']}"):
                        st.session_state._disk_delete_target = target
                        st.rerun()
        else:
            saved_files = _get_saved_files(user["username"])
            if saved_files:
                saved_names = [f["name"] for f in saved_files]
                selected_saved = st.selectbox(
                    "从已保存文件中添加",
                    ["— 不选择 —"] + saved_names,
                    key="saved_file_select",
                )
                if selected_saved != "— 不选择 —":
                    prev_selected = st.session_state.get("_prev_saved", "")
                    if prev_selected != selected_saved:
                        try:
                            target = next(f for f in saved_files if f["name"] == selected_saved)
                            with open(target["path"], "rb") as f:
                                file_bytes = f.read()
                            df_new = data_loader.load_file(file_bytes, selected_saved)
                            _register_dataset(selected_saved, df_new, f"已保存: {selected_saved}")
                            st.session_state._prev_saved = selected_saved
                            st.success(f"✅ 已加载: {selected_saved} ({len(df_new):,} 行)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {e}")
                    # Delete from disk button
                    target = next(f for f in saved_files if f["name"] == selected_saved)
                    if st.button(f"🗑️ 从磁盘删除 {target['name']}", key=f"del_{target['path']}"):
                        st.session_state._disk_delete_target = target
                        st.rerun()

        # ---- Disk file delete confirmation ----
        if st.session_state.get("_disk_delete_target"):
            target = st.session_state._disk_delete_target
            st.warning(f"⚠️ 从磁盘永久删除 **{target['name']}**？")
            c1, c2 = st.columns(2)
            if c1.button("✅ 确认删除", key="confirm_disk_del"):
                if _delete_saved_file(target["path"]):
                    st.success(f"已删除文件: {target['name']}")
                    # Also remove from loaded datasets
                    _remove_dataset(target["name"])
                    st.session_state.pop("_disk_delete_target", None)
                    st.session_state.pop("_prev_saved", None)
                    st.rerun()
                else:
                    st.error("删除失败")
            if c2.button("❌ 取消", key="cancel_disk_del"):
                st.session_state.pop("_disk_delete_target", None)
                st.rerun()

        # ---- Auto-load default if nothing loaded ----
        if len(st.session_state._datasets) == 0 and not uploaded_file:
            default_path = os.path.join(os.path.dirname(__file__), "data", "online_retail.csv")
            if os.path.exists(default_path):
                if "_default_loaded" not in st.session_state:
                    try:
                        df = pd.read_csv(default_path, encoding="utf-8")
                        _register_dataset("online_retail.csv", df, "默认: online_retail.csv")
                        st.session_state._default_loaded = True
                        st.info("📦 已自动加载默认数据集")
                    except Exception:
                        st.warning("默认数据集不可用，请上传文件。")

        st.divider()

        # ---- Loaded datasets list ----
        st.subheader("📂 已加载数据集")
        datasets = st.session_state._datasets
        if datasets:
            active = st.session_state._active_dataset
            for ds_name in list(st.session_state._dataset_order):
                if ds_name not in datasets:
                    continue
                ds = datasets[ds_name]
                is_active = (ds_name == active)
                label = f"{'🔵 ' if is_active else '⚪ '}{ds_name}"
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(label, key=f"switch_{ds_name}", width="stretch",
                                 help=f"切换到此数据集 ({len(ds['df_raw']):,} 行)",
                                 type="primary" if is_active else "secondary"):
                        st.session_state._active_dataset = ds_name
                        st.rerun()
                with col2:
                    if st.button("✕", key=f"remove_{ds_name}", help=f"移除 {ds_name}"):
                        _remove_dataset(ds_name)
                        st.rerun()
                if is_active:
                    st.caption(f"   📌 {ds['data_source']} | {len(ds['df_raw']):,} 行 × {len(ds['df_raw'].columns)} 列")
        else:
            st.caption("暂无数据集，请上传或选择已保存的文件")

        # ---- Merge datasets ----
        prepped_for_merge = {}
        for ds_name, ds in st.session_state._datasets.items():
            if ds.get("preprocessed") and ds.get("df_clean") is not None:
                prepped_for_merge[ds_name] = ds
        if len(prepped_for_merge) >= 2:
            st.divider()
            st.subheader("🔀 合并数据集")
            merge_pick = st.multiselect(
                "选择要合并的数据集（2个以上）",
                options=list(prepped_for_merge.keys()),
                key="merge_pick",
            )
            if len(merge_pick) >= 2 and st.button("合并选中数据集", width="stretch",
                                                    key="merge_btn"):
                with st.spinner("合并中..."):
                    frames = []
                    merged_names = []
                    for n in merge_pick:
                        df_c = prepped_for_merge[n]["df_clean"].copy()
                        df_c["_来源数据集"] = n
                        frames.append(df_c)
                        merged_names.append(n)
                    merged_df = pd.concat(frames, ignore_index=True)
                    merged_df = preprocessor.preprocess_generic(merged_df)
                    merge_name = "合并 (" + " + ".join(merged_names) + ")"
                    _register_dataset(merge_name, merged_df, f"合并: {merge_name}")
                    # Mark as preprocessed to avoid re-processing already-processed data
                    _ds_set("df_clean", merged_df)
                    _ds_set("rfm_df", preprocessor.get_rfm_table(merged_df))
                    _ds_set("preprocessed", True)
                    st.success(f"✅ 已创建合并数据集: {merge_name} ({len(merged_df):,} 行)")
                    st.rerun()

        st.divider()

        # Module 2: Preprocessing (for active dataset)
        st.subheader("🔧 数据预处理")
        if _ds_get("df_raw") is not None:
            col_a, col_b = st.columns([3, 1])
            ds_name = st.session_state._active_dataset
            with col_a:
                if st.button("运行预处理", width="stretch", key=f"preproc_{ds_name}"):
                    with st.spinner("处理中..."):
                        try:
                            df = _ds_get("df_raw").copy()
                            df = preprocessor.preprocess_online_retail(df)
                            _ds_set("df_clean", df)
                            _ds_set("rfm_df", preprocessor.get_rfm_table(df))
                            _ds_set("preprocessed", True)
                        except Exception as e:
                            st.error(f"预处理出错: {e}")
                            try:
                                df = _ds_get("df_raw").copy()
                                df = preprocessor.preprocess_generic(df)
                                _ds_set("df_clean", df)
                                _ds_set("rfm_df", preprocessor.get_rfm_table(df))
                                _ds_set("preprocessed", True)
                                st.info("已使用通用预处理")
                            except Exception as e2:
                                st.error(f"通用预处理也失败: {e2}")
            with col_b:
                if _ds_get("preprocessed") and st.button("重置", width="stretch", key=f"reset_{ds_name}"):
                    _ds_set("preprocessed", False)
                    _ds_set("df_clean", None)
                    _ds_set("rfm_df", None)
                    st.rerun()

            if _ds_get("preprocessed"):
                df_c = _ds_get("df_clean")
                rfm = _ds_get("rfm_df")
                if df_c is not None:
                    st.metric("有效记录", f"{len(df_c):,}")
                    st.metric("总销售额", f"{df_c['TotalPrice'].sum():,.0f}")
                    if rfm is not None:
                        st.metric("客户/Segment", f"{len(rfm):,}")
        else:
            st.caption("请先加载数据")

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
        if _ds_get("df_raw") is None:
            st.info("👈 请上传数据文件或用默认数据集。")
        else:
            df = _ds_get("df_raw")
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

            if _ds_get("preprocessed"):
                st.divider()
                st.subheader("预处理后数据")
                st.dataframe(_ds_get("df_clean").head(20), width="stretch")
                st.subheader("RFM 客户分层表")
                st.dataframe(_ds_get("rfm_df").head(20), width="stretch")

    # ========== Tab: Data Analysis ==========
    with tabs[1]:
        # Collect preprocessed datasets for comparison
        prepped = {}
        for ds_name, ds in st.session_state._datasets.items():
            if ds.get("preprocessed") and ds.get("df_clean") is not None:
                prepped[ds_name] = ds

        if not prepped:
            st.info("请先在左侧对至少一个数据集运行数据预处理。")
        else:
            active = st.session_state._active_dataset
            dataset_names = list(prepped.keys())

            # Comparison multi-select
            compare_names = st.multiselect(
                "对比数据集（可多选叠加对比）",
                options=dataset_names,
                default=[active] if active in prepped else [dataset_names[0]] if dataset_names else [],
            )
            if not compare_names:
                compare_names = [dataset_names[0]] if dataset_names else []

            is_multi = len(compare_names) > 1

            analysis_type = st.selectbox("选择分析方法", [
                "📈 月度销售趋势", "⏰ 时段分析", "📦 产品销售 Top-N",
                "🗺️ 国家/地区分析", "👥 RFM 客户分层", "🔗 购物篮关联", "🔍 产品搜索",
            ])
            st.divider()

            def _run_comparison(analysis_fn, **kwargs):
                frames = []
                for ds_name in compare_names:
                    df_c = preprocessor.get_clean_transactions(prepped[ds_name]["df_clean"])
                    result = analysis_fn(df_c, **kwargs)
                    result["数据集"] = ds_name
                    frames.append(result)
                return pd.concat(frames, ignore_index=True)

            # Single-dataset fallback references
            df_clean = preprocessor.get_clean_transactions(
                prepped.get(compare_names[0])["df_clean"]) if compare_names else None
            rfm_df = prepped.get(compare_names[0])["rfm_df"] if compare_names else None

            if analysis_type == "📈 月度销售趋势":
                if is_multi:
                    trend = _run_comparison(analyzer.monthly_trend)
                    has_txn = "TransactionCount" in trend.columns
                    st.plotly_chart(visualizer.plot_multi_line_bar(
                        trend, x="YearMonth", y_bar="TotalRevenue",
                        y_line="TransactionCount" if has_txn else None,
                        title="月度销售趋势对比"), width="stretch")
                else:
                    trend = analyzer.monthly_trend(df_clean)
                    st.plotly_chart(visualizer.plot_monthly_trend(trend), width="stretch")
                st.dataframe(trend, width="stretch")

            elif analysis_type == "⏰ 时段分析":
                c1, c2 = st.columns(2)
                if is_multi:
                    hourly = _run_comparison(analyzer.hourly_trend)
                    c1.plotly_chart(visualizer.plot_multi_line(
                        hourly, x="Hour", y="TotalRevenue",
                        title="每小时销售额对比"), width="stretch")
                    weekday = _run_comparison(analyzer.weekday_trend)
                    c2.plotly_chart(visualizer.plot_multi_bar(
                        weekday, x="WeekdayName", y="TotalRevenue",
                        title="工作日销售额对比"), width="stretch")
                else:
                    c1.plotly_chart(visualizer.plot_hourly(
                        analyzer.hourly_trend(df_clean)), width="stretch")
                    c2.plotly_chart(visualizer.plot_weekday(
                        analyzer.weekday_trend(df_clean)), width="stretch")

            elif analysis_type == "📦 产品销售 Top-N":
                n = st.slider("显示前 N 个", 5, 50, 15)
                if is_multi:
                    top = _run_comparison(analyzer.top_n_analysis, n=n)
                    st.plotly_chart(visualizer.plot_multi_bar(
                        top, x=top.columns[0], y="TotalRevenue",
                        title=f"Top {n} 产品销售对比"), width="stretch")
                else:
                    top = analyzer.top_n_analysis(df_clean, n=n)
                    st.plotly_chart(visualizer.plot_top_products(top), width="stretch")
                st.dataframe(top, width="stretch")

            elif analysis_type == "🗺️ 国家/地区分析":
                if is_multi:
                    country = _run_comparison(analyzer.country_analysis)
                    st.plotly_chart(visualizer.plot_multi_bar(
                        country.head(30), x="Country", y="TotalRevenue",
                        title="国家/地区营收对比"), width="stretch")
                else:
                    country = analyzer.country_analysis(df_clean)
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(visualizer.plot_country_bar(country), width="stretch")
                    c2.plotly_chart(visualizer.plot_country_revenue(country), width="stretch")
                st.dataframe(country, width="stretch")

            elif analysis_type == "👥 RFM 客户分层":
                if rfm_df is not None:
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
                            st.plotly_chart(visualizer.plot_cluster_scatter(
                                rfm_df, clust["cluster_labels"]), width="stretch")

            elif analysis_type == "🔗 购物篮关联":
                ms = st.slider("最小支持度", 5, 100, 20)
                rules = analyzer.market_basket_analysis(df_clean, min_support=ms)
                if rules:
                    st.plotly_chart(visualizer.plot_basket_associations(rules), width="stretch")
                    st.dataframe(pd.DataFrame(rules), width="stretch")
                else:
                    st.warning("未找到关联规则，请降低最小支持度。")

            elif analysis_type == "🔍 产品搜索":
                kw = st.text_input("关键词", placeholder="产品名称、描述关键词...")
                if kw:
                    if is_multi:
                        frames = []
                        for ds_name in compare_names:
                            df_c = preprocessor.get_clean_transactions(prepped[ds_name]["df_clean"])
                            r = analyzer.search_products(df_c, kw)
                            r["数据集"] = ds_name
                            frames.append(r)
                        result = pd.concat(frames, ignore_index=True)
                    else:
                        result = analyzer.search_products(df_clean, kw)
                    if len(result) > 0:
                        st.plotly_chart(visualizer.auto_chart(result.head(15), "bar"), width="stretch")
                        st.dataframe(result, width="stretch")

    # ========== Tab: Intelligent Q&A ==========
    with tabs[2]:
        # Collect preprocessed datasets for multi-dataset Q&A
        prepped = {}
        for ds_name, ds in st.session_state._datasets.items():
            if ds.get("preprocessed") and ds.get("df_clean") is not None:
                prepped[ds_name] = ds

        if not prepped:
            st.info("请先在左侧对至少一个数据集运行数据预处理。")
        else:
            active = st.session_state._active_dataset
            dataset_names = list(prepped.keys())

            st.subheader("💬 智能问答")

            # Multi-dataset selector
            qa_datasets = st.multiselect(
                "选择要提问的数据集（可多选，合并分析）",
                options=dataset_names,
                default=[active] if active in prepped else [dataset_names[0]] if dataset_names else [],
            )
            if not qa_datasets:
                qa_datasets = [dataset_names[0]] if dataset_names else []

            has_llm, llm_error = qa_engine.get_llm_status()
            if has_llm:
                st.success("🤖 DeepSeek AI 已连接 — 畅聊无限制，数据分析/电商咨询/通用对话皆可")
            else:
                if "余额" in llm_error:
                    st.warning(f"💰 {llm_error}")
                elif "DEEPSEEK_API_KEY" in llm_error:
                    st.info("💡 AI 未连接，使用规则引擎。设置 `DEEPSEEK_API_KEY` 可解锁无限问答能力")
                else:
                    st.warning(f"⚠️ DeepSeek 连接异常: {llm_error}")

            with st.expander("💡 示例问题"):
                for q in qa_engine.get_example_questions():
                    if st.button(q, key=f"ex_{q}"):
                        st.session_state.current_question = q

            default_q = st.session_state.get("current_question", "")
            query = st.chat_input("输入你的数据问题...") or default_q

            if query:
                # Merge data if multiple datasets selected
                if len(qa_datasets) > 1:
                    frames = [preprocessor.get_clean_transactions(prepped[n]["df_clean"]) for n in qa_datasets]
                    df_clean = pd.concat(frames, ignore_index=True)
                    # Merge RFM tables too
                    rfm_frames = [prepped[n]["rfm_df"] for n in qa_datasets]
                    rfm = pd.concat(rfm_frames, ignore_index=True) if rfm_frames else None
                    multi_context = "\n".join(
                        f"数据集 '{n}': {len(prepped[n]['df_clean'])} 行记录"
                        for n in qa_datasets
                    )
                else:
                    df_clean = preprocessor.get_clean_transactions(prepped[qa_datasets[0]]["df_clean"])
                    rfm = prepped[qa_datasets[0]]["rfm_df"]
                    multi_context = ""

                with st.spinner("分析中..." + (" (DeepSeek AI 思考中...)" if has_llm else "")):
                    result = qa_engine.parse_query(query, df_clean, rfm, extra_info=multi_context)

                if not has_llm and result.get("source") == "fallback":
                    st.info("💡 该问题超出了规则引擎范围。接入 DeepSeek AI 后可回答任意问题，设置 `DEEPSEEK_API_KEY` 即可。")

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
            admin_tab = st.radio("管理选项", ["👥 用户管理", "📋 问答日志", "📁 上传记录"], horizontal=True)

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

            elif admin_tab == "📁 上传记录":
                uploads = auth.get_upload_logs()
                st.metric("总上传次数", len(uploads))
                if uploads:
                    df_uploads = pd.DataFrame(uploads)
                    df_uploads.columns = ["ID", "用户名", "文件名", "行数", "列数", "文件大小(KB)", "上传时间"]
                    st.dataframe(df_uploads[["用户名", "文件名", "行数", "列数", "文件大小(KB)", "上传时间"]].head(50), width="stretch")

                    usernames_u = list(set(u["username"] for u in uploads))
                    filter_user_u = st.selectbox("按用户筛选", ["全部"] + usernames_u, key="filter_upload")
                    if filter_user_u != "全部":
                        filtered_u = [u for u in uploads if u["username"] == filter_user_u]
                        st.dataframe(pd.DataFrame(filtered_u)[["文件名", "行数", "列数", "文件大小(KB)", "上传时间"]].head(20), width="stretch")
                else:
                    st.info("暂无上传记录")


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
