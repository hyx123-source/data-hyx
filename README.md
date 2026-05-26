# 电商交易智能分析问答系统

## AI-Data-QA: E-commerce Smart Analytics Q&A System

高级Python程序设计期末课程项目
学号：2025201770  姓名：黄奕翔

---

## 项目简介

基于 **Streamlit** 的智能数据问答 Web 应用，以 **Online Retail**（英国在线零售）数据集为核心，实现了完整的「数据读取 → 数据预处理 → 数据分析 → 数据可视化 → 智能问答」五大模块。

核心创新：**DeepSeek LLM 驱动的自然语言问答**，用户用中文自由提问（不限范围），系统自动执行数据查询并生成交互式可视化图表。同时实现了完整的用户认证系统（注册/登录/管理员面板）和**通用数据预处理**（支持任意 CSV/Excel 上传）。

---

## 快速开始

### 1. 环境要求

- Python 3.9+
- pip

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动应用

```bash
cd final_project
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

### 4. 使用说明

- **登录系统**：默认管理员 `admin` / `admin123`，也可自行注册
- **默认数据集**：应用启动后自动加载 `data/online_retail.csv`，无文件时自动生成模拟数据
- **上传自定义数据**：左侧边栏上传 CSV/Excel/JSON 文件，系统自动检测列名并适配处理
- **运行预处理**：点击「运行预处理」按钮，自动识别数据集类型选择最佳处理管道
- **浏览分析**：在「数据分析」Tab 中选择分析方法（Top-N / 趋势 / RFM / 关联 / 聚类 / 搜索）
- **智能问答**：在「智能问答」Tab 中输入任意问题（数据分析 / 电商咨询 / 通用对话均可）
- **管理员面板**：管理员可查看所有用户和问答日志

---

## 项目结构

```
final_project/
├── app.py                  # Streamlit 主程序（页面路由 + 状态管理）
├── modules/
│   ├── __init__.py
│   ├── auth.py             # 用户认证模块（SQLite + SHA-256 加盐）
│   ├── data_loader.py      # 模块1：多格式数据读取
│   ├── preprocessor.py     # 模块2：数据预处理（在线零售 + 通用管道）
│   ├── analyzer.py         # 模块3：数据分析方法
│   ├── visualizer.py       # 模块4：交互式数据可视化
│   └── qa_engine.py        # 模块5：智能问答引擎（LLM 优先 + 规则兜底）
├── data/
│   └── online_retail.csv   # Online Retail 数据集（54万条）
├── generate_data.py        # 模拟数据生成脚本
├── test_auth.py            # 认证模块测试
├── test_pipeline.py        # 数据处理管道测试
├── requirements.txt        # Python 依赖
├── README.md               # 运行说明
├── AI_usage_report.md      # AI 使用情况说明书
├── PPT_outline.md          # 演示 PPT 大纲
└── demo_script.md          # 课堂演示脚本
```

---

## 五大模块功能

| 模块 | 功能 | 核心技术 | 代码来源 |
|------|------|----------|----------|
| 数据读取 | 多格式支持（CSV/Excel/JSON）、编码检测、数据概览 | Pandas, chardet | 学生+AI |
| 数据预处理 | 在线零售管道 + 通用管道、列名自动检测、RFM 计算 | Pandas, NumPy | 学生+AI |
| 数据分析 | RFM分层、购物篮关联、时间序列、KMeans聚类、Top-N | Scikit-learn, mlxtend | 学生+AI |
| 数据可视化 | 交互式图表（折线/柱状/饼图/散点/地图） | Plotly | AI |
| 智能问答 | DeepSeek LLM 优先 + 20+ 规则模板，不限范围对话 | OpenAI API + 正则引擎 | 学生+AI |

---

## 创新功能

1. **DeepSeek LLM 智能问答（不限范围）** — 数据分析、电商咨询、概念解释、通用对话均可
2. **通用数据预处理** — 列名自动检测映射，任意 CSV/Excel/JSON 上传不报错
3. **完整用户认证系统** — 注册/登录/密码修改/管理员面板，SHA-256 加盐密码
4. **智能图表推荐** — 根据问题类型自动选择最佳可视化
5. **RFM + KMeans 双重客户分析** — 业务规则分层 + 数据驱动聚类
6. **LLM + 规则双层兜底** — 在线智能，离线也能用
7. **购物篮关联分析** — 发现"经常一起购买"的商品组合

---

## 数据集说明

- **名称**：Online Retail (UCI Machine Learning Repository)
- **来源**：https://archive.ics.uci.edu/dataset/352/online%2Bretail
- **描述**：英国在线零售商 2010-2011 年交易记录
- **规模**：541,909 条，8 个原始字段
- **字段**：InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## 评分对标

| 评分项 | 占比 | 项目实现 |
|--------|------|----------|
| 代码质量与可执行性 | 30% | 6个独立模块，完整可运行，支持自定义数据测试 |
| 汇报表达与演示 | 30% | 3分钟PPT + 现场Demo，15页大纲 + 完整演示脚本 |
| 数据与分析设计 | 20% | 完整数据处理流程 + 多种分析方法 + 智能问答 |
| AI工具运用 | 20% | AI_usage_report.md 详细记录使用过程和反思 |
| 创新加分 | +10% | 通用预处理 + LLM不限范围问答 + 双重客户分析 |

---

## 已知限制

- LLM 依赖 DeepSeek API 网络连接，离线时自动回退到规则引擎
- 购物篮分析使用简化的配对计数方法，极大数据集下可能需要采样
- 通用预处理管道对高度非结构化的数据效果有限
