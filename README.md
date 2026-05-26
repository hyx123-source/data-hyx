# 电商交易智能分析问答系统

## AI-Data-QA: E-commerce Smart Analytics Q&A System

高级Python程序设计期末课程项目
学号：2025201770  姓名：黄奕翔

---

## 项目简介

基于 **Streamlit** 的智能数据问答 Web 应用，以 **Online Retail**（英国在线零售）数据集为核心，实现了完整的「数据读取 → 预处理 → 分析 → 可视化 → 智能问答」五大模块。

核心创新：**DeepSeek LLM 驱动的自然语言问答**，用户用中文自由提问，系统自动执行数据查询并生成交互式可视化图表。同时实现了完整的用户认证系统（注册/登录/管理员面板）。

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

**方式一：模块化版本**
```bash
cd final_project
streamlit run app.py
```

**方式二：单文件整合版本**
```bash
streamlit run 2025201770-黄奕翔-期末作业.py
```

浏览器会自动打开 http://localhost:8501

### 4. 使用说明

- **登录系统**：默认管理员 `admin` / `admin123`，也可自行注册
- **默认数据集**：应用启动后自动加载 `data/online_retail.csv`，无文件时自动生成模拟数据
- **上传自定义数据**：左侧边栏上传 CSV/Excel/JSON 文件
- **运行预处理**：点击「运行预处理」按钮
- **浏览分析**：在「数据分析」Tab 中选择分析方法
- **智能问答**：在「智能问答」Tab 中输入自然语言问题
- **管理员面板**：管理员可查看所有用户和问答日志

---

## 项目结构

```
final_project/
├── app.py                  # Streamlit 主程序（模块化版本）
├── modules/
│   ├── __init__.py
│   ├── auth.py             # 用户认证模块（SQLite + SHA-256）
│   ├── data_loader.py      # 模块1：数据文件读取
│   ├── preprocessor.py     # 模块2：数据预处理
│   ├── analyzer.py         # 模块3：数据分析方法
│   ├── visualizer.py       # 模块4：数据可视化
│   └── qa_engine.py        # 模块5：智能问答引擎（DeepSeek LLM + 规则引擎）
├── data/
│   └── online_retail.csv   # Online Retail 数据集
├── requirements.txt        # Python 依赖
├── README.md               # 本文件
├── AI_usage_report.md      # AI 使用情况说明书
├── PPT_outline.md          # 演示 PPT 大纲
└── demo_script.md          # 课堂演示脚本
```

---

## 五大模块功能

| 模块 | 功能 | 核心技术 |
|------|------|----------|
| 数据读取 | 多格式支持（CSV/Excel/JSON）、编码检测、数据概览 | Pandas, chardet |
| 数据预处理 | 缺失值处理、特征工程、RFM 计算 | Pandas, NumPy |
| 数据分析 | RFM分层、购物篮关联、时间序列、聚类、Top-N | Scikit-learn, mlxtend |
| 数据可视化 | 交互式图表（折线/柱状/饼图/散点/地图/3D散点） | Plotly |
| 智能问答 | DeepSeek LLM 优先 + 20+ 规则模板后备，自动图表推荐 | OpenAI API + 正则引擎 |

---

## 创新功能

1. **DeepSeek LLM 智能问答** — 优先使用大模型理解任意自然语言问题，规则引擎保障离线可用
2. **完整用户认证系统** — 注册/登录/管理员面板，SHA-256 加盐密码，SQLite 持久化
3. **智能图表推荐** — 根据问题类型自动选择最佳可视化
4. **RFM 客户分层 + KMeans 聚类** — 客户价值精细化分析
5. **购物篮关联分析** — 发现"经常一起购买"的商品组合
6. **模块化 + 单文件双版本** — 既可作为项目工程，也可作为单文件直接运行

---

## 数据集说明

- **名称**：Online Retail (UCI Machine Learning Repository)
- **来源**：https://archive.ics.uci.edu/dataset/352/online%2Bretail
- **描述**：英国在线零售商 2010-2011 年交易记录
- **规模**：541,909 条，8 个原始字段
- **字段**：InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

---

## 已知限制

- LLM 依赖 DeepSeek API 网络连接，离线时自动回退到规则引擎
- 购物篮分析使用简化的配对计数方法，极大数据集下可能需要采样
