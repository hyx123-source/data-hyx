"""
Module 4: Data Visualization
Plotly-based interactive charts for all analysis types.
Source: Student + AI collaboration.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


# ---- Color palette ----
PALETTE = px.colors.qualitative.Set2


def plot_top_products(top_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for top-N products by revenue."""
    col = top_df.columns[0]
    fig = px.bar(
        top_df.head(15),
        x="TotalRevenue", y=col, orientation="h",
        color="TotalRevenue",
        color_continuous_scale="Blues",
        title="Top Products by Revenue",
        text_auto=".2s",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
    return fig


def plot_monthly_trend(trend_df: pd.DataFrame) -> go.Figure:
    """Line + bar combo for monthly sales trend."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=trend_df["YearMonth"], y=trend_df["TotalRevenue"],
               name="Revenue", marker_color=PALETTE[0], opacity=0.8),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=trend_df["YearMonth"], y=trend_df["TransactionCount"],
                   name="Transactions", mode="lines+markers",
                   line=dict(color=PALETTE[1], width=2)),
        secondary_y=True,
    )
    fig.update_layout(
        title="Monthly Sales Trend",
        height=450,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Revenue (£)", secondary_y=False)
    fig.update_yaxes(title_text="Transactions", secondary_y=True)
    return fig


def plot_country_revenue(country_df: pd.DataFrame) -> go.Figure:
    """Choropleth map for country revenue (excl. UK for better scale)."""
    non_uk = country_df[country_df["Country"] != "United Kingdom"]
    fig = px.choropleth(
        non_uk,
        locations="Country",
        locationmode="country names",
        color="TotalRevenue",
        title="Revenue by Country (excl. UK)",
        color_continuous_scale="Blues",
        height=500,
    )
    return fig


def plot_country_bar(country_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart for top countries."""
    top = country_df.head(top_n)
    fig = px.bar(
        top, x="Country", y="TotalRevenue",
        color="TotalRevenue",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Countries by Revenue",
        text_auto=".2s",
        height=450,
    )
    return fig


def plot_rfm_distribution(rfm_df: pd.DataFrame) -> go.Figure:
    """Pie chart for RFM segments + bar chart for stats."""
    seg_counts = pd.DataFrame(
        rfm_df["Segment"].value_counts().reset_index()
    )
    seg_counts.columns = ["Segment", "Count"]

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Customer Segmentation (RFM)", "Avg Spend by Segment"),
    )

    fig.add_trace(
        go.Pie(labels=seg_counts["Segment"], values=seg_counts["Count"],
               marker=dict(colors=PALETTE[:len(seg_counts)])),
        row=1, col=1,
    )

    seg_avg = rfm_df.groupby("Segment")["Monetary"].mean().reset_index()
    fig.add_trace(
        go.Bar(x=seg_avg["Segment"], y=seg_avg["Monetary"],
               marker_color=PALETTE[:len(seg_avg)],
               text=seg_avg["Monetary"].round(2), textposition="outside"),
        row=1, col=2,
    )

    fig.update_layout(height=450, showlegend=False)
    return fig


def plot_hourly(hourly_df: pd.DataFrame) -> go.Figure:
    """Bar chart for hourly sales pattern."""
    fig = px.bar(
        hourly_df, x="Hour", y="TotalRevenue",
        title="Sales by Hour of Day",
        color="TotalRevenue",
        color_continuous_scale="Blues",
        height=400,
    )
    return fig


def plot_weekday(weekday_df: pd.DataFrame) -> go.Figure:
    """Bar chart for sales by day of week."""
    fig = px.bar(
        weekday_df, x="WeekdayName", y="TotalRevenue",
        title="Sales by Day of Week",
        color="TotalRevenue",
        color_continuous_scale="Blues",
        height=400,
    )
    return fig


def plot_basket_associations(assoc_rules: list, top_n: int = 20) -> go.Figure:
    """Network/bar chart showing top product associations."""
    if not assoc_rules:
        fig = go.Figure()
        fig.update_layout(title="No association rules found (try lower min_support)")
        return fig

    df_rules = pd.DataFrame(assoc_rules[:top_n])
    df_rules["label"] = df_rules["item_a"] + " + " + df_rules["item_b"]

    fig = px.bar(
        df_rules, x="count", y="label", orientation="h",
        title="Top Product Associations (Market Basket)",
        color="count",
        color_continuous_scale="Blues",
        height=500,
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig


def plot_cluster_scatter(rfm_df: pd.DataFrame, cluster_labels: list) -> go.Figure:
    """3D scatter of customer clusters."""
    plot_df = rfm_df.copy()
    plot_df["Cluster"] = cluster_labels
    plot_df["Cluster"] = plot_df["Cluster"].astype(str)

    fig = px.scatter_3d(
        plot_df,
        x="Recency", y="Frequency", z="Monetary",
        color="Cluster",
        title="Customer Clusters (KMeans on RFM)",
        opacity=0.6,
        color_discrete_sequence=PALETTE,
        height=550,
    )
    fig.update_traces(marker=dict(size=3))
    return fig


def plot_metric_card(value: str, label: str, delta: str = None) -> go.Figure:
    """Metric card for dashboard KPIs."""
    fig = go.Figure(go.Indicator(
        mode="number+delta" if delta else "number",
        value=float(value) if isinstance(value, (int, float)) else 0,
        title={"text": label},
        delta={"reference": float(delta)} if delta else None,
    ))
    fig.update_layout(height=200)
    return fig


def plot_histogram(df: pd.DataFrame, col: str, title: str = None) -> go.Figure:
    """Distribution histogram for a numeric column."""
    fig = px.histogram(
        df, x=col, nbins=50,
        title=title or f"Distribution of {col}",
        marginal="box",
        height=400,
    )
    return fig


def plot_empty(message: str = "No data available") -> go.Figure:
    """Return an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, text=message, showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(height=300)
    return fig


def auto_chart(df: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
    """Auto-dispatch chart based on type string.

    Supported: 'bar', 'line', 'pie', 'scatter', 'histogram', 'map'
    Falls back to a table view if type unknown.
    """
    if df is None or df.empty:
        return plot_empty("Query returned no results")

    if chart_type == "bar" and len(df.columns) >= 2:
        return px.bar(df, x=df.columns[0], y=df.columns[1],
                      title=kwargs.get("title", ""), height=400)
    elif chart_type == "line" and len(df.columns) >= 2:
        return px.line(df, x=df.columns[0], y=df.columns[1],
                       title=kwargs.get("title", ""), height=400)
    elif chart_type == "pie" and len(df.columns) >= 2:
        return px.pie(df, names=df.columns[0], values=df.columns[1],
                      title=kwargs.get("title", ""), height=400)
    elif chart_type == "scatter" and len(df.columns) >= 2:
        return px.scatter(df, x=df.columns[0], y=df.columns[1],
                          title=kwargs.get("title", ""), height=400)
    elif chart_type == "map":
        return plot_country_bar(df, top_n=min(len(df), 20))
    else:
        # Default: bar of first two columns
        return px.bar(df.head(20), x=df.columns[0], y=df.columns[1] if len(df.columns) > 1 else None,
                      title=kwargs.get("title", "Results"), height=400)
