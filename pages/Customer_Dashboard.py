# pages/2_Customer_Dashboard.py
import streamlit as st
import pandas as pd
import duckdb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from genai.groq_client import generate_insight
from utils.connection import get_connection

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Customer Dashboard",
    layout="wide"
)

st.title("Customer Dashboard")

# ==================================================
# CONNECTION
# ==================================================

con = get_connection()

# ==================================================
# CACHE DATA
# ==================================================

@st.cache_data(ttl=600)
def load_customer_orders():
    """
    Lấy đơn hàng delivered + thông tin khách hàng.
    """
    df = con.execute("""
        SELECT
            sc.customer_unique_id AS customer_key,
            sc.customer_city,
            sc.customer_state,
            fo.order_id,
            fo.customer_id,
            fo.order_month_date,
            fo.total_order_value AS order_value,
            fo.order_status
        FROM olist_dw.gold_main_gold.fact_orders fo
        JOIN olist_dw.silver.customers sc
            ON fo.customer_id = sc.customer_id
        WHERE fo.order_status = 'delivered'
          AND fo.order_month_date IS NOT NULL
    """).df()

    df["order_month_date"] = pd.to_datetime(df["order_month_date"])
    return df

@st.cache_data(ttl=600)
def load_dim_customer():
    df = con.execute("""
        SELECT
            customer_unique_id AS customer_key,
            customer_city,
            customer_state
        FROM olist_dw.silver.customers
    """).df()
    return df

@st.cache_data(ttl=600)
def build_customer_summary(df_orders):
    g = (
        df_orders.groupby("customer_key")
        .agg(
            n_orders=("order_id", "nunique"),
            total_value=("order_value", "sum"),
            first_month=("order_month_date", "min"),
            last_month=("order_month_date", "max"),
            state=("customer_state", "first"),
            city=("customer_city", "first"),
        )
        .reset_index()
    )

    g["is_repeat"] = g["n_orders"] > 1
    g["avg_order_value"] = (
        g["total_value"] /
        g["n_orders"].replace(0, np.nan)
    )
    return g

# ==================================================
# LOAD DATA
# ==================================================

df_orders = load_customer_orders()
dim_cust = load_dim_customer()
cust = build_customer_summary(df_orders)

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Customer Retention",
    "Repeat Customers",
    "Customer Lifetime Value",
    "Geographic Distribution"
])

# ==================================================
# TAB 1: CUSTOMER RETENTION
# ==================================================

with tab1:
    # KPI CALCULATION
    total_customers = len(cust)
    repeat_rate = cust["is_repeat"].mean() * 100
    avg_orders = cust["n_orders"].mean()
    active_months = df_orders["order_month_date"].nunique()
    
    # KPI DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng số khách hàng", f"{total_customers:,}")
    with col2:
        st.metric("Tỷ lệ khách quay lại", f"{repeat_rate:.1f}%")
    with col3:
        st.metric("Số đơn TB / khách", f"{avg_orders:.2f}")
    with col4:
        st.metric("Số tháng có dữ liệu", f"{active_months}")
    
    st.divider()
    
    # New vs Returning theo tháng
    orders = df_orders.merge(cust[["customer_key", "first_month"]], on="customer_key")
    orders["visit_type"] = np.where(
        orders["order_month_date"] == orders["first_month"], "Khách mới", "Khách quay lại"
    )
    monthly = (orders.groupby(["order_month_date", "visit_type"])["order_id"]
               .nunique().reset_index(name="orders"))
    
    st.subheader("Khách hàng mới vs. Quay lại theo tháng")
    fig = px.bar(
        monthly,
        x="order_month_date",
        y="orders",
        color="visit_type",
        color_discrete_map={"Khách mới": "#3498db", "Khách quay lại": "#2ecc71"},
        barmode="stack"
    )
    fig.update_layout(
        height=400,
        xaxis_title="Tháng",
        yaxis_title="Số đơn hàng",
        legend_title=""
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Repeat customer + Order Frequency
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tỷ lệ khách hàng quay lại mua")
        
        repeat_df = pd.DataFrame({
            "Loại khách hàng": ["Mua 1 lần", "Mua lặp lại"],
            "Số lượng": [
                (~cust["is_repeat"]).sum(),
                cust["is_repeat"].sum()
            ]
        })
        
        fig = px.pie(
            repeat_df,
            names="Loại khách hàng",
            values="Số lượng",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Khách mua lặp lại thể hiện khả năng giữ chân khách hàng của doanh nghiệp.")
    
    with col2:
        st.subheader("Phân phối số đơn hàng mỗi khách")
        
        order_freq = cust["n_orders"]
        fig = px.histogram(
            order_freq,
            nbins=20,
            color_discrete_sequence=["#9b59b6"]
        )
        fig.update_layout(
            height=360,
            xaxis_title="Số đơn hàng",
            yaxis_title="Số khách hàng"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Biểu đồ cho thấy phần lớn khách hàng mua một lần hay nhiều lần.")
    
    # Cohort retention heatmap
    st.subheader("Cohort Retention theo tháng (%)")
    st.caption("Mỗi hàng là một nhóm khách hàng mua lần đầu trong cùng tháng (cohort). Cột thể hiện % khách trong cohort đó còn quay lại mua sau N tháng.")
    
    orders["cohort_month"] = orders["first_month"]
    orders["period_number"] = (
        (orders["order_month_date"].dt.year - orders["cohort_month"].dt.year) * 12 +
        (orders["order_month_date"].dt.month - orders["cohort_month"].dt.month)
    )
    cohort_counts = (orders.groupby(["cohort_month", "period_number"])["customer_key"]
                     .nunique().reset_index())
    cohort_pivot = cohort_counts.pivot(index="cohort_month", columns="period_number",
                                        values="customer_key")
    
    if 0 in cohort_pivot.columns:
        cohort_size = cohort_pivot[0]
        retention = cohort_pivot.divide(cohort_size, axis=0) * 100
        retention = retention.iloc[:, :12]
        retention.index = retention.index.strftime("%Y-%m")
        
        fig = go.Figure(data=go.Heatmap(
            z=retention.values,
            x=[f"Tháng {c}" for c in retention.columns],
            y=retention.index,
            colorscale="Blues",
            zmin=0,
            zmax=100,
            text=retention.round(1).values,
            texttemplate="%{text}%",
            hovertemplate="Cohort %{y} | %{x}: %{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            height=450,
            xaxis_title="Số tháng kể từ lần mua đầu",
            yaxis_title="Cohort (tháng mua đầu tiên)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Không đủ dữ liệu để dựng cohort retention.")

# ==================================================
# TAB 2: REPEAT CUSTOMERS
# ==================================================

with tab2:
    # KPI CALCULATION
    repeat_df = cust[cust["is_repeat"]]
    n_repeat = len(repeat_df)
    repeat_rate = n_repeat / len(cust) * 100
    avg_orders_repeat = repeat_df["n_orders"].mean() if n_repeat else 0
    revenue_share = (repeat_df["total_value"].sum() / cust["total_value"].sum() * 100
                      if cust["total_value"].sum() > 0 else 0)
    
    # KPI DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Khách mua lặp lại", f"{n_repeat:,}")
    with col2:
        st.metric("Tỷ lệ khách mua lặp lại", f"{repeat_rate:.1f}%")
    with col3:
        st.metric("Số đơn TB (khách lặp lại)", f"{avg_orders_repeat:.2f}")
    with col4:
        st.metric("% Doanh thu từ khách lặp lại", f"{revenue_share:.1f}%")
    st.caption("Số đơn tính trên toàn bộ trạng thái. Doanh thu chỉ tính đơn đã delivered.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố số đơn hàng / khách")
        
        def bucket(n):
            if n == 1:
                return "1 đơn"
            elif n == 2:
                return "2 đơn"
            elif n == 3:
                return "3 đơn"
            else:
                return "4+ đơn"
        
        cust_b = cust.copy()
        cust_b["order_bucket"] = cust_b["n_orders"].apply(bucket)
        order_b = ["1 đơn", "2 đơn", "3 đơn", "4+ đơn"]
        bucket_df = (cust_b.groupby("order_bucket")["customer_key"].count()
                     .reindex(order_b).reset_index(name="customers"))
        
        fig = px.bar(
            bucket_df,
            x="order_bucket",
            y="customers",
            text="customers",
            color="order_bucket",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False,
            height=380,
            xaxis_title="Số đơn hàng",
            yaxis_title="Số khách hàng"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Đóng góp doanh thu: 1 lần vs. Lặp lại")
        rev_df = cust.copy()
        rev_df["group"] = np.where(rev_df["is_repeat"], "Khách lặp lại", "Khách mua 1 lần")
        rev_g = rev_df.groupby("group")["total_value"].sum().reset_index()
        
        fig = px.pie(
            rev_g,
            names="group",
            values="total_value",
            hole=0.4,
            color="group",
            color_discrete_map={"Khách lặp lại": "#2ecc71", "Khách mua 1 lần": "#95a5a6"}
        )
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 20 khách hàng mua lặp lại nhiều nhất")
    top_repeat = (repeat_df.sort_values(["n_orders", "total_value"], ascending=False)
                  .head(20)[["customer_key", "state", "city", "n_orders",
                             "total_value", "avg_order_value"]])
    top_repeat = top_repeat.rename(columns={
        "customer_key": "Customer ID",
        "state": "State",
        "city": "City",
        "n_orders": "Số đơn",
        "total_value": "Tổng chi tiêu (R$)",
        "avg_order_value": "TB/đơn (R$)"
    })
    st.dataframe(top_repeat, use_container_width=True, hide_index=True)

# ==================================================
# TAB 3: CUSTOMER LIFETIME VALUE
# ==================================================

with tab3:
    # KPI CALCULATION
    avg_clv = cust["total_value"].mean()
    median_clv = cust["total_value"].median()
    
    cust_sorted = cust.sort_values("total_value", ascending=False).reset_index(drop=True)
    total_rev = cust_sorted["total_value"].sum()
    top10_n = max(int(len(cust_sorted) * 0.10), 1)
    top10_share = (cust_sorted.iloc[:top10_n]["total_value"].sum() / total_rev * 100
                   if total_rev > 0 else 0)
    
    # KPI DISPLAY
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CLV trung bình", f"R$ {avg_clv:,.0f}")
    with col2:
        st.metric("CLV trung vị", f"R$ {median_clv:,.0f}")
    with col3:
        st.metric("Top 10% KH đóng góp doanh thu", f"{top10_share:.1f}%")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân phối CLV")
        p95 = cust["total_value"].quantile(0.95)
        fig = px.histogram(
            cust[cust["total_value"] <= p95],
            x="total_value",
            nbins=40,
            color_discrete_sequence=["#3498db"]
        )
        fig.update_layout(
            height=380,
            xaxis_title="CLV (R$)",
            yaxis_title="Số khách hàng"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Đã loại 5% khách có CLV cao nhất (outlier) để biểu đồ dễ đọc hơn.")
    
    with col2:
        st.subheader("Đường cong Pareto: % Khách hàng vs. % Doanh thu")
        cust_sorted["cum_revenue_pct"] = cust_sorted["total_value"].cumsum() / total_rev * 100
        cust_sorted["cum_customer_pct"] = (cust_sorted.index + 1) / len(cust_sorted) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cust_sorted["cum_customer_pct"],
            y=cust_sorted["cum_revenue_pct"],
            mode="lines",
            name="Thực tế",
            line=dict(color="#e74c3c", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Phân bố đều",
            line=dict(color="gray", dash="dash")
        ))
        fig.update_layout(
            height=380,
            xaxis_title="% Khách hàng (tích lũy)",
            yaxis_title="% Doanh thu (tích lũy)",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Phân khúc khách hàng theo CLV")
    try:
        cust["clv_segment"] = pd.qcut(cust["total_value"], q=4,
                                       labels=["Q1 - Thấp", "Q2 - TB thấp", "Q3 - TB cao", "Q4 - Cao"])
    except ValueError:
        cust["clv_segment"] = pd.qcut(cust["total_value"], q=4, duplicates="drop")
    
    seg_df = cust.groupby("clv_segment").agg(
        customers=("customer_key", "count"),
        avg_clv=("total_value", "mean"),
        total_revenue=("total_value", "sum"),
    ).reset_index()
    
    fig = px.bar(
        seg_df,
        x="clv_segment",
        y="total_revenue",
        text="customers",
        color="clv_segment",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"total_revenue": "Tổng doanh thu (R$)", "clv_segment": "Phân khúc"}
    )
    fig.update_traces(texttemplate="%{text} khách", textposition="outside")
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 15 khách hàng theo CLV")
    top_clv = cust_sorted.head(15)[["customer_key", "state", "city", "n_orders", "total_value"]]
    top_clv = top_clv.rename(columns={
        "customer_key": "Customer ID",
        "state": "State",
        "city": "City",
        "n_orders": "Số đơn",
        "total_value": "CLV (R$)"
    })
    st.dataframe(top_clv, use_container_width=True, hide_index=True)

# ==================================================
# TAB 4: GEOGRAPHIC DISTRIBUTION
# ==================================================

with tab4:
    # KPI CALCULATION
    state_summary = cust.groupby("state").agg(
        customers=("customer_key", "count"),
        total_orders=("n_orders", "sum"),
        total_revenue=("total_value", "sum"),
        avg_clv=("total_value", "mean"),
    ).reset_index().sort_values("total_revenue", ascending=False)
    
    n_states = state_summary["state"].nunique()
    top_state_customers = state_summary.iloc[0]["state"] if len(state_summary) else "-"
    top_state_revenue = state_summary.sort_values("total_revenue", ascending=False).iloc[0]["state"] if len(state_summary) else "-"
    
    # KPI DISPLAY
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Số bang/vùng có khách hàng", f"{n_states}")
    with col2:
        st.metric("Bang dẫn đầu về doanh thu", f"{top_state_revenue}")
    with col3:
        st.metric("Tổng doanh thu", f"R$ {state_summary['total_revenue'].sum():,.0f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 bang theo số lượng khách hàng")
        top_by_cust = state_summary.sort_values("customers", ascending=False).head(15)
        fig = px.bar(
            top_by_cust,
            x="state",
            y="customers",
            text="customers",
            color="customers",
            color_continuous_scale="Blues"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=400,
            xaxis_title="State",
            yaxis_title="Số khách hàng",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 15 bang theo doanh thu")
        top_by_rev = state_summary.head(15)
        fig = px.bar(
            top_by_rev,
            x="state",
            y="total_revenue",
            text="total_revenue",
            color="total_revenue",
            color_continuous_scale="Greens"
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig.update_layout(
            height=400,
            xaxis_title="State",
            yaxis_title="Doanh thu (R$)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Phân bố khách hàng theo Bang → Thành phố (Top 50 thành phố)")
    city_summary = cust.groupby(["state", "city"])["customer_key"].count().reset_index(name="customers")
    top_cities = city_summary.sort_values("customers", ascending=False).head(50)
    fig = px.treemap(
        top_cities,
        path=["state", "city"],
        values="customers",
        color="customers",
        color_continuous_scale="Blues"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Bảng tổng hợp theo Bang")
    table = state_summary.rename(columns={
        "state": "State",
        "customers": "Số khách hàng",
        "total_orders": "Tổng đơn hàng",
        "total_revenue": "Tổng doanh thu (R$)",
        "avg_clv": "CLV TB (R$)"
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

# ==================================================
# AI INSIGHT (DÙNG CHUNG CHO CẢ 4 TAB)
# ==================================================

st.divider()
st.subheader("AI Insight")

# Chọn context dựa trên tab đang active
tab_context = {
    "Customer Retention": "Phân tích tỷ lệ giữ chân khách hàng (Customer Retention) bao gồm: tổng số khách hàng, tỷ lệ khách quay lại, số đơn trung bình mỗi khách, và xu hướng khách mới vs quay lại theo tháng",
    "Repeat Customers": "Phân tích nhóm khách hàng mua lặp lại (Repeat Customers) bao gồm: số lượng và tỷ lệ khách mua lặp lại, số đơn trung bình, đóng góp doanh thu, và top khách hàng mua nhiều nhất",
    "Customer Lifetime Value": "Phân tích Customer Lifetime Value (CLV) bao gồm: CLV trung bình và trung vị, đóng góp của top 10% khách hàng, phân phối CLV, và phân khúc khách hàng theo CLV",
    "Geographic Distribution": "Phân tích phân bố địa lý của khách hàng (Geographic Distribution) bao gồm: số bang có khách hàng, bang dẫn đầu về doanh thu, top bang theo số lượng khách hàng và doanh thu, và phân bố khách hàng theo thành phố"
}

# Dropdown để chọn tab cho AI insight
ai_tab = st.selectbox(
    "Chọn phân tích cho AI:",
    ["Customer Retention", "Repeat Customers", "Customer Lifetime Value", "Geographic Distribution"],
    index=0
)

if st.button("Phân tích với AI", key="insight_general"):
    _context = tab_context.get(ai_tab, "Phân tích dữ liệu khách hàng")
    
    # Lấy dữ liệu tương ứng
    if ai_tab == "Customer Retention":
        # Prepare retention data
        orders = df_orders.merge(cust[["customer_key", "first_month"]], on="customer_key")
        orders["visit_type"] = np.where(
            orders["order_month_date"] == orders["first_month"], "Khách mới", "Khách quay lại"
        )
        monthly = (orders.groupby(["order_month_date", "visit_type"])["order_id"]
                   .nunique().reset_index(name="orders"))
        _data = pd.DataFrame({
            "total_customers": [total_customers],
            "repeat_rate": [repeat_rate],
            "avg_orders": [avg_orders],
            "active_months": [active_months]
        }).to_string(index=False)
        
    elif ai_tab == "Repeat Customers":
        top_repeat = (repeat_df.sort_values(["n_orders", "total_value"], ascending=False)
                      .head(10)[["customer_key", "n_orders", "total_value"]])
        _data = pd.DataFrame({
            "n_repeat": [n_repeat],
            "repeat_rate": [repeat_rate],
            "avg_orders_repeat": [avg_orders_repeat],
            "revenue_share": [revenue_share],
            "top_repeat_customers": [top_repeat.to_string(index=False)]
        }).to_string(index=False)
        
    elif ai_tab == "Customer Lifetime Value":
        _data = pd.DataFrame({
            "avg_clv": [avg_clv],
            "median_clv": [median_clv],
            "top10_share": [top10_share],
            "total_revenue": [total_rev]
        }).to_string(index=False)
        
    else:  # Geographic Distribution
        top_states = state_summary.head(10)[["state", "customers", "total_revenue"]]
        _data = pd.DataFrame({
            "n_states": [n_states],
            "top_state": [top_state_revenue],
            "total_revenue": [state_summary['total_revenue'].sum()],
            "top_states": [top_states.to_string(index=False)]
        }).to_string(index=False)
    
    with st.spinner("Đang phân tích..."):
        insight = generate_insight(_context, _data)
    st.markdown(insight)