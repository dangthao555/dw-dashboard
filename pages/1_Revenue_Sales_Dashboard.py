# pages/1_Revenue_and_Sales_Dashboard.py
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
from utils.connection import get_connection
from genai import generate_insight

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Revenue and Sales Dashboard",
    layout="wide"
)

st.title("Revenue and Sales Dashboard")

# ==================================================
# CONNECTION
# ==================================================

con = get_connection()

# ==================================================
# CACHE DATA
# ==================================================

@st.cache_data(ttl=600)
def load_years():
    return con.execute("""
        SELECT DISTINCT order_year
        FROM gold_main_gold.fact_payments
        ORDER BY order_year
    """).fetchdf()

@st.cache_data(ttl=600)
def load_revenue_data(selected_year):
    if selected_year == "All":
        return con.execute("""
            SELECT
                order_month_date,
                SUM(payment_value) AS revenue,
                COUNT(DISTINCT order_id) AS total_orders
            FROM gold_main_gold.fact_payments
            WHERE order_status = 'delivered'
            GROUP BY order_month_date
            ORDER BY order_month_date
        """).fetchdf()
    else:
        return con.execute(f"""
            SELECT
                order_month_date,
                SUM(payment_value) AS revenue,
                COUNT(DISTINCT order_id) AS total_orders
            FROM gold_main_gold.fact_payments
            WHERE order_status = 'delivered'
              AND order_year = {selected_year}
            GROUP BY order_month_date
            ORDER BY order_month_date
        """).fetchdf()

@st.cache_data(ttl=600)
def load_category_data():
    return con.execute("""
        SELECT
            dp.product_category,
            COUNT(*) AS total_items_sold,
            COUNT(DISTINCT oi.order_id) AS total_orders,
            SUM(oi.price) AS revenue
        FROM silver.order_items oi
        JOIN gold_main_gold.dim_product dp
            ON oi.product_id = dp.product_id
        WHERE dp.product_category IS NOT NULL
        GROUP BY dp.product_category
        ORDER BY revenue DESC
    """).fetchdf()

@st.cache_data(ttl=600)
def load_payment_data():
    return con.execute("""
        SELECT
            payment_type,
            payment_installments,
            payment_value,
            order_id
        FROM gold_main_gold.fact_payments
        WHERE order_status = 'delivered'
    """).fetchdf()

@st.cache_data(ttl=600)
def load_yearly_revenue():
    return con.execute("""
        SELECT
            order_year,
            SUM(payment_value) AS revenue
        FROM gold_main_gold.fact_payments
        WHERE order_status = 'delivered'
        GROUP BY order_year
        ORDER BY order_year
    """).fetchdf()

# ==================================================
# SIDEBAR FILTER
# ==================================================

years_df = load_years()
year_list = ["All"] + years_df["order_year"].astype(str).tolist()

selected_year = st.sidebar.selectbox(
    "Select Year",
    year_list
)

# ==================================================
# LOAD DATA
# ==================================================

revenue_df = load_revenue_data(selected_year)
category_df = load_category_data()
payment_df = load_payment_data()
yearly_df = load_yearly_revenue()

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3 = st.tabs([
    "Revenue Analysis",
    "Category Analysis",
    "Payment Analysis"
])

# ==================================================
# TAB 1: REVENUE ANALYSIS
# ==================================================

with tab1:
    # KPI CALCULATION
    total_revenue = revenue_df["revenue"].sum()
    total_orders = revenue_df["total_orders"].sum()
    aov = total_revenue / total_orders if total_orders > 0 else 0
    best_month_row = revenue_df.loc[revenue_df["revenue"].idxmax()] if not revenue_df.empty else None
    best_month = best_month_row["order_month_date"] if best_month_row is not None else "N/A"
    best_revenue = best_month_row["revenue"] if best_month_row is not None else 0
    
    # KPI DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"{total_revenue:,.2f} BRL")
    with col2:
        st.metric("Total Orders", f"{total_orders:,}")
    with col3:
        st.metric("Average Order Value", f"{aov:,.2f} BRL")
    with col4:
        st.metric("Best Month Revenue", f"{best_revenue:,.2f} BRL")
    
    st.divider()
    
    # MONTHLY REVENUE TREND
    st.subheader("Monthly Revenue Trend")
    fig_line = px.line(
        revenue_df,
        x="order_month_date",
        y="revenue",
        markers=True
    )
    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Revenue (BRL)",
        height=500
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    # REVENUE BY YEAR
    st.subheader("Revenue by Year")
    fig_bar = px.bar(
        yearly_df,
        x="order_year",
        y="revenue",
        text_auto=".2s"
    )
    fig_bar.update_layout(
        xaxis_title="Year",
        yaxis_title="Revenue (BRL)",
        height=450
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # REVENUE DETAIL TABLE
    st.subheader("Revenue Detail")
    display_df = revenue_df.copy()
    display_df["revenue"] = display_df["revenue"].round(2)
    st.dataframe(display_df, use_container_width=True)

# ==================================================
# TAB 2: CATEGORY ANALYSIS
# ==================================================

with tab2:
    # KPI CALCULATION
    top_category = category_df.iloc[0]["product_category"] if not category_df.empty else "N/A"
    top_revenue = category_df.iloc[0]["revenue"] if not category_df.empty else 0
    total_categories = len(category_df)
    total_revenue_cat = category_df["revenue"].sum()
    top_share = (top_revenue / total_revenue_cat * 100) if total_revenue_cat > 0 else 0
    
    # KPI DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Categories", f"{total_categories:,}")
    with col2:
        st.metric("Top Category", top_category)
    with col3:
        st.metric("Top Category Revenue", f"{top_revenue:,.2f} BRL")
    with col4:
        st.metric("Top Category Share", f"{top_share:.2f}%")
    
    st.divider()
    
    # TOP 10 CATEGORIES BY REVENUE
    st.subheader("Top 10 Categories by Revenue")
    top10_revenue = category_df.sort_values("revenue", ascending=False).head(10)
    fig_revenue = px.bar(
        top10_revenue,
        x="revenue",
        y="product_category",
        orientation="h",
        text_auto=".2s"
    )
    fig_revenue.update_layout(
        xaxis_title="Revenue (BRL)",
        yaxis_title="Category",
        height=550
    )
    st.plotly_chart(fig_revenue, use_container_width=True)
    
    # TOP 10 CATEGORIES BY SALES VOLUME
    st.subheader("Top 10 Categories by Items Sold")
    top10_sales = category_df.sort_values("total_items_sold", ascending=False).head(10)
    fig_sales = px.bar(
        top10_sales,
        x="total_items_sold",
        y="product_category",
        orientation="h",
        text_auto=True
    )
    fig_sales.update_layout(
        xaxis_title="Items Sold",
        yaxis_title="Category",
        height=550
    )
    st.plotly_chart(fig_sales, use_container_width=True)
    
    # REVENUE SHARE
    st.subheader("Revenue Share of Top Categories")
    pie_df = category_df.sort_values("revenue", ascending=False)
    top20 = pie_df.head(20)
    others_revenue = pie_df.iloc[20:]["revenue"].sum()
    others_row = pd.DataFrame({
        "product_category": ["Others"],
        "revenue": [others_revenue]
    })
    pie_df_final = pd.concat([top20, others_row], ignore_index=True)
    fig_pie = px.pie(
        pie_df_final,
        names="product_category",
        values="revenue"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # DETAIL TABLE
    st.subheader("Category Detail")
    display_df = category_df.sort_values("revenue", ascending=False)
    display_df["revenue"] = display_df["revenue"].round(2)
    display_df["revenue_share"] = (display_df["revenue"] / display_df["revenue"].sum() * 100).round(2)
    st.dataframe(display_df, use_container_width=True)

# ==================================================
# TAB 3: PAYMENT ANALYSIS
# ==================================================

with tab3:
    # KPI CALCULATION
    total_revenue_pay = payment_df["payment_value"].sum()
    total_orders_pay = payment_df["order_id"].nunique()
    top_method = payment_df["payment_type"].value_counts().idxmax() if not payment_df.empty else "N/A"
    avg_installments = payment_df["payment_installments"].mean()
    
    # KPI DISPLAY
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Revenue", f"{total_revenue_pay:,.2f} BRL")
    with col2:
        st.metric("Total Orders", f"{total_orders_pay:,}")
    with col3:
        st.metric("Most Used Method", top_method)
    with col4:
        st.metric("Avg Installments", f"{avg_installments:.2f}")
    
    st.divider()
    
    # PAYMENT METHOD DISTRIBUTION
    st.subheader("Payment Method Distribution")
    method_df = payment_df.groupby("payment_type").agg(
        total_orders=("order_id", "count")
    ).reset_index()
    fig_pie = px.pie(
        method_df,
        names="payment_type",
        values="total_orders"
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # REVENUE BY PAYMENT METHOD
    st.subheader("Revenue by Payment Method")
    revenue_method_df = payment_df.groupby("payment_type").agg(
        revenue=("payment_value", "sum")
    ).reset_index().sort_values("revenue", ascending=False)
    fig_bar = px.bar(
        revenue_method_df,
        x="payment_type",
        y="revenue",
        text_auto=".2s"
    )
    fig_bar.update_layout(
        xaxis_title="Payment Method",
        yaxis_title="Revenue (BRL)",
        height=500
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # INSTALLMENT ANALYSIS
    st.subheader("Installment Analysis")
    fig_hist = px.histogram(
        payment_df,
        x="payment_installments",
        nbins=24
    )
    fig_hist.update_layout(
        xaxis_title="Installments",
        yaxis_title="Number of Payments",
        height=500
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # DETAIL TABLE
    st.subheader("Payment Method Detail")
    detail_df = payment_df.groupby("payment_type").agg(
        total_orders=("order_id", "count"),
        total_revenue=("payment_value", "sum"),
        avg_installments=("payment_installments", "mean")
    ).reset_index()
    detail_df["total_revenue"] = detail_df["total_revenue"].round(2)
    detail_df["avg_installments"] = detail_df["avg_installments"].round(2)
    st.dataframe(detail_df, use_container_width=True)

# ==================================================
# AI INSIGHT (DÙNG CHUNG CHO CẢ 3 TAB)
# ==================================================

st.divider()
st.subheader("AI Insight")

# Chọn context dựa trên tab đang active
tab_context = {
    "Revenue Analysis": "Phân tích doanh thu theo tháng và năm",
    "Category Analysis": "Phân tích danh mục sản phẩm theo doanh thu và số lượng bán",
    "Payment Analysis": "Phân tích phương thức thanh toán theo số đơn, doanh thu và số kỳ trả góp"
}

# Dropdown để chọn tab cho AI insight
ai_tab = st.selectbox(
    "Chọn phân tích cho AI:",
    ["Revenue Analysis", "Category Analysis", "Payment Analysis"],
    index=0
)

if st.button("Phân tích với AI", key="insight_general"):
    _context = tab_context.get(ai_tab, "Phân tích dữ liệu")
    
    # Lấy dữ liệu tương ứng
    if ai_tab == "Revenue Analysis":
        _data = revenue_df.to_string(index=False)
    elif ai_tab == "Category Analysis":
        _data = category_df.head(20).to_string(index=False)
    else:  # Payment Analysis
        _data = detail_df.to_string(index=False)
    
    with st.spinner("Đang phân tích..."):
        insight = generate_insight(_context, _data)
    st.markdown(insight)