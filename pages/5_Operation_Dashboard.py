# pages/5_Operations_Dashboard.py
import streamlit as st
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Import kết nối chung từ utils
from utils.connection import get_connection

# Khởi tạo kết nối
con = get_connection()

# Định nghĩa schema
GOLD = "gold_main_gold"
SILVER = "silver"

# =============================================
# HÀM LOAD DATA (có cache)
# =============================================

@st.cache_data(ttl=600)
def load_order_status():
    return con.execute(f"""
        SELECT order_status,
               COUNT(*) AS order_count,
               ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2) AS pct
        FROM {GOLD}.fact_orders
        GROUP BY 1 ORDER BY order_count DESC
    """).df()

@st.cache_data(ttl=600)
def load_delivery():
    return con.execute(f"""
        SELECT delivery_status,
               COUNT(*)                            AS orders,
               ROUND(AVG(actual_delivery_days), 1) AS avg_days,
               ROUND(AVG(delivery_delay_days),  1) AS avg_delay,
               ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2) AS pct
        FROM {GOLD}.fact_orders
        WHERE order_status = 'delivered'
          AND actual_delivery_days IS NOT NULL
        GROUP BY 1
    """).df()

@st.cache_data(ttl=600)
def load_delivery_trend():
    return con.execute(f"""
        SELECT order_month_date,
               ROUND(AVG(actual_delivery_days), 1) AS avg_delivery_days,
               ROUND(AVG(delivery_delay_days),  1) AS avg_delay_days,
               ROUND(SUM(CASE WHEN delivery_status='on_time'
                         THEN 1.0 ELSE 0 END)*100/COUNT(*), 1) AS on_time_pct
        FROM {GOLD}.fact_orders
        WHERE order_status = 'delivered'
          AND order_month_date IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()

@st.cache_data(ttl=600)
def load_review():
    return con.execute(f"""
        SELECT review_score,
               COUNT(*) AS count,
               ROUND(COUNT(*)*100.0/SUM(COUNT(*)) OVER(), 2) AS pct,
               ROUND(AVG(delivery_delay_days), 1) AS avg_delay_days
        FROM {GOLD}.fact_orders
        WHERE review_score IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).df()

@st.cache_data(ttl=600)
def load_review_by_category():
    return con.execute(f"""
        SELECT dp.product_category,
               ROUND(AVG(fo.review_score), 2) AS avg_score,
               COUNT(*) AS total_reviews
        FROM {GOLD}.fact_orders fo
        JOIN {SILVER}.order_items oi USING(order_id)
        JOIN {GOLD}.dim_product dp USING(product_id)
        WHERE fo.review_score IS NOT NULL
        GROUP BY 1
        HAVING COUNT(*) > 100
        ORDER BY avg_score DESC
        LIMIT 15
    """).df()

@st.cache_data(ttl=600)
def load_rfm_raw():
    df = con.execute(f"""
        SELECT
            c.customer_unique_id,
            MAX(c.customer_state)                 AS customer_state,
            COUNT(DISTINCT fo.order_id)            AS frequency,
            MAX(fo.order_purchase_timestamp)       AS last_order_date,
            ROUND(SUM(fo.total_order_value), 2)    AS monetary
        FROM {GOLD}.fact_orders fo
        JOIN {SILVER}.customers c ON fo.customer_id = c.customer_id
        WHERE fo.order_status = 'delivered'
        GROUP BY c.customer_unique_id
    """).df()

    snapshot_date = df["last_order_date"].max()
    df["recency"] = (snapshot_date - df["last_order_date"]).dt.days
    df = df.drop(columns=["last_order_date"])
    return df

# =============================================
# GIAO DIỆN CHÍNH
# =============================================

st.title("Operations Dashboard")
st.caption("Overview of order processing, delivery performance, and customer reviews")

# Tạo tabs để tổ chức nội dung
tab1, tab2, tab3, tab4 = st.tabs([
    "Order Status",
    "Delivery Performance",
    "Review Score",
    "RFM Segmentation"
])

# =============================================
# TAB 1: ORDER STATUS
# =============================================
with tab1:
    st.subheader("Order Status Analysis")
    
    df = load_order_status()
    total = df["order_count"].sum()
    delivered = df[df["order_status"]=="delivered"]["order_count"].sum()
    cancelled = df[df["order_status"]=="canceled"]["order_count"].sum() if "canceled" in df["order_status"].values else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{total:,}")
    col2.metric("Delivered", f"{delivered:,}", f"{delivered/total*100:.1f}%")
    col3.metric("Cancelled", f"{cancelled:,}", f"-{cancelled/total*100:.1f}%", delta_color="inverse")
    col4.metric("Remaining Orders", f"{total-delivered-cancelled:,}")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        fig = px.bar(df, x="order_status", y="order_count", text="pct",
                     color="order_status", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=400, xaxis_title="Status", yaxis_title="Orders")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        fig = px.pie(df, names="order_status", values="order_count",
                     color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Detail Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

# =============================================
# TAB 2: DELIVERY PERFORMANCE
# =============================================
with tab2:
    st.subheader("Delivery Performance")
    
    df_del = load_delivery()
    df_trend = load_delivery_trend()
    
    on_time_row = df_del[df_del["delivery_status"]=="on_time"]
    late_row = df_del[df_del["delivery_status"]=="late"]
    on_time_pct = on_time_row["pct"].values[0] if len(on_time_row) else 0
    avg_days = on_time_row["avg_days"].values[0] if len(on_time_row) else 0
    avg_delay = late_row["avg_delay"].values[0] if len(late_row) else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("On-time Delivery", f"{on_time_pct}%")
    col2.metric("Avg Delivery Time", f"{avg_days} days")
    col3.metric("Avg Delay (late orders)", f"{avg_delay} days", delta_color="inverse")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        colors = {"on_time": "#2ecc71", "late": "#e74c3c"}
        fig = px.pie(df_del, names="delivery_status", values="orders",
                     color="delivery_status", color_discrete_map=colors, hole=0.4)
        fig.update_traces(textinfo="label+percent")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        fig = px.line(df_trend, x="order_month_date", y="on_time_pct",
                      markers=True, color_discrete_sequence=["#2ecc71"])
        fig.add_hline(y=90, line_dash="dash", line_color="red", annotation_text="Target 90%")
        fig.update_layout(height=380, xaxis_title="Month", yaxis_title="% On-time")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Delivery Time & Delay by Month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_trend["order_month_date"], y=df_trend["avg_delivery_days"],
        name="Avg Delivery Time (days)", line=dict(color="#3498db")))
    fig.add_trace(go.Bar(
        x=df_trend["order_month_date"], y=df_trend["avg_delay_days"],
        name="Avg Delay (days)", marker_color="#e74c3c", opacity=0.5))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(height=380, xaxis_title="Month", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# TAB 3: REVIEW SCORE
# =============================================
with tab3:
    st.subheader("Review Score Analysis")
    
    df_rev = load_review()
    df_cat = load_review_by_category()
    
    avg_score = (df_rev["review_score"]*df_rev["count"]).sum()/df_rev["count"].sum()
    pct_5_star = df_rev[df_rev["review_score"]==5]["pct"].values[0] if len(df_rev[df_rev["review_score"]==5]) else 0
    pct_1_star = df_rev[df_rev["review_score"]==1]["pct"].values[0] if len(df_rev[df_rev["review_score"]==1]) else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Review Score", f"{avg_score:.2f} / 5")
    col2.metric("5-star Rate", f"{pct_5_star}%")
    col3.metric("1-star Rate", f"{pct_1_star}%", delta_color="inverse")
    
    st.divider()
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        colors = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#27ae60"]
        fig = px.bar(df_rev, x="review_score", y="count", text="pct",
                     color="review_score", color_discrete_sequence=colors)
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, height=380, xaxis_title="Score", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        fig = px.bar(df_rev, x="review_score", y="avg_delay_days",
                     color="avg_delay_days", color_continuous_scale=["#2ecc71","#e74c3c"],
                     labels={"avg_delay_days":"Avg Delay (days)"})
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(height=380, xaxis_title="Review Score", yaxis_title="Avg Delay (days)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Avg Score by Product Category (Top 15)")
    fig = px.bar(df_cat.sort_values("avg_score"),
                 x="avg_score", y="product_category", orientation="h",
                 text="avg_score", color="avg_score",
                 color_continuous_scale=["#e74c3c","#f1c40f","#2ecc71"],
                 range_color=[3.5, 5])
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=500, xaxis_range=[3, 5.2],
                      yaxis_title="", xaxis_title="Avg Score")
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# TAB 4: CUSTOMER SEGMENTATION
# =============================================
with tab4:

    st.subheader("Customer Segmentation")

    st.caption(
        "RFM analysis to identify customer groups, revenue contribution, and retention opportunities."
    )

    with st.spinner("Loading data..."):
        df = load_rfm_raw().copy()

    # ── RFM Scoring ──────────────────────────────
    df["R_score"] = pd.qcut(
        df["recency"].rank(method="first"),
        5,
        labels=[5,4,3,2,1]
    ).astype(int)

    df["F_score"] = pd.qcut(
        df["frequency"].rank(method="first"),
        5,
        labels=[1,2,3,4,5]
    ).astype(int)

    df["M_score"] = pd.qcut(
        df["monetary"].rank(method="first"),
        5,
        labels=[1,2,3,4,5]
    ).astype(int)

    df["RFM_score"] = (
        df["R_score"]
        + df["F_score"]
        + df["M_score"]
    )

    # ── KMeans cố định K=6 ───────────────────────
    K = 6

    X = StandardScaler().fit_transform(
        df[["recency","frequency","monetary"]]
    )

    df["cluster"] = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10
    ).fit_predict(X)

    cluster_profile = (
        df.groupby("cluster")
          .agg(
              Avg_Recency=("recency","mean"),
              Avg_Frequency=("frequency","mean"),
              Avg_Monetary=("monetary","mean"),
              Customers=("customer_unique_id","count")
          )
          .round(2)
          .reset_index()
    )

    # ── Segment Mapping ──────────────────────────
    label_map = {
        2: "Champions",
        3: "Loyal",
        4: "Returning",
        1: "At Risk",
        0: "New",
        5: "Lost"
    }

    df["segment"] = df["cluster"].map(label_map)

    cluster_profile["Segment"] = (
        cluster_profile["cluster"]
        .map(label_map)
    )

    cluster_profile = cluster_profile[
        [
            "cluster",
            "Segment",
            "Avg_Recency",
            "Avg_Frequency",
            "Avg_Monetary",
            "Customers"
        ]
    ]

    SEGMENT_ORDER = [
        "Champions",
        "Loyal",
        "Returning",
        "New",
        "At Risk",
        "Lost"
    ]

    # ── Cluster Profile ──────────────────────────
    st.subheader("Cluster Profile")

    st.dataframe(
        cluster_profile,
        use_container_width=True,
        hide_index=True
    )

    # ── KPI ──────────────────────────────────────
    seg_counts = (
        df["segment"]
          .value_counts()
          .reindex(SEGMENT_ORDER)
          .dropna()
    )

    st.metric(
        "Total Customers",
        f"{len(df):,}"
    )

    cols = st.columns(len(seg_counts))

    for i, (seg, cnt) in enumerate(seg_counts.items()):
        cols[i].metric(
            seg,
            f"{cnt:,}",
            f"{cnt/len(df)*100:.1f}%"
        )

    st.divider()

    # ── Charts ───────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Customer Distribution by Segment")

        fig = px.pie(
            df,
            names="segment",
            hole=0.4,
            category_orders={
                "segment": SEGMENT_ORDER
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_traces(
            textinfo="label+percent"
        )

        fig.update_layout(height=380)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Monetary vs Recency")

        sample = df.sample(
            min(3000, len(df)),
            random_state=42
        )

        fig = px.scatter(
            sample,
            x="recency",
            y="monetary",
            color="segment",
            size="frequency",
            opacity=0.7,
            category_orders={
                "segment": SEGMENT_ORDER
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={
                "recency":"Recency (days)",
                "monetary":"Monetary (R$)"
            }
        )

        fig.update_xaxes(
            autorange="reversed"
        )

        fig.update_layout(height=380)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ── Segment Characteristics ─────────────────
    st.subheader("Segment Characteristics")

    summary = (
        df.groupby("segment")
          .agg(
              Customers=("customer_unique_id","count"),
              Avg_Recency=("recency","mean"),
              Avg_Frequency=("frequency","mean"),
              Avg_Monetary=("monetary","mean"),
              Total_Revenue=("monetary","sum")
          )
          .round(1)
          .reset_index()
    )

    summary["segment"] = pd.Categorical(
        summary["segment"],
        categories=SEGMENT_ORDER,
        ordered=True
    )

    summary = summary.sort_values("segment")

    summary["Total_Revenue"] = (
        summary["Total_Revenue"]
        .apply(lambda x: f"R${x:,.0f}")
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # ── Revenue Contribution ────────────────────
    st.subheader("Revenue Contribution by Segment")

    revenue_seg = (
        df.groupby("segment")["monetary"]
          .sum()
          .reset_index()
    )

    revenue_seg["segment"] = pd.Categorical(
        revenue_seg["segment"],
        categories=SEGMENT_ORDER,
        ordered=True
    )

    revenue_seg = revenue_seg.sort_values("segment")

    revenue_seg["pct"] = (
        revenue_seg["monetary"]
        / revenue_seg["monetary"].sum()
        * 100
    ).round(1)

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            revenue_seg,
            x="segment",
            y="monetary",
            color="segment",
            text="pct",
            category_orders={
                "segment": SEGMENT_ORDER
            },
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside"
        )

        fig.update_layout(
            height=380,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.pie(
            revenue_seg,
            names="segment",
            values="monetary",
            category_orders={
                "segment": SEGMENT_ORDER
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )

        fig.update_layout(height=380)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ── RFM Score Distribution ──────────────────
    st.subheader("RFM Score Distribution by Segment")

    fig = px.box(
        df,
        x="segment",
        y="RFM_score",
        color="segment",
        category_orders={
            "segment": SEGMENT_ORDER
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        height=380,
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ── Marketing Action Recommendations ────────
    st.subheader("Marketing Action Recommendations")

    actions = {
        "Champions": "Offer VIP perks, invite to loyalty program, encourage reviews",
        "Loyal": "Upsell premium products, send birthday offers, retain with reward points",
        "Returning": "Remind about viewed products, suggest related items",
        "New": "Good onboarding, send guides, offer discount on second purchase",
        "At Risk": "Send win-back emails, offer discount vouchers to re-engage",
        "Lost": "Final re-engagement campaign, stop marketing if no response",
    }

    for seg in SEGMENT_ORDER:
        if seg in seg_counts.index:
            st.info(f"**{seg}** — {actions[seg]}")

    # ── Export ──────────────────────────────────
    st.divider()

    st.subheader("Export Results")

    export_df = df[
        [
            "customer_unique_id",
            "customer_state",
            "recency",
            "frequency",
            "monetary",
            "R_score",
            "F_score",
            "M_score",
            "RFM_score",
            "segment"
        ]
    ]

    csv = export_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download RFM Results (.csv)",
        data=csv,
        file_name="rfm_segments.csv",
        mime="text/csv"
    )