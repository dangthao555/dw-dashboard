import streamlit as st
from utils.connection import get_connection
from genai import generate_sql, generate_insight

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Natural Language Query",
    layout="wide"
)

st.title("Natural Language Query")

st.markdown(
    "Đặt câu hỏi bằng tiếng Việt. "
    "AI sẽ tự sinh SQL, truy vấn dữ liệu và phân tích kết quả."
)

# ==================================================
# CONNECTION
# ==================================================

con = get_connection()

# ==================================================
# EXAMPLE QUESTIONS
# ==================================================

EXAMPLES = [
    "Top 10 danh mục doanh thu cao nhất",
    "Doanh thu theo từng tháng",
    "Top 5 tiểu bang có nhiều khách hàng nhất",
    "Phương thức thanh toán phổ biến nhất",
    "Top 10 seller doanh thu cao nhất",
    "Tỷ lệ đơn hàng theo trạng thái",
]

st.markdown("**Câu hỏi gợi ý:**")

cols = st.columns(3)

for i, example in enumerate(EXAMPLES):
    with cols[i % 3]:
        if st.button(example, key=f"ex_{i}", use_container_width=True):
            st.session_state["nl_question"] = example

# ==================================================
# INPUT
# ==================================================

st.divider()

question = st.text_input(
    "Câu hỏi của bạn",
    value=st.session_state.get("nl_question", ""),
    placeholder="Ví dụ: Top 10 danh mục doanh thu cao nhất",
    key="nl_question_input"
)

run = st.button("Phân tích", type="primary")

# ==================================================
# QUERY + INSIGHT
# ==================================================

if run and question.strip():

    # Step 1: Generate SQL
    with st.spinner("Đang sinh câu lệnh SQL..."):
        try:
            sql = generate_sql(question)
        except ValueError as e:
            st.error(f"Không thể sinh SQL hợp lệ: {e}")
            st.stop()

    with st.expander("Xem câu lệnh SQL"):
        st.code(sql, language="sql")

    # Step 2: Execute
    with st.spinner("Đang truy vấn dữ liệu..."):
        try:
            df = con.execute(sql).fetchdf()
        except Exception as e:
            st.error(f"Lỗi khi thực thi SQL: {e}")
            st.markdown("Hãy thử diễn đạt lại câu hỏi hoặc chọn câu hỏi gợi ý.")
            st.stop()

    if df.empty:
        st.warning("Truy vấn không trả về kết quả.")
        st.stop()

    st.subheader("Kết quả")
    st.dataframe(df, use_container_width=True)

    # Step 3: Generate Insight
    with st.spinner("Đang phân tích..."):
        insight = generate_insight(question, df.to_string(index=False))

    st.subheader("Phân tích AI")
    st.markdown(insight)

elif run and not question.strip():
    st.warning("Vui lòng nhập câu hỏi.")