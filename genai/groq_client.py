# genai/groq_client.py
# Wrapper cho Groq API — generate_sql và generate_insight
# Gọi từ bất kỳ Streamlit page nào

import streamlit as st
from groq import Groq
from genai.olist_context import build_sql_system_prompt, build_insight_system_prompt

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 512


@st.cache_resource
def _get_client() -> Groq:
    """Khởi tạo Groq client một lần duy nhất, dùng lại cho toàn session."""
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


def generate_sql(question: str) -> str:
    """
    Nhận câu hỏi tiếng Việt, trả về SQL string.
    Raise ValueError nếu response rỗng hoặc chứa ký tự nghi ngờ không phải SQL.
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": build_sql_system_prompt(),
            },
            {
                "role": "user",
                "content": f"QUESTION: {question}\n\nGenerate SQL. Return only SQL.",
            },
        ],
    )

    sql = response.choices[0].message.content.strip()

    # Strip markdown fences nếu LLM vẫn thêm vào
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.lower().startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()

    if not sql.lower().startswith("select"):
        raise ValueError(f"LLM không trả về SQL hợp lệ:\n{sql}")

    return sql


def generate_insight(question: str, df_str: str) -> str:
    """
    Nhận câu hỏi + kết quả dạng string, trả về insight tiếng Việt.
    df_str nên là df.to_string(index=False) hoặc df.to_markdown().
    """
    client = _get_client()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": build_insight_system_prompt(),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nData:\n{df_str}",
            },
        ],
    )

    return response.choices[0].message.content.strip()