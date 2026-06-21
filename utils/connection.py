import streamlit as st
import duckdb

@st.cache_resource
def get_connection():

    token = st.secrets["MOTHERDUCK_TOKEN"]

    return duckdb.connect(
        f"md:olist_dw?motherduck_token={token}"
    )