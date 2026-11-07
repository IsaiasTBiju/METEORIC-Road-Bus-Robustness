"""METEORIC app entry point: a menu with a synthetic prototype and the real-data page."""
import streamlit as st

st.set_page_config(page_title="METEORIC", layout="wide")

pages = [
    st.Page("views/real_data.py", title="Real data (Dubai)", default=True),
    st.Page("views/synthetic_prototype.py", title="Synthetic prototype"),
]
st.navigation(pages).run()
