import streamlit as st

st.set_page_config(page_title="METEORIC", layout="wide")

pages = [
    st.Page("views/m0_robustness.py", title="M0 robustness (Dubai)", default=True),
    st.Page("views/real_data.py", title="Real data (Dubai)"),
    st.Page("views/synthetic_prototype.py", title="Synthetic prototype"),
]
st.navigation(pages).run()