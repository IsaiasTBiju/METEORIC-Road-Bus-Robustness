"""Real-data page (Dubai). Placeholder until the cleaned, aggregated data exists."""
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

st.title("METEORIC: Dubai road-bus network")
st.info("Under construction. This page will show the real network once the data has been cleaned and aggregated.")

found = sorted(p.name for p in PROCESSED.glob("*")) if PROCESSED.exists() else []
if found:
    st.write("Files in data/processed:", found)
else:
    st.write("No cleaned data in `data/processed/` yet. The next step is the cleaning script.")
