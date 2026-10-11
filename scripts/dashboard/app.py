"""BI/analytics dashboard for the GeoSuitability AI Analytics project.

Reads the data lake (datasets/raw), the processed warehouse/mart layer (datasets/processed) and the
results of the modeling experiments (scripts/modeling). No separate database or backend.

Each page is its own script and only runs while it is open, so heavy pages (EDA, AI Lab) cost nothing
until they are visited.

Run with dashboard.ipynb, or from this folder (so the theme in .streamlit/config.toml is used):
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="GeoSuitability AI", page_icon=":material/agriculture:", layout="wide")

import common as C  # noqa: E402

C.apply_style()
ASSETS = C.ROOT / "scripts" / "dashboard" / "assets"
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

pages = {
    "Overview": [
        st.Page("page_overview.py", title="Overview", icon=":material/dashboard:", default=True),
    ],
    "Suitability": [
        st.Page("page_map.py", title="Suitability Map", icon=":material/map:"),
        st.Page("page_gaez_gcs.py", title="GAEZ x GCS 2024", icon=":material/compare_arrows:"),
    ],
    "Data": [
        st.Page("page_lake.py", title="Data Lake & Warehouse", icon=":material/database:"),
        st.Page("page_eda.py", title="Exploratory Analysis", icon=":material/query_stats:"),
        st.Page("page_explorer.py", title="Data Explorer", icon=":material/table_view:"),
    ],
    "AI": [
        st.Page("page_experiments.py", title="ML/DL Experiments", icon=":material/science:"),
        st.Page("page_ai_lab.py", title="AI Lab", icon=":material/smart_toy:"),
    ],
    "Analytics": [
        st.Page("page_graph.py", title="Graph Analytics", icon=":material/hub:"),
    ],
}

nav = st.navigation(pages)
C.rule_controls()
st.sidebar.caption("Banten Province · 9,333 cells of 1 km · UTM 48S")
nav.run()
C.footer()
