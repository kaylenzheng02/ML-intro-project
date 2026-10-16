from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DATA_PATH = Path(__file__).parent / "data" / "penguins.csv"
MEASUREMENTS = {
    "Bill length (mm)": "bill_length_mm",
    "Bill depth (mm)": "bill_depth_mm",
    "Flipper length (mm)": "flipper_length_mm",
    "Body mass (g)": "body_mass_g",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


st.set_page_config(page_title="Penguin Explorer", page_icon=None, layout="wide")
st.title("Explore the Palmer Penguins")
st.write(
    "Each point is one penguin. Choose measurements to compare and see how "
    "the three species cluster."
)

data = load_data()
species = sorted(data["species"].dropna().unique().tolist())

with st.sidebar:
    st.header("Graph settings")
    x_label = st.selectbox("Horizontal axis", MEASUREMENTS, index=0)
    y_label = st.selectbox("Vertical axis", MEASUREMENTS, index=2)
    selected_species = st.multiselect(
        "Species to show",
        options=species,
        default=species,
    )

if not selected_species:
    st.warning("Select at least one species to display the graph.")
    st.stop()

x_column = MEASUREMENTS[x_label]
y_column = MEASUREMENTS[y_label]
chart_data = data.loc[
    data["species"].isin(selected_species),
    ["species", "island", "sex", x_column, y_column],
].dropna(subset=[x_column, y_column])

if chart_data.empty:
    st.warning("There are no penguins with both selected measurements.")
    st.stop()

figure = px.scatter(
    chart_data,
    x=x_column,
    y=y_column,
    color="species",
    hover_data=["island", "sex"],
    labels={x_column: x_label, y_column: y_label, "species": "Species"},
    title=f"{y_label} vs. {x_label}",
)
figure.update_layout(legend_title_text="Species")
st.plotly_chart(figure)
st.caption(
    f"Showing {len(chart_data)} of {len(data)} penguins. "
    "Penguins missing either selected measurement are left out of the graph."
)

with st.expander("Preview the dataset"):
    st.dataframe(data)
