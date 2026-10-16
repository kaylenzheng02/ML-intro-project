from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from train import FEATURES, TARGET, build_model


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


@st.cache_resource
def load_prediction_model():
    training_data = load_data().dropna(subset=[TARGET])
    model = build_model()
    model.fit(training_data[FEATURES], training_data[TARGET])
    return model


st.set_page_config(page_title="Penguin Explorer", page_icon=None, layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    h1 {
        font-size: 2rem !important;
        line-height: 1.2 !important;
        letter-spacing: -0.025em;
        font-weight: 550;
    }
    .eyebrow {
        color: #526A73;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
    }
    div[data-testid="stMetricLabel"] {
        color: #526A73;
        font-size: 0.85rem;
    }
    div[data-testid="stMetricValue"] {
        color: #243B53;
        font-size: 1.6rem;
        font-weight: 500;
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid #D9E2E8;
    }
    div[data-testid="stSelectbox"] label {
        color: #526A73;
    }
    div[data-testid="stSelectbox"] [role="group"] {
        border-radius: 0 !important;
    }
    div[data-testid="stNumberInputContainer"],
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stDateInput"] [role="group"],
    div[data-testid="stTimeInput"] [role="group"] {
        border-radius: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="eyebrow">PALMER PENGUINS · DATA EXPLORATION</p>', unsafe_allow_html=True)
st.title("Penguin measurements by species")
st.write(
    "Compare physical measurements across species. Each point represents one "
    "penguin; select variables and species to examine the data."
)

data = load_data()
species = sorted(data["species"].dropna().unique().tolist())
all_species_label = "All species"

with st.sidebar:
    st.header("Plot settings")
    x_label = st.selectbox("X-axis", MEASUREMENTS, index=0)
    y_label = st.selectbox("Y-axis", MEASUREMENTS, index=2)
    species_selection = st.selectbox(
        "Species",
        options=[all_species_label, *species],
        index=0,
    )

x_column = MEASUREMENTS[x_label]
y_column = MEASUREMENTS[y_label]
species_mask = (
    data["species"].isin(species)
    if species_selection == all_species_label
    else data["species"].eq(species_selection)
)
chart_data = data.loc[
    species_mask,
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
    color_discrete_map={
        "Adelie": "#0072B2",
        "Chinstrap": "#D55E00",
        "Gentoo": "#009E73",
    },
    hover_data=["island", "sex"],
    labels={x_column: x_label, y_column: y_label, "species": "Species"},
)
figure.update_layout(
    template="plotly_white",
    legend_title_text="Species",
    font={"family": "Arial, sans-serif", "color": "#243B53"},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin={"l": 20, "r": 20, "t": 35, "b": 20},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 1, "xanchor": "right"},
)
figure.update_traces(marker={"size": 8, "opacity": 0.8, "line": {"width": 0.5, "color": "#FFFFFF"}})
figure.update_xaxes(showgrid=True, gridcolor="#E6EBEF", zeroline=False, linecolor="#B8C5CC")
figure.update_yaxes(showgrid=True, gridcolor="#E6EBEF", zeroline=False, linecolor="#B8C5CC")

metric_columns = st.columns(3)
metric_columns[0].metric("Penguins on graph", len(chart_data))
metric_columns[1].metric(
    "Species shown",
    len(species) if species_selection == all_species_label else 1,
)
metric_columns[2].metric("Islands represented", chart_data["island"].nunique())

st.subheader(f"{y_label} vs. {x_label}")
st.plotly_chart(figure)
st.caption(
    f"Showing {len(chart_data)} of {len(data)} penguins. "
    "Rows missing either selected measurement are omitted."
)

st.divider()
st.subheader("Try a prediction")
st.write(
    "The model learned from penguins with known species. Enter measurements "
    "for one penguin to see what species it predicts."
)

measurement_columns = st.columns(4)
bill_length = measurement_columns[0].number_input(
    "Bill length (mm)", min_value=0.0, value=46.1, step=0.1
)
bill_depth = measurement_columns[1].number_input(
    "Bill depth (mm)", min_value=0.0, value=13.2, step=0.1
)
flipper_length = measurement_columns[2].number_input(
    "Flipper length (mm)", min_value=0.0, value=211.0, step=1.0
)
body_mass = measurement_columns[3].number_input(
    "Body mass (g)", min_value=0.0, value=4500.0, step=50.0
)

input_penguin = pd.DataFrame(
    [[bill_length, bill_depth, flipper_length, body_mass]],
    columns=FEATURES,
)
prediction_model = load_prediction_model()
predicted_species = prediction_model.predict(input_penguin)[0]
probabilities = prediction_model.predict_proba(input_penguin)[0]
probability_by_species = pd.Series(
    probabilities,
    index=prediction_model.classes_,
).sort_values(ascending=False)

result_columns = st.columns([1, 2])
result_columns[0].metric("Model predicts", predicted_species)
result_columns[0].caption(
    f"Estimated probability: {probability_by_species.iloc[0]:.0%}. "
    "This is the model's estimate, not a guarantee."
)
result_columns[1].caption("Estimated probability by species")
probability_figure = px.bar(
    pd.DataFrame(
        {
            "Species": probability_by_species.index,
            "Probability": probability_by_species.values,
        }
    ),
    x="Probability",
    y="Species",
    orientation="h",
    range_x=[0, 1],
    color_discrete_sequence=["#0072B2"],
)
probability_figure.update_traces(
    texttemplate="%{x:.0%}",
    textposition="outside",
    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
)
probability_figure.update_layout(
    height=210,
    margin={"l": 10, "r": 45, "t": 5, "b": 30},
    showlegend=False,
    xaxis={"title": "Estimated probability", "tickformat": ".0%", "gridcolor": "#E6EBEF"},
    yaxis={"title": None, "categoryorder": "total ascending"},
    font={"family": "Arial, sans-serif", "color": "#243B53"},
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
result_columns[1].plotly_chart(probability_figure)

with st.expander("Preview the dataset"):
    st.dataframe(data)
