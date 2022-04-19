import pandas as pd
import plotly.graph_objects as plty
import plotly.express as px
import streamlit as st

from utils.tables import create_cfd_df


translucent = "rgba(240,255,255,0.0000001)"

my_pal = {
    "EXTREME": "red",
    "CRITICAL_LOSS": "orange",
    "SUBSTANTIAL_LOSS": "yellow",
    "WALL_LOSS": "lightgreen",
    "PRISTINE": "green",
    "none": translucent,
    "OUTLIER_VALUE": "black",
}


color_scale = [
    (0.0, translucent),
    (0.2, "red"),
    (0.5, "orange"),
    (0.7, "yellow"),
    (0.9, "green"),
    (1, "darkgreen"),
]


@st.cache(suppress_st_warning=True)
def block_grid_heatmap(xyz: pd.DataFrame) -> plty.Figure:

    "Plots a heatmap the xyz data."

    fig = px.scatter(
        data_frame=xyz,
        x="x",
        y="y",
        color="z",
        title="HP Sep Wall Thickness (in)",
        range_color=[0.0, 1.6],
        color_continuous_scale=color_scale,
        height=725,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        range=[xyz["x"].min(), xyz["x"].max()],
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
    )
    fig.update_yaxes(
        range=[xyz["y"].min(), xyz["y"].max()],
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
    )
    return fig


@st.cache(suppress_st_warning=True)
def problem_area_heatmap(
    xyz: pd.DataFrame,
    problem_areas: pd.DataFrame,
    title: str = None,
    selection: bool = False,
    selection_title: list = None,
    selected_clusters: pd.DataFrame = None,
    couple: bool = False,
    couples: pd.DataFrame = None,
) -> plty.Figure:

    """
    Plots a heatmap the xyz data.
    Adds annotations to the heatmap that call out problem areas
    """

    if selection:
        title = "Selected Clusters:" + " " + ", ".join(str(i) for i in selection_title)
        if couple:
            first_cluster = problem_areas.loc[
                problem_areas.cluster == couples.first_cluster.values[0]
            ]
            second_cluster = problem_areas.loc[
                problem_areas.cluster == couples.second_cluster.values[0]
            ]
            problem_areas = pd.concat([first_cluster, second_cluster])
        else:
            problem_areas = problem_areas.loc[selected_clusters]
    fig = px.scatter(
        xyz,
        x="x",
        y="y",
        color="z",
        title=title,
        range_color=[0.0, 1.6],
        color_continuous_scale=color_scale,
        height=725,
    )
    for _, row in problem_areas.iterrows():
        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=row["xmin"] - 1,
            y0=row["ymin"] - 2,
            x1=row["xmax"] + 1,
            y1=row["ymax"] + 2,
            opacity=0.8,
            line_color="black",
        )
        fig.add_annotation(
            x=row["xcenter"],
            y=row["ycenter"],
            xref="x",
            yref="y",
            text=row["cluster"],
            font=dict(size=12, color="#B8B8B8"),
            arrowcolor="black",
        )
    fig.update_layout(showlegend=False,)
    fig.update_xaxes(
        range=[xyz["x"].min(), xyz["x"].max()],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
    )
    fig.update_yaxes(
        range=[xyz["y"].min(), xyz["y"].max()],
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
    )

    return fig


@st.cache(suppress_st_warning=True)
def ctp_graph(
    local_loss: pd.DataFrame,
    cluster_idx: int,
    ctp_type: pd.DataFrame,
    minimum_required_thickness: float,
    title: str,
    xaxis: str,
    yaxis: str,
    yaxis_title: str,
) -> plty.Figure:

    "Creates a ctp plot from the local loss data for each ctp type."

    fig = px.scatter(ctp_type, x=xaxis, y=yaxis)
    fig.data[0].update(mode="markers+lines")
    fig.add_shape(
        type="line",
        x0=ctp_type[xaxis].min(),
        y0=minimum_required_thickness,
        x1=ctp_type[xaxis].max(),
        y1=minimum_required_thickness,
        line=dict(color="Red",),
        xref="x",
        yref="y",
    )
    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        yaxis_title=yaxis_title,
        legend_title="Legend Title",
    )

    fig.update_xaxes(
        range=[
            local_loss["xmin"].iloc[cluster_idx],
            local_loss["xmax"].iloc[cluster_idx],
        ],
        zeroline=False,
    )
    fig.update_yaxes(range=[0, 1.5])
    return fig


@st.cache(suppress_st_warning=True)
def binary_heat_map(binary: pd.DataFrame) -> plty.Figure:

    "Creates a binary heatmap using the stacked binary data."

    fig = px.scatter(
        data_frame=binary,
        x="x",
        y="y",
        color="binary",
        title="Binary Wall Thickness",
        range_color=[0, 1],
        color_continuous_scale=[
            (0.0, "green"),
            (0.5, "green"),
            (0.5, "red"),
            (1, "red"),
        ],
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
        title_text="x",
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="gray", gridcolor="gray", title_text="y",
    )
    return fig


@st.cache(suppress_st_warning=True)
def prob_graph(df: pd.DataFrame, step_change: int) -> plty.Figure:

    """
    Creates a cfd plot by converting the raw data into a grid block and determining
    the average wall thickness and the associated CFD value.
    """

    data = create_cfd_df(df, step_change)
    fig = px.line(
        data_frame=data, x="average_value", y="CFD", title="CFD vs. Wall Thickness"
    )
    fig.update_traces(line_color="orange")
    fig.update_xaxes(
        range=[data["average_value"].min(), data["average_value"].max()],
        zeroline=False,
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
        title_text="Wall Thickness (in)",
    )
    fig.update_yaxes(
        range=[data["CFD"].min(), data["CFD"].max()],
        showline=True,
        linewidth=1,
        linecolor="gray",
        gridcolor="gray",
        title_text="CFD",
    )
    return fig
