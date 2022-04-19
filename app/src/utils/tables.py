# %%
import numpy as np
import pandas as pd
import streamlit as st

pd.options.mode.chained_assignment = None
pd.set_option("display.max_columns", None)
from scipy.spatial import distance_matrix

# %%
@st.cache(suppress_st_warning=True)
def excel_to_df(file: str) -> pd.DataFrame:
    "Converts excel file containing raw data to pandas dataframe."

    if file:
        df = pd.read_excel(file).set_index(0.00).replace(0, np.nan)
    else:
        df = None
    return df


# %%
@st.cache(suppress_st_warning=True)
def create_stack_df(df: pd.DataFrame) -> pd.DataFrame:

    """
    Stacks the raw data df so that x, y, & z values are in the same rows.
    This df is used to locate pit cluster and local loss coordinates.
    """

    stack_df = pd.DataFrame(df.stack())
    stack_df = (
        pd.DataFrame(stack_df.to_records())
        .rename(columns={"0.0": "y", "level_0": "x", "0": "z"})
        .sort_values(["x", "y"], ascending=True)
    )
    stack_df["x"] = stack_df["x"].astype(float)
    return stack_df


@st.cache(suppress_st_warning=True)
def convert_grid_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Converts the raw data df to xyz data and classifies each point based on the z value.
    The z value is the wall thickness at the xy coordinate.
    """

    xyz_df = (
        df.fillna(0)
        .reset_index()
        .set_index(0.00)
        .stack()
        .reset_index(name="z")
        .rename(columns={"level_1": "x", 0.0: "y"})
    )
    xyz_df["TARGET"] = None
    xyz_df["TARGET_NUMBER"] = 0
    xyz_df["x"] = xyz_df["x"].astype(float)

    for idx, row in xyz_df.iterrows():
        if row["z"] < 0.25:
            xyz_df.at[idx, "TARGET"] = "OUTLIER_VALUE"
            xyz_df.at[idx, "TARGET_NUMBER"] = 0
        elif 0.25 <= row["z"] < 0.75:
            xyz_df.at[idx, "TARGET"] = "EXTREME"
            xyz_df.at[idx, "TARGET_NUMBER"] = 5
        elif 0.75 <= row["z"] < 1.0:
            xyz_df.at[idx, "TARGET"] = "CRITICAL_LOSS"
            xyz_df.at[idx, "TARGET_NUMBER"] = 4
        elif 1.0 <= row["z"] < 1.25:
            xyz_df.at[idx, "TARGET"] = "SUBSTANTIAL_LOSS"
            xyz_df.at[idx, "TARGET_NUMBER"] = 3
        elif 1.25 <= row["z"] < 1.5:
            xyz_df.at[idx, "TARGET"] = "WALL_LOSS"
            xyz_df.at[idx, "TARGET_NUMBER"] = 2
        elif row["z"] >= 1.5:
            xyz_df.at[idx, "TARGET"] = "PRISTINE"
            xyz_df.at[idx, "TARGET_NUMBER"] = 1
        elif row["z"] == "NaN":
            xyz_df.at[idx, "TARGET"] = "NO_DATA"
            xyz_df.at[idx, "TARGET_NUMBER"] = 0.1

    xyz_df = xyz_df.sort_values(["x", "y"], ascending=[True, True])

    return xyz_df


@st.cache(suppress_st_warning=True)
def create_binary_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.array]:

    """
    Converts the xyz df and replaces the target numbers with binary values associated with xy coordinates.
    The stacked binary df is used to plot a heatmap of the vessel calling out all areas with a target number >= 3.
    The binary df and grid np array are used in locating cluster regions.
    """

    target_df = df.pivot_table(columns="x", index="y", values="TARGET_NUMBER").iloc[
        ::-1
    ]
    binary_df = target_df.replace({0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1})
    stack_df = (
        binary_df.stack()
        .reset_index(name="binary")
        .rename(columns={"level_1": "x", 0.0: "y"})
    )
    grid = binary_df.to_numpy()
    return binary_df, stack_df, grid


def create_region_df(df: pd.DataFrame, grid: np.array) -> pd.DataFrame:

    "Using the binary data, formulate region parameters for pit clusters and local loss areas."

    rows = int(len(df.index))
    cols = int(len(df.columns))
    mat = grid
    ones = set()
    for row in range(rows):
        for col in range(cols):
            if mat[row][col] == 1:
                ones.add((row, col))
    regions = []
    count = 0
    d = {
        "cluster": [],
        "size": [],
        "region": [],
        "row_min_cluster": [],
        "row_max_cluster": [],
        "column_min_cluster": [],
        "column_max_cluster": [],
    }
    while ones:
        n = ones.pop()
        region = [n]
        regions.append(region)
        unchecked = [n]
        count = count + 1
        while unchecked:
            xr, xc = unchecked.pop()
            for r in [-1, 0, 1]:
                for c in [-1, 0, 1]:
                    p = (xr + r, xc + c)
                    if p in ones:
                        ones.remove(p)
                        region.append(p)
                        unchecked.append(p)
        d["cluster"].append(count)
        d["size"].append(len(region))
        d["region"].append(region)
        d["row_min_cluster"].append(min(region, key=lambda x: x[0])[0])
        d["row_max_cluster"].append(max(region, key=lambda x: x[0])[0])
        d["column_min_cluster"].append(min(region, key=lambda x: x[1])[1])
        d["column_max_cluster"].append(max(region, key=lambda x: x[1])[1])
    regions_df = pd.DataFrame(d)
    regions_df = regions_df.sort_values("size", ascending=False)
    regions_df["area"] = regions_df["size"] * (0.25 ** 2)
    for idx, row in regions_df.iterrows():
        regions_df.at[idx, "region"] = str(regions_df.at[idx, "region"])
    return regions_df


def adapt_region_df(
    df: pd.DataFrame, binary: pd.DataFrame, grid: np.array
) -> pd.DataFrame:

    "Locate pit cluster and local loss region coordinates by associating the binary data and grid np array with the raw data."

    x_index = [float(i) for i in df.columns.tolist()]
    x_value_min = min(x_index)
    x_value_max = max(x_index)
    y_value_min = df.index.min()
    y_value_max = df.index.max()
    first_adaptation = create_region_df(binary, grid).apply(
        lambda y: y_value_max - ((y_value_max - y_value_min) * (y / len(df.index)))
        if y.name in ["row_min_cluster", "row_max_cluster"]
        else y
    )
    second_adaptation = first_adaptation.apply(
        lambda x: x_value_min + ((x_value_max - x_value_min) * (x / len(df.columns)))
        if x.name in ["column_min_cluster", "column_max_cluster"]
        else x
    )
    second_adaptation = second_adaptation.rename(
        columns={
            "row_min_cluster": "ymax",
            "row_max_cluster": "ymin",
            "column_min_cluster": "xmin",
            "column_max_cluster": "xmax",
        }
    )
    second_adaptation["s"] = second_adaptation["xmax"] - second_adaptation["xmin"]
    second_adaptation["c"] = second_adaptation["ymax"] - second_adaptation["ymin"]
    second_adaptation["xcenter"] = (
        second_adaptation["xmax"] + second_adaptation["xmin"]
    ) / 2
    second_adaptation["ycenter"] = (
        second_adaptation["ymax"] + second_adaptation["ymin"]
    ) / 2
    second_adaptation = second_adaptation.sort_values(by=["cluster"], ascending=True)
    return second_adaptation


# %%
def set_increments(df: pd.DataFrame, grid: str) -> float:

    "From the raw data, get the horizontal(x) and the vertical(y) increments of of the scans."

    axis_values = df.columns.tolist() if "x_" in grid else df.index.tolist()
    filtered_grid = [num for num in axis_values if isinstance(num, (int, float))]
    diff = [
        filtered_grid[n] - filtered_grid[n - 1] for n in range(1, len(filtered_grid))
    ]
    increments = pd.DataFrame(diff, columns=[grid])
    median_increment = round(abs(increments[grid].median()), 3)
    return median_increment


# %%
def create_summary_table(keys: list, values: list) -> pd.DataFrame:

    "Create a summary table by pairing two lists and converting to a dataframe."

    table = dict(zip(keys, values))
    df = pd.DataFrame(data=table)
    return df


def create_cfd_df(df: pd.DataFrame, step_change: int) -> pd.DataFrame:

    """
    Convert the raw data into a grid block.
    Determine the average wall thickness and the associated CFD value.
    Create a new df containing the average wall thickness values and CFD values.
    """

    old_cols = list(df.columns)
    new_cols = list(range(0, len(df.index), step_change))
    d = {}
    for k, v in zip(old_cols, new_cols):
        d[k] = v

    grid_block_df = df.reset_index(drop=True)
    grid_block_df.rename(columns=d, inplace=True)
    grid_block_df = grid_block_df.reset_index(drop=True)

    value_list = grid_block_df.values.flatten().tolist()
    cfd_df = (
        pd.DataFrame(value_list, columns=["average_value"])
        .dropna()
        .sort_values("average_value", ascending=True)
        .reset_index()
        .drop("index", axis=1)
    )
    cfd_df["CFD"] = cfd_df.index / (len(cfd_df) - 1)

    return cfd_df


def find_local_zone(
    cluster_areas: pd.DataFrame, stack: pd.DataFrame, cluster_number: int
) -> pd.DataFrame:

    """
    Locates a specific cluster in the cluster area df using the assigned cluster number and assign the cluster data to a new df.
    Using the x & y values from the cluster area df, find the associated values in the stacked df to form a localized xyz df for the zone the cluster is in.
    """

    df = cluster_areas[cluster_areas["cluster"] == cluster_number]
    local_zone_df = stack[
        (stack["x"] >= df["xmin"].iloc[0])
        & (stack["x"] <= df["xmax"].iloc[0])
        & (stack["y"] >= df["ymin"].iloc[0])
        & (stack["y"] <= df["ymax"].iloc[0])
    ]
    return local_zone_df[["x", "y", "z"]]


def create_cluster_df(
    cluster_areas: pd.DataFrame,
    stack: pd.DataFrame,
    minimum_required_thickness: float,
    future_corrosion_allowance: float,
    inside_diameter: float,
    measured_length_to_discontinuity: float,
    design_condition_pressure: float,
    rsfa: float,
    calc: classmethod,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list, int
]:

    """
    Adds various parameters to the cluster area df based on problem class.
    Separates pitting, local loss, pit couple, and problem area dfs to be used as visualization within the app.
    Also returns df which contains all cluster data.
    """

    problem_classes = ["PITTING", "LOCAL_LOSS"]
    df = cluster_areas
    df["problem_class"] = None
    df["t_mm"] = None
    df["R_t"] = None
    df["lambda"] = None
    df["M_t"] = None
    df["RSF"] = None
    df["c/d"] = None
    df["limiting_flaw_size_criterium_1"] = None
    df["limiting_flaw_size_criterium_2"] = None
    df["limiting_flaw_size_criterium_3"] = None
    df["critical_groove_radius"] = None
    df["groove_depth"] = None
    df["groove_criterium_1"] = None
    df["groove_criterium_2"] = None
    df["MAWPr"] = None

    for idx, row in df.iterrows():
        if row["area"] >= 1:
            df.at[idx, "problem_class"] = "LOCAL_LOSS"
        elif 1 >= row["area"] > 0.0625:
            df.at[idx, "problem_class"] = "PITTING"
        elif row["size"] == 1:
            df.at[idx, "problem_class"] = "OUTLIER"

    for idx, row in df.iterrows():
        cluster = calc.cluster(
            df[df["problem_class"].isin(problem_classes)], (df.at[idx, "cluster"])
        )
        s_local_loss = calc.s_local_loss(cluster)
        c_local_loss = calc.c_local_loss(cluster)
        minimum_measured_thickness_local = calc.minimum_measured_thickness_local(
            find_local_zone(cluster_areas, stack, row["cluster"])
        )
        remaining_thickness_ratio_local = calc.remaining_thickness_ratio_local(
            minimum_measured_thickness_local,
            future_corrosion_allowance,
            minimum_required_thickness,
        )
        shell_parameter_local = calc.shell_parameter_local(
            s_local_loss, inside_diameter, minimum_required_thickness
        )
        mt_local = calc.mt_local(s_local_loss, shell_parameter_local)
        rsf_local = calc.rsf_local(
            s_local_loss, remaining_thickness_ratio_local, mt_local
        )
        MAWPr = calc.MAWPr(design_condition_pressure, rsf_local, rsfa)
        c_over_d = calc.c_over_d(c_local_loss, inside_diameter)
        limiting_flaw_size_criterium_1 = calc.limiting_flaw_size_criterium_1(
            remaining_thickness_ratio_local
        )
        limiting_flaw_size_criterium_2 = calc.limiting_flaw_size_criterium_2(
            minimum_measured_thickness_local
        )
        limiting_flaw_size_criterium_3 = calc.limiting_flaw_size_criterium_3(
            measured_length_to_discontinuity,
            inside_diameter,
            minimum_required_thickness,
        )
        critical_groove_radius = calc.critical_groove_radius(minimum_required_thickness)
        groove_length = calc.groove_length(s_local_loss)
        groove_width = calc.groove_width(c_local_loss)
        groove_depth = calc.groove_depth(minimum_measured_thickness_local)
        groove_radius = calc.groove_radius(groove_depth)
        groove_criterium_1 = calc.groove_criterium_1(
            groove_radius, critical_groove_radius
        )
        groove_criterium_2 = calc.groove_criterium_2(
            groove_radius, remaining_thickness_ratio_local, minimum_required_thickness
        )

        if row["s"] == 0.0:
            df.at[idx, "problem_class"] = "OUTLIER"

        if row["problem_class"] == "PITTING":
            df.at[idx, "t_mm"] = minimum_measured_thickness_local
            df.at[idx, "R_t"] = remaining_thickness_ratio_local
            df.at[idx, "lambda"] = shell_parameter_local
            df.at[idx, "M_t"] = mt_local
            df.at[idx, "RSF"] = rsf_local
            df.at[idx, "MAWPr"] = MAWPr

        if row["problem_class"] == "LOCAL_LOSS":
            df.at[idx, "t_mm"] = minimum_measured_thickness_local
            df.at[idx, "R_t"] = remaining_thickness_ratio_local
            df.at[idx, "lambda"] = shell_parameter_local
            df.at[idx, "M_t"] = mt_local
            df.at[idx, "RSF"] = rsf_local
            df.at[idx, "MAWPr"] = MAWPr
            df.at[idx, "c/d"] = c_over_d
            df.at[
                idx, "limiting_flaw_size_criterium_1"
            ] = limiting_flaw_size_criterium_1
            df.at[
                idx, "limiting_flaw_size_criterium_2"
            ] = limiting_flaw_size_criterium_2
            df.at[
                idx, "limiting_flaw_size_criterium_3"
            ] = limiting_flaw_size_criterium_3
            df.at[idx, "critical_groove_radius"] = critical_groove_radius
            df.at[idx, "groove_length"] = groove_length
            df.at[idx, "groove_width"] = groove_width
            df.at[idx, "groove_depth"] = groove_depth
            df.at[idx, "groove_radius"] = groove_radius
            df.at[idx, "groove_criterium_1"] = groove_criterium_1
            df.at[idx, "groove_criterium_2"] = groove_criterium_2

    problem_areas_df = df[["cluster", "problem_class", "area", "xcenter", "ycenter"]]
    local_loss_df = df[df["problem_class"] == "LOCAL_LOSS"]
    pitting_df = df[df["problem_class"] == "PITTING"]
    pit_centers = pitting_df[["cluster", "xcenter", "ycenter"]].set_index("cluster")
    cluster_matrices = pd.DataFrame(
        distance_matrix(pit_centers.values, pit_centers.values),
        index=pit_centers.index,
        columns=pit_centers.index,
    )
    pit_couples = pd.DataFrame(cluster_matrices.stack())
    pit_couples = pit_couples.rename(columns={0: "distance"})
    pit_couples = pit_couples.reset_index(level=0)
    pit_couples = pit_couples.rename(columns={"cluster": "cluster2"})
    pit_couples = pd.DataFrame(pit_couples.to_records())
    pit_couples = (
        pit_couples[pit_couples["distance"] > 0].sort_values("distance").reset_index()
    )
    cluster_list = df["cluster"].tolist()
    number_of_pits = len(cluster_list)
    return (
        df,
        pitting_df,
        local_loss_df,
        pit_couples,
        problem_areas_df,
        cluster_list,
        number_of_pits,
    )


def get_min_clusters(
    pit_couples: pd.DataFrame,
    calc: classmethod,
    clusters: pd.DataFrame,
    xyz: pd.DataFrame,
    current_wall_thickness: float,
    corrosion_allowance_const: float,
    minimum_required_thickness: float,
    strength_factor_const: float,
    radius_circ: float,
) -> pd.DataFrame:

    """
    Locates cluster couples under the minimum distance threshold and moves them to a separate df.
    Claculates various parameters of each cluster and adds to the new df previously created.
    """

    first_cluster = []
    second_cluster = []
    distances = []
    for row in pit_couples.itertuples():
        if (
            (row[2] not in first_cluster)
            and (row[3] not in first_cluster)
            and (row[2] not in second_cluster)
            and (row[3] not in second_cluster)
        ):
            first_cluster.append(row[2])
            second_cluster.append(row[3])
            distances.append(row[4])
    df = pd.DataFrame(
        {
            "first_cluster": first_cluster,
            "second_cluster": second_cluster,
            "distance": distances,
        }
    )
    df["wi"] = None
    df["wj"] = None
    df["di"] = None
    df["dj"] = None

    for idx, row in df.iterrows():
        df.at[idx, "wi"] = calc.w_pit(clusters, df.at[idx, "first_cluster"], xyz)
        df.at[idx, "wj"] = calc.w_pit(clusters, df.at[idx, "second_cluster"], xyz)

        first_pass = clusters[clusters["cluster"] == df.at[idx, "first_cluster"]]
        df.at[idx, "di"] = max(first_pass["s"].iloc[0], first_pass["c"].iloc[0])

        second_pass = clusters[clusters["cluster"] == df.at[idx, "second_cluster"]]
        df.at[idx, "dj"] = max(second_pass["s"].iloc[0], second_pass["c"].iloc[0])

    df["actual_depth_i"] = df["wi"] - (
        current_wall_thickness - corrosion_allowance_const - minimum_required_thickness
    )
    df["actual_depth_j"] = df["wj"] - (
        current_wall_thickness - corrosion_allowance_const - minimum_required_thickness
    )
    df["average_depth"] = (df["actual_depth_i"] + df["actual_depth_j"]) / 2
    df["average_diameter"] = (df["di"] + df["dj"]) / 2
    df["remaining_thickness_ratio_i"] = (
        minimum_required_thickness - df["actual_depth_i"] - corrosion_allowance_const
    ) / minimum_required_thickness

    for idx, row in df.iterrows():
        if df.at[idx, "remaining_thickness_ratio_i"] < 0:
            df.at[idx, "remaining_thickness_ratio_i"] = 0

    df["remaining_thickness_ratio_j"] = (
        minimum_required_thickness - df["actual_depth_j"] - corrosion_allowance_const
    ) / minimum_required_thickness

    for idx, row in df.iterrows():
        if df.at[idx, "remaining_thickness_ratio_j"] < 0:
            df.at[idx, "remaining_thickness_ratio_j"] = 0

    df["Q_i"] = 1.123 * (
        (
            (
                (
                    (1 - df["remaining_thickness_ratio_i"])
                    / (1 - df["remaining_thickness_ratio_i"] / strength_factor_const)
                )
                ** 2
            )
            - 1
        )
        ** 0.5
    )
    df["Q_j"] = 1.123 * (
        (
            (
                (
                    (1 - df["remaining_thickness_ratio_j"])
                    / (1 - df["remaining_thickness_ratio_j"] / strength_factor_const)
                )
                ** 2
            )
            - 1
        )
        ** 0.5
    )
    df["Q_i*sqrt(id*tmin)"] = df["Q_i"] * (
        (radius_circ * minimum_required_thickness) ** 0.5
    )
    df["Q_j*sqrt(id*tmin)"] = df["Q_j"] * (
        (radius_circ * minimum_required_thickness) ** 0.5
    )
    df["cond1_pit_width_i"] = df["wi"] <= df["Q_i*sqrt(id*tmin)"]
    df["cond1_pit_width_j"] = df["wj"] <= df["Q_j*sqrt(id*tmin)"]
    df["cond2_pit_depth_i"] = df["di"] >= 0.2
    df["cond2_pit_depth_j"] = df["dj"] >= 0.2

    return df


# %%
def create_local_zone_df(
    cluster_number: int, cluster_areas: pd.DataFrame, stack: pd.DataFrame
) -> pd.DataFrame:

    """
    Identifies areas of local loss and creates a new xyz df.
    Using the new xyz df, a grid df is created with the associated longitudinal CTP and horizontal ctp.
    """

    df = cluster_areas[cluster_areas["cluster"] == cluster_number]
    local_zone = stack[
        (stack["x"] >= df["xmin"].iloc[0])
        & (stack["x"] <= df["xmax"].iloc[0])
        & (stack["y"] >= df["ymin"].iloc[0])
        & (stack["y"] <= df["ymax"].iloc[0])
    ]
    local_zone = local_zone[["x", "y", "z"]]
    grid = local_zone.drop_duplicates().pivot_table(columns="x", index="y", values="z")
    grid.loc["Longitudinal CTP"] = grid.min()
    grid["Circumferential CTP"] = grid.min(axis=1)
    return grid


# %%
def create_ctp_df(zone: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Using the local loss zone df, create to separate dfs. Fisrt one for longitudinal ctp, and the second one for
    horizontal ctp.
    """

    long_df = pd.DataFrame(zone.loc["Longitudinal CTP"])
    long_df.columns = long_df.columns.get_level_values(0)
    long_df = long_df.reset_index()
    long_df = long_df.drop(long_df.tail(1).index, inplace=False)

    circ_df = zone.iloc[:, -1:].reset_index()
    circ_df = circ_df.drop(circ_df.tail(1).index, inplace=False)
    return long_df, circ_df
