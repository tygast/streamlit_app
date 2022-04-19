# %%
import streamlit as st

st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_page_config(layout="wide")

from templates.plots import (
    block_grid_heatmap,
    problem_area_heatmap,
    ctp_graph,
    binary_heat_map,
    prob_graph,
)
from utils.calculations import (
    ClusterCalcs,
    LocalLossCalcs,
    SeparatorAttrs,
)
from utils.constants import (
    DESIGN_COND_PRESSURE,
    DESIGN_COND_TEMP,
    INSIDE_DIAMETER,
    FUTURE_CORROSION_ALLOWANCE,
    WELD_JOINT_EFFICIENCY,
    ALLOWABLE_TENSILE_STRESS,
    RSFa,
    NOMINAL_WALL_THICKNESS,
)
from utils.tables import (
    excel_to_df,
    create_stack_df,
    create_binary_df,
    adapt_region_df,
    convert_grid_data,
    set_increments,
    create_summary_table,
    create_cluster_df,
    get_min_clusters,
    create_local_zone_df,
    create_ctp_df,
)

from utils.reports import convert_to_image, PDF

# %%
st.markdown(
    "<h1 style='text-align: center;'>AUT Insepction App</h1> <p style='text-align: center;'><i>Data Web app used to compute HPS integrity</i></p>",
    unsafe_allow_html=True,
)

with st.sidebar.expander("Inspection Info"):
    inspector_name = st.text_input("Enter Inspector Name")
    inspection_date = st.date_input("Enter Inspection Date")
    vessel_id = st.number_input("Enter Vessel ID", min_value=0, step=1, value=10000)
    vessel_sn = st.number_input(
        "Enter Manufacturing SN", min_value=0, step=1, value=141000000
    )
    vessel_start_date = st.date_input("Enter Vessel Start Date")

with st.sidebar.expander("Manual Design Inputs"):
    future_corrosion_allowance = st.number_input(
        "Enter Future Corrosion Allowance (in)",
        min_value=0.0,
        step=0.05,
        value=FUTURE_CORROSION_ALLOWANCE,
    )
    design_cond_pressure = st.number_input(
        "Enter Design Condition Pressure (psi)",
        min_value=0,
        step=10,
        value=DESIGN_COND_PRESSURE,
    )
    rsfa = st.number_input(
        "Enter Allowable Remaining Strength Factor",
        min_value=0.0,
        step=0.05,
        value=RSFa,
    )
    inside_diameter = st.number_input(
        "Enter Vessel Inside Diameter (in)", min_value=0, step=12, value=INSIDE_DIAMETER
    )
    weld_joint_efficiency = st.number_input(
        "Enter Weld Joint Efficiency)",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        value=WELD_JOINT_EFFICIENCY,
    )
    measured_length_to_discontinuity = st.number_input(
        "Enter Measured Length to Nearest Discontinuity (in)",
        min_value=0,
        step=1,
        value=36,
    )

excel_file = st.sidebar.file_uploader(
    label="Excel File Upload", type="xlsx", accept_multiple_files=False
)

if not excel_file:
    st.warning("Upload a data file to begin AUT inspection analysis.")
else:
    # %%
    cc = ClusterCalcs()
    llc = LocalLossCalcs()
    sa = SeparatorAttrs()

    # %%
    raw_data = excel_to_df(excel_file)
    stacked_data = create_stack_df(raw_data)

    # %%
    x_median_increment, y_median_increment = (
        set_increments(raw_data, "x_increments"),
        set_increments(raw_data, "y_increments"),
    )

    # %%
    xyz_data = convert_grid_data(raw_data)
    binary_data, stack, grid = create_binary_df(xyz_data)
    cluster_area_data = adapt_region_df(raw_data, binary_data, grid)

    # %%
    wall_loss = sa.wall_loss(raw_data)
    total_number_of_scans = sa.total_scans(wall_loss)
    absolute_minimum_reading = sa.absolute_minimum_reading(xyz_data)
    average_wall_thickness = sa.average_wall_thickness(xyz_data)
    average_wall_loss = sa.average_wall_loss(wall_loss)
    uniform_metal_loss = sa.uniform_metal_loss(average_wall_loss)
    t = sa.t(average_wall_loss)
    radius_circ = sa.radius_circ(
        inside_diameter, uniform_metal_loss, future_corrosion_allowance
    )
    minimum_required_thickness_circ = sa.minimum_required_thickness_circ(
        design_cond_pressure, radius_circ, weld_joint_efficiency
    )
    minimum_required_thickness_long = sa.minimum_required_thickness_long(
        design_cond_pressure, radius_circ, weld_joint_efficiency
    )
    minimum_required_thickness = sa.minimum_required_thickness(
        minimum_required_thickness_circ, minimum_required_thickness_long
    )

    # %%
    (
        cluster_data,
        pitting_data,
        local_loss_data,
        pit_couples,
        problem_areas,
        cluster_list,
        number_of_pits,
    ) = create_cluster_df(
        cluster_area_data,
        stacked_data,
        minimum_required_thickness,
        future_corrosion_allowance,
        inside_diameter,
        measured_length_to_discontinuity,
        design_cond_pressure,
        rsfa,
        llc,
    )

    # %%
    min_cluster_table = get_min_clusters(
        pit_couples,
        cc,
        cluster_data,
        xyz_data,
        t,
        future_corrosion_allowance,
        minimum_required_thickness,
        rsfa,
        radius_circ,
    )
    # %%
    rsf = cc.rsf(
        min_cluster_table, minimum_required_thickness, t, future_corrosion_allowance
    )
    MAWPr = cc.MAWPr(design_cond_pressure, rsf)
    pit_status = cc.pit_status(MAWPr, design_cond_pressure)

    min_cluster_table.insert(
        0, "pit_couple", list(range(1, len(min_cluster_table) + 1))
    )
    # %%
    sep_params_table = create_summary_table(
        ["Parameter", "Value"],
        [
            [
                "Pristine Wall Thickness (in)",
                "Future Corrosion Allowance (in)",
                "Weld Joint Efficiency",
                "Allowable Tensile Stress",
                "Allowable Remaining Strength Factor",
                "Inside Diameter (in)",
                "Design Condition Pressure (psig)",
                "Design Condition Temperature (F)",
            ],
            [
                NOMINAL_WALL_THICKNESS,
                future_corrosion_allowance,
                weld_joint_efficiency,
                ALLOWABLE_TENSILE_STRESS,
                rsfa,
                inside_diameter,
                design_cond_pressure,
                DESIGN_COND_TEMP,
            ],
        ],
    )

    # %%
    st.write(
        """
    ## General Info
    """
    )
    gen_met1, gen_met2, gen_met3, gen_met4, gen_met5, gen_met6 = st.columns(6)
    gen_met1.metric(label="Scans", value=round(total_number_of_scans, 3), delta=None)
    gen_met2.metric(
        label="Horizontal Increments (in)",
        value=round(x_median_increment, 3),
        delta=None,
    )
    gen_met3.metric(
        label="Vertical Increments (in)", value=round(y_median_increment, 3), delta=None
    )
    gen_met4.metric(
        label="Min Req Thickness (circ)",
        value=round(minimum_required_thickness_circ, 3),
        delta=None,
    )
    gen_met5.metric(
        label="Min Req Thickness (long)",
        value=round(minimum_required_thickness_long, 3),
        delta=None,
    )
    gen_met6.metric(
        label="Min Req Thickness",
        value=round(minimum_required_thickness, 3),
        delta=None,
    )

    sp_chart, sp_table = st.columns([1, 1])
    sp_chart.plotly_chart(block_grid_heatmap(xyz_data), use_container_width=True)
    sp_table.write("")
    sp_table.write("")
    sp_table.write("")
    sp_table.write("")
    sp_table.write("")
    sp_table.write("")
    sp_table.write(sep_params_table)

    st.write(
        """
    ## Categorizing Problem Areas
    """
    )
    st.markdown(
        "_The table below shows current distinction between **Local Metal Loss** and **Pitting**_"
    )

    pa_chart, pa_table = st.columns([1, 1])
    selected_indices = pa_table.multiselect(
        label="",
        options=problem_areas.cluster,
        format_func=lambda opt: "Cluster " + str(opt),
    )
    if selected_indices:
        cluster_idx = [
            problem_areas.loc[problem_areas.cluster == i].index[0]
            for i in selected_indices
        ]
        selected_rows = problem_areas.loc[cluster_idx]
        pa_chart.plotly_chart(
            problem_area_heatmap(
                xyz_data,
                cluster_area_data,
                selection=True,
                selection_title=selected_indices,
                selected_clusters=cluster_idx,
            ),
            use_container_width=True,
        )
        pa_table.dataframe(selected_rows)

    else:
        pa_chart.plotly_chart(
            problem_area_heatmap(xyz_data, cluster_area_data, "All Problem Areas"),
            use_container_width=True,
        )
        pa_table.dataframe(problem_areas)

    st.write(
        """
    ## Local Metal Loss
    """
    )
    if not local_loss_data.empty:
        ll_chart, ll_table = st.columns([1, 1])
        cluster_number = ll_table.multiselect(
            label="",
            options=local_loss_data.cluster,
            format_func=lambda opt: "Cluster " + str(opt),
        )
        if cluster_number:
            cluster_idx = [
                local_loss_data.loc[local_loss_data.cluster == i].index[0]
                for i in cluster_number
            ]
            selected_rows = local_loss_data.loc[cluster_idx]
            ll_chart.plotly_chart(
                problem_area_heatmap(
                    xyz_data,
                    local_loss_data,
                    selection=True,
                    selection_title=cluster_number,
                    selected_clusters=cluster_idx,
                ),
                use_container_width=True,
            )
            ll_table.dataframe(selected_rows)

            local_loss_zones = create_local_zone_df(
                cluster_number[0], cluster_area_data, stacked_data
            )
            ctp_long, ctp_circ = create_ctp_df(local_loss_zones)
            sub_chart1, sub_chart2 = st.columns(2)
            sub_chart1.plotly_chart(
                ctp_graph(
                    local_loss_data,
                    cluster_idx,
                    ctp_long,
                    minimum_required_thickness,
                    "Longitudinal CTP inches",
                    "x",
                    "Longitudinal CTP",
                    "Long Min Wall Thickness (in)",
                ),
                use_container_width=True,
            )
            sub_chart2.plotly_chart(
                ctp_graph(
                    local_loss_data,
                    cluster_idx,
                    ctp_circ,
                    minimum_required_thickness,
                    "Circumferential CTP inches",
                    "y",
                    "Circumferential CTP",
                    "Circ Min Wall Thickness (in)",
                ),
                use_container_width=True,
            )
        else:
            ll_chart.plotly_chart(
                problem_area_heatmap(
                    xyz_data, local_loss_data, "Local Metal Loss Areas"
                ),
                use_container_width=True,
            )
            ll_table.write(local_loss_data)
    else:
        st.markdown("<i>No local metal loss</i> :sunglasses:", unsafe_allow_html=True)

    st.write(
        """
    ## Pitting LTA
    """
    )
    pitting_data

    st.write(
        """
    ## Pit Couples
    """
    )

    pc_chart, pc_table = st.columns([1, 1])
    selected_indices = pc_table.multiselect(
        label="",
        options=min_cluster_table.pit_couple,
        format_func=lambda opt: "Pit Couple " + str(opt),
    )
    if selected_indices:
        cluster_idx = [
            min_cluster_table.loc[min_cluster_table.pit_couple == i].index[0]
            for i in selected_indices
        ]
        selected_rows = min_cluster_table.loc[cluster_idx]
        pc_table.dataframe(selected_rows)
        pc_chart.plotly_chart(
            problem_area_heatmap(
                xyz_data,
                pitting_data,
                selection=True,
                selection_title=selected_indices,
                selected_clusters=None,
                couple=True,
                couple_table=selected_rows,
            ),
            use_container_width=True,
        )

    else:
        pc_table.dataframe(min_cluster_table)
        pc_chart.plotly_chart(
            problem_area_heatmap(xyz_data, pitting_data, "Pitting Areas"),
            use_container_width=True,
        )

    pit_met1, pit_met2, pit_met3 = st.columns(3)
    pit_met1.metric(
        label="Remaining Strength Factor",
        value=round(rsf, 3) if rsf > 0 else 0,
        delta=None,
    )
    # print(f"rsf: {rsf}\nMAWPr: {MAWPr}\nrsfa: {rsfa}")
    pit_met2.metric(
        label="Maximum Allowable Working Pressure",
        value=round(MAWPr, 3) if MAWPr > 0 else 0,
        delta=None,
    )
    pit_met3.metric(
        label="Overall Pit Status", value=pit_status, delta=None,
    )

    st.write(
        """
    ## Additional
    """
    )
    bin_col, cfd_col = st.columns(2)
    bin_col.plotly_chart(binary_heat_map(stack), use_container_width=True)
    cfd_col.plotly_chart(prob_graph(raw_data, 1), use_container_width=True)

y_values = list(range(0, 297, 9))
x_values = list(range(0, 210, 10))
font_color = [131, 139, 139]

if st.sidebar.button("Results Summary"):
    pdf = PDF()
    pdf.add_page()
    pdf.titles(f"{inspection_date} AUT INSPECTION REPORT: {int(vessel_id)}")
    pdf.texts(
        txt=f"Vessel Manufacturing SN: {int(vessel_sn)}",
        x=10,
        y=30,
        color=font_color,
        weight="B",
        size=8,
    )
    pdf.texts(
        txt=f"Vessel Start Date: {vessel_start_date}",
        x=10,
        y=33.5,
        color=font_color,
        weight="B",
        size=8,
    )
    pdf.texts(txt="Scans", x=10, y=44, color=font_color, weight="", size=8)
    pdf.texts(
        txt=f"{int(total_number_of_scans)}",
        x=10,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Horizontal Increments (in)",
        x=30,
        y=44,
        color=font_color,
        weight="",
        size=8,
    )
    pdf.texts(
        txt=f"{round(x_median_increment, 3)}",
        x=30,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Vertical Increments (in)", x=67, y=44, color=font_color, weight="", size=8
    )
    pdf.texts(
        txt=f"{round(y_median_increment, 3)}",
        x=67,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Min Req Thickness (circ)", x=101, y=44, color=font_color, weight="", size=8
    )
    pdf.texts(
        txt=f"{round(minimum_required_thickness_circ, 3)}",
        x=101,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Min Req Thickness (long)", x=137, y=44, color=font_color, weight="", size=8
    )
    pdf.texts(
        txt=f"{round(minimum_required_thickness_long, 3)}",
        x=137,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(txt="Min Req Thickness", x=174, y=44, color=font_color, weight="", size=8)
    pdf.texts(
        txt=f"{round(minimum_required_thickness, 3)}",
        x=174,
        y=48.5,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.img(
        x=8,
        y=52,
        w=120,
        h=80,
        pltx=convert_to_image(file="bg_heatmap.png", pltx=block_grid_heatmap(xyz_data)),
    )
    pdf.img(
        x=130,
        y=66,
        w=70,
        h=45,
        tablex=convert_to_image(file="sp_table.png", tablex=sep_params_table),
    )
    pdf.texts(
        txt="Remaining Strength Factor",
        x=10,
        y=135.5,
        color=font_color,
        weight="",
        size=8,
    )
    pdf.texts(
        txt=f"{round(rsf, 3) if rsf > 0 else 0}",
        x=10,
        y=140,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Maximum Allowable Working Pressure",
        x=80,
        y=135.5,
        color=font_color,
        weight="",
        size=8,
    )
    pdf.texts(
        txt=f"{round(MAWPr, 3) if MAWPr > 0 else 0}",
        x=80,
        y=140,
        color=font_color,
        weight="",
        size=14,
    )
    pdf.texts(
        txt="Overall Pit Status", x=165, y=135.5, color=font_color, weight="", size=8
    )
    pdf.texts(txt=f"{pit_status}", x=165, y=140, color=font_color, weight="", size=14)
    pdf.img(
        x=8,
        y=148,
        w=100,
        h=80,
        pltx=convert_to_image(file="bin_heatmap.png", pltx=binary_heat_map(stack)),
    )
    pdf.img(
        x=110,
        y=148,
        w=100,
        h=80,
        pltx=convert_to_image(file="prob_graph.png", pltx=prob_graph(raw_data, 1)),
    )
    pdf.set_author(inspector_name)
    pdf.output(f"{inspection_date}_AUT_INSPECTION_REPORT_{int(vessel_id)}.pdf", "F")
