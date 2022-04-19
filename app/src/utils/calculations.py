import pandas as pd
from typing import Callable, Union
from utils.constants import (
    ALLOWABLE_TENSILE_STRESS,
    NOMINAL_WALL_THICKNESS,
    RSFa,
)


class ClusterCalcs:

    """
    A set of calculations using the dfs for clusters, cluster couples under the minimum distance threshold, and xyz to
    determine pit dimensions, the remaining strength factor, maximum allowable working pressure and whether the
    vessel is fit for continued use.
    """

    def w_pit(
        self, clusters: pd.DataFrame, cluster_number: int, xyz: pd.DataFrame
    ) -> float:
        clusters = clusters[clusters["cluster"] == cluster_number]
        local_zones = xyz[
            (xyz["x"] >= clusters["xmin"].iloc[0])
            & (xyz["x"] <= clusters["xmax"].iloc[0])
            & (xyz["y"] >= clusters["ymin"].iloc[0])
            & (xyz["y"] <= clusters["ymax"].iloc[0])
        ]
        min_thickness = local_zones[["x", "y", "z"]]["z"].min()
        return NOMINAL_WALL_THICKNESS - min_thickness

    def rsf(
        self,
        min_clusters: pd.DataFrame,
        minimum_required_thickness: float,
        current_wall_thickness: float,
        future_corrosion_allowance: float,
    ) -> int:
        avg_all_pits_depth = min_clusters["average_depth"].mean()
        avg_all_pits_diameter = min_clusters["average_diameter"].mean()
        avg_pit_couple_spacing = min_clusters["distance"].quantile(0.5)
        mavg = (avg_pit_couple_spacing - avg_all_pits_diameter) / avg_pit_couple_spacing
        if mavg < 0:
            mavg = 0
        eavg = ((3 ** 0.5) * mavg) / 2
        rsf = min(
            1
            - (avg_all_pits_depth / minimum_required_thickness)
            + (
                (
                    eavg
                    * (
                        current_wall_thickness
                        - future_corrosion_allowance
                        + avg_all_pits_depth
                        - minimum_required_thickness
                    )
                )
                / minimum_required_thickness
            ),
            1,
        )
        return rsf

    def MAWPr(self, design_cond_pressure, rsf) -> float:
        return design_cond_pressure * (rsf / RSFa)

    def pit_status(self, MAWPr, design_cond_pressure) -> str:
        if MAWPr < design_cond_pressure:
            pit_status = "UNACCEPTABLE"
        else:
            pit_status = "ACCEPTABLE"
        return pit_status


class LocalLossCalcs:

    """
    A set of calculations using the local loss df to determine loss criteria and conditions
    for safe use for each compromised zone.
    """

    def cluster(self, local_loss: pd.DataFrame, cluster_number: int) -> pd.DataFrame:
        return local_loss[local_loss["cluster"] == cluster_number]

    def s_local_loss(self, cluster: pd.DataFrame) -> Union[float, None]:
        return cluster["s"].iloc[0] if not cluster.empty else None

    def c_local_loss(self, cluster: pd.DataFrame) -> Union[float, None]:
        return cluster["c"].iloc[0] if not cluster.empty else None

    def minimum_measured_thickness_local(
        self, find_local_zone: Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame]
    ) -> float:
        return find_local_zone["z"].min()

    def remaining_thickness_ratio_local(
        self,
        minimum_measured_thickness_local: float,
        future_corrosion_allowance: float,
        minimum_required_thickness: float,
    ) -> float:
        return (
            (minimum_measured_thickness_local - future_corrosion_allowance)
            / minimum_required_thickness
            if (minimum_measured_thickness_local - future_corrosion_allowance)
            / minimum_required_thickness
            > 0
            else 0
        )

    def shell_parameter_local(
        self,
        s_local_loss: Union[float, None],
        inside_diameter: float,
        minimum_required_thickness: float,
    ) -> Union[float, None]:
        return (
            (1.285 * s_local_loss)
            / ((inside_diameter * minimum_required_thickness) ** 0.5)
            if s_local_loss != None
            else None
        )

    def mt_local(
        self, s_local_loss: Union[float, None], shell_parameter_local: float
    ) -> Union[float, None]:
        if s_local_loss:
            if s_local_loss != 0.0:
                mt_local = (1 + 0.48 * shell_parameter_local ** 2) ** 0.5
            else:
                mt_local = 0.0
        else:
            mt_local = None
        return mt_local

    def rsf_local(
        self,
        s_local_loss: Union[float, None],
        remaining_thickness_ratio_local: float,
        mt_local: float,
    ) -> Union[float, None]:
        if s_local_loss:
            if s_local_loss != 0.0:
                rsf_local = remaining_thickness_ratio_local / (
                    1 - (1 / mt_local) * (1 - remaining_thickness_ratio_local)
                )
            else:
                rsf_local = 0.0
        else:
            rsf_local = None
        return rsf_local

    def MAWPr(
        self, design_condition_pressure: float, rsf_local: float, rsfa: float
    ) -> Union[float, None]:
        return (
            design_condition_pressure * (rsf_local * rsfa)
            if rsf_local != None
            else None
        )

    def c_over_d(
        self, c_local_loss: Union[float, None], inside_diameter: float
    ) -> Union[float, None]:
        return c_local_loss / inside_diameter if c_local_loss != None else None

    def limiting_flaw_size_criterium_1(
        self, remaining_thickness_ratio_local: float
    ) -> bool:
        return True if remaining_thickness_ratio_local >= 0.2 else False

    def limiting_flaw_size_criterium_2(
        self, minimum_measured_thickness_local: float
    ) -> bool:
        return True if minimum_measured_thickness_local >= 0.1 else False

    def limiting_flaw_size_criterium_3(
        self,
        measured_length_to_discontinuity: float,
        inside_diameter: float,
        minimum_required_thickness: float,
    ) -> bool:
        return (
            True
            if measured_length_to_discontinuity
            >= (1.8 * inside_diameter * minimum_required_thickness)
            else False
        )

    def critical_groove_radius(self, minimum_required_thickness: float) -> float:
        return max(0.25 * minimum_required_thickness, 0.25)

    def groove_length(self, s_local_loss: Union[float, None]) -> Union[float, None]:
        return s_local_loss if s_local_loss != None else None

    def groove_width(self, c_local_loss: Union[float, None]) -> Union[float, None]:
        return c_local_loss if c_local_loss != None else None

    def groove_depth(self, minimum_measured_thickness_local: float) -> float:
        return NOMINAL_WALL_THICKNESS - minimum_measured_thickness_local

    def groove_radius(self, groove_depth: float) -> float:
        return groove_depth

    def groove_criterium_1(
        self, groove_radius: float, critical_groove_radius: float
    ) -> bool:
        return True if groove_radius >= critical_groove_radius else False

    def groove_criterium_2(
        self,
        groove_radius: float,
        remaining_thickness_ratio_local: float,
        minimum_required_thickness: float,
    ) -> bool:
        return (
            True
            if (
                groove_radius
                / (1 - remaining_thickness_ratio_local * minimum_required_thickness)
            )
            >= 1
            else False
        )


class SeparatorAttrs:

    "Calculates various separator attributes where df is the raw daw and xyz is the xyz data"

    def wall_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        return NOMINAL_WALL_THICKNESS - df

    def total_scans(self, wall_loss: pd.DataFrame) -> int:
        return int(len(wall_loss.stack()))

    def absolute_minimum_reading(self, xyz: pd.DataFrame) -> float:
        return xyz["z"].min()

    def average_wall_thickness(self, xyz: pd.DataFrame) -> float:
        return xyz["z"].mean()

    def average_wall_loss(self, wall_loss: pd.DataFrame) -> float:
        return wall_loss.stack().mean()

    def uniform_metal_loss(self, average_wall_loss: float) -> float:
        return average_wall_loss

    def t(self, average_wall_loss: float) -> float:
        return NOMINAL_WALL_THICKNESS - average_wall_loss

    def radius_circ(
        self,
        inside_diameter: float,
        uniform_metal_loss: float,
        future_corrosion_allowance: float,
    ) -> float:
        return (inside_diameter / 2) + uniform_metal_loss + future_corrosion_allowance

    def minimum_required_thickness_circ(
        self,
        design_cond_pressure: float,
        radius_circ: float,
        weld_joint_efficiency: float,
    ) -> float:
        return (design_cond_pressure * radius_circ) / (
            (ALLOWABLE_TENSILE_STRESS * weld_joint_efficiency)
            - 0.6 * design_cond_pressure
        )

    def minimum_required_thickness_long(
        self,
        design_cond_pressure: float,
        radius_circ: float,
        weld_joint_efficiency: float,
    ) -> float:
        return (design_cond_pressure * radius_circ) / (
            (2 * ALLOWABLE_TENSILE_STRESS * weld_joint_efficiency)
            + 0.4 * design_cond_pressure
        )

    def minimum_required_thickness(
        self,
        minimum_required_thickness_circ: float,
        minimum_required_thickness_long: float,
    ) -> float:
        return max(minimum_required_thickness_circ, minimum_required_thickness_long)
