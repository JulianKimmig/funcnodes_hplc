from typing import List, Optional, Tuple
from funcnodes_span.peaks import PeakProperties, calculate_peak_symmetricity
from funcnodes_span.fitting import peaks_from_fitted
from funcnodes_span._curves import estimate_noise
import numpy as np
import pandas as pd
import funcnodes as fn
from scipy.integrate import trapezoid

from .signals import calculate_resolution
from .hplc_run import estimate_t0


def hplc_report(
    x,
    y,
    peaks: List[PeakProperties],
    fitted_peaks: Optional[List[PeakProperties]] = None,
    t0: Optional[float] = None,
    noise: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if t0 is None:
        t0 = estimate_t0(x, y)

    if noise is None:
        noise = estimate_noise(x, y)

    if fitted_peaks is None:
        try:
            fitted_peaks = peaks_from_fitted(peaks)
        except Exception:
            fitted_peaks = [None] * len(peaks)

    if len(peaks) != len(fitted_peaks):
        raise ValueError(
            "if fitted peaks are provided they must match one to one with peaks"
        )

    rundata = {
        "t0": t0,
        "noise": noise,
        "n_signals": len(peaks),
        "total_area_over_curve": trapezoid(y[y > 0], x[y > 0]),
        "total_area_under_curve": trapezoid(y[y < 0], x[y < 0]),
    }

    total_peak_area = 0
    total_peak_area_fitted = 0

    def calc_for_peak(p: PeakProperties):
        h10t, h10l, h10r, *_ = p.get_width_at_height(0.1)
        p.add_serializable_property("retention_time", p.x_at_index - t0)
        p.add_serializable_property(
            "retention_factor", p.retention_time / t0 if t0 > 0 else np.nan
        )
        p.add_serializable_property(
            "efficiency_hm", 5.54 * (p.retention_time / p.fwhm) ** 2
        )
        p.add_serializable_property(
            "efficiency_b", 16 * (p.retention_time / p.get_width_at_height(0)[0]) ** 2
        )
        p.add_serializable_property("signal_to_noise_ratio", p.y_at_index / noise)

        p.add_serializable_property("tailing_factor", (h10l + h10r) / (2 * h10l))

        calculate_peak_symmetricity(p, "h5p", True)

    def calc_data_for_peak(
        p: PeakProperties,
        p_before: Optional[PeakProperties],
        p_after: Optional[PeakProperties],
    ):
        data = p.to_dict()
        data["fwhm"] = p.fwhm
        data["Width5%"] = p.get_width_at_height(0.05)[0]
        data["resolution_before"] = (
            calculate_resolution(p, p_before) if p_before else np.nan
        )
        data["resolution_after"] = (
            calculate_resolution(p, p_after) if p_after else np.nan
        )
        data["selectivity_before"] = (
            p.retention_factor / p_before.retention_factor if p_before else np.nan
        )
        data["selectivity_after"] = (
            p_after.retention_factor / p.retention_factor if p_after else np.nan
        )
        data["fundamental_resolution_before"] = (
            1
            / 4
            * np.sqrt(p.efficiency_hm)
            * ((data["selectivity_before"] - 1) / data["selectivity_before"])
            * p.retention_factor
            / (1 + p.retention_factor)
        )
        data["fundamental_resolution_after"] = (
            1
            / 4
            * np.sqrt(p.efficiency_hm)
            * ((data["selectivity_after"] - 1) / data["selectivity_after"])
            * p.retention_factor
            / (1 + p.retention_factor)
        )

        return data

    datas = []
    for p, pf in zip(peaks, fitted_peaks):
        calc_for_peak(p)
        if pf:
            calc_for_peak(pf)

    for i, (p, pf) in enumerate(zip(peaks, fitted_peaks)):
        p_before = peaks[i - 1] if i > 0 else None
        p_after = peaks[i + 1] if i < len(peaks) - 1 else None
        datas.append(calc_data_for_peak(p, p_before, p_after))
        total_peak_area += p.area
        if pf is not None:
            pf_before = fitted_peaks[i - 1] if i > 0 else None
            pf_after = fitted_peaks[i + 1] if i < len(peaks) - 1 else None
            datas.append(calc_data_for_peak(pf, pf_before, pf_after))
            total_peak_area_fitted += pf.area

    rundata["total_peak_area"] = total_peak_area
    rundata["peak_area_ratio"] = (
        rundata["total_peak_area"] / rundata["total_area_over_curve"]
    )

    if total_peak_area_fitted > 0:
        rundata["total_peak_area_fitted"] = total_peak_area_fitted
        rundata["peak_area_ratio_fitted"] = (
            rundata["total_peak_area_fitted"] / rundata["total_area_over_curve"]
        )
        rundata["fitted_area_ratio"] = (
            rundata["total_peak_area_fitted"] / rundata["total_peak_area"]
        )

    return pd.DataFrame([rundata]), pd.DataFrame(datas)


hplc_report_node = fn.NodeDecorator(
    node_id="fnhplc.report.hplc_report",
    name="HPLC Report",
    description="Calculates HPLC report data from peaks and fitted peaks.",
    outputs=[
        {"name": "rundata"},
        {"name": "peakdata"},
    ],
)(hplc_report)
REPORT_SHELF = fn.Shelf(
    nodes=[
        hplc_report_node,
    ],
    subshelves=[],
    name="HPLC Report",
    description="HPLC Report Nodes",
)
