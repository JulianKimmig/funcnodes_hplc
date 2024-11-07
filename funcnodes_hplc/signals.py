from funcnodes_span.peaks import PeakProperties


def calculate_resolution(peak_a: PeakProperties, peak_b: PeakProperties) -> float:
    return (
        2
        * (peak_a.x_at_index - peak_b.x_at_index)
        / (peak_a.get_width_at_height(0.0)[0] + peak_b.get_width_at_height(0.0)[0])
    )
