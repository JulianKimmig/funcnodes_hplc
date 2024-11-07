import unittest
import funcnodes_hplc
import funcnodes_pandas as fnpd
import os
import pandas as pd
import funcnodes_span as fnsp
import funcnodes as fn

fn.config.IN_NODE_TEST = True


class TestEval(unittest.IsolatedAsyncioTestCase):
    async def test_eval1(self):
        with open(os.path.join(os.path.dirname(__file__), "data.csv"), "rb") as f:
            data = f.read()

        read_csv = fnpd.from_csv_str()
        read_csv["source"] < data
        await read_csv
        self.assertEqual(read_csv["df"].value.shape, (20699, 2))

        xcol = fnpd.get_column()
        xcol["df"] > read_csv["df"]
        xcol["column"] < "x"
        await xcol

        ycol = fnpd.get_column()
        ycol["df"] > read_csv["df"]
        ycol["column"] < "y"
        await ycol

        self.assertIsInstance(xcol["series"].value, pd.Series)
        self.assertIsInstance(ycol["series"].value, pd.Series)

        flatfit = fnsp.baseline.flatfit()

        flatfit["x_data"] > xcol["series"]
        flatfit["data"] > ycol["series"]

        await flatfit

        self.assertEqual(flatfit["baseline_corrected"].value.shape, (20699,))

        smooth = fnsp.smoothing._smooth()

        smooth["y"] > flatfit["baseline_corrected"]
        smooth["x"] > xcol["series"]
        smooth["window"] < 0.01333
        smooth["mode"] < "gaussian"

        await smooth

        self.assertEqual(smooth["smoothed"].value.shape, (20699,))

        peak_finder = fnsp.peak_analysis.peak_finder()

        peak_finder["on"] > smooth["smoothed"]
        peak_finder["x"] > xcol["series"]
        peak_finder["y"] > flatfit["baseline_corrected"]

        await peak_finder

        self.assertEqual(len(peak_finder["peaks"].value), 2)

        fit_peaks = fnsp.peak_analysis.fit_peaks_node()

        fit_peaks["peaks"] > peak_finder["peaks"]
        fit_peaks["x"] > xcol["series"]
        fit_peaks["y"] > flatfit["baseline_corrected"]

        await fit_peaks

        self.assertEqual(len(fit_peaks["fitted_peaks"].value), 2)

        hplc_report = funcnodes_hplc.report.hplc_report_node()
        hplc_report["x"] > xcol["series"]
        hplc_report["y"] > flatfit["baseline_corrected"]
        hplc_report["peaks"] > peak_finder["peaks"]
        hplc_report["fitted_peaks"] > fit_peaks["fitted_peaks"]

        await hplc_report

        self.assertEqual(hplc_report["rundata"].value.shape, (1, 10))
        self.assertEqual(hplc_report["peakdata"].value.shape, (4, 26))
