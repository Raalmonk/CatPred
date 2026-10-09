"""Check measurement boundaries and arithmetic without drawing or loading models."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('stage_plot', Path(__file__).with_name('plot_comparison.py'))
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class StageRowsTests(unittest.TestCase):
    def summary(self):
        def arm(seconds):
            return {'seconds': [seconds + 0.1, seconds, seconds - 0.1], 'median_seconds': seconds}
        return {'status': 'COMPLETE', 'all_reported_timings_passed_exact_bits': True,
                'arms': {'Original': arm(4), 'S_STREAM_K1': arm(1)},
                'preparation': {'esm_seconds': 99, 'model_load_seconds': 12,
                    'esm_comparison': {'status': 'COMPLETE', 'all_exact_bits': True,
                                      'arms': {'Original': arm(3), 'Optimized': arm(2)}}}}

    def test_stages_use_their_own_measured_medians(self):
        rows = plot.stage_rows(self.summary())
        self.assertEqual(rows, [
            {'Stage': 'ESM features + loading', 'Original (s)': 3, 'Optimized (s)': 2, 'Speedup': 1.5},
            {'Stage': 'Prediction (models loaded)', 'Original (s)': 4, 'Optimized (s)': 1, 'Speedup': 4},
        ])

    def test_shared_feature_time_cannot_stand_in_for_an_esm_comparison(self):
        summary = self.summary()
        del summary['preparation']['esm_comparison']
        with self.assertRaises(KeyError):
            plot.stage_rows(summary)

    def test_failed_precision_or_incomplete_stage_cannot_produce_a_plot(self):
        for target, key, value in [('root', 'status', 'FAILED'), ('root', 'all_reported_timings_passed_exact_bits', False),
                                   ('esm', 'status', 'FAILED'), ('esm', 'all_exact_bits', False)]:
            summary = self.summary()
            record = summary if target == 'root' else summary['preparation']['esm_comparison']
            record[key] = value
            with self.assertRaises(ValueError):
                plot.stage_rows(summary)

    def test_invalid_timings_and_wrong_medians_are_rejected(self):
        for changes in ({'seconds': []}, {'seconds': [float('nan')]}, {'seconds': [0]}, {'median_seconds': 100}):
            summary = copy.deepcopy(self.summary())
            summary['preparation']['esm_comparison']['arms']['Optimized'].update(changes)
            with self.assertRaises(ValueError):
                plot.stage_rows(summary)


if __name__ == '__main__':
    unittest.main()
