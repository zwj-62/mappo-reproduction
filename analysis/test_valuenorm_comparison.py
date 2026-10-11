import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_valuenorm_comparison import align_curves, load_curve, load_record


class AnalysisTests(unittest.TestCase):
    def test_absolute_tensorboardx_key(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp) / "logs"
            logs.mkdir()
            data = {r"D:\old\logs\average_episode_rewards\average_episode_rewards": [[1, 200, -200], [2, 1200, -180]]}
            (logs / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            steps, values = load_curve(temp)
            np.testing.assert_array_equal(steps, [200, 1200])
            np.testing.assert_array_equal(values, [-200, -180])

    def test_exact_common_steps_without_interpolation(self):
        curves = [(np.array([200, 1200, 2200]), np.array([-3, -2, -1])),
                  (np.array([200, 2200]), np.array([-5, -3]))]
        steps, values = align_curves(curves)
        np.testing.assert_array_equal(steps, [200, 2200])
        np.testing.assert_array_equal(values, [[-3, -1], [-5, -3]])
        np.testing.assert_array_equal(values.mean(axis=0), [-4, -2])
        np.testing.assert_array_equal(values.std(axis=0, ddof=0), [1, 1])

    def test_reject_nonfinite_and_duplicate_steps(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp) / "logs"
            logs.mkdir()
            for events in ([[1, 200, -200], [2, 200, -180]], [[1, 200, float("nan")]]):
                (logs / "summary.json").write_text(json.dumps({"average_episode_rewards/average_episode_rewards": events}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_curve(temp)

    def test_reject_missing_and_disjoint_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            logs = Path(temp) / "logs"
            logs.mkdir()
            (logs / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_curve(temp)
        with self.assertRaises(ValueError):
            align_curves([(np.array([200]), np.array([-1])), (np.array([1200]), np.array([-2]))])

    def test_only_valuenorm_changes(self):
        on = load_record("on", 1)["resolved_config"]
        off = load_record("off", 1)["resolved_config"]
        differences = {key for key in on.keys() | off.keys() if on.get(key) != off.get(key)}
        self.assertEqual(differences - {"experiment_name"}, {"use_valuenorm"})
        self.assertTrue(on["use_valuenorm"])
        self.assertFalse(off["use_valuenorm"])
        self.assertTrue(on["use_centralized_V"])
        self.assertFalse(on["use_popart"])
        self.assertFalse(on["use_eval"])


if __name__ == "__main__":
    unittest.main()
