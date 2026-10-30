import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.generate_final_report import (
    PROJECT_ROOT,
    _validate_binary_values,
    edge_connected_zero_mask,
    ensure_output_available,
    foreground_intensity_proxy,
    generate_final_report,
    make_border_mask,
    region_attention_metrics,
    sha256_file,
    stratified_bootstrap_intervals,
    wilson_interval,
)


class GenerateFinalReportTest(unittest.TestCase):
    def test_binary_validation_rejects_fractional_values(self):
        _validate_binary_values(np.array([0, 1, 0.0, 1.0]), "values")
        with self.assertRaisesRegex(ValueError, "exact binary"):
            _validate_binary_values(np.array([0, 0.5, 1]), "values")

    def test_wilson_interval_handles_boundary_proportions(self):
        lower, upper = wilson_interval(14, 14)
        self.assertAlmostEqual(lower, 0.784689197, places=7)
        self.assertAlmostEqual(upper, 1.0)
        lower, upper = wilson_interval(0, 14)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(upper, 0.215310803, places=7)

    def test_wilson_interval_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            wilson_interval(0, 0)
        with self.assertRaisesRegex(ValueError, "between zero and total"):
            wilson_interval(4, 3)

    def test_border_mask_has_expected_area(self):
        mask = make_border_mask((224, 224), 32)
        expected = 224 * 224 - (224 - 64) * (224 - 64)
        self.assertEqual(int(mask.sum()), expected)
        self.assertTrue(mask[0, 100])
        self.assertFalse(mask[100, 100])

    def test_edge_connected_zero_mask_excludes_enclosed_zero_island(self):
        image = np.ones((7, 7), dtype=np.float32)
        image[0, :] = 0.0
        image[1, 0] = 0.0
        image[3, 3] = 0.0
        mask = edge_connected_zero_mask(image)
        self.assertTrue(mask[0, 4])
        self.assertTrue(mask[1, 0])
        self.assertFalse(mask[3, 3])

    def test_foreground_proxy_excludes_edge_background(self):
        image = np.zeros((20, 20), dtype=np.float32)
        image[5:15, 5:15] = np.linspace(0.1, 1.0, 100).reshape(10, 10)
        padding = edge_connected_zero_mask(image)
        foreground = foreground_intensity_proxy(image, padding)
        self.assertIsNotNone(foreground)
        self.assertFalse(np.any(foreground & padding))
        self.assertTrue(foreground[10, 10])

    def test_region_metrics_handle_zero_heatmap(self):
        heatmap = np.zeros((4, 4), dtype=np.float32)
        region = np.zeros((4, 4), dtype=bool)
        region[:2, :] = True
        top = np.zeros((4, 4), dtype=bool)
        top[0, 0] = True
        result = region_attention_metrics(heatmap, region, top)
        self.assertEqual(result["area_fraction"], 0.5)
        self.assertIsNone(result["activation_mass_fraction"])
        self.assertIsNone(result["activation_enrichment"])
        self.assertIsNone(result["argmax_inside"])
        self.assertIsNone(result["top10_fraction_inside"])

    def test_bootstrap_audit_is_deterministic(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.6, 0.4, 0.8, 0.9])
        first = stratified_bootstrap_intervals(labels, scores, iterations=25, seed=7)
        second = stratified_bootstrap_intervals(labels, scores, iterations=25, seed=7)
        self.assertEqual(first, second)

    def test_existing_report_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            output.mkdir()
            ensure_output_available(output)
            (output / "report.md").write_text("frozen", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                ensure_output_available(output)

    def test_reporting_script_has_no_model_runtime_imports(self):
        script = PROJECT_ROOT / "scripts" / "generate_final_report.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        forbidden = {
            "keras",
            "tensorflow",
            "src.modeling",
            "scripts.evaluate_models",
            "scripts.generate_gradcam",
        }
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        violations = {
            name
            for name in imported
            if any(name == forbidden_name or name.startswith(f"{forbidden_name}.") for forbidden_name in forbidden)
        }
        self.assertFalse(violations, violations)

    def test_report_output_cannot_overlap_an_input_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "evaluation"
            evaluation.mkdir()
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                generate_final_report(
                    evaluation,
                    root / "explainability",
                    root / "processed",
                    evaluation / "report",
                )

    @unittest.skipUnless(
        (PROJECT_ROOT / "outputs" / "evaluation" / "test" / "metrics.json").is_file()
        and (PROJECT_ROOT / "outputs" / "explainability" / "test" / "case_index.csv").is_file(),
        "Frozen evaluation and Grad-CAM artifacts are not available.",
    )
    def test_frozen_artifacts_generate_report_without_source_changes(self):
        source = PROJECT_ROOT / "outputs" / "evaluation" / "test" / "patient_predictions.csv"
        before = sha256_file(source)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "final"
            result = generate_final_report(
                PROJECT_ROOT / "outputs" / "evaluation",
                PROJECT_ROOT / "outputs" / "explainability" / "test",
                PROJECT_ROOT / "data" / "processed",
                output,
            )
            self.assertEqual(result, output.resolve())
            for relative in (
                "report.md",
                "statistical_results.csv",
                "agreement_summary.csv",
                "gradcam_per_map.csv",
                "gradcam_summary.csv",
                "integrity_checks.csv",
                "run_provenance.json",
                "checksums.sha256",
                "figures/test_roc_curves.png",
                "figures/test_confusion_matrices.png",
                "figures/test_metrics_forest.png",
                "figures/model_agreement_matrix.png",
                "figures/gradcam_artifact_attention.png",
            ):
                self.assertTrue((result / relative).is_file(), relative)
            provenance = json.loads(
                (result / "run_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["reporting_script_sha256"],
                sha256_file(PROJECT_ROOT / "scripts" / "generate_final_report.py"),
            )
            manifest = json.loads(
                (result / "input_manifest.json").read_text(encoding="utf-8")
            )
            for record in manifest["files"]:
                path = PROJECT_ROOT / record["path"]
                self.assertTrue(path.is_file(), record["path"])
                self.assertEqual(sha256_file(path), record["sha256"])
            integrity = (result / "integrity_checks.csv").read_text(encoding="utf-8")
            self.assertIn("gradcam_heatmap_authenticity,warning", integrity)
        self.assertEqual(sha256_file(source), before)


if __name__ == "__main__":
    unittest.main()
