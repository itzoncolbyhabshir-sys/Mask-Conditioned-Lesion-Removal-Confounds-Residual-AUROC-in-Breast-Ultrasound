"""Regression checks for the main-text controls."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
LOCKS = {'fixed_shape_mask': '773df101049c6ad70232031481483a4ae3bc2d6ee7a61b8124d1b59b08ed64b3', 'summarize': '34ed93ff7751b3f3c836a354572d7b2be9f017c1b22aa0bfd2b716a80d5d65a2', 'audit_fixed_shape_coverage': '64001ac24177f5f1bcd54a4601ccb6539490774f6f45a9071dc40ad6a43a5698', 'generate_c4_inpaint_variants': 'ee687ca24f031ac41dea7f46740c3c015d6b35446a126bc11d5d8662975c277d', 'align_patient_predictions': '673ed4ae0981b543d11d443933b935deb7123e6f7874544d2a66d8341a79d41c', 'endpoint_statistics': 'fe0b2ee0f1c3f235ef156cc9faa8fdd2adfee3e8cf89d48676faf15edda51bc0', 'compare_conditions': '840a0d79585372f7d925d89ecc87b344a19fbafb9bca6d3e887b29ba4daf9a44'}
GOLDEN_SHAPES = {'0/circle': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11', '0/square': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11', '0/ellipse': '09cec5a5bd8afffbb758753810a20c55ccb06a46d7bf54eda69ecd2ad645ef11', '1/circle': '716abec4aafeb69f08cbcb7853d4b3a8accaa31e20ba2ba5acf8c35f399dd63d', '1/square': '4dab6ea6e6dda410f0082a43d44401c5bafb826d8788eab68b57f2c0f1214cd0', '1/ellipse': 'e120630696763eed5622bff007d5dc136e153b7d4f5fcc02c69e58f99cfec036', '2/circle': '3ca591c9b0a0256cafafe7a0bc82c17fb984f095479abfc510e77927626cf4b7', '2/square': '67ede60953da0b3269636c133d1bf1d75a0129180f6e9491abae7bfd6021d455', '2/ellipse': '9d147893c4bc8f00d137d819a94f765a634730d0b23a81e4c89d69067012780a', '3/circle': '2165e31ee0d0f64111e4d48269e884408b036c29ee5c92017cb476f1d1294ea2', '3/square': 'e772d46aa36e3b0634be405414b4c0724e3d8aa9fdf4108d76904f916f674540', '3/ellipse': '1e70f2089c72cd0b39f911c5ac5b9d5e4e7055121eee180068a69ff1e95b22f2'}


def canonical_ast(node):
    if isinstance(node, ast.AST):
        return [type(node).__name__, [(key, canonical_ast(value)) for key, value in ast.iter_fields(node)
                                      if key != "type_params" and value is not None and value != []]]
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    return node


def function_node(filename, name):
    tree = ast.parse((ROOT / "scripts" / filename).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def load_function(filename, name, namespace):
    node = function_node(filename, name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class MainTextControlsTests(unittest.TestCase):
    def test_retained_computations_match_recorded_source(self):
        locations = {
            "fixed_shape_mask": "run_fixed_shape_controls.py",
            "summarize": "audit_fixed_shape_coverage.py",
            "audit_fixed_shape_coverage": "audit_fixed_shape_coverage.py",
            "generate_c4_inpaint_variants": "run_inpaint_sensitivity.py",
            "align_patient_predictions": "run_inpaint_sensitivity.py",
            "endpoint_statistics": "refresh_bibm_fixed_shape_statistics.py",
            "compare_conditions": "refresh_bibm_fixed_shape_statistics.py",
        }
        for name, filename in locations.items():
            with self.subTest(function=name):
                canonical = json.dumps(canonical_ast(function_node(filename, name)), ensure_ascii=True, separators=(",", ":"))
                self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), LOCKS[name])

    def test_fixed_shape_entry_runs_only_three_main_text_conditions(self):
        class Column(list):
            def __eq__(self, other):
                return [value == other for value in self]
            def nunique(self):
                return len(set(self))
        class Frame:
            empty = False
            def __getitem__(self, item):
                if isinstance(item, list):
                    return self
                return Column(["BUS-BRA" if item == "dataset_name" else "patient-1"])
            def copy(self):
                return self
            def reset_index(self, **kwargs):
                return self
            def itertuples(self, **kwargs):
                return iter([SimpleNamespace(image_path="data/sample.png", dilated_mask_path="data/mask.png")])
            def __len__(self):
                return 1
        class OutputPath:
            def __init__(self, path=""):
                self.path = path
            def __truediv__(self, part):
                return OutputPath(self.path + "/" + str(part))
            def mkdir(self, **kwargs):
                pass
            def write_text(self, contents, **kwargs):
                written[self.path] = json.loads(contents)
            def __str__(self):
                return self.path
        conditions = {"circle": "C10_fixed-circle-inpaint", "square": "C11_fixed-square-inpaint", "ellipse": "C12_fixed-ellipse-inpaint"}
        written = {}
        extract, probe, save = Mock(), Mock(), Mock()
        namespace = {
            "pd": SimpleNamespace(read_csv=lambda path: Frame()),
            "ROOT": Path("."), "Path": Path, "IMAGES_DIR": OutputPath("images"), "RESULTS_DIR": OutputPath("results"),
            "FIXED_CONDITIONS": conditions, "C4_CONDITION": "C4_lesion-removed-inpaint", "SEEDS": [2026, 2027, 2028, 2029, 2030],
            "load_image": Mock(return_value="image"), "load_mask": Mock(return_value="mask"),
            "fixed_shape_mask": Mock(side_effect=lambda mask, shape: shape), "inpaint_telea": Mock(return_value="edited"),
            "Image": SimpleNamespace(fromarray=lambda value: SimpleNamespace(save=save)),
            "run_extraction": extract, "run_probe_experiment": probe,
            "endpoint_statistics": Mock(return_value={"auroc": 0.8}), "compare_conditions": Mock(return_value={"delta_auroc": 0.05}),
            "json": json,
        }
        report = load_function("run_fixed_shape_controls.py", "main", namespace)()
        self.assertEqual([call.args[2] for call in extract.call_args_list], list(conditions.values()))
        self.assertEqual([call.args[2] for call in probe.call_args_list], list(conditions.values()))
        self.assertEqual(save.call_count, 3)
        self.assertEqual(set(report["fixed_shape_removal"]), set(conditions))
        self.assertEqual(set(report), {"model", "dataset", "n_images", "n_target_patients", "fixed_shape_removal", "fixed_shape_statistics_note"})
        self.assertIn("results/fixed_shape_controls.json", written)

    def test_fixed_shape_masks_match_pre_cleanup_numerical_baseline(self):
        if not all(importlib.util.find_spec(name) for name in ("numpy", "cv2")):
            self.skipTest("NumPy and OpenCV are required for numerical regression")
        import numpy as np
        import cv2
        fixed_shape_mask = load_function("run_fixed_shape_controls.py", "fixed_shape_mask", {"np": np, "cv2": cv2, "math": math})
        for index, region in enumerate([(0, 0, 0, 0), (12, 28, 16, 31), (0, 12, 0, 8), (30, 40, 35, 45)]):
            mask = np.zeros((40, 45), dtype=np.uint8)
            y0, y1, x0, x1 = region
            mask[y0:y1, x0:x1] = 1
            for shape in ("circle", "square", "ellipse"):
                with self.subTest(mask=index, shape=shape):
                    actual = hashlib.sha256(fixed_shape_mask(mask, shape).tobytes()).hexdigest()
                    self.assertEqual(actual, GOLDEN_SHAPES[f"{index}/{shape}"])

    def test_no_removed_module_dependencies(self):
        for filename in ("run_fixed_shape_controls.py", "audit_fixed_shape_coverage.py", "run_inpaint_sensitivity.py", "refresh_bibm_fixed_shape_statistics.py", "make_bibm_figures.py"):
            text = (ROOT / "scripts" / filename).read_text(encoding="utf-8")
            for removed in ("src.exp4_localization", "run_c4_source_attribution", "audit_bibm_closure_validity", "c4_source_attribution", "C9_mask-swap", "C13_pseudo"):
                with self.subTest(file=filename, removed=removed):
                    self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
