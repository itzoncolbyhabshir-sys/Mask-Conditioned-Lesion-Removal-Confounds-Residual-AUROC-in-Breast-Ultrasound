"""Regression locks for retained main-text algorithms; no ML dependencies needed."""
import ast
import hashlib
import importlib.util
import json
from unittest.mock import patch
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = {'src/conditions.py': {'condition_output_name': '48dfe99d3dbaccb7f8ceaf9ffb8c4c89a4bef616a9f76b9e09f224899b308136',
                       'generate_C1_full': 'c582719577508f1df6f1ae1acbe8a93c9688252fe08363c5d9a42e77c79c6cfd',
                       'generate_C2_lesion_only': 'df887c0d9e7c934fa88a0a68fdaa2f1bbc1f3721bacba20ef31fbc28fd623704',
                       'generate_C4_lesion_removed_inpaint': 'df4b5bfa8911a3040f630a247ab23beb685be9c538f56f18b3b55c615def6133',
                       'generate_C5_mask_only': 'f4419df5ef0290d51eb60175aee28580eb16029f877db228e55075c78a44369a',
                       'generate_C6_shape_only': '4694692df9b3923674f624a1b0a66d6c5c59cfd8327797901b022b4f1952e730',
                       'inpaint_navier_stokes': 'd4afce4e119b951a89d630a81397b5e73138e0ffab45e69e8f2f689da07f9294',
                       'inpaint_telea': 'a1c0a69be50eb52f37802fcccc0f220a26bdb8eddb6976cba02734c7e3a90dde',
                       'load_image': '2d43ba810d83e558e7259689ff535d95460aa6ae454d8f9ccdb45cbcaec02d72',
                       'load_mask': 'e9003f8adc1a4225b9c495a31eba03eaec2571f6edfa1976841f43b655e6a497'},
 'src/manifest.py': {'_case_insensitive_child': 'ae25e948fcd840c08c57a67770a31eeea12d265cbfd6e79d1608760e4f58d39f',
                     '_find_bus_bra_layout': 'd3fb545c40c53a3abdeb37134b454904c4094881a76d97cc81e66295ec009eb8',
                     '_index_files_by_stem': '5efdff1236c550fd0456ebf861353a2bbb653c2de7d369f51c62b30f2db2087b',
                     'compute_bbox': 'eafa278ddd576fe8ab4059d6f180137b84c1724c459b53b6986787985e2cecca',
                     'compute_lesion_area_ratio': 'd0706daff001ea7363805d20fb541a577bfcddc99b627bbc47fe2791df36448c',
                     'infer_bus_uclm_label': '67e7c5b7bcbd849681f9053872141f0b4287feffc96ff645d8a2a7243809523f',
                     'load_mask_binary': '0d7fd715227e14b3e52891bbcedefcdf008ba20422f45beda7d6091e3e5f057c',
                     'md5': '14dafce97032d9768908b69a58bc13c0caaf00c9942e90572e2973cfb52426eb',
                     'namespaced_patient_id': '4eed96524401e17a485f1334fbe2e38ac255cc6becdd0bdd17755205414b9933',
                     'normalize_label': '016abedde474ac553a4491cd3577e499bf1d627277181b9cf1088039df3f902a',
                     'quality_check': '1755abd00b220a5daa4771686f3ee7a6a03a9bffba97387f19f47433c3ac8d60'}}

def canonical_ast(node):
    """Ignore empty/version-specific AST fields while preserving algorithm structure."""
    if isinstance(node, ast.AST):
        return [type(node).__name__, [(key, canonical_ast(value)) for key, value in ast.iter_fields(node)
                                      if key != "type_params" and value is not None and value != []]]
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    if isinstance(node, bytes):
        return ["bytes", node.hex()]
    return node


class RetainedAlgorithmsTests(unittest.TestCase):
    def test_retained_algorithm_definitions_match_original_snapshot(self):
        for relative, functions in BASELINE.items():
            tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
            current = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
            for name, expected in functions.items():
                with self.subTest(file=relative, function=name):
                    canonical = json.dumps(canonical_ast(current[name]), ensure_ascii=True, separators=(",", ":"))
                    digest = hashlib.sha256(canonical.encode()).hexdigest()
                    self.assertEqual(digest, expected)


class MainTextScopeTests(unittest.TestCase):
    def test_only_paper_datasets_models_and_conditions_are_registered(self):
        tree = ast.parse((ROOT / "src/config.py").read_text(encoding="utf-8"))
        assignments = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)}
        def keys(name):
            return {key.value for key in assignments[name].keys}
        self.assertEqual(keys("DATASETS"), {"BUS-BRA", "BUS-UCLM"})
        self.assertEqual(keys("MODELS"), {"ResNet-50", "DINOv3", "MedSAM2", "UltraSam"})
        self.assertEqual(keys("CONDITIONS"), {"C1_full", "C2_lesion-only", "C4_lesion-removed-inpaint", "C5_mask-only", "C6_shape-only", "C5d_dilated-mask-only", "C6d_dilated-geometry-rendering"})
        self.assertEqual(keys("EXPERIMENTS"), {"exp1_primary", "exp2_controls"})

    def test_cli_has_only_main_text_stages_and_propagates_failure(self):
        spec = importlib.util.spec_from_file_location("maintext_pipeline", ROOT / "run_pipeline.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(set(module.STAGE_MAP), {"infra", "exp1", "exp2"})
        with patch.object(module, "run_stage", return_value={"status": "FAILED"}):
            self.assertEqual(module.main(["--stage", "exp1"]), 1)
        with patch.object(module, "run_stage", return_value={"status": "OK"}):
            self.assertEqual(module.main(["--stage", "exp1"]), 0)


if __name__ == "__main__":
    unittest.main()
