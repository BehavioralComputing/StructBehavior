"""Small CPU checks for TwiBot-20 material graph preparation and loading.

Run with ``python -m unittest discover -s tests -v``. These tests use synthetic
tensors only; importing this module never imports a training entry point.
"""

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "Data" / "TwiBot20" / "material_data" / "build_graph.py"
RUN_PATH = BUILD_PATH.with_name("run.py")
SPEC = importlib.util.spec_from_file_location("twibot20_material_graph_test_target", BUILD_PATH)
build = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build
SPEC.loader.exec_module(build)

FEATURE_FILES = (
    "num_properties_tensor.pt",
    "tweets_tensor.pt",
    "cat_properties_tensor.pt",
    "des_tensor.pt",
)
INPUT_FILES = ("edge_index.pt", "edge_type.pt") + FEATURE_FILES
CACHE_ERRORS = (FileNotFoundError, RuntimeError, ValueError)


def triples(edges, relations):
    return {(int(u), int(v), int(r)) for (u, v), r in zip(edges.t(), relations)}


def feature_fixture():
    # Node 3 is each basis vector's unique nearest neighbor. Node 3's top
    # choice is tied among 0, 1 and 2; all legal choices give the same support.
    return torch.tensor([[1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.], [1., 1., 1.]], dtype=torch.float64)


def write_inputs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    edges = torch.tensor([[0, 3, 3, 2], [3, 0, 0, 2]])
    relations = torch.tensor([0, 0, 4, 4])
    torch.save(edges, directory / "edge_index.pt")
    torch.save(relations, directory / "edge_type.pt")
    feature_columns = torch.cat((feature_fixture().float(), torch.zeros(4, 1)), dim=1)
    for index, filename in enumerate(FEATURE_FILES):
        torch.save(feature_columns[:, index:index + 1], directory / filename)
    return edges, relations


def settings(**overrides):
    values = dict(max_neighbors=2, query_chunk_size=2,
                  reference_chunk_size=2, device="cpu")
    values.update(overrides)
    return values


def extract_training_loader():
    """Execute the loader definition without importing DGL or starting training."""
    source = ROOT / "main_Twibot20.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ("load_material_graph_data", "relational_undirected")]
    namespace = dict(torch=torch, os=os, Path=Path,
                     load_prepared_material_graph=build.load_prepared_material_graph)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["load_material_graph_data"]


class CosineNeighborsTests(unittest.TestCase):
    def assert_exact_topk(self, features, indices, epsilon):
        self.assertEqual(indices.shape[0], len(features))
        norms = torch.linalg.vector_norm(features.double(), dim=1)
        scores = features.double() @ features.double().t()
        scores /= norms[:, None] * norms[None, :] + epsilon
        for index, chosen in enumerate(indices.tolist()):
            self.assertNotIn(index, chosen)
            self.assertEqual(len(chosen), len(set(chosen)))
            remaining = [j for j in range(len(features))
                         if j != index and j not in chosen]
            if chosen and remaining:
                self.assertGreaterEqual(float(scores[index, chosen].min()) + 1e-6,
                                        float(scores[index, remaining].max()))

    def test_cosine_excludes_self_and_accepts_zero_vector_ties(self):
        features = torch.tensor([[1., 0.], [2., 1.], [0., 1.],
                                 [-1., 0.], [0., 0.]])
        for query_chunk, reference_chunk in ((1, 1), (2, 3), (20, 20)):
            with self.subTest(query_chunk=query_chunk, reference_chunk=reference_chunk):
                neighbors = build.exact_cosine_topk(
                    features, 2, query_chunk_size=query_chunk,
                    reference_chunk_size=reference_chunk, device="cpu")
                self.assertEqual(tuple(neighbors.shape), (5, 2))
                self.assert_exact_topk(features, neighbors, 1e-8)

    def test_cosine_regularization_is_added_to_pair_norm_product(self):
        # With epsilon=1, neighbor 1 outranks the short, collinear node 2.
        # Normalizing each row first would incorrectly choose node 2.
        features = torch.tensor([[1., 0.], [2., .5], [.2, 0.], [0., 1.]])
        neighbors = build.exact_cosine_topk(features, 1, cosine_epsilon=1., device="cpu",
                                           query_chunk_size=2, reference_chunk_size=2)
        self.assertEqual(int(neighbors[0, 0]), 1)
        self.assert_exact_topk(features, neighbors, 1.)

    def test_topk_respects_population_limit(self):
        features = feature_fixture()
        neighbors = build.exact_cosine_topk(features, 50, device="cpu")
        self.assertEqual(tuple(neighbors.shape), (4, 3))
        self.assert_exact_topk(features, neighbors, 1e-8)


class EntropyTests(unittest.TestCase):
    def test_pearson_smoothing_and_entropy_match_hand_calculation(self):
        features = feature_fixture()
        native = torch.tensor([[0, 1], [1, 0]])
        neighbors = torch.tensor([[3], [3], [3], [0]])
        observed = build.weighted_entropy_for_k(native, features, neighbors, 1, device="cpu")
        # The support is a 3-leaf star plus native pair (0,1). Shifted
        # correlations are (1,1,1,1/2), their mean is 7/8, epsilon_w=7/64.
        # Weighted degrees: (55,55,35.5,106.5)/32; volume=252/32.
        # The tiny regularizer changes the resulting entropy by < 1e-8.
        self.assertAlmostEqual(observed, 1.88199355134047, places=7)

    def test_native_duplicates_and_self_edges_do_not_change_entropy_support(self):
        features = feature_fixture()
        neighbors = torch.tensor([[3], [3], [3], [0]])
        native = torch.tensor([[0, 1], [1, 0]])
        repeated = torch.tensor([[0, 1, 0, 1, 2], [1, 0, 1, 0, 2]])
        clean = build.weighted_entropy_for_k(native, features, neighbors, 1, device="cpu")
        duplicate = build.weighted_entropy_for_k(repeated, features, neighbors, 1, device="cpu")
        self.assertAlmostEqual(clean, duplicate, places=12)

    def test_constant_features_have_finite_uniform_cycle_entropy(self):
        features = torch.ones(4, 3, dtype=torch.float64)
        native = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        neighbors = torch.tensor([[1], [2], [3], [0]])
        entropy = build.weighted_entropy_for_k(native, features, neighbors, 1, device="cpu")
        self.assertTrue(math.isfinite(entropy))
        self.assertAlmostEqual(entropy, 2., places=12)


class SearchTests(unittest.TestCase):
    def test_increment_dependent_five_three_one_steps(self):
        values = {1: 1., 6: 1.05, 9: 1.055, 10: 1.0555}
        visits = []
        selected, history = build.select_neighbor_count(
            lambda k: visits.append(k) or values[k], 20)
        self.assertEqual(visits, [1, 6, 9, 10])
        self.assertEqual(selected, 10)
        self.assertEqual([row["k"] for row in history], visits)

    def test_first_entropy_uses_zero_reference_and_tolerance_equality_stops(self):
        visits = []
        selected, history = build.select_neighbor_count(
            lambda k: visits.append(k) or .001, 20, entropy_tolerance=.001)
        self.assertEqual(selected, 1)
        self.assertEqual(visits, [1])
        self.assertAlmostEqual(history[0]["delta"], .001)

    def test_point_one_boundary_uses_three_step(self):
        visits = []
        selected, _ = build.select_neighbor_count(
            lambda k: visits.append(k) or {1: .1, 4: .1005}[k], 20)
        self.assertEqual(visits, [1, 4])
        self.assertEqual(selected, 4)

    def test_point_zero_one_boundary_uses_one_step(self):
        visits = []
        selected, _ = build.select_neighbor_count(
            lambda k: visits.append(k) or {1: .01, 2: .0105}[k], 20)
        self.assertEqual(visits, [1, 2])
        self.assertEqual(selected, 2)

    def test_search_evaluates_clamped_limit(self):
        visits = []
        selected, history = build.select_neighbor_count(
            lambda k: visits.append(k) or float(k), 3)
        self.assertEqual(visits, [1, 3])
        self.assertEqual(selected, 3)
        self.assertEqual(history[-1]["k"], 3)

    def test_negative_increment_selects_current_searched_support(self):
        selected, history = build.select_neighbor_count(
            lambda k: {1: 1., 6: .9}[k], 20)
        self.assertEqual(selected, 6)
        self.assertLess(history[-1]["delta"], 0)


class AugmentedGraphTests(unittest.TestCase):
    def test_native_ids_are_preserved_and_feature_overlap_is_not_added_twice(self):
        native = torch.tensor([[0, 3, 3, 2], [3, 0, 0, 2]])
        relations = torch.tensor([0, 0, 4, 4])
        saved_edges, saved_types = native.clone(), relations.clone()
        edges, types, _ = build.build_augmented_graph(
            native, relations, feature_fixture(), **settings(max_neighbors=1))
        actual = triples(edges, types)
        expected = {(0, 3, 0), (3, 0, 0), (0, 3, 4), (3, 0, 4),
                    (1, 3, 5), (3, 1, 5), (2, 3, 5), (3, 2, 5)}
        self.assertEqual(actual, expected)
        self.assertEqual(edges.shape[1], len(actual))
        self.assertTrue(torch.equal(native, saved_edges))
        self.assertTrue(torch.equal(relations, saved_types))
        self.assertTrue(all(u != v for u, v, _ in actual))
        self.assertTrue(all((v, u, r) in actual for u, v, r in actual))

    def test_fractional_native_endpoints_and_relations_are_rejected(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        relations = torch.tensor([0, 0])
        with self.assertRaises(ValueError):
            build.build_augmented_graph(edges.float() + .2, relations, feature_fixture(), **settings())
        with self.assertRaises(ValueError):
            build.build_augmented_graph(edges, relations.float() + .2, feature_fixture(), **settings())

    def test_invalid_native_bounds_and_negative_relations_are_rejected(self):
        with self.assertRaises(ValueError):
            build.build_augmented_graph(torch.tensor([[0, 4], [1, 0]]), torch.tensor([0, 0]),
                                        feature_fixture(), **settings())
        with self.assertRaises(ValueError):
            build.build_augmented_graph(torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, -1]),
                                        feature_fixture(), **settings())

    def test_nonfinite_features_are_rejected(self):
        features = feature_fixture()
        features[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            build.build_augmented_graph(torch.tensor([[0, 1], [1, 0]]), torch.tensor([0, 0]),
                                        features, **settings())


class PreparedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        write_inputs(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **overrides):
        return build.prepare_material_graph(self.directory, **settings(**overrides))

    def test_prepare_preserves_native_inputs_and_strict_loader_roundtrip(self):
        before = {name: (self.directory / name).read_bytes() for name in INPUT_FILES}
        manifest = self.prepare()
        edges, types, loaded = build.load_prepared_material_graph(self.directory)
        self.assertEqual({k: v for k, v in manifest.items() if k != "cache_reused"}, loaded)
        self.assertTrue(all((self.directory / name).read_bytes() == value
                            for name, value in before.items()))
        self.assertEqual(edges.shape[1], len(types))
        self.assertEqual(edges.shape[1], len(triples(edges, types)))

    def test_loader_refuses_native_only_without_manifest(self):
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)

    def test_all_six_changed_inputs_invalidate_prepared_cache(self):
        for name in INPUT_FILES:
            with self.subTest(input=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                write_inputs(directory)
                build.prepare_material_graph(directory, **settings())
                value = torch.load(directory / name, map_location="cpu", weights_only=True)
                if name == "edge_index.pt":
                    value[0, 0] = 1
                elif name == "edge_type.pt":
                    value[0] = 2
                else:
                    value[0, 0] += .25
                torch.save(value, directory / name)
                with self.assertRaises(CACHE_ERRORS):
                    build.load_prepared_material_graph(directory)

    def test_corrupted_output_is_rejected(self):
        self.prepare()
        edges, types, _ = build.load_prepared_material_graph(self.directory)
        # Discover the new edge artifact from its tensor shape, without
        # assuming a public filename beyond the manifest contract.
        candidates = [p for p in self.directory.glob("*.pt") if p.name not in INPUT_FILES]
        target = next(p for p in candidates if torch.equal(
            torch.load(p, map_location="cpu", weights_only=True), edges))
        torch.save(edges[:, :-1], target)
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)

    def test_corrupted_relation_output_is_rejected(self):
        self.prepare()
        path = self.directory / build.TYPE_FILE
        types = torch.load(path, map_location="cpu", weights_only=True)
        types[0] += 1
        torch.save(types, path)
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)

    def test_same_inputs_and_settings_reuse_verified_cache(self):
        self.prepare()
        path = self.directory / build.MANIFEST_FILE
        before = path.read_bytes()
        reused = self.prepare()
        self.assertTrue(reused["cache_reused"])
        self.assertEqual(before, path.read_bytes())

    def test_invalid_algorithm_version_is_rejected(self):
        self.prepare()
        path = self.directory / build.MANIFEST_FILE
        manifest = json.loads(path.read_text())
        manifest["algorithm_version"] = "old-native-only-version"
        path.write_text(json.dumps(manifest))
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)

    def test_settings_inconsistent_with_prepared_history_are_rejected(self):
        self.prepare()
        path = self.directory / build.MANIFEST_FILE
        manifest = json.loads(path.read_text())
        manifest["settings"]["max_neighbors"] = 1
        path.write_text(json.dumps(manifest))
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)

    def test_cuda_prepared_cache_can_be_read_on_cpu_without_reconstruction(self):
        self.prepare()
        path = self.directory / build.MANIFEST_FILE
        manifest = json.loads(path.read_text())
        manifest["settings"]["device"] = "cuda:0"
        path.write_text(json.dumps(manifest))
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(build, "build_augmented_graph", side_effect=AssertionError("must not reconstruct")):
            edges, types, loaded = build.load_prepared_material_graph(self.directory)
        self.assertEqual(edges.device.type, "cpu")
        self.assertEqual(types.device.type, "cpu")
        self.assertEqual(loaded["settings"]["device"], "cuda:0")

    def test_cuda_construction_requires_available_cuda(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(ValueError):
                self.prepare(device="cuda:0", force=True)

    def test_parameter_change_rebuilds_instead_of_reusing_old_manifest(self):
        first = self.prepare(max_neighbors=1)
        second = self.prepare(max_neighbors=3)
        self.assertNotEqual(first, second)
        _, _, manifest = build.load_prepared_material_graph(self.directory)
        self.assertEqual({k: v for k, v in second.items() if k != "cache_reused"}, manifest)

    def test_cli_reports_missing_inputs_and_can_recover_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as missing:
            command = [sys.executable, str(RUN_PATH), "--data-dir", missing, "--device", "cpu"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("edge_index.pt", result.stdout + result.stderr)
        self.prepare()
        generated = [p for p in self.directory.glob("*.pt") if p.name not in INPUT_FILES]
        self.assertEqual(len(generated), 2)
        generated[0].unlink()
        with self.assertRaises(CACHE_ERRORS):
            build.load_prepared_material_graph(self.directory)
        command = [sys.executable, str(RUN_PATH), "--data-dir", str(self.directory),
                   "--max-neighbors", "2", "--query-chunk-size", "2",
                   "--reference-chunk-size", "2", "--device", "cpu"]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        build.load_prepared_material_graph(self.directory)

    def test_training_wrapper_requires_augmented_default_and_native_is_explicit(self):
        loader = extract_training_loader()
        with self.assertRaises(CACHE_ERRORS):
            loader(self.directory)
        native = loader(self.directory, graph_mode="native")
        self.assertGreaterEqual(len(native), 2)
        self.assertEqual(triples(native[0], native[1]),
                         {(0, 3, 0), (3, 0, 0), (0, 3, 4), (3, 0, 4)})
        self.prepare()
        prepared = loader(self.directory)
        expected_edges, expected_types, _ = build.load_prepared_material_graph(self.directory)
        self.assertTrue(torch.equal(prepared[0], expected_edges))
        self.assertTrue(torch.equal(prepared[1], expected_types))


if __name__ == "__main__":
    unittest.main()
