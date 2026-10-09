"""Offline, exact feature-neighbor augmentation of downloaded TwiBot-20 edges.

Native inputs are never overwritten. The prepared graph is an undirected,
unit-weight propagation graph with native relation IDs retained and one extra
relation for newly selected feature-neighbor connections. Entropy is evaluated
on unique undirected support using Pearson-derived weights, not propagation
weights. Building this graph does not rerun or certify manuscript experiments.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
import uuid

import torch

SCHEMA_VERSION = 1
ALGORITHM_VERSION = "twibot20-exact-material-v1"
FEATURE_FILES = (
    "num_properties_tensor.pt", "tweets_tensor.pt",
    "cat_properties_tensor.pt", "des_tensor.pt",
)
INPUT_FILES = ("edge_index.pt", "edge_type.pt") + FEATURE_FILES
EDGE_FILE = "edge_index_augmented.pt"
TYPE_FILE = "edge_type_augmented.pt"
MANIFEST_FILE = "material_graph_manifest.json"
DEFAULT_SETTINGS = {
    "max_neighbors": 50, "entropy_tolerance": 1e-3,
    "query_chunk_size": 256, "reference_chunk_size": 8192,
    "edge_chunk_size": 8192, "device": "auto",
    "cosine_epsilon": 1e-8, "pearson_epsilon": 1e-8,
}


class MaterialGraphError(RuntimeError):
    """Missing, invalid, or stale prepared graph; native fallback is forbidden."""


def _instruction(data_dir):
    script = Path(__file__).with_name("run.py")
    return f"python {shlex.quote(str(script))} --data-dir {shlex.quote(str(data_dir))}"


def _device(value, *, require_available=True):
    if str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        result = torch.device(value)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"Invalid construction device: {value}") from error
    if require_available and result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    return result


def _settings(values):
    unknown = set(values) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown construction settings: {sorted(unknown)}")
    result = {**DEFAULT_SETTINGS, **values}
    for key in ("max_neighbors", "query_chunk_size", "reference_chunk_size", "edge_chunk_size"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    for key in ("entropy_tolerance", "cosine_epsilon", "pearson_epsilon"):
        if isinstance(result[key], bool) or not isinstance(result[key], (int, float)) or not math.isfinite(result[key]):
            raise ValueError(f"{key} must be finite.")
        if result[key] < 0 or (key != "entropy_tolerance" and result[key] == 0):
            raise ValueError(f"{key} must be positive (tolerance may be zero).")
    result["device"] = str(result["device"])
    # A verified cache can be loaded on another machine without its build device.
    _device(result["device"], require_available=False)
    return result


def _validate_features(features):
    if not torch.is_tensor(features) or features.ndim != 2:
        raise ValueError("features must be a two-dimensional tensor.")
    if features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("At least two instances and one feature are required.")
    if not torch.isfinite(features).all():
        raise ValueError("Features contain non-finite values.")


def exact_cosine_topk(features, max_neighbors, *, query_chunk_size=256,
                      reference_chunk_size=8192, device="auto", cosine_epsilon=1e-8):
    """Return CPU indices [N,k], ranked by exact cosine, excluding self.

    Scores use dot/(norm_product+epsilon). Zero vectors are allowed; tied
    neighbors may be returned in any valid top-k order. Both score dimensions
    are bounded by chunks smaller than N, so no N-by-N matrix is allocated.
    """
    _validate_features(features)
    cfg = _settings({"max_neighbors": max_neighbors, "query_chunk_size": query_chunk_size,
                     "reference_chunk_size": reference_chunk_size, "device": str(device),
                     "cosine_epsilon": cosine_epsilon})
    n = features.shape[0]
    k = min(max_neighbors, n - 1)
    target = _device(cfg["device"])
    x = features.detach().to(target, dtype=torch.float32)
    norms = torch.linalg.vector_norm(x, dim=1)
    q_chunk = min(query_chunk_size, max(1, n // 2))
    r_chunk = min(reference_chunk_size, max(1, n // 2))
    output = torch.empty((n, k), dtype=torch.long)
    with torch.no_grad():
        for q0 in range(0, n, q_chunk):
            q1 = min(q0 + q_chunk, n)
            best_scores = torch.empty((q1 - q0, 0), device=target)
            best_indices = torch.empty((q1 - q0, 0), dtype=torch.long, device=target)
            queries = torch.arange(q0, q1, device=target)
            for r0 in range(0, n, r_chunk):
                r1 = min(r0 + r_chunk, n)
                candidates = torch.arange(r0, r1, device=target)
                scores = x[q0:q1] @ x[r0:r1].t()
                denominator = norms[q0:q1, None] * norms[None, r0:r1] + cosine_epsilon
                scores = scores / denominator
                scores.masked_fill_(queries[:, None] == candidates[None, :], float("-inf"))
                take = min(k, r1 - r0)
                values, positions = scores.topk(take, dim=1, sorted=True)
                indices = positions + r0
                combined_scores = torch.cat((best_scores, values), dim=1)
                combined_indices = torch.cat((best_indices, indices), dim=1)
                keep = min(k, combined_scores.shape[1])
                best_scores, chosen = combined_scores.topk(keep, dim=1, sorted=True)
                best_indices = combined_indices.gather(1, chosen)
            if not torch.isfinite(best_scores).all():
                raise ValueError("Cosine search failed to find enough finite nonself neighbors.")
            output[q0:q1] = best_indices.cpu()
    return output


def _native_support(edge_index, n):
    if not torch.is_tensor(edge_index) or edge_index.dtype != torch.long:
        raise ValueError("native_edge_index must be an int64 tensor; values are not coerced.")
    edges = edge_index.detach().cpu()
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("native_edge_index must have shape [2,E].")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("Native edge endpoints are outside the feature population.")
    lo = torch.minimum(edges[0], edges[1])
    hi = torch.maximum(edges[0], edges[1])
    return torch.unique((lo * n + hi)[lo != hi], sorted=True)


def _feature_support(neighbors, k, n):
    if not torch.is_tensor(neighbors) or neighbors.dtype != torch.long:
        raise ValueError("Neighbor indices must be an int64 tensor.")
    if neighbors.ndim != 2 or neighbors.shape[0] != n or not 1 <= k <= neighbors.shape[1]:
        raise ValueError("Invalid neighbor matrix or selected k.")
    destinations = neighbors[:, :k].reshape(-1).cpu()
    sources = torch.arange(n).repeat_interleave(k)
    if destinations.numel() and (destinations.min() < 0 or destinations.max() >= n):
        raise ValueError("Neighbor endpoints are outside the instance population.")
    if torch.any(sources == destinations):
        raise ValueError("Feature neighbors must exclude the instance itself.")
    return torch.unique(torch.minimum(sources, destinations) * n
                        + torch.maximum(sources, destinations), sorted=True)


class _EntropyContext:
    def __init__(self, native_edge_index, features, *, device="auto",
                 edge_chunk_size=8192, pearson_epsilon=1e-8):
        _validate_features(features)
        self.n = features.shape[0]
        self.native = _native_support(native_edge_index, self.n)
        self.device = _device(device)
        self.chunk = edge_chunk_size
        self.epsilon = pearson_epsilon
        # Double precision preserves small positive shifts near anticorrelation.
        x = features.detach().to(self.device, dtype=torch.float64, copy=True)
        x.sub_(x.mean(dim=1, keepdim=True))
        self.centered = x
        self.norms = torch.linalg.vector_norm(self.centered, dim=1)

    def evaluate(self, neighbors, k):
        feature_pairs = _feature_support(neighbors, k, self.n)
        pairs = torch.unique(torch.cat((self.native, feature_pairs)), sorted=True)
        degree = torch.zeros(self.n, dtype=torch.float64, device=self.device)
        edge_counts = torch.zeros_like(degree)
        shifted_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for part in pairs.split(self.chunk):
                u = (part // self.n).to(self.device)
                v = (part % self.n).to(self.device)
                numerator = (self.centered[u] * self.centered[v]).sum(dim=1)
                denominator = self.norms[u] * self.norms[v] + self.epsilon
                correlation = (numerator / denominator).clamp(-1.0, 1.0)
                shifted = 1.0 + correlation
                degree.index_add_(0, u, shifted)
                degree.index_add_(0, v, shifted)
                ones = torch.ones_like(shifted)
                edge_counts.index_add_(0, u, ones)
                edge_counts.index_add_(0, v, ones)
                shifted_sum += shifted.sum()
            smoothing = shifted_sum / pairs.numel() / (2 * self.n)
            degree += smoothing * edge_counts
            volume = degree.sum()
            if not torch.isfinite(volume) or volume <= 0:
                raise ValueError("Entropy requires a positive finite support volume.")
            occupancy = degree[degree > 0] / volume
            return float(-(occupancy * torch.log2(occupancy)).sum().cpu())


def weighted_entropy_for_k(native_edge_index, features, neighbor_indices, k, *,
                           device="auto", edge_chunk_size=8192, pearson_epsilon=1e-8):
    """Entropy on unique undirected union, using 1+Pearson+mean/(2N)."""
    cfg = _settings({"device": str(device), "edge_chunk_size": edge_chunk_size,
                     "pearson_epsilon": pearson_epsilon})
    context = _EntropyContext(native_edge_index, features, **{
        key: cfg[key] for key in ("device", "edge_chunk_size", "pearson_epsilon")})
    return context.evaluate(neighbor_indices, k)


def _step(delta):
    return 5 if delta > 0.1 else 3 if delta > 0.01 else 1


def select_neighbor_count(entropy_for_k, max_neighbors, *, entropy_tolerance=1e-3):
    """Search k=1 upward, comparing the initial entropy with zero.

    A positive increment above .1/.01/.001 advances by 5/3/1. Search stops
    when delta <= tolerance or at the evaluated upper bound; no k exceeds it.
    """
    _settings({"max_neighbors": max_neighbors, "entropy_tolerance": entropy_tolerance})
    k, previous, history = 1, 0.0, []
    while True:
        entropy = float(entropy_for_k(k))
        if not math.isfinite(entropy):
            raise ValueError("Entropy search returned a non-finite value.")
        delta = entropy - previous
        history.append({"k": k, "entropy": entropy, "delta": delta})
        if delta <= entropy_tolerance or k == max_neighbors:
            return k, history
        previous = entropy
        k = min(max_neighbors, k + _step(delta))


def build_augmented_graph(native_edge_index, native_edge_type, features, **settings):
    """Return (undirected unit-weight edge_index, edge_type, construction details)."""
    cfg = _settings(settings)
    _validate_features(features)
    n = features.shape[0]
    native_support = _native_support(native_edge_index, n)
    native = native_edge_index.detach().cpu()
    if not torch.is_tensor(native_edge_type) or native_edge_type.dtype != torch.long:
        raise ValueError("native_edge_type must be an int64 tensor; values are not coerced.")
    types = native_edge_type.detach().cpu()
    if types.ndim != 1 or types.numel() != native.shape[1] or (types.numel() and types.min() < 0):
        raise ValueError("Native edge types must be nonnegative and match native edges.")
    cap = min(cfg["max_neighbors"], n - 1)
    neighbors = exact_cosine_topk(features, cap, **{
        key: cfg[key] for key in ("query_chunk_size", "reference_chunk_size", "device", "cosine_epsilon")})
    context = _EntropyContext(native, features, **{
        key: cfg[key] for key in ("device", "edge_chunk_size", "pearson_epsilon")})
    selected, history = select_neighbor_count(
        lambda k: context.evaluate(neighbors, k), cap,
        entropy_tolerance=cfg["entropy_tolerance"])
    pairs = _feature_support(neighbors, selected, n)
    if native_support.numel():
        positions = torch.searchsorted(native_support, pairs)
        present = positions < native_support.numel()
        present &= native_support[positions.clamp(max=native_support.numel() - 1)] == pairs
        pairs = pairs[~present]
    mask = native[0] != native[1]
    native_triplets = torch.cat((native[:, mask].t(), types[mask, None]), dim=1)
    reverse = native_triplets[:, [1, 0, 2]]
    native_triplets = torch.unique(torch.cat((native_triplets, reverse)), dim=0, sorted=True)
    new_type = int(types.max()) + 1 if types.numel() else 0
    u, v = pairs // n, pairs % n
    feature_edges = torch.stack((torch.cat((u, v)), torch.cat((v, u))))
    output_edges = torch.cat((native_triplets[:, :2].t(), feature_edges), dim=1).contiguous()
    output_types = torch.cat((native_triplets[:, 2], torch.full((feature_edges.shape[1],), new_type)))
    details = {
        "num_instances": n, "feature_width": features.shape[1],
        "effective_max_neighbors": cap, "selected_k": selected,
        "feature_relation_id": new_type, "native_input_edges": native.shape[1],
        "native_output_edges": native_triplets.shape[0],
        "new_feature_edges": feature_edges.shape[1], "output_edges": output_edges.shape[1],
        "entropy_history": history, "propagation_weights": "unit",
    }
    return output_edges, output_types, details


def _sha_handle(handle):
    digest = hashlib.sha256()
    while True:
        block = handle.read(1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _sha(path):
    with Path(path).open("rb") as handle:
        return _sha_handle(handle)


def _load_tensor(handle):
    return torch.load(handle, map_location="cpu", weights_only=True)


def _checked_load(path, expected_hash):
    # Hash and load the same open inode, avoiding a rename-between-check/load race.
    with Path(path).open("rb") as handle:
        if _sha_handle(handle) != expected_hash:
            raise MaterialGraphError(f"Hash mismatch: {path}")
        handle.seek(0)
        tensor = _load_tensor(handle)
    if not torch.is_tensor(tensor):
        raise MaterialGraphError(f"Expected tensor: {path}")
    return tensor


def _read_manifest(data_dir):
    directory = Path(data_dir)
    try:
        raw = (directory / MANIFEST_FILE).read_bytes()
        manifest = json.loads(raw)
        if manifest["schema_version"] != SCHEMA_VERSION or manifest["algorithm_version"] != ALGORITHM_VERSION:
            raise ValueError("Unsupported construction version.")
        if set(manifest["settings"]) != set(DEFAULT_SETTINGS):
            raise ValueError("Missing or unrecognized construction settings.")
        _settings(manifest["settings"])
        if set(manifest["inputs"]) != set(INPUT_FILES) or set(manifest["outputs"]) != {EDGE_FILE, TYPE_FILE}:
            raise ValueError("Manifest file set does not match the construction contract.")
        details = manifest["details"]
        cap = min(manifest["settings"]["max_neighbors"], details["num_instances"] - 1)
        if details["effective_max_neighbors"] != cap or not 1 <= details["selected_k"] <= cap:
            raise ValueError("Inconsistent neighbor bounds.")
        history = details["entropy_history"]
        expected_k, previous = 1, 0.0
        for index, row in enumerate(history):
            if row["k"] != expected_k or not math.isfinite(row["entropy"]):
                raise ValueError("Invalid entropy history.")
            delta = row["entropy"] - previous
            if not math.isclose(delta, row["delta"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Inconsistent entropy increments.")
            stopped = delta <= manifest["settings"]["entropy_tolerance"] or expected_k == cap
            if stopped != (index == len(history) - 1):
                raise ValueError("Inconsistent entropy stopping point.")
            previous = row["entropy"]
            expected_k = min(cap, expected_k + _step(delta))
        if not history or history[-1]["k"] != details["selected_k"]:
            raise ValueError("Missing selected-neighborhood history.")
        for name in INPUT_FILES:
            entry = manifest["inputs"][name]
            if (directory / name).stat().st_size != entry["bytes"] or _sha(directory / name) != entry["sha256"]:
                raise ValueError(f"Input changed: {name}")
        return manifest, raw
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise MaterialGraphError(f"Prepared material graph missing, invalid, or stale: {error}. Run: {_instruction(directory)}") from error


def load_prepared_material_graph(data_dir):
    """Read only verified prepared edges; never silently fall back to native."""
    directory = Path(data_dir)
    manifest, original = _read_manifest(directory)
    try:
        edges = _checked_load(directory / EDGE_FILE, manifest["outputs"][EDGE_FILE]["sha256"])
        types = _checked_load(directory / TYPE_FILE, manifest["outputs"][TYPE_FILE]["sha256"])
        if edges.dtype != torch.long or types.dtype != torch.long or edges.shape != (2, types.numel()) or types.ndim != 1:
            raise ValueError("Invalid prepared tensor shapes or dtypes.")
        n = manifest["details"]["num_instances"]
        if edges.shape[1] != manifest["details"]["output_edges"] or (edges.numel() and (edges.min() < 0 or edges.max() >= n)):
            raise ValueError("Prepared edge count or endpoints disagree with manifest.")
        if torch.any(edges[0] == edges[1]) or torch.any(types < 0):
            raise ValueError("Prepared graph contains self edges or negative relation IDs.")
        if (directory / MANIFEST_FILE).read_bytes() != original:
            raise ValueError("Construction changed during loading; retry after preparation completes.")
        return edges, types, manifest
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise MaterialGraphError(f"Cannot load verified material graph: {error}. Run: {_instruction(directory)}") from error


@contextmanager
def _build_lock(directory):
    import fcntl
    with (directory / ".material_graph_build.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MaterialGraphError("Another material graph preparation is running.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare_material_graph(data_dir, *, force=False, **settings):
    """Build or reuse a SHA-verified offline graph without touching native inputs."""
    directory = Path(data_dir).resolve()
    cfg = _settings(settings)
    missing = [name for name in INPUT_FILES if not (directory / name).is_file()]
    if missing:
        raise MaterialGraphError(f"Missing native inputs: {', '.join(missing)}. Place the downloaded tensors in {directory}; then run: {_instruction(directory)}")
    with _build_lock(directory):
        if not force:
            try:
                existing, _ = _read_manifest(directory)
                if existing["settings"] == cfg:
                    for name in (EDGE_FILE, TYPE_FILE):
                        if _sha(directory / name) != existing["outputs"][name]["sha256"]:
                            raise MaterialGraphError(f"Output changed: {name}")
                    return {**existing, "cache_reused": True}
            except (OSError, MaterialGraphError):
                pass
        inputs, tensors = {}, {}
        for name in INPUT_FILES:
            with (directory / name).open("rb") as handle:
                digest = _sha_handle(handle)
                size = handle.tell()
                handle.seek(0)
                tensors[name] = _load_tensor(handle)
            if not torch.is_tensor(tensors[name]):
                raise MaterialGraphError(f"Expected input tensor: {name}")
            inputs[name] = {"sha256": digest, "bytes": size}
        blocks = [tensors[name] for name in FEATURE_FILES]
        if any(block.ndim != 2 for block in blocks) or len({block.shape[0] for block in blocks}) != 1:
            raise MaterialGraphError("All four feature tensors must be 2D with equal instance counts.")
        features = torch.cat([block.float() for block in blocks], dim=1)
        edges, types, details = build_augmented_graph(tensors["edge_index.pt"], tensors["edge_type.pt"], features, **cfg)
        manifest = {
            "schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
            "generation_id": str(uuid.uuid4()), "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": cfg, "inputs": inputs, "outputs": {}, "details": details,
        }
        with tempfile.TemporaryDirectory(prefix=".material_graph_staging_", dir=directory) as staging:
            staging = Path(staging)
            for name, tensor in ((EDGE_FILE, edges), (TYPE_FILE, types)):
                with (staging / name).open("wb") as handle:
                    torch.save(tensor, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                manifest["outputs"][name] = {"sha256": _sha(staging / name), "bytes": (staging / name).stat().st_size}
            # Detect inputs changed during a long exact-neighbor construction.
            for name in INPUT_FILES:
                if _sha(directory / name) != inputs[name]["sha256"]:
                    raise MaterialGraphError(f"Input changed during construction: {name}; no new manifest published.")
            manifest_path = staging / MANIFEST_FILE
            with manifest_path.open("w") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging / EDGE_FILE, directory / EDGE_FILE)
            os.replace(staging / TYPE_FILE, directory / TYPE_FILE)
            os.replace(manifest_path, directory / MANIFEST_FILE)
        return {**manifest, "cache_reused": False}
