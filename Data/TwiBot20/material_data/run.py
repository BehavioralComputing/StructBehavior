"""Prepare exact feature-neighbor material edges from downloaded native tensors."""

import argparse
from pathlib import Path
import sys

try:
    from Data.TwiBot20.material_data.build_graph import MaterialGraphError, prepare_material_graph
except ModuleNotFoundError:
    from build_graph import MaterialGraphError, prepare_material_graph


def main(argv=()):
    # Data/runme.py calls main() without forwarding its own --dataset arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--max-neighbors", type=int, default=50, help="Maximum nonself neighbors; capped at N-1.")
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--reference-chunk-size", type=int, default=8192)
    parser.add_argument("--edge-chunk-size", type=int, default=8192)
    parser.add_argument("--entropy-tolerance", type=float, default=1e-3)
    parser.add_argument("--cosine-epsilon", type=float, default=1e-8)
    parser.add_argument("--pearson-epsilon", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true", help="Rebuild even when all input/output hashes match.")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_material_graph(
            args.data_dir, force=args.force, device=args.device,
            max_neighbors=args.max_neighbors, query_chunk_size=args.query_chunk_size,
            reference_chunk_size=args.reference_chunk_size, edge_chunk_size=args.edge_chunk_size,
            entropy_tolerance=args.entropy_tolerance, cosine_epsilon=args.cosine_epsilon,
            pearson_epsilon=args.pearson_epsilon,
        )
    except (MaterialGraphError, ValueError, OSError) as error:
        parser.exit(2, f"Material graph preparation failed: {error}\n")
    details = manifest["details"]
    action = "Reused verified" if manifest["cache_reused"] else "Prepared"
    print(f"{action} material graph: k={details['selected_k']}, "
          f"edges={details['output_edges']}, feature_relation={details['feature_relation_id']}")
    print("Native input files were preserved. No experiment training was run.")
    return manifest


if __name__ == "__main__":
    main(sys.argv[1:])
