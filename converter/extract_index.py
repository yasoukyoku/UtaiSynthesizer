"""Extract feature vectors from a FAISS .index file to .npy format.

RVC models ship with FAISS index files for feature retrieval.
This script extracts the raw vectors so the Rust backend can
load them without a FAISS dependency.

Usage:
    python extract_index.py --input model.index --output model.npy

Requires: faiss-cpu (pip install faiss-cpu)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def extract(input_path: Path, output_path: Path):
    try:
        import faiss
    except ImportError:
        print("ERROR: faiss-cpu is required. Install with: pip install faiss-cpu", file=sys.stderr)
        sys.exit(1)

    index = faiss.read_index(str(input_path))
    n = index.ntotal
    dim = index.d

    vectors = index.reconstruct_n(0, n).astype(np.float32)
    np.save(str(output_path), vectors)

    print(f"Extracted {n} vectors x {dim} dim -> {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Extract FAISS index to .npy")
    parser.add_argument("--input", type=str, required=True, help="Input .index file")
    parser.add_argument("--output", type=str, required=True, help="Output .npy file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    extract(input_path, output_path)


if __name__ == "__main__":
    main()
