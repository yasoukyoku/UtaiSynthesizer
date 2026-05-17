"""Index file generation for RVC voice models.

Generates a .npy KNN index from HuBERT features extracted during training.
This index is used at inference time for feature retrieval (timbre matching).

CRITICAL: This must run on ANY training stop (completion, user stop, error).
"""

import numpy as np
from pathlib import Path

from .progress import ProgressReporter


def generate_index(
    features_dir: Path,
    output_path: Path,
    reporter: ProgressReporter | None = None,
) -> bool:
    """Generate .npy index from extracted features.

    Args:
        features_dir: Directory containing .npy feature files from training
        output_path: Path to write the final index .npy file
        reporter: Optional progress reporter

    Returns:
        True if index was generated successfully
    """
    if reporter:
        reporter.report_state("generating_index")

    feature_files = sorted(features_dir.glob("*.npy"))
    if not feature_files:
        return False

    # Load all features into a single matrix
    all_features = []
    for f in feature_files:
        feat = np.load(str(f))
        if feat.ndim == 2:
            all_features.append(feat)
        elif feat.ndim == 1:
            all_features.append(feat.reshape(1, -1))

    if not all_features:
        return False

    features_matrix = np.concatenate(all_features, axis=0).astype(np.float32)

    # Limit index size for memory efficiency
    max_entries = 100000
    if features_matrix.shape[0] > max_entries:
        indices = np.random.choice(features_matrix.shape[0], max_entries, replace=False)
        indices.sort()
        features_matrix = features_matrix[indices]

    # Save as .npy (simple KNN index — at inference time, Rust does brute-force or ANN)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), features_matrix)

    return True
