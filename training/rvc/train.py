"""RVC v2 training script.

Called by the Rust sidecar manager with a JSON config on --config.
Communicates progress via JSON lines on stdout.

Exit behavior:
- Normal completion: save checkpoint → generate index → report completed
- Stop signal received: finish batch → save checkpoint → generate index → report stopped
- Error/OOM: attempt to save checkpoint → generate index → report error
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common.progress import ProgressReporter
from common.stop_signal import should_stop, cleanup
from common.index_gen import generate_index
from common.augment import AugmentConfig, augment_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = json.loads(args.config)
    reporter = ProgressReporter()

    model_name = config["model_name"]
    dataset_path = Path(config["dataset_path"])
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    sample_rate = config["sample_rate"]
    save_interval = config.get("save_interval", 25)
    augmentation = config.get("augmentation")
    continuation = config.get("continuation", "Fresh")

    output_dir = Path("training/output") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    try:
        reporter.report_state("preparing")

        # Handle continuation mode
        if continuation == "Fresh":
            # Clean start — remove existing checkpoints
            for f in output_dir.glob("*.pth"):
                f.unlink()
        # "Continue" mode keeps existing checkpoints

        # Data augmentation
        processed_dir = output_dir / "processed_audio"
        if augmentation and augmentation.get("intensity", 0) > 0:
            reporter.report_state("preprocessing")
            aug_config = AugmentConfig.from_intensity(augmentation["intensity"])
            augment_dataset(dataset_path, processed_dir, aug_config, sample_rate)
            training_data_dir = processed_dir
        else:
            training_data_dir = dataset_path

        reporter.report_state("preprocessing")
        # Feature extraction (HuBERT, F0)
        preprocess(training_data_dir, features_dir, sample_rate)

        reporter.report_state("training")

        # Training loop
        for epoch in range(1, epochs + 1):
            if should_stop():
                save_checkpoint(output_dir, model_name, epoch - 1)
                generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
                reporter.report_stopped()
                cleanup()
                return

            loss = train_one_epoch(epoch, batch_size, features_dir)
            reporter.report_epoch(epoch, epochs, loss)

            if epoch % save_interval == 0:
                save_checkpoint(output_dir, model_name, epoch)

        # Final save
        save_checkpoint(output_dir, model_name, epochs)
        generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
        reporter.report_completed()

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        # Always try to generate index even on error
        try:
            generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
        except Exception:
            pass
        reporter.report_state(f"error: {e}")
    finally:
        cleanup()


def preprocess(audio_dir: Path, features_dir: Path, sample_rate: int):
    """Extract HuBERT features and F0 from audio files."""
    # TODO: Implement actual feature extraction
    # 1. Load each audio file
    # 2. Extract HuBERT-base features (requires pretrained HuBERT model)
    # 3. Extract F0 using RMVPE/CREPE
    # 4. Save as .npy files in features_dir
    pass


def train_one_epoch(epoch: int, batch_size: int, features_dir: Path) -> float:
    """Train one epoch of RVC model.

    Returns the average loss for this epoch.
    """
    # TODO: Implement actual RVC training loop
    # This would involve:
    # 1. Load VITS generator + discriminator
    # 2. Create dataloader from features
    # 3. Forward pass through generator
    # 4. Compute adversarial + reconstruction + KL losses
    # 5. Backward pass + optimizer step
    import time
    time.sleep(0.01)  # Placeholder
    import random
    return 0.5 * (0.95 ** epoch) + random.uniform(-0.01, 0.01)


def save_checkpoint(output_dir: Path, model_name: str, epoch: int):
    """Save model checkpoint."""
    # TODO: Save actual PyTorch state dict
    checkpoint_path = output_dir / f"{model_name}_e{epoch}.pth"
    checkpoint_path.touch()


if __name__ == "__main__":
    main()
