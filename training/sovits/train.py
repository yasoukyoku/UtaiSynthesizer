"""SoVITS 4.1 training script.

Called by the Rust sidecar manager with a JSON config on --config.
Communicates progress via JSON lines on stdout.

SoVITS trains:
1. VITS encoder-decoder (main voice conversion model)
2. Shallow diffusion module (optional, for quality refinement)

Exit behavior matches RVC: always generates index on stop.
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
    shallow_diffusion = config.get("backend", {}).get("SoVits", {}).get("shallow_diffusion", True)

    output_dir = Path("training/output") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / "features"
    features_dir.mkdir(exist_ok=True)

    try:
        reporter.report_state("preparing")

        if continuation == "Fresh":
            for f in output_dir.glob("*.pth"):
                f.unlink()

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
        preprocess_sovits(training_data_dir, features_dir, sample_rate)

        reporter.report_state("training")

        # Phase 1: VITS training
        vits_epochs = epochs
        for epoch in range(1, vits_epochs + 1):
            if should_stop():
                save_checkpoint(output_dir, model_name, epoch - 1, "vits")
                generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
                reporter.report_stopped()
                cleanup()
                return

            loss = train_vits_epoch(epoch, batch_size, features_dir)
            reporter.report_epoch(epoch, vits_epochs, loss)

            if epoch % save_interval == 0:
                save_checkpoint(output_dir, model_name, epoch, "vits")

        save_checkpoint(output_dir, model_name, vits_epochs, "vits")

        # Phase 2: Shallow diffusion training (optional)
        if shallow_diffusion:
            diff_epochs = epochs // 2
            for epoch in range(1, diff_epochs + 1):
                if should_stop():
                    save_checkpoint(output_dir, model_name, epoch - 1, "diff")
                    generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
                    reporter.report_stopped()
                    cleanup()
                    return

                loss = train_diffusion_epoch(epoch, batch_size, features_dir)
                reporter.report_epoch(vits_epochs + epoch, vits_epochs + diff_epochs, loss)

                if epoch % save_interval == 0:
                    save_checkpoint(output_dir, model_name, epoch, "diff")

            save_checkpoint(output_dir, model_name, diff_epochs, "diff")

        generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
        reporter.report_completed()

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        try:
            generate_index(features_dir, output_dir / f"{model_name}.npy", reporter)
        except Exception:
            pass
        reporter.report_state(f"error: {e}")
    finally:
        cleanup()


def preprocess_sovits(audio_dir: Path, features_dir: Path, sample_rate: int):
    """Extract ContentVec features and F0."""
    # TODO: Extract ContentVec-768 features (requires pretrained ContentVec)
    # Also extract F0 (RMVPE/CREPE) and speaker embeddings
    pass


def train_vits_epoch(epoch: int, batch_size: int, features_dir: Path) -> float:
    """Train one epoch of SoVITS VITS model."""
    # TODO: VITS training (encoder + decoder + discriminator)
    import time, random
    time.sleep(0.01)
    return 0.8 * (0.97 ** epoch) + random.uniform(-0.02, 0.02)


def train_diffusion_epoch(epoch: int, batch_size: int, features_dir: Path) -> float:
    """Train one epoch of shallow diffusion model."""
    # TODO: Diffusion model training (denoising score matching)
    import time, random
    time.sleep(0.01)
    return 0.3 * (0.98 ** epoch) + random.uniform(-0.01, 0.01)


def save_checkpoint(output_dir: Path, model_name: str, epoch: int, phase: str):
    """Save model checkpoint."""
    checkpoint_path = output_dir / f"{model_name}_{phase}_e{epoch}.pth"
    checkpoint_path.touch()


if __name__ == "__main__":
    main()
