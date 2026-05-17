"""Data augmentation pipeline for voice model training.

Controlled by a single 'intensity' parameter (0.0 - 1.0) that scales:
- Noise injection (Gaussian + environmental)
- Pitch shifting (random ± semitones)
- Time stretching (slight speed changes)

Higher intensity = more aggressive augmentation = more augmented copies.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AugmentConfig:
    intensity: float
    noise_enabled: bool
    noise_min_snr_db: float
    noise_max_snr_db: float
    pitch_enabled: bool
    pitch_max_semitones: float
    stretch_enabled: bool
    stretch_max_ratio: float
    copies: int

    @classmethod
    def from_intensity(cls, intensity: float) -> "AugmentConfig":
        intensity = max(0.0, min(1.0, intensity))
        return cls(
            intensity=intensity,
            noise_enabled=intensity > 0.1,
            noise_min_snr_db=_lerp(30.0, 15.0, intensity),
            noise_max_snr_db=_lerp(50.0, 25.0, intensity),
            pitch_enabled=intensity > 0.05,
            pitch_max_semitones=_lerp(0.5, 2.0, intensity),
            stretch_enabled=intensity > 0.2,
            stretch_max_ratio=_lerp(1.02, 1.10, intensity),
            copies=int(_lerp(1.0, 4.0, intensity)),
        )


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    config: AugmentConfig,
    sample_rate: int = 40000,
) -> int:
    """Augment all audio files in input_dir, write augmented copies to output_dir.

    Returns the number of augmented files created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    created = 0

    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.flac"))

    for audio_path in audio_files:
        audio, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)

        # Always copy original
        out_path = output_dir / audio_path.name
        sf.write(str(out_path), audio, sample_rate)
        created += 1

        # Create augmented copies
        for copy_idx in range(config.copies):
            augmented = audio.copy()

            if config.noise_enabled:
                augmented = _add_noise(augmented, config, rng)

            if config.pitch_enabled:
                augmented = _pitch_shift(augmented, sample_rate, config, rng)

            if config.stretch_enabled:
                augmented = _time_stretch(augmented, config, rng)

            stem = audio_path.stem
            aug_name = f"{stem}_aug{copy_idx}.wav"
            sf.write(str(output_dir / aug_name), augmented, sample_rate)
            created += 1

    return created


def _add_noise(audio: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    snr_db = rng.uniform(config.noise_min_snr_db, config.noise_max_snr_db)
    signal_power = np.mean(audio**2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return audio + noise


def _pitch_shift(
    audio: np.ndarray, sr: int, config: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    semitones = rng.uniform(-config.pitch_max_semitones, config.pitch_max_semitones)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def _time_stretch(audio: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    ratio = rng.uniform(1.0 / config.stretch_max_ratio, config.stretch_max_ratio)
    return librosa.effects.time_stretch(audio, rate=ratio)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t
