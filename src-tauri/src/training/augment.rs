use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentConfig {
    pub intensity: f32,
    pub noise: NoiseConfig,
    pub pitch_shift: PitchShiftConfig,
    pub time_stretch: TimeStretchConfig,
    pub augmented_copies: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub enabled: bool,
    pub min_snr_db: f32,
    pub max_snr_db: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PitchShiftConfig {
    pub enabled: bool,
    pub max_semitones: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeStretchConfig {
    pub enabled: bool,
    pub max_ratio: f32,
}

impl AugmentConfig {
    pub fn from_intensity(intensity: f32) -> Self {
        let intensity = intensity.clamp(0.0, 1.0);

        Self {
            intensity,
            noise: NoiseConfig {
                enabled: intensity > 0.1,
                min_snr_db: lerp(30.0, 15.0, intensity),
                max_snr_db: lerp(50.0, 25.0, intensity),
            },
            pitch_shift: PitchShiftConfig {
                enabled: intensity > 0.05,
                max_semitones: lerp(0.5, 2.0, intensity),
            },
            time_stretch: TimeStretchConfig {
                enabled: intensity > 0.2,
                max_ratio: lerp(1.02, 1.10, intensity),
            },
            augmented_copies: lerp(1.0, 4.0, intensity) as u32,
        }
    }
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self::from_intensity(0.3)
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
