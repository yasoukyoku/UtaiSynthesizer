pub mod autosave;
pub mod undo;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::{Result, UtaiError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub metadata: ProjectMetadata,
    pub tracks: Vec<Track>,
    pub tempo: f64,
    pub time_signature: (u8, u8),
    #[serde(skip)]
    pub file_path: Option<PathBuf>,
    #[serde(skip)]
    pub dirty: bool,
    #[serde(skip)]
    pub undo_stack: undo::UndoStack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub name: String,
    pub version: String,
    pub created_at: String,
    pub modified_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaneControl {
    pub volume_db: f32,
    pub pan: f32,
    pub muted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Track {
    pub id: String,
    pub name: String,
    pub track_type: TrackType,
    pub segments: Vec<Segment>,
    pub volume_db: f32,
    pub pan: f32,
    pub muted: bool,
    pub solo: bool,
    #[serde(default)]
    pub expanded: bool,
    #[serde(default)]
    pub lane_controls: HashMap<String, LaneControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TrackType {
    Vocal { voice_model: Option<String> },
    Audio,
    Instrument,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessedOutput {
    pub lane_label: String,
    pub audio_path: String,
    pub total_duration_ms: f64,
    #[serde(default)]
    pub waveform_peaks: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub id: String,
    pub start_tick: u64,
    pub duration_ticks: u64,
    pub content: SegmentContent,
    pub workflow: Option<Workflow>,
    #[serde(default)]
    pub processed_outputs: Option<Vec<ProcessedOutput>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum SegmentContent {
    Notes { notes: Vec<Note> },
    AudioClip {
        source_path: String,
        offset_ms: f64,
        total_duration_ms: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub id: String,
    pub tick: u64,
    pub duration: u64,
    pub pitch: u8,
    pub lyric: String,
    pub phoneme: Option<String>,
    pub velocity: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub nodes: Vec<WorkflowNode>,
    pub connections: Vec<WorkflowConnection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowNode {
    pub id: String,
    pub node_type: WorkflowNodeType,
    pub position: Position,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WorkflowNodeType {
    Input,
    Output,
    Rvc,
    #[serde(rename = "sovits")]
    SoVits,
    PitchShift,
    FormantShift,
    AudioEnhance,
    MsstSeparation,
    Split,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkflowConnection {
    pub from_node: String,
    pub from_port: u32,
    pub to_node: String,
    pub to_port: u32,
}

const TICKS_PER_BEAT: u64 = 480;

impl Project {
    pub fn new(name: &str) -> Self {
        let now = chrono_now();
        Self {
            metadata: ProjectMetadata {
                name: name.to_string(),
                version: "2.0.0".to_string(),
                created_at: now.clone(),
                modified_at: now,
            },
            tracks: Vec::new(),
            tempo: 120.0,
            time_signature: (4, 4),
            file_path: None,
            dirty: false,
            undo_stack: undo::UndoStack::new(),
        }
    }

    pub fn save(&mut self, path: &Path) -> Result<()> {
        self.metadata.modified_at = chrono_now();
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        self.file_path = Some(path.to_owned());
        self.dirty = false;
        tracing::info!("Project saved: {}", path.display());
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut project: Self =
            serde_json::from_str(&content).map_err(|e| UtaiError::Project(e.to_string()))?;
        project.file_path = Some(path.to_owned());
        project.dirty = false;
        tracing::info!("Project loaded: {}", path.display());
        Ok(project)
    }

    pub fn add_track(&mut self, track: Track) {
        self.tracks.push(track);
        self.mark_dirty();
    }

    pub fn ticks_to_seconds(&self, ticks: u64) -> f64 {
        let beats = ticks as f64 / TICKS_PER_BEAT as f64;
        beats * 60.0 / self.tempo
    }

    pub fn seconds_to_ticks(&self, seconds: f64) -> u64 {
        let beats = seconds * self.tempo / 60.0;
        (beats * TICKS_PER_BEAT as f64) as u64
    }

    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.metadata.modified_at = chrono_now();
    }
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}
