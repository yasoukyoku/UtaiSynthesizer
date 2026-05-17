use std::path::PathBuf;
use std::sync::Arc;
use tauri::State;

use crate::project::Project;
use crate::AppState;

#[derive(serde::Serialize)]
pub struct ProjectInfo {
    pub name: String,
    pub path: Option<String>,
    pub dirty: bool,
    pub track_count: usize,
    pub tempo: f64,
}

#[tauri::command]
pub async fn new_project(
    state: State<'_, Arc<AppState>>,
    name: String,
) -> Result<ProjectInfo, String> {
    let project = Project::new(&name);
    let info = project_to_info(&project);
    *state.project.write() = Some(project);
    Ok(info)
}

#[tauri::command]
pub async fn open_project(
    state: State<'_, Arc<AppState>>,
    path: String,
) -> Result<ProjectInfo, String> {
    let project = Project::load(&PathBuf::from(&path)).map_err(|e| e.to_string())?;
    let info = project_to_info(&project);
    *state.project.write() = Some(project);
    Ok(info)
}

#[tauri::command]
pub async fn save_project(
    state: State<'_, Arc<AppState>>,
    path: Option<String>,
) -> Result<(), String> {
    let mut proj = state.project.write();
    let project = proj
        .as_mut()
        .ok_or_else(|| "No project open".to_string())?;

    let save_path = path
        .map(PathBuf::from)
        .or_else(|| project.file_path.clone())
        .ok_or_else(|| "No save path specified".to_string())?;

    project.save(&save_path).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn get_project_state(
    state: State<'_, Arc<AppState>>,
) -> Result<Option<ProjectInfo>, String> {
    let proj = state.project.read();
    Ok(proj.as_ref().map(project_to_info))
}

fn project_to_info(project: &Project) -> ProjectInfo {
    ProjectInfo {
        name: project.metadata.name.clone(),
        path: project.file_path.as_ref().map(|p| p.to_string_lossy().to_string()),
        dirty: project.dirty,
        track_count: project.tracks.len(),
        tempo: project.tempo,
    }
}
