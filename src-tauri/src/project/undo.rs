use serde::{Deserialize, Serialize};

const MAX_UNDO_HISTORY: usize = 100;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UndoStack {
    #[serde(skip)]
    history: Vec<UndoEntry>,
    #[serde(skip)]
    position: usize,
}

#[derive(Debug, Clone)]
struct UndoEntry {
    description: String,
    snapshot: String,
}

impl UndoStack {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            position: 0,
        }
    }

    pub fn push(&mut self, description: &str, state_json: String) {
        self.history.truncate(self.position);

        self.history.push(UndoEntry {
            description: description.to_string(),
            snapshot: state_json,
        });

        if self.history.len() > MAX_UNDO_HISTORY {
            self.history.remove(0);
        }

        self.position = self.history.len();
    }

    pub fn undo(&mut self) -> Option<&str> {
        if self.position > 1 {
            self.position -= 1;
            Some(&self.history[self.position - 1].snapshot)
        } else {
            None
        }
    }

    pub fn redo(&mut self) -> Option<&str> {
        if self.position < self.history.len() {
            self.position += 1;
            Some(&self.history[self.position - 1].snapshot)
        } else {
            None
        }
    }

    pub fn can_undo(&self) -> bool {
        self.position > 1
    }

    pub fn can_redo(&self) -> bool {
        self.position < self.history.len()
    }

    pub fn clear(&mut self) {
        self.history.clear();
        self.position = 0;
    }
}
