export interface Note {
  pitch: number;
  startTime: number;
  duration: number;
  velocity: number;
}

export function quantizeNotes(notes: Note[], gridSize: number): Note[] {
  return notes.map(note => ({
    ...note,
    startTime: Math.round(note.startTime / gridSize) * gridSize,
    duration: Math.max(gridSize, Math.round(note.duration / gridSize) * gridSize)
  }));
}

export function humanizeNotes(
  notes: Note[],
  options: {
    timingVariation?: number;
    velocityVariation?: number;
  } = {}
): Note[] {
  const { timingVariation = 0.02, velocityVariation = 10 } = options;

  return notes.map(note => ({
    ...note,
    startTime: note.startTime + (Math.random() - 0.5) * timingVariation,
    velocity: Math.max(1, Math.min(127, 
      Math.round(note.velocity + (Math.random() - 0.5) * velocityVariation)
    ))
  }));
}

export function applyVelocityCurve(
  notes: Note[],
  curveType: 'crescendo' | 'diminuendo' | 'exponential'
): Note[] {
  if (notes.length === 0) return notes;

  const sortedNotes = [...notes].sort((a, b) => a.startTime - b.startTime);
  const firstNote = sortedNotes[0];
  const lastNote = sortedNotes[sortedNotes.length - 1];
  if (!firstNote || !lastNote) return notes;
  
  const minTime = firstNote.startTime;
  const maxTime = lastNote.startTime;
  const timeRange = maxTime - minTime;

  if (timeRange === 0) return notes;

  return notes.map(note => {
    const normalizedTime = (note.startTime - minTime) / timeRange;
    let factor: number;

    switch (curveType) {
      case 'crescendo':
        factor = normalizedTime;
        break;
      case 'diminuendo':
        factor = 1 - normalizedTime;
        break;
      case 'exponential':
        factor = Math.pow(normalizedTime, 2);
        break;
      default:
        factor = 1;
    }

    const minVelocity = 40;
    const maxVelocity = 127;
    const newVelocity = Math.round(minVelocity + (maxVelocity - minVelocity) * factor);

    return {
      ...note,
      velocity: Math.max(1, Math.min(127, newVelocity))
    };
  });
}

export function transposeNotes(notes: Note[], semitones: number): Note[] {
  return notes.map(note => ({
    ...note,
    pitch: Math.max(0, Math.min(127, note.pitch + semitones))
  }));
}

export function scaleVelocity(notes: Note[], factor: number): Note[] {
  return notes.map(note => ({
    ...note,
    velocity: Math.max(1, Math.min(127, Math.round(note.velocity * factor)))
  }));
}

export function filterNotesByPitchRange(
  notes: Note[],
  minPitch: number,
  maxPitch: number
): Note[] {
  return notes.filter(note => note.pitch >= minPitch && note.pitch <= maxPitch);
}

export function getNoteStatistics(notes: Note[]): {
  count: number;
  averageVelocity: number;
  pitchRange: [number, number];
  durationRange: [number, number];
  totalDuration: number;
} {
  if (notes.length === 0) {
    return {
      count: 0,
      averageVelocity: 0,
      pitchRange: [0, 0],
      durationRange: [0, 0],
      totalDuration: 0
    };
  }

  const velocities = notes.map(n => n.velocity);
  const pitches = notes.map(n => n.pitch);
  const durations = notes.map(n => n.duration);

  const averageVelocity = velocities.reduce((a, b) => a + b, 0) / velocities.length;
  const minPitch = Math.min(...pitches);
  const maxPitch = Math.max(...pitches);
  const minDuration = Math.min(...durations);
  const maxDuration = Math.max(...durations);
  const totalDuration = durations.reduce((a, b) => a + b, 0);

  return {
    count: notes.length,
    averageVelocity: Math.round(averageVelocity),
    pitchRange: [minPitch, maxPitch],
    durationRange: [minDuration, maxDuration],
    totalDuration
  };
}
