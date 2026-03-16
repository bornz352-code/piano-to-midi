"""Pitch detection using pyin, returning a list of NoteEvents."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import librosa

SR = 22050
FRAME_LENGTH = 2048   # ~93 ms — good for piano attack
HOP_LENGTH = 512      # ~23 ms hop — smooth tracking
MIN_NOTE_DURATION = 0.08  # seconds; filter sub-80ms pyin artefacts


@dataclass
class NoteEvent:
    start_sec: float
    end_sec: float
    midi_note: int   # 0-127
    confidence: float


def detect(
    y: np.ndarray,
    sr: int = SR,
    min_note_duration: float = MIN_NOTE_DURATION,
    on_progress=None,
) -> list[NoteEvent]:
    """Run pyin pitch detection and return a list of NoteEvents.

    Args:
        y: Mono float32 audio samples.
        sr: Sample rate (should be 22050).
        min_note_duration: Minimum note length in seconds.
        on_progress: Optional callable(percent: int, message: str).
    """
    def progress(pct, msg):
        if on_progress:
            on_progress(pct, msg)

    progress(0, "Running pitch detection…")
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A0"),   # 27.5 Hz — lowest piano key
        fmax=librosa.note_to_hz("C8"),   # 4186 Hz — highest piano key
        sr=sr,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    progress(80, "Processing pitch frames…")

    frame_dur = HOP_LENGTH / sr          # seconds per frame
    min_frames = max(1, int(min_note_duration / frame_dur))

    events: list[NoteEvent] = []
    n = len(f0)
    i = 0
    while i < n:
        if not voiced_flag[i] or np.isnan(f0[i]):
            i += 1
            continue
        # Gather consecutive voiced frames with the same rounded MIDI note
        note = _hz_to_midi(f0[i])
        j = i + 1
        conf_sum = float(voiced_probs[i])
        while j < n and voiced_flag[j] and not np.isnan(f0[j]) and _hz_to_midi(f0[j]) == note:
            conf_sum += float(voiced_probs[j])
            j += 1
        length = j - i
        if length >= min_frames:
            events.append(NoteEvent(
                start_sec=i * frame_dur,
                end_sec=j * frame_dur,
                midi_note=note,
                confidence=conf_sum / length,
            ))
        i = j

    progress(100, f"Found {len(events)} note(s).")
    return events


def _hz_to_midi(hz: float) -> int:
    """Convert Hz to the nearest MIDI note number (clamped 0–127)."""
    return int(np.clip(round(69 + 12 * np.log2(hz / 440.0)), 0, 127))
