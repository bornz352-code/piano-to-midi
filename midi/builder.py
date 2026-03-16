"""Build a mido MidiFile from a list of NoteEvents."""
from __future__ import annotations
import mido
from mido import MidiFile, MidiTrack, Message
from pitch.detector import NoteEvent

TICKS_PER_BEAT = 480


def build(events: list[NoteEvent], tempo_bpm: int = 120, velocity: int = 80) -> MidiFile:
    """Convert NoteEvents into a MidiFile object.

    Args:
        events: Note events from the pitch detector.
        tempo_bpm: Playback tempo in BPM.
        velocity: MIDI velocity for all notes (0–127).

    Returns:
        A mido.MidiFile ready to be saved.
    """
    tempo_us = mido.bpm2tempo(tempo_bpm)
    mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
    track.append(mido.MetaMessage("track_name", name="Piano", time=0))
    track.append(Message("program_change", program=0, channel=0, time=0))  # Acoustic Grand Piano

    # Collect all events as (abs_tick, priority, msg_type, note, vel)
    # priority 0 = note_off (fires before note_on at same tick)
    raw: list[tuple[int, int, str, int, int]] = []
    for ev in events:
        start_tick = int(mido.second2tick(ev.start_sec, TICKS_PER_BEAT, tempo_us))
        end_tick = int(mido.second2tick(ev.end_sec, TICKS_PER_BEAT, tempo_us))
        # Guarantee at least 1 tick of note length
        end_tick = max(end_tick, start_tick + 1)
        raw.append((start_tick, 1, "note_on", ev.midi_note, velocity))
        raw.append((end_tick, 0, "note_off", ev.midi_note, 0))

    raw.sort()

    prev_tick = 0
    for tick, _, msg_type, note, vel in raw:
        delta = max(0, tick - prev_tick)
        track.append(Message(msg_type, note=note, velocity=vel, channel=0, time=delta))
        prev_tick = tick

    track.append(mido.MetaMessage("end_of_track", time=0))
    return mid


def save(mid: MidiFile, path: str) -> None:
    mid.save(path)
