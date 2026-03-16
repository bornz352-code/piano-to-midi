"""Flask web server for Piano → MIDI converter."""
from __future__ import annotations
import sys
import os
import io
import re
import json
import base64
import secrets
import tempfile
import threading
import functools
import datetime
from collections import deque
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf
from flask import (
    Flask, render_template, request, send_file,
    jsonify, session, redirect, url_for, Response,
)
from audio.loader import load as load_audio
from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB
app.secret_key = secrets.token_hex(32)

# ── Password ──────────────────────────────────────────────────────────────────
# Set via env var PIANO_PASSWORD; a random one is generated and printed if unset.
_PASSWORD = os.environ.get("PIANO_PASSWORD", "")
if not _PASSWORD:
    _PASSWORD = secrets.token_urlsafe(10)
    print(f"  Password: {_PASSWORD}  (set PIANO_PASSWORD env var to choose your own)")


# ── Chord helpers ─────────────────────────────────────────────────────────────
_NOTE_SEMITONES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_CHORD_INTERVALS = {
    '':      [0, 4, 7], 'maj':  [0, 4, 7],  'M':   [0, 4, 7],
    'm':     [0, 3, 7], 'min':  [0, 3, 7],  '-':   [0, 3, 7],
    '5':     [0, 7],
    '6':     [0, 4, 7, 9],  'm6':   [0, 3, 7, 9],
    '7':     [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'M7': [0, 4, 7, 11],
    'm7':    [0, 3, 7, 10], 'mM7':  [0, 3, 7, 11],
    'dim':   [0, 3, 6],     'o':    [0, 3, 6],
    'dim7':  [0, 3, 6, 9],  'o7':   [0, 3, 6, 9],
    'm7b5':  [0, 3, 6, 10], 'ø7':   [0, 3, 6, 10],
    'aug':   [0, 4, 8],     '+':    [0, 4, 8],
    'sus2':  [0, 2, 7],     'sus4': [0, 5, 7],     'sus': [0, 5, 7],
    'add9':  [0, 4, 7, 14],
    '9':     [0, 4, 7, 10, 14],  'maj9': [0, 4, 7, 11, 14], 'm9': [0, 3, 7, 10, 14],
    '11':    [0, 4, 7, 10, 14, 17],
    '13':    [0, 4, 7, 10, 14, 17, 21],
}


def _chord_to_notes(chord_name: str, octave: int = 4) -> list:
    """Convert a chord name like 'Cmaj7' or 'C7b9' to MIDI note numbers."""
    m = re.match(r'^([A-G][#b]?)(.*?)(?:/[A-G][#b]?)?$', chord_name.strip())
    if not m:
        return [60, 64, 67]
    root_str, quality = m.group(1), m.group(2)
    semitone = _NOTE_SEMITONES.get(root_str[0], 0)
    if len(root_str) > 1:
        semitone += (1 if root_str[1] == '#' else -1)
    semitone = semitone % 12
    base = 12 * (octave + 1) + semitone  # C4 = 60
    # Strip altered extensions (b9, #11, b13, etc.) — try progressively shorter quality
    q = quality
    while q and q not in _CHORD_INTERVALS:
        q = re.sub(r'[#b]\d+$', '', q).rstrip('()')
    intervals = _CHORD_INTERVALS.get(q, [0, 4, 7])
    return [base + i for i in intervals if 0 <= base + i <= 127]


def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if secrets.compare_digest(request.form.get("password", ""), _PASSWORD):
            session["authenticated"] = True
            return redirect(request.args.get("next") or url_for("home"))
        error = "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ── URL audio cache (id → wav path) ──────────────────────────────────────────
_url_cache: dict[str, str] = {}
_url_cache_lock = threading.Lock()


def _cleanup_url(audio_id: str):
    with _url_cache_lock:
        path = _url_cache.pop(audio_id, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ── History store (last 10 conversions) ──────────────────────────────────────
_history: deque = deque(maxlen=10)
_history_lock = threading.Lock()


def _add_to_history(filename: str, note_count: int, midi_bytes: bytes) -> str:
    entry_id = secrets.token_hex(8)
    with _history_lock:
        _history.appendleft({
            "id": entry_id,
            "filename": filename,
            "note_count": note_count,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "midi_bytes": midi_bytes,
        })
    return entry_id


# ── Progress store (keyed by job_id) ─────────────────────────────────────────
_progress: dict[str, dict] = {}
_lock = threading.Lock()

# ── Perc loop store (job_id → {variations, wav_cache}) ───────────────────────
_perc_store: dict[str, dict] = {}
_perc_store_lock = threading.Lock()


def _set_progress(job_id: str, pct: int, msg: str):
    with _lock:
        _progress[job_id] = {"pct": pct, "msg": msg, "done": False, "error": None}


def _set_done(job_id: str, msg: str = "Done.", notes: list = None):
    with _lock:
        _progress[job_id] = {"pct": 100, "msg": msg, "done": True, "error": None, "notes": notes or []}


def _set_error(job_id: str, err: str):
    with _lock:
        _progress[job_id] = {"pct": 0, "msg": err, "done": True, "error": err}


# ── App routes ────────────────────────────────────────────────────────────────
@app.route("/")
@login_required
def home():
    return render_template("home.html")


@app.route("/audio-to-midi")
@login_required
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
@login_required
def convert():
    audio_file = request.files.get("audio")
    tempo = int(request.form.get("tempo", 120))
    job_id = request.form.get("job_id", "default")

    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided."}), 400

    ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp_wav = None
    try:
        audio_file.save(tmp_in.name)
        tmp_in.close()

        _set_progress(job_id, 10, "Loading audio…")
        y, sr = load_audio(tmp_in.name)
        duration = len(y) / sr

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, y, sr)
        tmp_wav.close()

        _set_progress(job_id, 25, f"Loaded {duration:.1f}s — transcribing (polyphonic)…")
        _, midi_data, _ = bp_predict(tmp_wav.name)

        midi_data.initial_tempo = float(tempo)
        note_count = sum(len(inst.notes) for inst in midi_data.instruments)
        _set_progress(job_id, 90, f"Building MIDI from {note_count} note(s)…")

        buf = io.BytesIO()
        midi_data.write(buf)
        buf.seek(0)

        midi_bytes = buf.getvalue()
        original_name = os.path.splitext(audio_file.filename)[0] + ".mid"
        _add_to_history(original_name, note_count, midi_bytes)

        notes_data = [
            {"pitch": n.pitch, "start": round(n.start, 3), "end": round(n.end, 3)}
            for inst in midi_data.instruments for n in inst.notes
        ]
        _set_done(job_id, f"Converted {note_count} notes.", notes=notes_data)
        return send_file(io.BytesIO(midi_bytes), as_attachment=True, download_name=original_name, mimetype="audio/midi")
    except Exception as exc:
        _set_error(job_id, str(exc))
        return jsonify({"error": str(exc)}), 500
    finally:
        for tmp in (tmp_in, tmp_wav):
            if tmp:
                try:
                    os.unlink(tmp.name)
                except OSError:
                    pass


@app.route("/detect-bpm", methods=["POST"])
@login_required
def detect_bpm():
    import librosa
    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided."}), 400

    ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        audio_file.save(tmp_in.name)
        tmp_in.close()
        y, sr = load_audio(tmp_in.name)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(np.atleast_1d(tempo)[0])))
        return jsonify({"bpm": bpm})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass


@app.route("/history")
@login_required
def history():
    with _history_lock:
        entries = [{"id": e["id"], "filename": e["filename"],
                    "note_count": e["note_count"], "created_at": e["created_at"]}
                   for e in _history]
    return jsonify(entries)


@app.route("/download/<entry_id>")
@login_required
def download(entry_id: str):
    with _history_lock:
        entry = next((e for e in _history if e["id"] == entry_id), None)
    if not entry:
        return jsonify({"error": "Not found."}), 404
    return send_file(
        io.BytesIO(entry["midi_bytes"]),
        as_attachment=True,
        download_name=entry["filename"],
        mimetype="audio/midi",
    )


@app.route("/progress/<job_id>")
@login_required
def progress(job_id: str):
    with _lock:
        data = _progress.get(job_id, {"pct": 0, "msg": "Waiting…", "done": False, "error": None})
    return jsonify(data)


@app.route("/fetch-url", methods=["POST"])
@login_required
def fetch_url():
    """Fetch audio from a URL (Instagram, YouTube, etc.) using yt-dlp."""
    import yt_dlp
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    audio_id = secrets.token_hex(8)
    tmp_dir = tempfile.mkdtemp()
    out_template = os.path.join(tmp_dir, "audio.%(ext)s")

    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": ffmpeg_exe,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "audio")

        wav_path = os.path.join(tmp_dir, "audio.wav")
        if not os.path.exists(wav_path):
            # fallback: find whatever was downloaded
            files = [f for f in os.listdir(tmp_dir) if os.path.isfile(os.path.join(tmp_dir, f))]
            if not files:
                return jsonify({"error": "No audio extracted."}), 500
            wav_path = os.path.join(tmp_dir, files[0])

        with _url_cache_lock:
            _url_cache[audio_id] = wav_path

        duration = round(len(__import__('soundfile').read(wav_path)[0]) / __import__('soundfile').read(wav_path)[1], 1)
        return jsonify({"id": audio_id, "title": title, "duration": duration})

    except Exception as exc:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/stream/<audio_id>")
@login_required
def stream(audio_id: str):
    """Stream fetched audio to the browser."""
    with _url_cache_lock:
        path = _url_cache.get(audio_id)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Audio not found."}), 404

    ext = os.path.splitext(path)[1].lower()
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "m4a": "audio/mp4"}.get(ext[1:], "audio/wav")
    return send_file(path, mimetype=mime)


@app.route("/convert-url", methods=["POST"])
@login_required
def convert_url():
    """Convert a previously fetched URL audio to MIDI."""
    audio_id = request.form.get("audio_id", "")
    tempo = int(request.form.get("tempo", 120))
    job_id = request.form.get("job_id", "default")

    with _url_cache_lock:
        path = _url_cache.get(audio_id)
    if not path or not os.path.exists(path):
        return jsonify({"error": "Audio not found. Fetch the URL again."}), 404

    tmp_wav = None
    try:
        _set_progress(job_id, 10, "Loading audio…")
        y, sr = load_audio(path)
        duration = len(y) / sr

        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, y, sr)
        tmp_wav.close()

        _set_progress(job_id, 25, f"Loaded {duration:.1f}s — transcribing…")
        _, midi_data, _ = bp_predict(tmp_wav.name)

        midi_data.initial_tempo = float(tempo)
        note_count = sum(len(inst.notes) for inst in midi_data.instruments)
        _set_progress(job_id, 90, f"Building MIDI from {note_count} note(s)…")

        buf = io.BytesIO()
        midi_data.write(buf)
        buf.seek(0)
        midi_bytes = buf.getvalue()

        _add_to_history("url_audio.mid", note_count, midi_bytes)
        _set_done(job_id, f"Converted {note_count} notes.")
        return send_file(io.BytesIO(midi_bytes), as_attachment=True, download_name="output.mid", mimetype="audio/midi")

    except Exception as exc:
        _set_error(job_id, str(exc))
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav.name)
            except OSError:
                pass


@app.route("/chord-generator")
@login_required
def chord_generator():
    return render_template("chords.html")


@app.route("/generate-chords", methods=["POST"])
@login_required
def generate_chords():
    import anthropic
    import pretty_midi

    genre = request.form.get("genre", "Jazz")
    mood = request.form.get("mood", "Happy")
    key = request.form.get("key", "C")
    scale = request.form.get("scale", "Major")
    bars = max(1, min(32, int(request.form.get("bars", 4))))
    tempo = max(20, min(300, int(request.form.get("tempo", 120))))

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return jsonify({"error": "ANTHROPIC_API_KEY is not configured."}), 500

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"Create a {bars}-bar chord progression for a {genre} song.\n"
        f"Mood: {mood}\nKey: {key} {scale}\nTime signature: 4/4\n\n"
        "Return ONLY a JSON array. Each element must have:\n"
        '- "chord": chord name (e.g. "Cmaj7", "Am7", "F#m", "Bb7")\n'
        '- "beats": integer beats (1, 2, or 4)\n'
        f"Total beats must equal exactly {bars * 4}.\n"
        "Return only the JSON array, no explanation or markdown."
    )

    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        match = re.search(r'\[[\s\S]*\]', text)
        if not match:
            return jsonify({"error": "Could not parse chord progression from Claude response."}), 500
        chords = json.loads(match.group())
    except Exception as exc:
        return jsonify({"error": f"Claude API error: {exc}"}), 500

    try:
        pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
        instrument = pretty_midi.Instrument(program=0, name="Piano")
        beats_per_second = tempo / 60.0
        current_time = 0.0
        for entry in chords:
            chord_name = entry.get("chord", "C")
            beats = max(1, int(entry.get("beats", 4)))
            duration = beats / beats_per_second
            for pitch in _chord_to_notes(chord_name):
                instrument.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch,
                    start=current_time, end=current_time + duration - 0.05,
                ))
            current_time += duration
        pm.instruments.append(instrument)
        buf = io.BytesIO()
        pm.write(buf)
        midi_bytes = buf.getvalue()
    except Exception as exc:
        return jsonify({"error": f"MIDI generation error: {exc}"}), 500

    filename = f"chords_{genre.lower()}_{key}{scale[0].lower()}.mid"
    _add_to_history(filename, len(chords), midi_bytes)
    return jsonify({
        "chords": chords,
        "midi_b64": base64.b64encode(midi_bytes).decode(),
        "filename": filename,
    })


@app.route("/rebuild-midi", methods=["POST"])
@login_required
def rebuild_midi():
    """Rebuild MIDI from chords list with per-chord inversions."""
    import pretty_midi
    data = request.get_json()
    chords = data.get("chords", [])
    tempo = max(20, min(300, int(data.get("tempo", 120))))

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(tempo))
    instrument = pretty_midi.Instrument(program=0, name="Piano")
    beats_per_second = tempo / 60.0
    current_time = 0.0

    for entry in chords:
        chord_name = entry.get("chord", "C")
        beats = max(1, int(entry.get("beats", 4)))
        inversion = max(0, int(entry.get("inversion", 0)))
        duration = beats / beats_per_second

        notes = sorted(_chord_to_notes(chord_name))
        for i in range(min(inversion, len(notes) - 1)):
            notes.append(notes.pop(0) + 12)

        for pitch in notes:
            if 0 <= pitch <= 127:
                instrument.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch,
                    start=current_time, end=current_time + duration - 0.05,
                ))
        current_time += duration

    pm.instruments.append(instrument)
    buf = io.BytesIO()
    pm.write(buf)
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="chords_updated.mid", mimetype="audio/midi")


# ── Perc loop: audio analysis ────────────────────────────────────────────────
def _analyze_audio_for_perc(path: str) -> dict:
    import librosa as _lr
    y, sr = _lr.load(path, sr=22050, mono=True)

    tempo, _ = _lr.beat.beat_track(y=y, sr=sr)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)

    rms = _lr.feature.rms(y=y)[0]
    energy_mean = float(np.mean(rms))
    energy_level = "high" if energy_mean > 0.07 else "medium" if energy_mean > 0.02 else "low"

    centroid = _lr.feature.spectral_centroid(y=y, sr=sr)[0]
    dom_freq = int(np.mean(centroid))
    freq_char = "bright/treble" if dom_freq > 3000 else "mid-range" if dom_freq > 1000 else "bass-heavy"

    onsets = _lr.onset.onset_detect(y=y, sr=sr, units="time")
    duration = len(y) / sr
    density = round(len(onsets) / max(duration, 0.1), 2)

    groove = "straight"
    if len(onsets) > 6:
        ioi = np.diff(onsets)
        if len(ioi) >= 4:
            p60 = np.percentile(ioi, 60)
            short_ioi = ioi[ioi < p60]
            long_ioi  = ioi[ioi >= p60]
            if len(short_ioi) > 0 and len(long_ioi) > 0:
                ratio = float(np.mean(long_ioi)) / (float(np.mean(short_ioi)) + 1e-8)
                groove = "swung" if ratio > 1.4 else "straight"

    return {
        "bpm": bpm,
        "energy_level": energy_level,
        "groove_feel": groove,
        "dominant_freq_hz": dom_freq,
        "freq_character": freq_char,
        "rhythmic_density": density,
        "duration_s": round(duration, 2),
    }


def _auto_style(analysis: dict) -> str:
    bpm = analysis["bpm"]
    if bpm < 85:   return "boom bap"
    if bpm < 100:  return "boom bap / lo-fi"
    if bpm < 115:  return "R&B / neo soul"
    if bpm < 128:  return "afrobeat"
    if bpm < 140:  return "house"
    if bpm < 160:  return "trap"
    if bpm < 175:  return "jungle"
    return "drum and bass"


def _build_perc_prompt(analysis: dict, style: str, bars: int) -> str:
    steps = 16 * bars
    groove_note = (
        "Use syncopation, swing feel, and 'behind the beat' placement"
        if analysis["groove_feel"] == "swung"
        else "Keep patterns tight to the grid, straight time"
    )
    density_note = (
        "dense, driving, lots of activity" if analysis["energy_level"] == "high"
        else "moderate complexity, balanced" if analysis["energy_level"] == "medium"
        else "sparse and minimal"
    )
    return (
        f"You are an expert music producer. I analyzed a reference audio track:\n"
        f"BPM: {analysis['bpm']}\n"
        f"Energy: {analysis['energy_level']}\n"
        f"Groove: {analysis['groove_feel']}\n"
        f"Frequency character: {analysis['freq_character']} ({analysis['dominant_freq_hz']}Hz spectral centroid)\n"
        f"Rhythmic density: {analysis['rhythmic_density']} onsets/sec\n"
        f"Style: {style}\n"
        f"Bars: {bars} ({steps} 16th-note steps per instrument)\n\n"
        f"Generate exactly 4 unique percussion loop variations that COMPLEMENT this track.\n"
        f"Rules:\n"
        f"- Match BPM exactly: {analysis['bpm']}\n"
        f"- Groove: {groove_note}\n"
        f"- Energy: {density_note}\n"
        f"- Each variation must have a clearly distinct character\n\n"
        f"Velocity codes: 0=silent, 1=ghost note (soft), 2=normal hit, 3=accent (loud)\n\n"
        f"Return ONLY valid JSON in this exact format (no markdown, no explanation):\n"
        '{{\n  "variations": [\n    {{\n'
        '      "name": "Short creative name",\n'
        '      "description": "One sentence about the groove",\n'
        '      "instruments": {{\n'
        f'        "kick":         [exactly {steps} integers 0-3],\n'
        f'        "snare":        [exactly {steps} integers 0-3],\n'
        f'        "hihat_closed": [exactly {steps} integers 0-3],\n'
        f'        "hihat_open":   [exactly {steps} integers 0-3],\n'
        f'        "clap":         [exactly {steps} integers 0-3]\n'
        '      }}\n    }}\n  ]\n}}\n'
        "Generate exactly 4 variations."
    )


# ── Perc loop: drum synthesis ─────────────────────────────────────────────────
_PERC_SR = 44100


def _synth_kick(velocity=1.0):
    sr, dur = _PERC_SR, 0.55
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    freq = 150 * np.exp(-t * 22) + 42
    phase = np.cumsum(2 * np.pi * freq / sr)
    tone = np.sin(phase) * np.exp(-t * 9)
    click = np.random.randn(len(t)) * np.exp(-t * 250) * 0.15
    return np.clip((tone + click) * velocity, -1.0, 1.0)


def _synth_snare(velocity=1.0):
    from scipy.signal import sosfilt, butter
    sr, dur = _PERC_SR, 0.22
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t)) * np.exp(-t * 20)
    tone = np.sin(2 * np.pi * 200 * t) * np.exp(-t * 30) * 0.35
    sos = butter(2, 300 / (sr / 2), btype="high", output="sos")
    return np.clip((sosfilt(sos, noise) + tone) * velocity * 0.75, -1.0, 1.0)


def _synth_hihat_closed(velocity=1.0):
    from scipy.signal import sosfilt, butter
    sr, dur = _PERC_SR, 0.07
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t)) * np.exp(-t * 100)
    sos = butter(2, 7000 / (sr / 2), btype="high", output="sos")
    return np.clip(sosfilt(sos, noise) * velocity * 0.55, -1.0, 1.0)


def _synth_hihat_open(velocity=1.0):
    from scipy.signal import sosfilt, butter
    sr, dur = _PERC_SR, 0.38
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    noise = np.random.randn(len(t)) * np.exp(-t * 7)
    sos = butter(2, 6000 / (sr / 2), btype="high", output="sos")
    return np.clip(sosfilt(sos, noise) * velocity * 0.45, -1.0, 1.0)


def _synth_clap(velocity=1.0):
    from scipy.signal import sosfilt, butter
    sr, dur = _PERC_SR, 0.20
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    env = (np.exp(-t * 80)
           + 0.6 * np.exp(-((t - 0.008) ** 2) * 80000)
           + 0.4 * np.exp(-((t - 0.016) ** 2) * 80000))
    noise = np.random.randn(len(t))
    sos = butter(2, 1000 / (sr / 2), btype="high", output="sos")
    return np.clip(sosfilt(sos, noise * env) * velocity * 0.65, -1.0, 1.0)


_DRUM_SYNTHS = {
    "kick":         _synth_kick,
    "snare":        _synth_snare,
    "hihat_closed": _synth_hihat_closed,
    "hihat_open":   _synth_hihat_open,
    "clap":         _synth_clap,
}
_VEL_MAP = {1: 0.38, 2: 0.70, 3: 1.0}


def _pattern_to_wav_bytes(instruments: dict, bpm: float, bars: int) -> bytes:
    sr = _PERC_SR
    sp16 = (60.0 / bpm) / 4          # seconds per 16th note
    steps = 16 * bars
    total = int((steps * sp16 + 1.2) * sr)
    mix = np.zeros(total, dtype=np.float64)

    for instr, pattern in instruments.items():
        fn = _DRUM_SYNTHS.get(instr)
        if not fn:
            continue
        for i, vel_code in enumerate(pattern[:steps]):
            if not vel_code:
                continue
            vel = _VEL_MAP.get(int(vel_code), 0.70)
            hit = fn(vel)
            s = int(i * sp16 * sr)
            e = min(s + len(hit), total)
            mix[s:e] += hit[: e - s]

    peak = np.max(np.abs(mix))
    if peak > 1e-6:
        mix = mix / peak * 0.92

    buf = io.BytesIO()
    sf.write(buf, (np.clip(mix, -1, 1) * 32767).astype(np.int16), sr,
             format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.getvalue()


# ── Perc loop routes ──────────────────────────────────────────────────────────
@app.route("/perc-loop")
@login_required
def perc_loop():
    return render_template("perc_loop.html")


@app.route("/perc-generate", methods=["POST"])
@login_required
def perc_generate():
    import anthropic

    audio_file = request.files.get("audio")
    style = request.form.get("style", "auto-detect")
    bars = max(1, min(4, int(request.form.get("bars", 2))))

    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided."}), 400

    ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        audio_file.save(tmp.name)
        tmp.close()

        analysis = _analyze_audio_for_perc(tmp.name)
        if style == "auto-detect":
            style = _auto_style(analysis)

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return jsonify({"error": "ANTHROPIC_API_KEY not configured."}), 500

        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": _build_perc_prompt(analysis, style, bars)}],
        )
        text = msg.content[0].text.strip()
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return jsonify({"error": "Could not parse patterns from Claude."}), 500
        variations = json.loads(match.group()).get("variations", [])[:4]

        wav_cache = {}
        for idx, var in enumerate(variations):
            wav_cache[idx] = _pattern_to_wav_bytes(
                var.get("instruments", {}), analysis["bpm"], bars
            )

        job_id = secrets.token_hex(8)
        with _perc_store_lock:
            if len(_perc_store) >= 20:
                del _perc_store[next(iter(_perc_store))]
            _perc_store[job_id] = {"variations": variations, "wav_cache": wav_cache}

        return jsonify({"job_id": job_id, "analysis": analysis, "style": style,
                        "variations": variations})

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@app.route("/perc-audio/<job_id>/<int:idx>")
@login_required
def perc_audio(job_id: str, idx: int):
    with _perc_store_lock:
        job = _perc_store.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
    wav = job["wav_cache"].get(idx)
    if wav is None:
        return jsonify({"error": "Variation not found."}), 404
    return send_file(
        io.BytesIO(wav), mimetype="audio/wav",
        as_attachment=False, download_name=f"perc_loop_v{idx + 1}.wav",
    )


if __name__ == "__main__":
    import webbrowser
    port = 5050
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
