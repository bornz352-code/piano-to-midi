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
    """Convert a chord name like 'Cmaj7' to MIDI note numbers."""
    m = re.match(r'^([A-G][#b]?)(.*?)(?:/[A-G][#b]?)?$', chord_name.strip())
    if not m:
        return [60, 64, 67]
    root_str, quality = m.group(1), m.group(2)
    semitone = _NOTE_SEMITONES.get(root_str[0], 0)
    if len(root_str) > 1:
        semitone += (1 if root_str[1] == '#' else -1)
    semitone = semitone % 12
    base = 12 * (octave + 1) + semitone  # C4 = 60
    intervals = _CHORD_INTERVALS.get(quality, [0, 4, 7])
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


if __name__ == "__main__":
    import webbrowser
    port = 5050
    webbrowser.open(f"http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
