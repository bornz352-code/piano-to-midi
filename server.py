"""Flask web server for Piano → MIDI converter."""
from __future__ import annotations
import sys
import os
import io
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, send_file, jsonify, Response
from audio.loader import load as load_audio
from pitch.detector import detect
from midi.builder import build as build_midi

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ── Progress store (keyed by job_id) ─────────────────────────────────────────
_progress: dict[str, dict] = {}
_lock = threading.Lock()


def _set_progress(job_id: str, pct: int, msg: str):
    with _lock:
        _progress[job_id] = {"pct": pct, "msg": msg, "done": False, "error": None}


def _set_done(job_id: str, msg: str = "Done."):
    with _lock:
        _progress[job_id] = {"pct": 100, "msg": msg, "done": True, "error": None}


def _set_error(job_id: str, err: str):
    with _lock:
        _progress[job_id] = {"pct": 0, "msg": err, "done": True, "error": err}


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    audio_file = request.files.get("audio")
    tempo = int(request.form.get("tempo", 120))
    job_id = request.form.get("job_id", "default")

    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided."}), 400

    ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    tmp_in = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        audio_file.save(tmp_in.name)
        tmp_in.close()

        _set_progress(job_id, 10, "Loading audio…")
        y, sr = load_audio(tmp_in.name)
        duration = len(y) / sr

        _set_progress(job_id, 20, f"Loaded {duration:.1f}s — detecting pitches…")

        def on_pitch_progress(p, m):
            _set_progress(job_id, 20 + int(p * 0.6), m)

        events = detect(y, sr, on_progress=on_pitch_progress)

        _set_progress(job_id, 85, f"Building MIDI from {len(events)} note(s)…")
        mid = build_midi(events, tempo_bpm=tempo)

        buf = io.BytesIO()
        mid.save(file=buf)
        buf.seek(0)

        _set_done(job_id, f"Converted {len(events)} notes.")
        return send_file(
            buf,
            as_attachment=True,
            download_name="output.mid",
            mimetype="audio/midi",
        )
    except Exception as exc:
        _set_error(job_id, str(exc))
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass


@app.route("/detect-bpm", methods=["POST"])
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
        bpm = int(round(float(tempo)))
        return jsonify({"bpm": bpm})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_in.name)
        except OSError:
            pass


@app.route("/progress/<job_id>")
def progress(job_id: str):
    with _lock:
        data = _progress.get(job_id, {"pct": 0, "msg": "Waiting…", "done": False, "error": None})
    return jsonify(data)


if __name__ == "__main__":
    import webbrowser
    port = 5050
    webbrowser.open(f"http://localhost:{port}")
    app.run(port=port, debug=False)
