"""Flask web server for Piano → MIDI converter."""
from __future__ import annotations
import sys
import os
import io
import tempfile
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import soundfile as sf
from flask import Flask, render_template, request, send_file, jsonify, Response
from audio.loader import load as load_audio
from basic_pitch.inference import predict as bp_predict
from basic_pitch import ICASSP_2022_MODEL_PATH

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

        # basic-pitch needs a WAV file; write a temp one if the input isn't WAV
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, y, sr)
        tmp_wav.close()

        _set_progress(job_id, 25, f"Loaded {duration:.1f}s — transcribing (polyphonic)…")
        _, midi_data, note_events = bp_predict(tmp_wav.name)
        os.unlink(tmp_wav.name)

        # Apply the chosen tempo to the MIDI file
        midi_data.initial_tempo = float(tempo)

        note_count = sum(len(inst.notes) for inst in midi_data.instruments)
        _set_progress(job_id, 90, f"Building MIDI from {note_count} note(s)…")

        buf = io.BytesIO()
        midi_data.write(buf)
        buf.seek(0)

        _set_done(job_id, f"Converted {note_count} notes.")
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
        # tempo may be a 0-d array in newer librosa versions
        bpm = int(round(float(np.atleast_1d(tempo)[0])))
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
