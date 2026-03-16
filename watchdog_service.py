#!/usr/bin/env python3
"""
Watchdog service — monitors /input for audio/video files and auto-converts to MIDI.
Converted files land in /output. Originals are moved to /input/done/.
"""
import os
import sys
import time
import logging
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, '/app')

import soundfile as sf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from audio.loader import load as load_audio
from basic_pitch.inference import predict as bp_predict

INPUT_DIR  = Path(os.environ.get('INPUT_DIR',  '/input'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', '/output'))
DONE_DIR   = INPUT_DIR / 'done'

AUDIO_EXT = {
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.m4a',
    '.mp4', '.mov', '.mkv', '.webm', '.aac', '.wma',
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [watchdog] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


def wait_until_stable(path: Path, timeout: int = 60) -> bool:
    """Block until a file stops growing (fully written to disk)."""
    prev = -1
    for _ in range(timeout):
        try:
            cur = path.stat().st_size
        except FileNotFoundError:
            return False
        if cur == prev and cur > 0:
            return True
        prev = cur
        time.sleep(1)
    return False


def convert(path: Path) -> None:
    log.info(f"Converting: {path.name}")
    output_path = OUTPUT_DIR / (path.stem + ".mid")
    tmp_wav = None
    try:
        y, sr = load_audio(str(path))
        tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp_wav.name, y, sr)
        tmp_wav.close()

        _, midi_data, _ = bp_predict(tmp_wav.name)
        midi_data.write(str(output_path))
        log.info(f"Saved: {output_path.name}")

        DONE_DIR.mkdir(exist_ok=True)
        shutil.move(str(path), str(DONE_DIR / path.name))
        log.info(f"Moved original → done/{path.name}")

    except Exception as exc:
        log.error(f"Failed to convert {path.name}: {exc}")
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav.name)
            except OSError:
                pass


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.parent != INPUT_DIR:
            return  # ignore subdirs (e.g. done/)
        if path.suffix.lower() in AUDIO_EXT:
            if wait_until_stable(path):
                convert(path)


if __name__ == "__main__":
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process any files already sitting in the input folder at startup
    for f in sorted(INPUT_DIR.iterdir()):
        if f.is_file() and f.suffix.lower() in AUDIO_EXT:
            log.info(f"Processing existing file: {f.name}")
            convert(f)

    observer = Observer()
    observer.schedule(Handler(), str(INPUT_DIR), recursive=False)
    observer.start()
    log.info(f"Watching {INPUT_DIR} …  Output → {OUTPUT_DIR}")

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
