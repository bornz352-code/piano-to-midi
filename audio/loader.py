"""Load audio files and normalize to mono 22050 Hz float32."""
import os
import shutil
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 22050


def _ensure_ffmpeg() -> None:
    """Add imageio-ffmpeg's bundled binary to PATH if system ffmpeg is missing."""
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        pass


def load(path: str) -> tuple[np.ndarray, int]:
    """Return (samples, sample_rate) normalized to mono float32 at TARGET_SR.

    Tries soundfile first (fast, handles WAV/FLAC/OGG), then falls back to
    librosa/audioread which handles WebM, MP3, M4A, etc. via ffmpeg.
    """
    _ensure_ffmpeg()
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        return y, sr
    except Exception:
        # Fall back to librosa (uses audioread → ffmpeg for MP4, WebM, MP3, M4A…)
        y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
        return y, sr
