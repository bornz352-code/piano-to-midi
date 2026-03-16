"""Microphone recording via sounddevice."""
import threading
import numpy as np
import sounddevice as sd

TARGET_SR = 22050


class Recorder:
    """Record audio from the default input device.

    Usage:
        rec = Recorder()
        rec.start()
        ...
        audio = rec.stop()  # returns np.ndarray, or None if nothing recorded
    """

    def __init__(self, sample_rate: int = TARGET_SR):
        self._sr = sample_rate
        self._chunks: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray | None:
        if self._stream is None:
            return None
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            if not self._chunks:
                return None
            return np.concatenate(self._chunks, axis=0).flatten()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata.copy())

    @property
    def sample_rate(self) -> int:
        return self._sr
