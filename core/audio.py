"""Audio I/O: loading, resampling, silence-aware 30 s chunking, mic capture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30


def to_mono_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype.kind == "i":
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    elif audio.dtype.kind == "u":
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    audio = audio.astype(np.float32)
    if audio.ndim == 2:
        # Gradio gives (samples, channels); soundfile too.
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    return audio


def resample(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return audio.astype(np.float32)
    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore

        g = gcd(sr, target)
        return resample_poly(audio, target // g, sr // g).astype(np.float32)
    except Exception:
        # Pure-numpy fallback: low-pass (moving average) then linear interpolation.
        if sr > target:
            k = max(1, int(round(sr / target)))
            audio = np.convolve(audio, np.ones(k) / k, mode="same")
        n_out = int(round(len(audio) * target / sr))
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """Load any wav/flac/ogg/mp3 file as mono float32 @ 16 kHz."""
    import soundfile as sf

    data, sr = sf.read(str(path), always_2d=False)
    return resample(to_mono_float(data), sr)


def from_gradio(value) -> np.ndarray | None:
    """Accept a Gradio audio value (filepath or (sr, ndarray))."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        return load_audio(value)
    sr, data = value
    return resample(to_mono_float(data), sr)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12)) if len(x) else 0.0


def is_silent(x: np.ndarray, threshold: float = 0.004) -> bool:
    """Whisper hallucinates on silence ("Thank you."), so we skip near-silent chunks."""
    return rms(x) < threshold


def smart_chunks(audio: np.ndarray, sr: int = SAMPLE_RATE, max_s: int = CHUNK_SECONDS, search_s: float = 5.0):
    """Split audio into <=30 s chunks, cutting at the quietest 100 ms in the last `search_s` seconds.

    Cutting in pauses instead of every exact 30 s avoids chopping words in half.
    Yields (start_seconds, chunk).
    """
    n = len(audio)
    max_len = max_s * sr
    win = int(0.1 * sr)
    pos = 0
    while pos < n:
        end = min(pos + max_len, n)
        if end < n:
            lo = max(pos + int((max_s - search_s) * sr), pos + win)
            seg = audio[lo:end]
            if len(seg) > win:
                frames = len(seg) // win
                energy = np.square(seg[: frames * win]).reshape(frames, win).mean(axis=1)
                end = lo + int(np.argmin(energy)) * win + win // 2
        yield pos / sr, audio[pos:end]
        pos = end


def record(seconds: float, device: int | None = None) -> np.ndarray:
    """Blocking microphone capture for the CLI (requires `sounddevice`)."""
    import sounddevice as sd  # type: ignore

    data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
    sd.wait()
    return data[:, 0]
