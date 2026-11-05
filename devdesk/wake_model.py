"""The exported "자비스" model: its input (`power_frames`, torchaudio's STFT does not export to ONNX) and `load_model`.

An adapter without Qt, so the app's listener, ml/scripts/export_wake.py and ml/streamlit_app.py share one copy.
numpy and onnxruntime are imported only when used: a broken install turns off only the feature that needs them.
"""
import functools
import hashlib
import json
from pathlib import Path

N_FFT, STFT_HOP = 480, 160  # = ml.wake.LogMel's MelSpectrogram
MODEL_DIR = Path(__file__).resolve().parent.parent / "ml" / "models"  # the checkout the app runs from


def power_frames(x, np):
    """(WIN,) float32 waveform → (1, N_FFT // 2 + 1, frames) power spectrogram, as torchaudio's Spectrogram
    (periodic Hann window, centered with reflect padding, power 2). np = numpy (passed in: imported lazily)."""
    x = np.pad(np.asarray(x, np.float32), N_FFT // 2, mode="reflect")
    idx, window = _stft_consts(np, len(x))
    spec = np.fft.rfft(x[idx] * window, n=N_FFT, axis=1)
    return (np.abs(spec) ** 2).T[None].astype(np.float32)


@functools.cache
def _stft_consts(np, padded_len):
    """Frame index matrix and Hann window, built once per length (the listener scores 10 times a second).
    shortcut: the whole 1.5 s STFT is recomputed per hop (~1.6 ms with the CNN); reuse the old frames if CPU matters."""
    frames = 1 + (padded_len - N_FFT) // STFT_HOP
    idx = np.arange(N_FFT)[None, :] + STFT_HOP * np.arange(frames)[:, None]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    return idx, window


def checked_session(model, sha256):
    """An onnxruntime session on one core over the model's bytes, if they hash to sha256 (its json names the model).
    Raises when they come from another export."""
    import onnxruntime as ort

    if hashlib.sha256(model).hexdigest() != sha256:  # missing = an older export: re-run it
        raise ValueError("the json does not belong to this model (re-run its export script)")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1  # ~2-3 ms per window on one core
    return ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])  # the bytes the hash checked


def load_model(model_dir=MODEL_DIR):
    """(score(window) -> logit, threshold). Raises when a file or package is missing or broken, or when wake.json
    belongs to another export than wake.onnx."""
    return load_pair((Path(model_dir) / "wake.onnx").read_bytes(), (Path(model_dir) / "wake.json").read_text(encoding="utf-8"))


def load_pair(model, meta_text):
    """load_model over the two files' contents, from memory (the deployed page has them in its secrets)."""
    import numpy as np

    meta = json.loads(meta_text)
    threshold = float(meta["threshold_logit"])
    session = checked_session(model, meta.get("onnx_sha256"))
    name = session.get_inputs()[0].name

    def score(window):
        return float(session.run(None, {name: power_frames(window, np)})[0].reshape(-1)[0])

    return score, threshold
