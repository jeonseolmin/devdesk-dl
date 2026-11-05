"""Wake word "자비스": data sets, augmentation, the scratch CNN and the false-alarm stream metric.

Data (paths under one root, see train_wake.py):
  wake/tts/pos, wake/tts/neg   synth_wake.py output, <speaker>_<n>.wav
  wake/real/<speaker>          record_wake.py output (positives), wake/real_neg/<speaker> (negatives)
  raw/speech_commands          Speech Commands v0.02 (English words = negatives, _background_noise_ = noise)
Splits are by speaker, never by clip, so a voice in test was never heard in training.
"""
import random
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram

from devdesk.wake import HOP, RATE, REFRACTORY, WIN, Trigger, refractory_hops  # noqa: F401  (shared with the app)


def speaker_split(name, val=0.15, test=0.15):
    """'train' / 'val' / 'test' from a stable hash of the speaker name (same answer on every run and PC)."""
    h = (zlib.crc32(name.encode("utf-8")) % 1000) / 1000
    return "test" if h < test else "val" if h < test + val else "train"


def tts_speaker(path):
    return Path(path).stem.rsplit("_", 1)[0]


def read(path):
    """Mono float16 (half the RAM; ~30 k clips are held in memory)."""
    x, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, f"{path}: {rate} Hz"
    return x.mean(axis=1).astype(np.float16)


def place(x, rng, center=False):
    """x (any float dtype) into a float32 WIN-long window: random offset (or centered), random crop if longer."""
    out = np.zeros(WIN, np.float32)
    if len(x) >= WIN:
        start = (len(x) - WIN) // 2 if center else rng.randrange(len(x) - WIN + 1)
        return x[start : start + WIN].astype(np.float32)
    start = (WIN - len(x)) // 2 if center else rng.randrange(WIN - len(x) + 1)
    out[start : start + len(x)] = x
    return out


def mix_noise(x, noise, snr_db):
    """Add noise at snr_db relative to x's power (silent x: noise at -30 dBFS)."""
    px = float(np.mean(x**2))
    pn = float(np.mean(noise**2)) + 1e-12
    scale = np.sqrt((px if px > 1e-8 else 1e-3) / (pn * 10 ** (snr_db / 10)))
    return x + scale * noise


class WakeSet(torch.utils.data.Dataset):
    """(waveform WIN, label) with on-the-fly augmentation when `augment` (offset, gain, noise mix)."""

    def __init__(self, clips, noise, augment, seed=0):
        self.clips, self.noise, self.augment = clips, noise, augment  # clips: [(array, label)]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.clips)

    def noise_crop(self):
        n = self.noise[self.rng.randrange(len(self.noise))]
        s = self.rng.randrange(len(n) - WIN + 1)
        return n[s : s + WIN].astype(np.float32)  # float16 squares underflow in mix_noise

    def __getitem__(self, i):
        x, y = self.clips[i]
        if not self.augment:
            return torch.from_numpy(place(x, self.rng, center=True)), torch.tensor(float(y))
        w = place(x, self.rng) * 10 ** (self.rng.uniform(-10, 6) / 20)
        if self.rng.random() < 0.8:
            w = mix_noise(w, self.noise_crop(), self.rng.uniform(0, 20))
        return torch.from_numpy(np.clip(w, -1, 1).astype(np.float32)), torch.tensor(float(y))


class LogMel(nn.Module):
    """Waveform (B, T) → log-mel (B, 1, 40, frames), mean-normalized per clip (mic gain does not matter)."""

    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogram(RATE, n_fft=480, hop_length=160, n_mels=40)

    def forward(self, x):
        m = torch.log(self.mel(x) + 1e-6)
        return (m - m.mean(dim=(1, 2), keepdim=True)).unsqueeze(1)


def _block(cin, cout, pool=True):
    layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]
    return layers + [nn.MaxPool2d(2)] if pool else layers


class WakeCNN(nn.Module):
    """Small scratch CNN on log-mel; waveform in, one logit out (~60 k parameters)."""

    def __init__(self):
        super().__init__()
        self.front = LogMel()
        self.body = nn.Sequential(
            *_block(1, 16), *_block(16, 32), *_block(32, 64), *_block(64, 64, pool=False),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.body(self.front(x)).squeeze(1)


OWW_STEP = 1280  # samples per embedding step (80 ms)
OWW_FRAMES = 9  # embeddings per 1.5 s window


class OwwEmbed:
    """Frozen pretrained features: openWakeWord's ONNX melspectrogram + Google speech_embedding (Apache-2.0), CPU.

    Only these two feature models are used; openWakeWord's own wake-word models (CC BY-NC-SA) are not.
    """

    def __init__(self, folder, ncpu=4):
        from openwakeword.utils import AudioFeatures

        folder = Path(folder)
        self.f = AudioFeatures(str(folder / "melspectrogram.onnx"), str(folder / "embedding_model.onnx"), ncpu=ncpu)
        self.ncpu = ncpu

    @staticmethod
    def pcm(x):
        """float [-1, 1] → int16 PCM; int16 passes through untouched."""
        if np.asarray(x).dtype == np.int16:
            return x
        return (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype(np.int16)

    def clips(self, waves, batch=256):
        """(N, WIN) float waves → (N, 9, 96) float32."""
        return self.f.embed_clips(self.pcm(waves), batch_size=batch, ncpu=self.ncpu).astype(np.float32)

    def stream(self, x, batch=4096):
        """1-D float stream → (n, 9, 96), one 1.5 s window every 80 ms, each embedded on its own like a clip.

        Not one long pass: the ONNX melspectrogram floors quiet bins relative to its input's peak, so a frame's
        value depends on what else is in the input (measured: embeddings 5-20 % apart). Per window = what training
        saw and what a live demo with a rolling 1.5 s buffer computes. ~5 ms per window on 4 CPU threads.
        """
        w = stream_windows(self.pcm(x), hop=OWW_STEP)  # converted once; windows are views into it
        return np.concatenate([self.clips(w[i : i + batch]) for i in range(0, len(w), batch)])


class WakeHead(nn.Module):
    """Small classifier on the frozen (9, 96) embeddings; the transfer-learning model."""

    def __init__(self, dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LayerNorm(OWW_FRAMES * dim), nn.Linear(OWW_FRAMES * dim, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def count_triggers(scores, threshold, hop_s=HOP / RATE, refractory_s=REFRACTORY):
    """Triggers in a score stream: a score >= threshold fires, then nothing for refractory_s.

    A score held above the threshold fires again after each refractory period, as the live demo would.
    Counted in whole hops, so float rounding of i * hop_s never decides a trigger.
    """
    quiet = refractory_hops(hop_s, refractory_s)
    fired, next_i = 0, 0
    for i in np.flatnonzero(np.asarray(scores) >= threshold):  # only the hops above threshold are visited
        if i >= next_i:
            fired += 1
            next_i = i + quiet
    return fired


def stream_windows(stream, hop=HOP):
    """WIN-long windows every `hop` samples over a 1-D stream, as a (n, WIN) view."""
    n = (len(stream) - WIN) // hop + 1
    return np.lib.stride_tricks.as_strided(stream, (n, WIN), (stream.strides[0] * hop, stream.strides[0]))


def threshold_for_fa(scores, hours, max_fa_per_hour, hop_s=HOP / RATE):
    """Lowest threshold whose stream false alarms per hour stay <= max_fa_per_hour, exact over the scores' values.

    Pass logits, not sigmoid outputs: float32 sigmoid is exactly 1.0 above ~17, which hides the order of confident
    scores (measured 2026-10-31: the oww head's stream scores sat at 0.999+, a 0.05-0.999 grid could not separate
    them). The count only falls as the threshold rises (the greedy refractory count is the most triggers a subset
    of hops can hold), so a bisection over the sorted distinct scores finds it in ~17 counts. If even the top score
    fires too often, the next float above it is returned (fires on nothing in this stream).
    """
    cand = np.unique(np.asarray(scores, np.float64))
    ok = lambda th: count_triggers(scores, th, hop_s) / hours <= max_fa_per_hour  # noqa: E731
    lo, hi = 0, len(cand)  # answer index in [lo, hi]; hi = "none of the scores"
    while lo < hi:
        mid = (lo + hi) // 2
        if ok(cand[mid]):
            hi = mid
        else:
            lo = mid + 1
    return float(cand[lo]) if lo < len(cand) else float(np.nextafter(cand[-1], np.inf))
