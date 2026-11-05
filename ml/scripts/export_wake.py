"""Export a trained wake-word CNN for the app: ml/models/<name>.pt → <out>/wake.onnx + wake.json.

Usage: uv run --group ml --group wake python -I ml/scripts/export_wake.py [--name wake_cnn_confuse] [--out ml/models]
The app (devdesk/wake_listener.py) reads <repo it runs from>/ml/models, so export into the main checkout's ml/models
when the model was trained in a worktree. torchaudio's STFT does not export to ONNX: the graph takes the power
spectrogram (devdesk.wake_model.power_frames, numpy) and does mel, log, normalization and the CNN. The export is
checked against the torch model on clips from wake/tts (and random noise) and refused if the logits differ.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from devdesk.wake_model import power_frames  # noqa: E402
from ml import wake  # noqa: E402

TOLERANCE = 1e-3  # logits; float32 STFT in numpy vs torch differ ~1e-5


class FromPower(torch.nn.Module):
    """WakeCNN after its STFT: power (B, 241, frames) → logit, with the same mel matrix and normalization."""

    def __init__(self, cnn):
        super().__init__()
        self.register_buffer("fb", cnn.front.mel.mel_scale.fb.clone())  # (241, 40)
        self.body = cnn.body

    def forward(self, power):
        mel = torch.matmul(power.transpose(1, 2), self.fb).transpose(1, 2)
        m = torch.log(mel + 1e-6)
        return self.body((m - m.mean(dim=(1, 2), keepdim=True)).unsqueeze(1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wake_cnn_confuse")
    ap.add_argument("--out", type=Path, default=REPO / "ml" / "models")
    ap.add_argument("--data", type=Path, default=REPO / "ml" / "data", help="folder with wake/tts (the check clips)")
    a = ap.parse_args()
    models = REPO / "ml" / "models"
    metrics = json.loads((models / f"{a.name}_metrics.json").read_text(encoding="utf-8"))
    if metrics["model"] != "cnn":
        ap.error(f"{a.name} is a {metrics['model']} model; only the CNN runs in the app")
    cnn = wake.WakeCNN()
    cnn.load_state_dict(torch.load(models / f"{a.name}.pt", map_location="cpu"))
    cnn.eval()
    graph = FromPower(cnn).eval()

    pos = sorted((a.data / "wake" / "tts" / "pos").glob("*.wav"))[:20]
    neg = sorted((a.data / "wake" / "tts" / "neg").glob("*.wav"))[:20]
    # both sides of the threshold: noise or negatives alone score far below it and would hide a front-end mismatch
    if len(pos) < 10 or len(neg) < 10:
        ap.error(f"need 10+ wake/tts/pos and neg clips under {a.data} to check the export (found {len(pos)}/{len(neg)})")
    clips = pos + neg
    rng = np.random.default_rng(0)
    windows = [wake.place(wake.read(p), None, center=True) for p in clips]
    windows += [(rng.standard_normal(wake.WIN) * 0.05).astype(np.float32) for _ in range(5)]

    a.out.mkdir(parents=True, exist_ok=True)
    onnx_path = a.out / "wake.onnx"
    tmp = a.out / "wake.onnx.tmp"
    example = torch.from_numpy(power_frames(windows[0], np))
    torch.onnx.export(graph, (example,), str(tmp), input_names=["power"], output_names=["logit"], opset_version=17,
                      dynamo=False)

    import onnxruntime as ort

    session = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    worst, wants = 0.0, []
    with torch.no_grad():
        for w in windows:
            want = cnn(torch.from_numpy(w[None])).item()
            got = session.run(None, {"power": power_frames(w, np)})[0].item()
            worst, wants = max(worst, abs(want - got)), wants + [want]
    threshold = metrics["at"]["fa_0.5"]["threshold_logit"]
    if worst > TOLERANCE or max(wants[: len(pos)]) < threshold:  # checked on both sides of the threshold
        tmp.unlink()
        raise SystemExit(f"export check failed: max diff {worst:.2e} logits (limit {TOLERANCE}), best positive "
                         f"{max(wants[: len(pos)]):.2f} vs threshold {threshold:.2f}; not written")
    # The json names the model by its hash, so the app refuses a model and a threshold from two exports (a run cut
    # between the two replaces below).
    meta = {"source": a.name, "seed": metrics.get("seed"), "threshold_logit": threshold,
            "onnx_sha256": hashlib.sha256(tmp.read_bytes()).hexdigest(),
            "note": "threshold = <= 0.5 false alarm/hour on the val stream (TTS + Speech Commands)"}
    meta_tmp = a.out / "wake.json.tmp"
    meta_tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(onnx_path)
    meta_tmp.replace(a.out / "wake.json")
    print(f"wrote {onnx_path} + wake.json from {a.name} ({len(windows)} check windows, max diff {worst:.1e})")


if __name__ == "__main__":
    main()
