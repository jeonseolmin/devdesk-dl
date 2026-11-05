"""Wake word "자비스": train one model and report the metrics the presentation compares.

Usage: uv run --group ml --group wake python -m ml.train_wake [--model cnn|oww] [--data ml/data] [--max-pos N] [--name NAME]
  cnn  scratch CNN on log-mel, trained end to end, augmentation on the fly
  oww  transfer: frozen openWakeWord/Google speech embedding (raw/oww/*.onnx) + a small head; augmented copies
       are embedded once up front (the embedding runs on CPU, ~20 ms per clip)
--data holds wake/ and raw/ (see ml/wake.py); --max-pos caps the training positives (50/100/300 comparison).
Writes ml/models/<name>.pt and <name>_metrics.json.

Main metric: miss rate on test positives at the threshold where the VAL negative stream (Speech Commands validation
words + the val speakers' TTS rest-of-sentence, ~2.8 h) gives <= 0.5 / <= 1 false alarm per hour; the test stream
(testing words + test speakers, ~3 h) then only measures that threshold (fa_per_hour_test).
Real recordings (wake/real, wake/real_neg) are test-only by default: they are what the demo will hear.
--real-train N moves each speaker's first N recordings (recording order, positives and negatives alike) into training,
each repeated --real-copies times (augmented anew each time); the rest stay test (name suffix _real<N>). --real-copies
0 trains on none of them but still tests only the rest (suffix _hold<N>): the no-real baseline on the same clips. The test
speaker is then a training speaker: miss_real measures that one person, not a new voice.
Near-miss words (wake/tts_confuse/<word>, "서비스", "자바", ...): the test speakers' clips give false_accept_confuse
per word; --confuse-train adds the train speakers' clips as negatives (name suffix _confuse).
"""
import argparse
import inspect
import json
import os
import random
import time
import zlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from ml import wake

ROOT = Path(__file__).resolve().parent
SEED = 42


def load(paths, label):
    return [(wake.read(p), label) for p in paths]


def gather(data, sc_train, sc_val, rng, real_first=0):
    """{'train'|'val'|'test': [(audio, label)]}, {'val'|'test': negative stream clips = all Speech Commands clips of
    that list + that split's TTS negatives}, real recordings (split_real(data, real_first)), noise clips."""
    split = {"train": [], "val": [], "test": []}
    for sub, label in (("pos", 1), ("neg", 0)):
        for p in sorted((data / "wake" / "tts" / sub).glob("*.wav")):
            split[wake.speaker_split(wake.tts_speaker(p))].append((p, label))
    clips = {k: [] for k in split}
    for k, items in split.items():
        for label in (1, 0):
            clips[k] += load([p for p, y in items if y == label], label)

    sc = data / "raw" / "speech_commands"
    listed = {name: set((sc / f"{name}_list.txt").read_text().split()) for name in ("validation", "testing")}
    words = sorted(
        str(p.relative_to(sc).as_posix()) for p in sc.glob("*/*.wav") if not p.parent.name.startswith("_")
    )
    train_sc = [w for w in words if w not in listed["validation"] and w not in listed["testing"]]
    clips["train"] += load([sc / w for w in rng.sample(train_sc, min(sc_train, len(train_sc)))], 0)
    stream_sc = {}
    for part, listname in (("val", "validation"), ("test", "testing")):
        names = sorted(listed[listname])
        rng.shuffle(names)  # clip-level negatives take a prefix: every word, not just "backward", "bed", ...
        stream_sc[part] = load([sc / w for w in names], 0)
    for part in stream_sc:  # the part's stream also carries its speakers' TTS negatives (Korean speech)
        stream_sc[part] = stream_sc[part] + [c for c in clips[part] if c[1] == 0]
    clips["val"] += stream_sc["val"][:sc_val]

    real = split_real(data, real_first)
    noise = [wake.read(p).astype(np.float32) for p in sorted((sc / "_background_noise_").glob("*.wav"))]
    noise = [n for n in noise if len(n) > wake.WIN]
    assert noise, f"no noise clip longer than 1.5 s in {sc / '_background_noise_'}"
    return clips, stream_sc, real, noise


def split_real(data, first):
    """Each speaker's first `first` recordings -> train, the rest -> test: {'pos': [(audio, 1)], 'neg': [(audio, 0)],
    'train': [(audio, label)]}. Every *.wav counts, in name order (record_wake.py numbers NNNN.wav in recording
    order); with `first`, other names are warned about, since they change which clips are first. A clip-level split
    within one speaker: the exception to "by speaker, never by clip" (one user's own assistant, 2026-11-02)."""
    out = {"pos": [], "neg": [], "train": []}
    for sub, key, label in (("real", "pos", 1), ("real_neg", "neg", 0)):
        for spk in sorted(p for p in (data / "wake" / sub).glob("*") if p.is_dir()):
            paths = sorted(spk.glob("*.wav"))
            odd = [p.name for p in paths if not (p.stem.isdigit() and len(p.stem) == 4)]
            if first and odd:
                print(f"warning: {sub}/{spk.name}: not record_wake.py names, sorted in by name: {', '.join(odd)}")
            if first and len(paths) <= first:
                print(f"warning: {sub}/{spk.name} has {len(paths)} recordings, all in training: none left to test")
            out["train"] += load(paths[:first], label)
            out[key] += load(paths[first:], label)
    return out


def run_name(a):
    """The run's file name: wake_<model>[_<max-pos>][_confuse][_real<N>[x<copies>] | _hold<N> (copies 0)][_s<seed>]."""
    if a.name:
        return a.name
    real_tag = (f"_real{a.real_train}" if a.real_copies else f"_hold{a.real_train}") if a.real_train else ""
    if a.real_train and a.real_copies not in (0, 20):  # the default's name stays: another count never overwrites it
        real_tag += f"x{a.real_copies}"
    return (f"wake_{a.model}" + (f"_{a.max_pos}" if a.max_pos else "") + ("_confuse" if a.confuse_train else "")
            + real_tag + (f"_s{a.seed}" if a.seed != SEED else ""))


def gather_confuse(data):
    """Near-miss TTS words (ml/scripts/synth_wake.py --word, wake/tts_confuse/<word>/pos) split by speaker like
    the wake word: {'train': [(audio, 0)], 'test': {word: [(audio, 0)]}}. 'test' = val AND test speakers: neither
    trains, and the threshold comes from the val stream, which holds no near-miss clip (a few clips per word only
    otherwise: 1-7 at --per-speaker 2, measured 2026-11-02)."""
    out = {"train": [], "test": {}}
    for d in sorted(p for p in (data / "wake" / "tts_confuse").glob("*") if p.is_dir()):
        for p in sorted((d / "pos").glob("*.wav")):
            if wake.speaker_split(wake.tts_speaker(p)) == "train":
                out["train"] += load([p], 0)
            else:
                out["test"].setdefault(d.name, []).extend(load([p], 0))
    return out


@torch.no_grad()
def scores(model, inputs, dev, batch=512):
    """Logits (not sigmoid: float32 sigmoid saturates to exactly 1.0, see wake.threshold_for_fa)."""
    model.eval()
    out = []
    for i in range(0, len(inputs), batch):
        out.append(model(torch.as_tensor(np.asarray(inputs[i : i + batch]), device=dev)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def centered(clips):
    rng = random.Random(0)
    return [wake.place(x, rng, center=True) for x, _ in clips]


def balanced_sampler(labels, pos_share=0.3, epoch_share=1.0):
    """Draws `epoch_share` x len(labels) items per epoch: with the real copies added, the same number of steps as
    without them (a _real run and its _hold baseline train equally long)."""
    labels = np.asarray(labels)
    weights = np.where(labels == 1, pos_share / labels.sum(), (1 - pos_share) / (1 - labels).sum())
    return torch.utils.data.WeightedRandomSampler(weights, num_samples=round(len(labels) * epoch_share),
                                                  replacement=True)


def setup(kind, a, clips, noise):
    """(model, train loader, inputs(waves), stream_inputs(stream), hop_s, cpu_ms_per_window) for one model kind."""
    labels = [y for _, y in clips["train"]]
    if kind == "cnn":
        train = wake.WakeSet(clips["train"], noise, augment=True, seed=a.seed)
        loader = torch.utils.data.DataLoader(train, batch_size=a.batch,
                                             sampler=balanced_sampler(labels, epoch_share=a.epoch_share))
        model = wake.WakeCNN()

        def cpu_ms(m):
            m = m.cpu().eval()
            x = torch.zeros(1, wake.WIN)
            with torch.no_grad():
                t = time.perf_counter()
                for _ in range(50):
                    m(x)
            return (time.perf_counter() - t) / 50 * 1000

        return model, loader, (lambda w: w), wake.stream_windows, wake.HOP / wake.RATE, cpu_ms

    emb = wake.OwwEmbed(a.data / "raw" / "oww")
    aug = wake.WakeSet(clips["train"], noise, augment=True, seed=a.seed)
    t0 = time.time()
    order = [i for i, y in enumerate(labels) for _ in range(a.pos_copies if y else 1)]
    ys = [labels[i] for i in order]
    # augment + embed 2048 windows at a time: the raw windows never all sit in RAM (96 KB each)
    feats = np.concatenate([
        emb.clips(np.stack([aug[i][0].numpy() for i in order[s : s + 2048]])) for s in range(0, len(order), 2048)
    ])
    print(f"embedded {len(order)} augmented clips in {time.time() - t0:.0f}s", flush=True)
    data = torch.utils.data.TensorDataset(torch.from_numpy(feats), torch.tensor(ys, dtype=torch.float32))
    # the real copies sit at the end of clips["train"]: the epoch is as long as the items made from the rest
    base = labels[:a.real_base]  # the count main() saw before adding the copies, not a ratio rounded back
    share = sum(a.pos_copies if y else 1 for y in base) / len(ys)
    loader = torch.utils.data.DataLoader(data, batch_size=a.batch, sampler=balanced_sampler(ys, epoch_share=share))

    def cpu_ms(m):
        m = m.cpu().eval()
        one = wake.OwwEmbed(a.data / "raw" / "oww", ncpu=1)  # ncpu = ONNX intra/inter-op threads: 1 like the CNN
        x = np.zeros((1, wake.WIN), np.float32)
        with torch.no_grad():
            t = time.perf_counter()
            for _ in range(50):
                m(torch.from_numpy(one.clips(x)))
        return (time.perf_counter() - t) / 50 * 1000

    def inputs(waves):
        return emb.clips(np.stack(waves)) if len(waves) else np.zeros((0, wake.OWW_FRAMES, 96), np.float32)

    return wake.WakeHead(), loader, inputs, emb.stream, wake.OWW_STEP / wake.RATE, cpu_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("cnn", "oww"), default="cnn")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--max-pos", type=int, default=0, help="cap on the TTS training positives (0 = all; "
                    "--real-train's clips come on top)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sc-train", type=int, default=12000)
    ap.add_argument("--sc-val", type=int, default=2000)
    ap.add_argument("--pos-copies", type=int, default=20, help="oww: augmented copies per training positive")
    ap.add_argument("--confuse-train", action="store_true",
                    help="train on the train speakers' near-miss words (wake/tts_confuse) as negatives")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (two runs of one seed still differ a "
                    "little on the GPU)")
    ap.add_argument("--real-train", type=int, default=0,
                    help="train on each speaker's first N real recordings (wake/real, wake/real_neg); the rest test")
    ap.add_argument("--real-copies", type=int, default=None,
                    help="times each real training clip is repeated (default 20; 0 = held out only: the baseline)")
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    if a.epochs < 1:
        ap.error("--epochs must be at least 1")
    if a.real_copies is not None and not a.real_train:
        ap.error("--real-copies needs --real-train")
    a.real_copies = 20 if a.real_copies is None else a.real_copies
    if a.real_train < 0 or a.real_copies < 0:
        ap.error("--real-train and --real-copies must be >= 0")
    name = run_name(a)
    rng = random.Random(SEED)  # data choice stays fixed; --seed varies only training (init, augmentation, order)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    clips, stream_sc, real, noise = gather(a.data, a.sc_train, a.sc_val, rng, a.real_train)
    if a.max_pos:
        pos = [c for c in clips["train"] if c[1] == 1]
        keep = {id(c[0]) for c in rng.sample(pos, min(a.max_pos, len(pos)))}
        clips["train"] = [c for c in clips["train"] if c[1] == 0 or id(c[0]) in keep]
    confuse = gather_confuse(a.data)
    if a.confuse_train:
        assert confuse["train"], "--confuse-train: no clips in wake/tts_confuse/*/pos (run synth_wake.py --word)"
        clips["train"] += confuse["train"]
    a.epoch_share, a.real_base = 1.0, len(clips["train"])  # nothing may be added to clips["train"] after the copies
    if a.real_train:
        assert real["train"], "--real-train: no recordings in wake/real/* or wake/real_neg/*"
        # same arrays, a fresh augmentation each draw; the epoch keeps its length without them (epoch_share), so a
        # _real run takes as many steps as its _hold baseline. oww also multiplies them by --pos-copies (cost only)
        clips["train"] += real["train"] * a.real_copies
        a.epoch_share = a.real_base / len(clips["train"])
    counts = {k: {"pos": sum(y for _, y in v), "neg": sum(1 - y for _, y in v)} for k, v in clips.items()}
    print(f"loaded in {time.time() - t0:.0f}s: {counts}, sc stream val/test "
          f"{len(stream_sc['val'])}/{len(stream_sc['test'])}, "
          f"real {len(real['pos'])}/{len(real['neg'])}", flush=True)
    assert counts["train"]["pos"] and counts["test"]["pos"], "no positives: run ml/scripts/synth_wake.py first"

    model, loader, inputs, stream_inputs, hop_s, cpu_ms = setup(a.model, a, clips, noise)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=a.epochs * len(loader))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    val_x, val_y = inputs(centered(clips["val"])), [y for _, y in clips["val"]]
    # the last epoch is kept (one-cycle schedule); val AP is only watched: clip-level val saturates near 1.0
    # (measured 2026-10-31), so it cannot pick an epoch
    for ep in range(a.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)
        ap_val = average_precision_score(val_y, scores(model, val_x, dev))
        print(f"epoch {ep + 1:2d}: loss {total / n:.4f}, val AP {ap_val:.4f}", flush=True)

    def stream_scores(part):
        """Scores over the part's negative stream (all its Speech Commands clips + its speakers' TTS negatives)."""
        stream = np.concatenate([x.astype(np.float32) for x, _ in stream_sc[part]])
        hours = len(stream) / wake.RATE / 3600
        if a.model != "oww":
            return scores(model, stream_inputs(stream), dev), hours
        # ~11 min of CPU embedding per stream; the key covers the audio, the backbone files, the feature code
        # (OwwEmbed + window constants) and the openwakeword version, so any of them changing re-embeds
        key = zlib.crc32(stream)
        for p in sorted((a.data / "raw" / "oww").glob("*.onnx")):
            key = zlib.crc32(p.read_bytes(), key)
        code = (inspect.getsource(wake.OwwEmbed), wake.WIN, wake.OWW_STEP, wake.OWW_FRAMES, version("openwakeword"))
        key = zlib.crc32(repr(code).encode(), key)
        cache = a.data / "cache" / f"oww_stream_{key:08x}.npy"
        if not cache.exists():
            cache.parent.mkdir(exist_ok=True)
            tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npy")  # per run: two runs never share a temp
            np.save(tmp, stream_inputs(stream))
            try:
                tmp.replace(cache)  # a run cut mid-save (power off) leaves only the .tmp, never a broken cache
            except PermissionError:  # Windows: another run made it meanwhile and has it open; same key = same data
                tmp.unlink()
        return scores(model, np.load(cache, mmap_mode="r"), dev), hours

    test_pos = [c for c in clips["test"] if c[1] == 1]
    test_neg = [c for c in clips["test"] if c[1] == 0]
    s_pos = scores(model, inputs(centered(test_pos)), dev)
    s_neg = scores(model, inputs(centered(test_neg) + centered(stream_sc["test"][:2000])), dev)
    s_real = scores(model, inputs(centered(real["pos"])), dev)
    s_real_neg = scores(model, inputs(centered(real["neg"])), dev)
    s_confuse = {w: scores(model, inputs(centered(c)), dev) for w, c in confuse["test"].items()}
    s_val_stream, val_hours = stream_scores("val")
    s_stream, hours = stream_scores("test")
    metrics = {
        "name": name,
        "model": a.model,
        "seed": a.seed,
        "params": sum(p.numel() for p in model.parameters()),
        "counts": counts,
        "val_ap": float(ap_val),
        "test_ap": float(average_precision_score([1] * len(s_pos) + [0] * len(s_neg), np.r_[s_pos, s_neg])),
        "stream_hours": {"val": val_hours, "test": hours},
        "confuse_train": len(confuse["train"]) if a.confuse_train else 0,
        "confuse_test": {w: len(s) for w, s in s_confuse.items()},
        # clips = real recordings trained on (0 for a _hold baseline); copies only with --real-train
        "real_train": {"first": a.real_train, "clips": len(real["train"]) if a.real_copies else 0,
                       "copies": a.real_copies if a.real_train else None,
                       "epoch_share": round(a.epoch_share, 4), "steps_per_epoch": len(loader)},
        "real_test": {"pos": len(real["pos"]), "neg": len(real["neg"])},
        "at": {},
    }

    def confused(th):  # share of the test speakers' near-miss clips that would wake the assistant, per word
        return {w: float(np.mean(s >= th)) for w, s in s_confuse.items()}
    # thresholds are picked on the val stream; test only measures them (fa_per_hour_test can exceed the target)
    for fa in (0.5, 1.0):
        th = wake.threshold_for_fa(s_val_stream, val_hours, fa, hop_s=hop_s)
        metrics["at"][f"fa_{fa}"] = {
            "threshold_logit": th,
            "fa_per_hour_test": wake.count_triggers(s_stream, th, hop_s) / hours,
            "miss_tts": float(np.mean(s_pos < th)),
            "miss_real": float(np.mean(s_real < th)) if len(s_real) else None,
            "false_accept_real_neg": float(np.mean(s_real_neg >= th)) if len(s_real_neg) else None,
            "false_accept_confuse": confused(th),
        }
    metrics["at"]["0.5"] = {  # probability 0.5 = logit 0, the untuned default
        "fa_per_hour_test": wake.count_triggers(s_stream, 0.0, hop_s) / hours,
        "miss_tts": float(np.mean(s_pos < 0.0)),
        "false_accept_clips": float(np.mean(s_neg >= 0.0)),
        "false_accept_confuse": confused(0.0),
    }
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out / f"{name}.pt")
    metrics["file_kb"] = round((out / f"{name}.pt").stat().st_size / 1024, 1)
    if a.model == "oww":  # the frozen backbone ships with the head
        metrics["backbone_kb"] = round(sum(p.stat().st_size for p in (a.data / "raw" / "oww").glob("*.onnx")) / 1024, 1)
    torch.set_num_threads(1)
    metrics["cpu_ms_per_window_1thread"] = round(cpu_ms(model), 2)
    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
