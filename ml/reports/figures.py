"""Presentation charts from ml/models/*_metrics.json → ml/reports/figures/*.png (static, for slides).

Usage: uv run --group ml python -m ml.reports.figures
Charts whose metrics file is missing are skipped with a note. Colors: categorical slots in fixed order
(validated default palette), text in ink colors, one y-axis per panel.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS, OUT = ROOT / "models", Path(__file__).resolve().parent / "figures"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # slots 1-4, in order, never cycled
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
for font in ("Malgun Gothic", "AppleGothic", "NanumGothic"):  # Hangul labels
    if any(f.name == font for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "axes.unicode_minus": False,
})


def load(name):
    p = MODELS / f"{name}_metrics.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


def seed_runs(base):
    """Metrics of one config over its training seeds: <base> (seed 42) + <base>_s<k> (ml.train_wake --seed k)."""
    return [m for m in (load(base), *(load(f"{base}_s{k}") for k in range(10))) if m]


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {OUT / name}.png")


def bars(ax, groups, series, values, err=None, fmt="{:.2f}"):
    """Grouped bars: values[s][g]; err[s][g] = (lo, hi) or None; value labels on top (ink, not series color)."""
    w = 0.8 / len(series)
    for i, s in enumerate(series):
        xs = [g + (i - (len(series) - 1) / 2) * w for g in range(len(groups))]
        ax.bar(xs, values[i], w - 0.02, color=SERIES[i], label=s, edgecolor=SURFACE, linewidth=1)
        for j, (x, v) in enumerate(zip(xs, values[i], strict=True)):
            e = err[i][j] if err else None
            if e:
                ax.errorbar(x, v, yerr=[[err_size(v - e[0])], [err_size(e[1] - v)]], color=INK2, capsize=3, lw=1)
            ax.annotate(fmt.format(v), (x, e[1] if e else v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK2)  # points: clear of the error cap at any scale
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series))  # under the title


def err_size(gap):
    """An error bar's length: rounding can put a mean a hair outside its own min..max (0), yerr must not be negative;
    a real gap is bad metrics and stays loud."""
    if gap < -1e-6:
        raise ValueError(f"a value lies {-gap} outside its error range")
    return max(gap, 0)


def title(ax, text):
    ax.set_title(text, pad=32)  # room for the legend row between title and plot


def wake_data_size():
    runs = {}
    for kind in ("cnn", "oww"):
        for cap in ("50", "100", "200", ""):
            ms = seed_runs(f"wake_{kind}" + (f"_{cap}" if cap else ""))
            if ms:
                runs.setdefault(kind, []).append(ms)
    if not runs:
        return print("skip wake_data_size: no wake_* metrics")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for i, (kind, points) in enumerate(runs.items()):
        points.sort(key=lambda ms: ms[0]["counts"]["train"]["pos"])
        x = [ms[0]["counts"]["train"]["pos"] for ms in points]
        label = {"cnn": "CNN (처음부터)", "oww": "전이학습 (openWakeWord 임베딩)"}[kind]
        for ax, get in ((a1, lambda m: 100 * m["at"]["fa_0.5"]["miss_tts"]),
                        (a2, lambda m: m["at"]["0.5"]["fa_per_hour_test"])):
            vals = [[get(m) for m in ms] for ms in points]
            ax.plot(x, [sum(v) / len(v) for v in vals], "o-", color=SERIES[i], lw=2, ms=8, label=label,
                    zorder=3 - i, mec=SURFACE, mew=2)  # CNN on top: lines can overlap at 0
            ax.fill_between(x, [min(v) for v in vals], [max(v) for v in vals], color=SERIES[i], alpha=0.15, lw=0)
    seeds = min(len(ms) for points in runs.values() for ms in points)
    a1.set(title="놓친 비율 (val에서 오경보 ≤0.5회/시간 기준)", xlabel="학습용 '자비스' 클립 수", ylabel="놓침 (%)")
    a2.set(title="오경보 / 시간 (튜닝 안 한 임계값 0.5)", xlabel="학습용 '자비스' 클립 수", ylabel="회/시간")
    fig.suptitle(f"웨이크워드: 학습 데이터 양 (선 = 시드 {seeds}개 평균, 띠 = 범위)", fontsize=12)
    for a in (a1, a2):
        a.set_ylim(bottom=0)
    a1.legend(frameon=False)
    save(fig, "wake_data_size")


def wake_confuse():
    label = {"wake_cnn": "CNN", "wake_cnn_confuse": "CNN + 비슷한 단어 학습",
             "wake_oww": "전이학습", "wake_oww_confuse": "전이학습 + 비슷한 단어 학습"}
    runs = {}  # config → its seeds' metrics
    for base in label:
        seeds = [m for m in seed_runs(base) if m.get("confuse_test")]
        if seeds:
            runs[base] = seeds
    if not runs:
        return print("skip wake_confuse: no run with wake/tts_confuse clips yet")
    n = {}
    for ms in runs.values():
        for m in ms:
            n.update(m["confuse_test"])
    words = sorted(n, key=lambda w: -n[w])

    def stat(ms, get):
        v = [100 * get(m) for m in ms]
        return sum(v) / len(v), (min(v), max(v))

    def at(w):
        return lambda m: m["at"]["fa_0.5"]["false_accept_confuse"].get(w, 0)

    cells = [[stat(ms, at(w)) for w in words] for ms in runs.values()]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    bars(ax, [f"{w}\n(n={n[w]})" for w in words], [label[b] for b in runs],
         [[c[0] for c in row] for row in cells], err=[[c[1] for c in row] for row in cells], fmt="{:.0f}")
    ax.set(ylabel="'자비스'로 깨어난 비율 (%)", ylim=(0, 110))
    seeds = min(len(ms) for ms in runs.values())
    title(ax, f"비슷한 단어에 깨어나는 비율 (학습에 안 쓴 화자; 오경보 ≤0.5회/시간 기준값; 시드 {seeds}개 평균, 막대 = 범위)")
    save(fig, "wake_confuse")


def wake_real():
    """Miss rate on the user's own held-out recordings: no real clip in training (_hold20) vs 20 of them (_real20)."""
    kinds = {"cnn": "CNN (처음부터)", "oww": "전이학습"}
    series = {"hold20": "TTS만 학습", "real20": "+ 내 녹음 20개"}
    runs = {(k, t): [m for m in seed_runs(f"wake_{k}_confuse_{t}")
                     if m.get("real_test") and m["at"]["fa_0.5"].get("miss_real") is not None]
            for k in kinds for t in series}  # runs with real test clips only (older files and real40 have none)
    kinds = {k: v for k, v in kinds.items() if all(runs[(k, t)] for t in series)}
    if not kinds:
        return print("skip wake_real: no wake_*_confuse_hold20 / _real20 metrics")
    sets = {json.dumps(m["real_test"], sort_keys=True) for k in kinds for t in series for m in runs[(k, t)]}
    if len(sets) > 1:  # a run made before recordings were added or deleted: its bar would mix two test sets
        return print(f"skip wake_real: the runs used different real test sets {sorted(sets)}; re-run them")

    def stat(ms):
        v = [100 * m["at"]["fa_0.5"]["miss_real"] for m in ms]
        return sum(v) / len(v), (min(v), max(v))

    cells = [[stat(runs[(k, t)]) for k in kinds] for t in series]
    n = runs[(next(iter(kinds)), "hold20")][0]["real_test"]["pos"]
    seeds = {k: min(len(runs[(k, t)]) for t in series) for k in kinds}  # the fewer of the two bars
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars(ax, [f"{v}\n(시드 {seeds[k]}개)" for k, v in kinds.items()], list(series.values()),
         [[c[0] for c in row] for row in cells], err=[[c[1] for c in row] for row in cells], fmt="{:.0f}")
    ax.set(ylabel="놓친 비율 (%)", ylim=(0, 115))
    title(ax, f"내 목소리 '자비스' {n}개를 놓친 비율 (학습에 안 쓴 녹음; 오경보 ≤0.5회/시간 기준값)")
    save(fig, "wake_real")


def next_app():
    m = load("next_app_behacom")
    if not m:
        return print("skip next_app: no next_app_behacom metrics")
    t = m["test"]
    names = list(t)
    label = {"back (A→B→A)": "되돌아가기\n(A→B→A)", "markov (per user)": "마르코프\n(사용자별)",
             "lightgbm": "LightGBM", "gru": "GRU"}
    fig, ax = plt.subplots(figsize=(9, 4))
    spread = t["gru"].get("seed_spread", {})
    err = [[tuple(spread[k]) if n == "gru" and k in spread else None for n in names] for k in ("top1", "top3")]
    bars(ax, [label.get(n, n) for n in names], ["top-1", "top-3"],
         [[t[n]["top1"] for n in names], [t[n]["top3"] for n in names]], err=err)
    ax.set(ylim=(0, 1), ylabel="정확도")
    title(ax, f"다음 앱 예측 (BEHACOM {m['users']}명, 테스트 {t['gru']['n']}건; GRU 오차 막대 = 시드 범위)")
    save(fig, "next_app")


def day_anomaly():
    m = load("day_anomaly_behacom")
    if not m:
        return print("skip day_anomaly: no day_anomaly_behacom metrics")
    methods = list(m["auc"])
    kinds = ["night_game", "shift", "swapped"]
    label = {"user mean distance": "평균과의 거리", "autoencoder (per user)": "오토인코더"}
    label |= {me: "PCA" for me in methods if me.startswith("pca")}  # the key carries the component count
    spread = m["auc"]["autoencoder (per user)"].get("seed_spread", {})
    err = [[tuple(spread[k]) if "autoencoder" in me and k in spread else None for k in kinds] for me in methods]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars(ax, ["밤에 게임", "시간대 밀림", "다른 사람의 하루"], [label.get(x, x) for x in methods],
         [[m["auc"][me][k] for k in kinds] for me in methods], err=err)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(-0.5, 0.47, "무작위 = 0.5", ha="left", va="top", fontsize=8, color=INK2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    ax.set(ylim=(0, 1.05), ylabel="AUC")
    title(ax, f"하루 패턴 이상 탐지 (BEHACOM, 테스트 {m['user_days']['test']}일; 오토인코더 오차 막대 = 시드 범위)")
    save(fig, "day_anomaly")


def work_share():
    meas, model = load("work_share_behacom"), load("work_share_model_behacom")
    if not meas or not model:
        return print("skip work_share: run ml.work_share and ml.train_work_share first")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    away = meas["away"]
    a1.bar(range(len(away["counts"])), away["counts"], 0.8, color=SERIES[0], edgecolor=SURFACE)
    a1.set_xticks(range(len(away["counts"])), [b.replace("-inf", "+") for b in away["bins_min"]])
    a1.set(xlabel="업무 앱을 떠나 있던 시간 (분)", ylabel="횟수", title="떠났다 돌아오기까지 (경계가 없음)")
    a1.grid(axis="x", visible=False)
    r = model["10"]
    names = list(r["mae_minutes"])
    label = {"user mean": "사람별 평균", "like the last H min": "직전 10분처럼", "lightgbm (base)": "LightGBM",
             "gru (base)": "GRU", "lightgbm (+extras)": "LightGBM\n+추가 정보", "gru (+extras)": "GRU\n+추가 정보"}
    color = [INK2 if n in ("user mean", "like the last H min") else SERIES[1] if "base" in n else SERIES[2]
             for n in names]
    vals = [r["mae_minutes"][n] for n in names]
    a2.bar(range(len(names)), vals, 0.7, color=color, edgecolor=SURFACE)
    for i, v in enumerate(vals):
        a2.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    a2.set_xticks(range(len(names)), [label.get(n, n) for n in names], fontsize=9)
    a2.set(ylabel="평균 오차 (분)", ylim=(0, max(vals) * 1.15),
           title=f"다음 10분 중 업무 시간 예측 (테스트 {r['points']['test']}건, 낮을수록 좋음)")
    a2.grid(axis="x", visible=False)
    save(fig, "work_share")


def main():
    for chart in (wake_data_size, wake_confuse, wake_real, next_app, day_anomaly, work_share):
        chart()


if __name__ == "__main__":
    main()
