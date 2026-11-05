"""DevDesk 딥러닝 과제를 한 페이지로 묶은 Streamlit 앱: 과제별 결과 + 차트 + 모델 코드 + 리포트 + "자비스" 실시간 추론.

Local:  uv run --group ml streamlit run ml/streamlit_app.py
Deploy: Streamlit Community Cloud, main file ml/streamlit_app.py (ml/requirements.txt beside it: no Qt, no torch).

The results come from ml/reports/ (nothing is restated here), the charts from ml/reports/figures/. Every model runs
here, each panel beside the test numbers of its own retraining (demos.json): the overview's table checks each one, each task's tab takes your input. The files come from
ml/scripts/export_wake.py and ml/scripts/export_demos.py. A missing or broken one turns off only its own panel
(a Cloud deployment has only what git has: the demo_* models, not wake.onnx, which is trained on the user's voice).
A clip goes from the browser to the machine running this page, by the user's own click, and stays in memory there:
never written or logged. On Community Cloud that machine is not the user's; the page says so.
"""
import ast
import io
import json
import re
import sys
import textwrap
import wave
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run` puts only ml/ on the path; the script reruns in one process
    sys.path.insert(0, str(ROOT))

from devdesk.wake import HOP, RATE, WIN, Trigger  # noqa: E402
from devdesk.wake_model import MODEL_DIR, checked_session, load_model  # noqa: E402
from ml import day_anomaly, next_app, work_share  # noqa: E402

REPORTS = ROOT / "ml" / "reports"
MAX_S = 30  # longest clip: a public deployment must not score for minutes on one upload
MAX_RATE, MAX_CHANNELS = 192000, 2  # with MAX_S: at most 23 MB of samples, whatever the header claims
# The upload itself is capped for every widget by .streamlit/config.toml (server.maxUploadSize).
REPORT_FILES = ("wake-word.md", "activity-dl.md", "ml1-behacom.md", "datasets.md")
TASKS = (  # tab, bullets of the report's section 2, charts, model code (file, class or function)
    ("웨이크워드", ("웨이크워드", "비슷한 단어", "실제 목소리", "재현성"),
     ("wake_data_size", "wake_confuse", "wake_real"), (("ml/wake.py", "WakeCNN"), ("ml/wake.py", "WakeHead"))),
    ("다음 앱 예측", ("다음 앱",), ("next_app",), (("ml/next_app.py", "gru_model"),)),
    ("하루 이상 탐지", ("하루 이상 탐지",), ("day_anomaly",), (("ml/day_anomaly.py", "autoencoder"),)),
    ("집중 이탈 예측", ("집중 이탈",), ("work_share",), (("ml/train_work_share.py", "Net"),)),
)


def section(md, n):
    """The body of `## <n>. ...` in a report."""
    return re.search(rf"^## {n}\. .*?\n(.*?)(?=^## |\Z)", md, re.S | re.M).group(1).strip()


def bullet(md, name):
    """One result of section 2 (nested lines included), without its pointer to the chart file shown beside it."""
    found = re.search(rf"^- \*\*{name}.*?(?=^- \*\*|\Z)", section(md, 2), re.S | re.M).group(0).strip()
    return re.sub(r"`figures/[^`]+`(, )?", "", found).replace(" ()", "")


def code_of(path, name):
    """The source of one class or function, so the page shows the model the report measured, not a copy of it."""
    src = (ROOT / path).read_text("utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name == name:
            return textwrap.dedent(ast.get_source_segment(src, node, padded=True))
    raise LookupError(f"{name} not in {path}")


def read_wav(data):
    """16-bit PCM WAV bytes → mono float32 at RATE. Untrusted input: anything else raises ValueError with a message."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            rate, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if width != 2:
                raise ValueError("16-bit PCM WAV만 읽을 수 있습니다.")
            if not 8000 <= rate <= MAX_RATE:
                raise ValueError("샘플레이트가 범위 밖입니다.")
            if channels > MAX_CHANNELS:
                raise ValueError("모노나 스테레오만 읽을 수 있습니다.")
            if n / rate > MAX_S:
                raise ValueError(f"{MAX_S}초까지만 분석합니다.")
            raw = w.readframes(n)
    except (wave.Error, EOFError, RuntimeError) as e:  # RuntimeError: chunk sizes that do not add up
        raise ValueError("16-bit PCM WAV만 읽을 수 있습니다.") from e
    x = np.frombuffer(raw[:len(raw) // 2 * 2], "<i2").astype(np.float32) / 32768
    x = x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    if rate != RATE:  # ponytail: linear, no anti-aliasing filter; good enough for a demo, the mic path needs none
        x = np.interp(np.arange(int(len(x) * RATE / rate)) * rate / RATE, np.arange(len(x)), x).astype(np.float32)
    if len(x) < HOP:
        raise ValueError("소리가 너무 짧습니다.")
    return x


def score_stream(x, score, threshold):
    """(logit per 100 ms hop, seconds where it fired): the app listener's window and Trigger, over a clip. Only the
    clip is scored: silence added after it fired where the clip's noise stopped (measured), so the page asks for a
    moment of recording after the word instead."""
    x = np.concatenate([np.zeros(WIN, np.float32), x])
    trigger, scores, fired = Trigger(threshold, HOP / RATE), [], []
    for end in range(WIN + HOP, len(x) + 1, HOP):
        scores.append(score(x[end - WIN:end]))
        if trigger.step(scores[-1]):
            fired.append((end - WIN) / RATE)
    return scores, fired


def load_demo(key, model_dir=MODEL_DIR):
    """(run(**inputs) -> output array, this model's part of demos.json), from ml/scripts/export_demos.py. Raises when a
    file is missing or the pair does not match."""
    meta = json.loads((Path(model_dir) / "demos.json").read_text("utf-8"))[key]
    session = checked_session(Path(model_dir) / f"demo_{key}.onnx", meta["sha256"])
    return (lambda **inputs: session.run(None, inputs)[0]), meta


def visit_chain(picked):
    """Picked apps (None = no visit), oldest first → a visit chain: a visit ends on a switch, so repeats in a row merge."""
    chain = []
    for app in picked:
        if app is not None and (not chain or chain[-1] != app):
            chain.append(app)
    return chain


def next_predict(run, meta, chain, minutes, hour, user):
    """Visit chain of app ids → (GRU's top 3 as (id, probability), "go back"'s top 3 ids); never the current app."""
    num = next_app.visit_features([minutes] * len(chain), [hour] * len(chain))[None]
    prob = run(apps=np.array([chain], np.int64), num=num, users=np.array([user], np.int64))[0]
    cur, prev = chain[-1], chain[-2] if len(chain) > 1 else -1
    order = next_app.mask_current(prob[None], [cur])[0]
    back = next_app.rank_back(pd.DataFrame({"prev": [prev], "cur": [cur]}), meta["fallback"])[0]
    return [(a, float(prob[a])) for a in order[:3]], back[:3]


def day_check(run, day, user):
    """(the model's reconstruction, its error): how far this day is from what the model expects of this user."""
    x = np.asarray(day, np.float32)
    recon = run(x=x[None], user=np.array([user], np.int64))[0]
    return recon, float(((recon - x) ** 2).mean())


def day_variant(days, user, kind):
    """The user's held-out day as it is, with an injected anomaly, or another user's day."""
    day = np.array(days[str(user)], np.float64)
    if kind == "swapped":
        others = sorted(int(u) for u in days if int(u) != user)
        return np.array(days[str(others[0])], np.float64) if others else day
    return day if kind == "normal" else day_anomaly.inject(day, kind, np.random.default_rng(0))


def focus_seq(work_minutes, away, switches):
    """The last SEQ minutes as the forecast sees them: `away` first, then work_minutes of IDE up to now, with input."""
    names = work_share.seq_names(False)
    seq = np.zeros((work_share.SEQ, len(names)), np.float32)
    seq[:, names.index("active")] = seq[:, names.index("present")] = 1
    seq[:, names.index("switches")] = np.log1p(switches)
    seq[1:, names.index("gap")] = 0.1  # one minute apart, as the export scales it
    start = work_share.SEQ - work_minutes
    seq[:start, names.index(f"cat_{away}")] = 1
    seq[start:, names.index("cat_ide")] = seq[start:, names.index("work")] = 1
    return seq


def focus_predict(run, meta, seq, user):
    """Minutes of the next horizon at work, by the GRU."""
    return float(run(seq=seq[None], user=np.array([user], np.int64))[0]) * meta["horizon"]


def check_rows(model_dir=MODEL_DIR):
    """One row per model: does it load and answer one input here? Any failure is that model's row, never the page's."""
    def wake_row():
        score, threshold = load_model(model_dir)
        return f"점수 {score(np.zeros(WIN, np.float32)):.2f}, 기준 {threshold:.2f}"

    def next_row():
        run, meta = load_demo("next_app", model_dir)
        top, _ = next_predict(run, meta, [1, 2], 5, 14, 0)
        return f"시험 top-1 {meta['gru']['top1']:.3f}, top-3 {meta['gru']['top3']:.3f} · 예시 1순위 {meta['apps'][top[0][0]]}"

    def day_row():
        run, meta = load_demo("day", model_dir)
        user = min(int(u) for u in meta["days"])
        _, error = day_check(run, meta["days"][str(user)], user)
        return f"시험 AUC 밤에 게임 {meta['auc']['night_game']:.2f} · 예시 오차 {error:.4f}"

    def focus_row():
        run, meta = load_demo("focus", model_dir)
        minutes = focus_predict(run, meta, focus_seq(15, "browser", 1), 0)
        return f"시험 MAE {meta['mae_gru']:.2f}분 (사람별 평균 {meta['mae_user_mean']:.2f}분) · 예시 {minutes:.1f}분"

    rows = []
    for name, check in (("웨이크워드 CNN", wake_row), ("다음 앱 GRU", next_row), ("하루 이상 오토인코더", day_row),
                        ("집중 이탈 GRU", focus_row)):
        try:
            rows.append({"모델": name, "상태": "✅ 동작", "확인 내용": check()})
        except Exception as e:  # fail-open: a row says what is wrong, without the exception's text
            rows.append({"모델": name, "상태": "❌ 사용 불가", "확인 내용": type(e).__name__})
    return rows


def panel(demo, st):
    """One try-it panel: whatever goes wrong in it stays in it (a stale demos.json, a graph with other inputs)."""
    try:
        demo(st)
    except Exception as e:  # st.stop / rerun are not Exceptions; the tabs after this one still draw
        st.error(f"이 패널을 실행하지 못했습니다 ({type(e).__name__}).")


def demo_or_info(st, key, model_dir):
    try:
        return st.cache_resource(load_demo)(key, str(model_dir))
    except Exception as e:  # fail-open: only this panel turns off
        st.info(f"이 모델을 쓸 수 없어 실행을 끕니다 ({type(e).__name__}). `ml/scripts/export_demos.py`로 만듭니다.")
        return None


def next_demo(st, model_dir=MODEL_DIR):
    st.subheader("직접 해 보기: 다음에 쓸 앱")
    loaded = demo_or_info(st, "next_app", model_dir)
    if loaded is None:
        return
    run, meta = loaded
    st.caption(f"재학습 시험 결과: GRU top-1 {meta['gru']['top1']:.3f}, top-3 {meta['gru']['top3']:.3f} / "
               f"되돌아가기 top-1 {meta['back']['top1']:.3f}, top-3 {meta['back']['top3']:.3f}")
    names, none = meta["apps"], "(없음)"
    picks = []
    for i, col in enumerate(st.columns(4)):
        label = "지금 앱" if i == 3 else f"{3 - i}번 전"
        # the first box starts empty, the rest as a back-and-forth
        picks.append(col.selectbox(label, (none, *names[1:]), index=(0, 2, 1, 2)[i], key=f"next_{i}"))
    chain = visit_chain([None if p == none else names.index(p) for p in picks])
    left, mid, right = st.columns(3)
    user = left.selectbox("사용자", range(meta["users"]), format_func=lambda u: f"사용자 {u}", key="next_user")
    minutes = mid.slider("앱마다 머문 시간 (분)", 1, 60, 5, key="next_minutes")
    hour = right.slider("시각", 0, 23, 14, key="next_hour")
    if picks[-1] == none:  # an earlier box must not silently become the current app
        st.warning("지금 앱을 고르세요.")
        return
    top, back = next_predict(run, meta, chain, minutes, hour, user)
    left, right = st.columns(2)
    left.markdown("**GRU 예측**")
    left.table([{"순위": i + 1, "앱": names[a], "확률": f"{p:.0%}"} for i, (a, p) in enumerate(top)])
    right.markdown("**비교: 되돌아가기 규칙**")
    right.table([{"순위": i + 1, "앱": names[a]} for i, a in enumerate(back)])


def day_demo(st, model_dir=MODEL_DIR):
    st.subheader("직접 해 보기: 이 하루가 평소 같은가")
    loaded = demo_or_info(st, "day", model_dir)
    if loaded is None:
        return
    run, meta = loaded
    st.caption(f"재학습 시험 AUC: 밤에 게임 {meta['auc']['night_game']:.2f}, 시간대 밀림 {meta['auc']['shift']:.2f}, "
               f"다른 사람의 하루 {meta['auc']['swapped']:.2f}. 학습에 쓰지 않은 실제 하루를 씁니다.")
    users = sorted(int(u) for u in meta["days"])
    left, right = st.columns(2)
    user = left.selectbox("사용자", users, format_func=lambda u: f"사용자 {u}", key="day_user")
    kinds = {"그대로": "normal", "밤에 게임 3시간 넣기": "night_game", "6시간 밀기": "shift", "다른 사람의 하루": "swapped"}
    kind = kinds[right.radio("이 하루를", tuple(kinds), key="day_kind")]
    x = day_variant(meta["days"], user, kind)
    recon, error = day_check(run, x, user)
    left, right = st.columns(2)
    limit = meta["normal_p95"][str(user)]
    left.metric(f"복원 오차 (이 사용자의 평소 95 %: {limit:.4f} 이하)", f"{error:.4f}")
    right.metric("판정", "평소와 다름" if error > limit else "평소 범위")
    apps = list(day_anomaly.CHANNELS[:-1])
    for col, title, values in ((left, "이 하루 (시간별 앱 비율)", x), (right, "모델이 복원한 하루", recon)):
        col.caption(title)
        col.bar_chart(pd.DataFrame(np.asarray(values).reshape(24, -1)[:, :len(apps)], columns=apps))


def focus_demo(st, model_dir=MODEL_DIR):
    st.subheader("직접 해 보기: 다음 10분 중 업무 시간")
    loaded = demo_or_info(st, "focus", model_dir)
    if loaded is None:
        return
    run, meta = loaded
    st.caption(f"재학습 시험 MAE: GRU {meta['mae_gru']:.2f}분, 사람별 평균 {meta['mae_user_mean']:.2f}분")
    users = sorted(int(u) for u in meta["user_mean"])
    left, mid, right = st.columns(3)
    user = left.selectbox("사용자", users, format_func=lambda u: f"사용자 {u}", key="focus_user")
    work = mid.slider("최근 30분 중 방금까지 이어진 업무 (분)", 1, work_share.SEQ, 15, key="focus_work")
    switches = right.slider("1분당 앱 전환", 0, 10, 1, key="focus_switches")
    away = st.radio("그 전에 쓴 앱", ("browser", "chat", "game", "other"), horizontal=True, key="focus_away")
    minutes = focus_predict(run, meta, focus_seq(work, away, switches), user)
    left, right = st.columns(2)
    left.metric(f"예측: 다음 {meta['horizon']}분 중 업무", f"{minutes:.1f}분")
    right.metric("비교: 이 사용자의 평균", f"{meta['user_mean'][str(user)] * meta['horizon']:.1f}분")


def wake_demo(st, model_dir=MODEL_DIR):
    st.subheader("직접 해 보기: \"자비스\"")
    try:
        score, threshold = st.cache_resource(load_model)(str(model_dir))
    except Exception as e:  # fail-open: only the demo turns off
        st.info(f"웨이크워드 모델을 쓸 수 없어 데모를 끕니다 ({type(e).__name__}). "
                "`ml/models/wake.onnx`와 `wake.json`이 필요합니다.")
        return
    st.caption("앱에 들어간 CNN(ONNX)이 CPU에서 100 ms마다 점수를 냅니다. 녹음이나 파일은 이 페이지를 실행하는 서버로 "
               "전송되어 메모리에서만 처리되며, 파일이나 로그로 남기지 않습니다. 학습에 쓴 실제 목소리는 한 사람뿐이라 "
               "다른 목소리에서는 잘 안 잡힐 수 있습니다. 말한 뒤 1초쯤 더 녹음하세요. WAV는 16 kHz가 가장 정확합니다.")
    if st.radio("입력", ("마이크로 녹음", "WAV 올리기"), horizontal=True) == "마이크로 녹음":  # one source: never a stale one
        clip = st.audio_input("마이크로 녹음", sample_rate=RATE)
    else:
        clip = st.file_uploader(f"WAV 올리기 (16-bit PCM, {MAX_S}초까지)", type="wav")
    if clip is None:
        return
    try:
        x = read_wav(clip.getvalue())
    except ValueError as e:
        st.error(str(e))
        return
    scores, fired = score_stream(x, score, threshold)
    left, right = st.columns(2)
    left.metric(f"가장 높은 점수 (logit, 기준 {threshold:.2f})", f"{max(scores):.2f}")
    right.metric("감지 횟수", len(fired))
    st.line_chart({"점수": scores, "기준값": [threshold] * len(scores)})
    st.caption("가로축: 100 ms 단위 구간 번호")
    if fired:
        st.success("감지: " + ", ".join(f"{s:.1f}초" for s in fired))
    else:
        st.warning("감지되지 않았습니다.")


def main():
    import streamlit as st

    st.set_page_config(page_title="DevDesk 딥러닝", layout="wide")
    st.title("DevDesk에 딥러닝 붙이기")
    md = (REPORTS / "presentation.md").read_text("utf-8")
    overview, *task_tabs, learned, next_steps, reports = st.tabs(["개요", *(t[0] for t in TASKS), "배운 것", "다음", "전체 리포트"])
    overview.markdown(section(md, 1))
    overview.subheader("모델 점검표")
    overview.table(st.cache_data(ttl=60)(check_rows)())  # not for good: a model that appears or breaks must show
    overview.caption("활동 모델은 BEHACOM(CC BY 4.0, DOI 10.17632/cg4br62535.2)으로 시드 42로 다시 학습했습니다. 각 과제 탭에서 직접 입력해 볼 수 있습니다.")
    demos = (wake_demo, next_demo, day_demo, focus_demo)  # one per TASKS row
    for tab, (_, bullets, figures, codes), demo in zip(task_tabs, TASKS, demos, strict=True):
        with tab:
            for name in bullets:
                st.markdown(bullet(md, name))
            for fig in figures:
                st.image(str(REPORTS / "figures" / f"{fig}.png"))
            with st.expander("모델 코드"):
                for path, name in codes:
                    st.caption(f"{path} · {name}")
                    st.code(code_of(path, name), language="python")
            st.fragment(panel)(demo, st)  # using one panel reruns only that panel
    learned.markdown(section(md, 3))
    next_steps.markdown(section(md, 4))
    with reports:
        st.fragment(lambda: st.markdown((REPORTS / st.selectbox("리포트", REPORT_FILES)).read_text("utf-8")))()  # no re-scoring of a held clip


if __name__ == "__main__":
    main()
