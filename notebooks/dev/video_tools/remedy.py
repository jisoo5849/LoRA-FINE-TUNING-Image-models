"""보완: 체크포인트 지표 → 증상 → 다음 실행의 FIXED 덮어쓰기 → run_many 형식 계획.

조사 노트 (영상 데이터 5절) 의 '증상 → 원인 → 조정표' 를 옮겼다. 기준값·조정 폭은 판단값이라 첫 파일럿에서 고친다.
  * step_metrics[step]: motion_ratio, flicker_rel, color_shift, drift_rel, id_pass, id_slope, memo, consistency
    (모두 선택, 못 잰 값·없는 키는 None 으로 본다). 0 단계(베이스)는 보지 않는다. consistency 는 표시용 (규칙 없음).
    color_shift·drift_rel 은 트리거 없는(class word) 클립에서 잰 값이다 (트리거 클립은 캐릭터 색이 바뀌는 게 정상).
  * val_loss: {step: 홀드아웃(학습에 안 쓴) 샘플의 고정 검증 손실}. 학습 샘플로 잰 손실은 암기해도 계속 내려가 과적합을
    못 보므로 None 으로 넘긴다. 그러면 손실 규칙은 모두 건너뛰고, 과적합은 암기(memo)·고른 단계 뒤 악화(그림체 드리프트 등)
    로만 보며, 추천 단계는 품질 지표로 고른다 (못 고르면 마지막 단계).
  * 판정: 점 지표 (움직임·깜빡임·색·그림체·클립 안 정체성) 는 pick_best_step 이 고른 단계에서, 정체성 추세
    (underfit / identity_weak) 와 과적합 (검증 손실 상승·암기·고른 단계 뒤 악화) 은 단계 전체에서 본다.
    입력이 없는 규칙은 건너뛴다.
  * 한 실행에 2개까지만 바꾼다 (효과를 가려내려고). PRIORITY 순서로, ALLOWED 범위로 자르고, 그대로인 값과 fixed 에 없는
    키는 건너뛴다 (계열이 쓰지 않는 키는 호출하는 쪽이 fixed 에서 뺀다). rank 를 바꾸면 alpha 도 같은 값
    (ComfyUI 는 alpha 없는 LoRA 를 alpha = rank 로 적용).
  * 다음 실행 계획 (next_run_plan) 은 이번 실행의 덮어쓰기 (base_overrides) 위에 이번 제안을 얹는다: 앞선 보완이 이어지고,
    이번 실행과 다른 값은 이번 제안 (2개까지) 뿐이다.
  * 움직임 손실의 target_scope "kvo": 정지 이미지로 어텐션 Q/K/V/O 를 학습하면 움직임이 줄고 K/V/O 만 학습하면 유지됐다
    (V-Warper, arXiv 2512.12375, CogVideoX-5B). Wan 에서는 실측 전.
자동 제안은 사람이 컨택트 시트를 보고 승인한 뒤 실행한다.
"""
import math
import numbers
import re
from collections.abc import Mapping
from pathlib import Path

THR = dict(
    id_pass=0.90,             # CCIP 통과율 (같은 캐릭터로 판정된 프레임 비율) 하한
    id_rise=0.02,             # 마지막 두 체크포인트 사이 통과율이 이보다 더 오르면 아직 배우는 중 (underfit)
    id_slope=0.002,           # 클립 안에서 CCIP 차이가 프레임마다 이보다 더 커지면 정체성 흔들림
    motion_low=0.6,           # 베이스 대비 움직임 비율 하한·상한
    motion_high=1.6,
    best_motion=(0.7, 1.3),   # pick_best_step 이 받아들이는 움직임 비율
    flicker_rel=1.5,          # 베이스 대비 깜빡임 (밝기 luma_flicker·대비 contrast_flicker 비율 가운데 큰 쪽)
    color_shift=0.10,         # class-word 클립의 채도·평균색 상대 변화 (베이스 대비)
    drift_rel=2.0,            # class-word 그림체 드리프트 / 베이스 시드 간 변동
    memo=0.95,                # 새 프롬프트 프레임과 가장 가까운 학습 프레임의 DINOv2 cos (암기)
    val_rise=0.02,            # 홀드아웃 검증 손실이 최저점보다 2% 넘게 오르면 과적합 (val_loss 가 None 이면 안 봄)
)
# min_motion 단위 = video_io subject_motion: 초당 화면 긴 변의 비율 (0.05 = 1초에 긴 변의 5%).
# 0.05 는 video_io.STATIC_MOTION 과 같은 값 = 노트북 EDA 의 '거의 정지한 클립' 기준 (832x480·16fps 에서 약 2.6px/프레임 아래)
MIN_MOTION_ON = 0.05
# 숫자 키는 (하한, 상한), 나머지는 고를 수 있는 값.
# min_motion 상한 0.2 (832x480·16fps 에서 약 10px/프레임): 그보다 높이면 보통 움직임 클립까지 빠진다
ALLOWED = dict(
    max_train_steps=(250, 6000), learning_rate=(1e-5, 3e-4), rank=(4, 64), low_steps_share=(0.5, 1.0),
    shift_low=(1.0, 5.0), shift_high=(3.0, 12.0), shift=(1.0, 12.0), image_share=(0.0, 1.0),
    high_uses_images=(False, True), min_motion=(0.0, 0.2), min_frames=(17, 81), caption_dropout=(0.0, 0.2),
    preservation=("off", "dop"), target_scope=("all", "attn", "kvo"),
)
_up = lambda v, x, cap: max(v, min(x, cap))   # x 로 올리되 cap 까지 (이미 cap 위면 그대로, 내리지 않음)
_rank_up = lambda v: _up(v, v * 2, 32)
_low_up = lambda v: _up(v, v + 0.1, 0.9)
# 증상 → [(키, 지금 값 → 다음 값)], 앞에서부터 쓴다
PLAYBOOK = {
    "overfit": [("max_train_steps", lambda v: int(v * 0.6)), ("learning_rate", lambda v: v * 0.5),
                ("caption_dropout", lambda v: max(v, 0.05)), ("preservation", lambda v: "dop")],
    "underfit": [("max_train_steps", lambda v: int(v * 1.5)), ("rank", _rank_up),
                 ("learning_rate", lambda v: _up(v, v * 1.5, 2e-4))],
    "identity_weak": [("low_steps_share", _low_up), ("rank", _rank_up),
                      ("shift_low", lambda v: v - 1.0), ("shift", lambda v: v - 1.0)],   # 낮은 shift = 저노이즈(디테일) 쪽
    "motion_loss": [("image_share", lambda v: v - 0.2),     # 섞인 데이터에서 클립을 더 뽑기 (이미지뿐이면 노트북이 이 키를 빼고 넘김)
                    ("high_uses_images", lambda v: False),                   # 이미지는 저노이즈 전문가만
                    ("target_scope", lambda v: "kvo"),                       # V-Warper: K/V/O 만 → 움직임 유지
                    ("min_motion", lambda v: max(v, MIN_MOTION_ON)),         # 거의 정지한 클립 빼기
                    ("shift_high", lambda v: v + 2.0), ("shift", lambda v: v + 1.0)],
    "identity_drift": [("min_frames", lambda v: v + 16), ("low_steps_share", _low_up), ("rank", _rank_up)],
    "flicker": [("learning_rate", lambda v: v * 0.5), ("min_frames", lambda v: v + 16)],
    "color_shift": [("learning_rate", lambda v: v * 0.7), ("preservation", lambda v: "dop")],
    "style_drift": [("preservation", lambda v: "dop"), ("max_train_steps", lambda v: int(v * 0.8))],
    "motion_excess": [("learning_rate", lambda v: v * 0.7), ("low_steps_share", _low_up)],
}
PRIORITY = ["overfit", "underfit", "identity_weak", "motion_loss", "identity_drift", "flicker", "color_shift",
            "style_drift", "motion_excess"]
MAX_CHANGES = 2
_SEV = ("low", "medium", "high")


def _num(v):
    """숫자로 읽을 수 있고 유한하면 float, 아니면 None"""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _step(s):
    """단계 키 (JSON 에서 읽은 '250' 도 됨). 정수가 아니면 None"""
    v = _num(s)
    return int(v) if v is not None and v == int(v) else None


def _steps(step_metrics):
    """{단계: 지표 dict} (0 단계·정수가 아닌 키는 뺌, dict 가 아닌 값은 빈 dict)"""
    return {k: (dict(m) if isinstance(m, Mapping) else {}) for s, m in (step_metrics or {}).items()
            if (k := _step(s)) is not None and k > 0}


def _losses(val_loss):
    return {k: v for s, x in (val_loss or {}).items() if (k := _step(s)) is not None and (v := _num(x)) is not None}


def _thr(thr):
    """기준값: 빠진 키는 THR 값으로 채운다"""
    return {**THR, **(thr or {})}


def _sev(ratio):
    """기준 대비 몇 배 나쁜가 → 정도"""
    return "high" if ratio >= 2.0 else "medium" if ratio >= 1.25 else "low"


def _point(m, thr):
    """한 체크포인트의 점 지표 판정: {이름: (근거, 정도)}. identity·memo 는 과적합 근거로만 쓴다."""
    g = lambda k: _num(m.get(k))
    r, f, c, d, sl, mm, p = (g(k) for k in ("motion_ratio", "flicker_rel", "color_shift", "drift_rel", "id_slope", "memo", "id_pass"))
    out = {}
    if r is not None and r < thr["motion_low"]:
        out["motion_loss"] = (f"베이스 대비 움직임 {r:.2f} < {thr['motion_low']}", _sev(thr["motion_low"] / max(r, 1e-6)))
    if r is not None and r > thr["motion_high"]:
        out["motion_excess"] = (f"베이스 대비 움직임 {r:.2f} > {thr['motion_high']}", _sev(r / thr["motion_high"]))
    if f is not None and f > thr["flicker_rel"]:
        out["flicker"] = (f"베이스 대비 밝기·대비 깜빡임 {f:.2f}배 > {thr['flicker_rel']}", _sev(f / thr["flicker_rel"]))
    if c is not None and c > thr["color_shift"]:
        out["color_shift"] = (f"class-word 채도·평균색 변화 {c:.1%} > {thr['color_shift']:.0%}", _sev(c / thr["color_shift"]))
    if d is not None and d > thr["drift_rel"]:
        out["style_drift"] = (f"class-word 그림체 변화가 시드 변동의 {d:.2f}배 > {thr['drift_rel']}", _sev(d / thr["drift_rel"]))
    if sl is not None and sl > thr["id_slope"]:
        out["identity_drift"] = (f"클립 안 CCIP 차이가 프레임마다 {sl:.4f} 증가 > {thr['id_slope']}", _sev(sl / thr["id_slope"]))
    if mm is not None and mm > thr["memo"]:
        out["memo"] = (f"학습 프레임과 DINOv2 cos {mm:.3f} > {thr['memo']} (암기)", _sev((1 - thr["memo"]) / max(1 - mm, 1e-6)))
    if p is not None and p < thr["id_pass"]:
        out["identity"] = (f"CCIP 통과율 {p:.2f} < {thr['id_pass']}", _sev(thr["id_pass"] / max(p, 1e-6)))
    return out


_QUALITY = ("motion_ratio", "flicker_rel", "drift_rel")


def _quality_ok(m, thr):
    r, f, d = (_num(m.get(k)) for k in _QUALITY)
    lo, hi = thr["best_motion"]
    return (r is None or lo <= r <= hi) and (f is None or f <= thr["flicker_rel"]) and (d is None or d <= thr["drift_rel"])


def pick_best_step(step_metrics, val_loss, thr=THR, after=None):
    """추천 체크포인트 단계 (없으면 None). 후보는 step_metrics 의 단계 (0 제외), 없으면 val_loss 의 단계.

    정체성(id_pass)을 쟀으면: 정체성 통과 + 품질 (움직임 비율 best_motion 안, 깜빡임·그림체 기준 안) 인 가장 이른 단계 →
    정체성만 통과한 가장 이른 단계 → (통과 단계 없음) 정체성이 가장 높은 단계 (품질 통과 단계 우선).
    after: A14B 처럼 전문가를 차례로 학습할 때 첫 전문가의 마지막 단계 (전문가가 하나면 None). 정체성으로 고를 때 그 뒤
    단계 (두 전문가의 LoRA 를 함께 쓴 미리보기) 에 정체성·품질 통과 단계가 있으면 그 가운데 가장 이른 단계를 먼저 고른다.
    정체성은 첫 전문가(저노이즈)가 배우므로, 이것이 없으면 학습이 잘될수록 뒤 전문가 LoRA 가 추천에서 빠진다.
    정체성을 안 쟀으면: '가장 이른 단계' 는 덜 배운 단계일 뿐이라, 품질 통과 단계 가운데 검증 손실이 가장 낮은 단계
    (같으면 늦은 단계, 검증 손실이 없으면 가장 늦은 단계). 품질 통과 단계도 없으면 검증 손실 최저, 그것도 없으면 마지막 단계.
    A14B 처럼 전문가를 차례로 학습하면 노트북이 첫 전문가의 마지막 검증 손실을 뒤 단계에 이어 붙이므로, 같은 손실이면
    뒤 전문가까지 학습한 단계를 고른다.
    val_loss 는 홀드아웃 손실일 때만 넘긴다 (None 이면 품질 지표만으로 고르고, 못 고르면 마지막 단계).
    이 실행에서 한 번도 못 잰 지표는 보지 않는다. 다른 단계에서는 잰 지표가 빠진 단계는 품질 통과로 치지 않는다.
    """
    thr = _thr(thr)
    sm, vl = _steps(step_metrics), _losses(val_loss)
    steps = sorted(sm) or sorted(s for s in vl if s > 0)
    if not steps:
        return None

    def by_loss(ss):
        known = [s for s in ss if s in vl]
        return min(known, key=lambda s: (vl[s], -s)) if known else max(ss)

    keys = [k for k in _QUALITY if any(_num(m.get(k)) is not None for m in sm.values())]
    full = [s for s in steps if all(_num(sm.get(s, {}).get(k)) is not None for k in keys)] or steps
    q = [s for s in full if _quality_ok(sm.get(s, {}), thr)]
    ids = {s: _num(sm.get(s, {}).get("id_pass")) for s in steps}
    measured = [s for s in steps if ids[s] is not None]
    if not measured:
        return by_loss(q or full)
    ok = [s for s in measured if ids[s] >= thr["id_pass"]]
    if ok:
        both = [s for s in ok if s in q and after is not None and s > after]
        return (both or [s for s in ok if s in q] or ok)[0]
    pool = [s for s in q if ids[s] is not None] or measured
    return max(pool, key=lambda s: (ids[s], -s))


def diagnose(step_metrics, val_loss, thr=THR, after=None):
    """증상 목록 [{symptom, evidence, severity}] (PRIORITY 순서). severity: low / medium / high (기준 대비 1.25배·2배).
    val_loss=None (홀드아웃 손실 없음) 이면 손실 근거는 쓰지 않는다: 과적합은 암기·고른 단계 뒤 악화로만 본다.
    after 는 pick_best_step 과 같다 (점 지표를 추천 단계에서 보도록 같은 값을 넘긴다)."""
    thr = _thr(thr)
    sm, vl = _steps(step_metrics), _losses(val_loss)
    best = pick_best_step(step_metrics, val_loss, thr, after)
    out = []
    add = lambda name, ev, sev: out.append(dict(symptom=name, evidence=ev, severity=sev))

    # 정체성 추세: 어느 단계도 기준에 못 미쳤으면 마지막 구간이 오르는 중이면 underfit, 아니면 identity_weak
    ids = [(s, _num(sm[s].get("id_pass"))) for s in sorted(sm)]
    ids = [(s, p) for s, p in ids if p is not None]
    underfit = False
    if ids and max(p for _, p in ids) < thr["id_pass"]:
        s1, p1 = ids[-1]
        rise = p1 - ids[-2][1] if len(ids) > 1 else 0.0
        sev = _sev(thr["id_pass"] / max(p1, 1e-6))
        if rise > thr["id_rise"]:
            underfit = True
            add("underfit", f"CCIP 통과율 {p1:.2f} < {thr['id_pass']} 인데 마지막 구간 {rise:+.2f} 오르는 중 ({s1}단계)", sev)
        else:
            add("identity_weak", f"CCIP 통과율 최고 {max(p for _, p in ids):.2f} < {thr['id_pass']}, "
                                 f"마지막 구간 {rise:+.2f} ({s1}단계)", sev)

    pts = _point(sm.get(best, {}), thr) if best is not None else {}

    # 과적합: 검증 손실 상승 (정체성이 아직 오르는 중이면 보지 않음), 암기, 고른 단계 뒤 악화
    ev = []
    if len(vl) >= 2 and not underfit:
        s_min, s_last = min(vl, key=lambda s: (vl[s], s)), max(vl)
        # 이어 붙인 값 (A14B: 첫 전문가의 마지막 손실을 뒤 단계에 복사) 은 처음 잰 단계로 적는다
        s_last = min((s for s in vl if s > s_min and vl[s] == vl[s_last]), default=s_last)
        rise = (vl[s_last] - vl[s_min]) / max(abs(vl[s_min]), 1e-12)
        if s_min < s_last and rise > thr["val_rise"]:
            ev.append((f"검증 손실 {s_min}단계 {vl[s_min]:.4f} → {s_last}단계 {vl[s_last]:.4f} ({rise:+.1%})",
                       _sev(rise / thr["val_rise"])))
    if "memo" in pts:
        ev.append((f"{best}단계 {pts['memo'][0]}", pts["memo"][1]))
    last = max((s for s, m in sm.items() if any(_num(v) is not None for v in m.values())), default=None)   # 잰 값이 있는 마지막 단계
    if best is not None and last is not None and best < last:
        later = _point(sm[last], thr)
        worse = [k for k in later if k not in pts]
        if worse:
            ev.append((f"{best}단계 뒤 나빠짐 ({last}단계: " + "; ".join(later[k][0] for k in worse) + ")", "medium"))
    if ev:
        add("overfit", " / ".join(e for e, _ in ev), max((s for _, s in ev), key=_SEV.index))

    for name in ("identity_drift", "motion_loss", "motion_excess", "flicker", "color_shift", "style_drift"):
        if name in pts:
            add(name, f"{best}단계: {pts[name][0]}", pts[name][1])
    return sorted(out, key=lambda s: PRIORITY.index(s["symptom"]))


def _isnum(v):
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def _clamp(key, v, allowed):
    """허용 범위로 자르고 숫자는 6자리로 다듬는다. 고를 수 있는 값이 아니면 None."""
    lim = allowed.get(key)
    if lim is None:
        return v
    if len(lim) == 2 and all(_isnum(x) for x in lim):
        v = min(max(v, lim[0]), lim[1])
        return int(v) if isinstance(v, numbers.Integral) else float(f"{float(v):.6g}")
    return v if v in lim else None


def propose(fixed, symptoms, allowed=ALLOWED, max_changes=MAX_CHANGES):
    """증상 → (fixed_overrides, changes). changes: [{key, old, new, symptom, evidence}].

    계열에서 쓰지 않는 키 (전문가가 하나인 계열의 low_steps_share·shift_low·shift_high·high_uses_images, I2V 의
    image_share 등) 는 fixed 에서 빼고 넘긴다. 없는 키·None·종류가 맞지 않는 값은 건너뛰므로 바꿀 기회를 헛쓰지 않는다.
    fixed·symptoms 가 None 이면 빈 것으로 본다.
    """
    found = {}
    for s in symptoms or ():
        if isinstance(s, Mapping) and s.get("symptom") in PLAYBOOK:
            found.setdefault(s["symptom"], s)
    changes, cur = [], dict(fixed or {})
    for name in (p for p in PRIORITY if p in found):
        for key, fn in PLAYBOOK.get(name, ()):
            if len(changes) >= max_changes:
                break
            if cur.get(key) is None or any(c["key"] == key for c in changes):
                continue
            old = cur[key]
            try:
                raw = fn(old)
                new = _clamp(key, raw, allowed)
            except (TypeError, ValueError):                # 종류가 다른 값 (예: 문자열 단계 수) → 건너뜀
                continue
            if new is None or new == old or (_isnum(old) and _isnum(raw) and (raw - old) * (new - old) <= 0):
                continue                                  # 그대로이거나, 범위 때문에 거꾸로 바뀌는 값 (예: 4단계 → 하한 250)
            cur[key] = new
            changes.append(dict(key=key, old=old, new=new, symptom=name, evidence=found[name].get("evidence", "")))
    overrides = {c["key"]: c["new"] for c in changes}
    if "rank" in overrides:
        overrides["alpha"] = overrides["rank"]
    return overrides, changes


def next_run_name(name):
    """niki_wan_v1 → niki_wan_v2, niki_v01 → niki_v02 (자릿수 유지), niki_v9 → niki_v10, niki → niki_v2"""
    m = re.fullmatch(r"(.*[_-][vV])(\d+)", name)
    return f"{m.group(1)}{int(m.group(2)) + 1:0{len(m.group(2))}d}" if m else f"{name}_v2"


def _abs(p):
    """지금 작업 폴더 기준 절대 경로 (빈 값은 그대로)"""
    return str(Path(p).expanduser().resolve()) if p else p


def next_run_plan(user, overrides, changes, best_step, base_overrides=None):
    """run_many.py 목록 형식 {"defaults": {}, "runs": [실행 1개]}. USER 값을 옮기고 실행 이름만 판을 올린다.
    USER 에 TARGET_SCOPE·EXPERT·CLASS_WORD·CAPTION_FIXED_WORDS 가 없거나 비어 있으면 빈 문자열 (run_many 는 건너뜀).

    경로는 지금 작업 폴더(노트북이 이 경로들을 읽은 곳) 기준 절대 경로로 바꾼다. run_many 는 상대 경로를 목록 파일
    (= 이번 실행 폴더) 기준으로 읽으므로, 그대로 두면 데이터셋을 못 찾거나 다음 결과가 이번 실행 폴더 안에 생긴다.
    DATASET_DIR·OUTPUT_DIR (비어 있으면 작업 폴더)·VALIDATION_FILE 은 늘, MODEL_PATH 는 로컬에 있을 때만
    (Hugging Face 저장소 이름은 그대로).
    fixed_overrides = base_overrides (이번 실행에 적용한 덮어쓰기 dict, 노트북의 OVERRIDES) 위에 overrides (이번 제안) 를
    덮은 것. 앞선 보완이 이어지고, 이번 실행과 다른 값은 이번 제안뿐이다. rank 가 있으면 alpha = rank.
    overrides·changes·base_overrides 가 None 이면 빈 것으로 본다.
    """
    opt = lambda k, d="": user.get(k) or d
    mp = user["MODEL_PATH"]
    fixed = {**dict(base_overrides or {}), **dict(overrides or {})}
    if "rank" in fixed:
        fixed["alpha"] = fixed["rank"]
    run = dict(family=user["MODEL_FAMILY"], model_path=_abs(mp) if mp and Path(mp).expanduser().exists() else mp,
               dataset_dir=_abs(user["DATASET_DIR"]), output_dir=_abs(opt("OUTPUT_DIR", ".")), trigger=user["TRIGGER"],
               run_name=next_run_name(user["RUN_NAME"]), target_scope=opt("TARGET_SCOPE"), expert=opt("EXPERT"),
               class_word=opt("CLASS_WORD"), validation_file=_abs(opt("VALIDATION_FILE")),
               caption_fixed_words=opt("CAPTION_FIXED_WORDS"), fixed_overrides=fixed,
               why=[{k: c.get(k) for k in ("key", "old", "new", "symptom", "evidence")} for c in changes or ()],
               previous=dict(run_name=user["RUN_NAME"], best_step=None if best_step is None else int(best_step)))
    return {"defaults": {}, "runs": [run]}
