"""remedy 시험: python test_remedy.py

시나리오마다 diagnose → pick_best_step → propose → next_run_plan 을 돌려 제안이 말이 되는지 본다.
움직임 손실 (섞인 데이터·이미지뿐·클립뿐·전문가 하나·I2V), 과적합, 문제 없음, 지표 없음·일부만·NaN·문자열 단계,
홀드아웃 손실 없음 (val_loss=None), 과소학습·정체성 약함, 범위 자르기, 우선순위·최대 2개, 실행 이름 (자릿수 유지),
None 입력, 다음 실행 계획 (절대 경로·캡션 고정 낱말·v1→v2→v3→v4 덮어쓰기 이어 가기, run_many 가 읽는지).
마지막에 REMEDY OK
"""
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remedy as RM

# 영상 노트북 COMMON_FIXED 의 보완 대상 값 + 상관없는 값 몇 개. propose 에는 그 계열이 쓰는 키만 넘긴다:
# A14B 는 shift (전문가 하나용) 를 빼고, 전문가가 하나인 계열은 A14B 전용 키를 빼고, I2V 는 image_share 를 뺀다
COMMON = dict(seed=42, rank=16, alpha=16, target_scope="all", learning_rate=1e-4, max_train_steps=2000, low_steps_share=0.75,
              high_uses_images=False, shift_low=1.0, shift_high=5.0, shift=3.0, image_share=0.5, caption_dropout=0.0,
              min_motion=0.0, min_frames=17, preservation="off", save_every=250)
FIXED = {k: v for k, v in COMMON.items() if k != "shift"}
SINGLE = {k: v for k, v in COMMON.items() if k not in ("low_steps_share", "shift_low", "shift_high", "high_uses_images")}
I2V = {k: v for k, v in FIXED.items() if k != "image_share"}
USER = dict(MODEL_FAMILY="wan22_t2v", MODEL_PATH="/srv/models/diffusers/Wan2.2-T2V-A14B-Diffusers", DATASET_DIR="dataset/niki",
            OUTPUT_DIR="outputs_video", TRIGGER="nkq7", RUN_NAME="niki_wan_v1", TARGET_SCOPE="all", EXPERT="auto",
            CLASS_WORD="girl", VALIDATION_FILE="", FIXED_OVERRIDES="", CAPTION_FIXED_WORDS="black bob, wine-red cape")
ab = lambda p: str(Path(p).resolve())          # next_run_plan 은 지금 작업 폴더 기준 절대 경로로 쓴다
STEPS = (250, 500, 750, 1000)
OK = dict(motion_ratio=0.95, flicker_rel=1.05, color_shift=0.03, drift_rel=1.1, id_pass=0.95, id_slope=0.0003, memo=0.85,
          consistency=0.97)


def metrics(**cols):
    """단계별 지표: 기본은 OK, cols 의 키는 단계마다 값 목록 (STEPS 순서)"""
    return {s: {**OK, **{k: v[i] for k, v in cols.items()}} for i, s in enumerate(STEPS)}


def run(name, sm, vl, fixed=FIXED):
    sym = RM.diagnose(sm, vl)
    best = RM.pick_best_step(sm, vl)
    ov, ch = RM.propose(fixed, sym)
    plan = RM.next_run_plan(USER, ov, ch, best)
    json.dumps(plan, ensure_ascii=False)                       # default=str 없이도 JSON 이 되어야 한다
    for s in sym:
        assert set(s) == {"symptom", "evidence", "severity"} and s["symptom"] in RM.PRIORITY and s["severity"] in RM._SEV, s
    assert len(ch) <= RM.MAX_CHANGES and all(c["key"] in fixed for c in ch)
    print(f"[{name}] best {best} | 증상 " + (", ".join(f"{s['symptom']}({s['severity']})" for s in sym) or "없음")
          + " | 바꿈 " + (", ".join(f"{c['key']} {c['old']}→{c['new']}" for c in ch) or "없음"))
    for s in sym:
        print("     ", s["symptom"], ":", s["evidence"])
    return sym, best, ov, ch, plan


names = lambda sym: [s["symptom"] for s in sym]
VAL_DOWN = {0: 0.300, 250: 0.280, 500: 0.270, 750: 0.265, 1000: 0.260}

# 1. 움직임 손실: 정체성은 500 단계부터 통과, 움직임은 모든 단계에서 0.6 아래
sm = metrics(id_pass=[0.85, 0.93, 0.95, 0.96], motion_ratio=[0.55, 0.50, 0.45, 0.42])
sym, best, ov, ch, plan = run("움직임 손실", sm, VAL_DOWN)
assert names(sym) == ["motion_loss"] and best == 500, (sym, best)
assert ov == {"image_share": 0.3, "target_scope": "kvo"}, ov             # high_uses_images 는 이미 False → 건너뜀
r = plan["runs"][0]
assert plan["defaults"] == {} and r["run_name"] == "niki_wan_v2" and r["fixed_overrides"] == ov
assert r["previous"] == {"run_name": "niki_wan_v1", "best_step": 500} and [w["symptom"] for w in r["why"]] == ["motion_loss"] * 2
# 경로는 절대 경로 (run_many 는 상대 경로를 목록 파일 = 이번 실행 폴더 기준으로 읽는다). 없는 MODEL_PATH 는 그대로
assert {k: r[k] for k in ("family", "model_path", "dataset_dir", "output_dir", "trigger", "target_scope", "expert", "class_word",
                          "validation_file", "caption_fixed_words")} == dict(
    family="wan22_t2v", model_path=USER["MODEL_PATH"], dataset_dir=ab("dataset/niki"), output_dir=ab("outputs_video"),
    trigger="nkq7", target_scope="all", expert="auto", class_word="girl", validation_file="",
    caption_fixed_words="black bob, wine-red cape"), r
assert all(isinstance(r[k], str) for k in r if k not in ("fixed_overrides", "why", "previous")), r   # run_many: 문자열 값
# 고노이즈도 이미지를 쓰던 실행 → 이미지를 저노이즈 전용으로
_, _, ov, _, _ = run("움직임 손실, 고노이즈가 이미지 사용", sm, VAL_DOWN, {**FIXED, "high_uses_images": True})
assert ov == {"image_share": 0.3, "high_uses_images": False}, ov
# min_motion 은 video_io subject_motion 단위 (초당 화면 긴 변의 비율): 0.05 = video_io.STATIC_MOTION
assert RM.MIN_MOTION_ON == 0.05 and RM.ALLOWED["min_motion"] == (0.0, 0.2)
# 이미지·클립이 섞였는데 image_share 1.0 (클립을 하나도 뽑지 않음): 이미지 비율부터 줄인다
_, _, ov, _, _ = run("움직임 손실, 섞인 데이터·image_share 1.0", sm, VAL_DOWN, {**FIXED, "image_share": 1.0})
assert ov == {"image_share": 0.8, "target_scope": "kvo"}, ov
# 이미지뿐인 A14B: 노트북 6-3 처럼 클립 전용 값과 학습하지 않은 고노이즈 값을 빼고 넘긴다 → K/V/O 만
IMG_ONLY = {k: v for k, v in FIXED.items() if k not in ("image_share", "min_motion", "min_frames", "window_mode",
                                                        "windows_per_segment", "ti2v_i2v_share", "shift_high",
                                                        "low_steps_share")}
_, _, ov, _, _ = run("움직임 손실, 이미지뿐 A14B", sm, VAL_DOWN, IMG_ONLY)
assert ov == {"target_scope": "kvo"}, ov
# 클립뿐 + 이미 kvo: 정지 클립 빼기, 고노이즈 shift 올리기
_, _, ov, _, _ = run("움직임 손실, 클립뿐·kvo", sm, VAL_DOWN, {**FIXED, "image_share": 0.0, "target_scope": "kvo"})
assert ov == {"min_motion": 0.05, "shift_high": 7.0}, ov
# 이미 min_motion 이 기준 위면 그대로 두고 다음 항목으로
_, _, ov, _, _ = run("움직임 손실, min_motion 이미 0.1", sm, VAL_DOWN, {**FIXED, "image_share": 0.0, "target_scope": "kvo", "min_motion": 0.1})
assert ov == {"shift_high": 7.0}, ov
# 전문가가 하나인 계열 (A14B 전용 키를 빼고 넘김)
_, _, ov, _, _ = run("움직임 손실, 전문가 하나", sm, VAL_DOWN, SINGLE)
assert ov == {"image_share": 0.3, "target_scope": "kvo"}, ov
# I2V (클립만 학습, image_share 를 빼고 넘김): 쓰지 않는 키에 바꿀 기회를 쓰지 않는다
_, _, ov, _, _ = run("움직임 손실, I2V", sm, VAL_DOWN, I2V)
assert ov == {"target_scope": "kvo", "min_motion": 0.05}, ov

# 2. 과적합: 500 단계가 가장 좋고 그 뒤 검증 손실이 오르고 암기가 늘어남
sm = metrics(id_pass=[0.80, 0.92, 0.95, 0.96], motion_ratio=[0.95, 0.90, 0.75, 0.62], memo=[0.90, 0.92, 0.95, 0.97])
sym, best, ov, ch, plan = run("과적합", sm, {0: 0.300, 250: 0.270, 500: 0.260, 750: 0.268, 1000: 0.280})
assert names(sym) == ["overfit"] and best == 500 and sym[0]["severity"] == "high", sym
assert "검증 손실" in sym[0]["evidence"] and "1000단계" in sym[0]["evidence"] and "암기" in sym[0]["evidence"], sym
assert ov == {"max_train_steps": 1200, "learning_rate": 5e-05} and isinstance(ov["max_train_steps"], int), ov
assert plan["runs"][0]["previous"]["best_step"] == 500

# 3. 문제 없음: 바꿀 것 없음, 정체성·품질을 모두 통과한 가장 이른 단계
sym, best, ov, ch, plan = run("문제 없음", metrics(), VAL_DOWN)
assert sym == [] and best == 250 and ov == {} and ch == [] and plan["runs"][0]["fixed_overrides"] == {}
assert plan["runs"][0]["why"] == [] and plan["runs"][0]["run_name"] == "niki_wan_v2"

# 4. 지표 없음·일부만
none = {s: {k: None for k in OK} for s in STEPS}
sym, best, ov, ch, _ = run("지표 모두 None, 검증 손실 없음", none, None)
assert sym == [] and best == 1000 and ov == {}, (sym, best)
sym, best, ov, ch, plan = run("아무것도 없음", {}, None)
assert sym == [] and best is None and ov == {} and plan["runs"][0]["previous"]["best_step"] is None
sym, best, _, _, _ = run("검증 손실만", {}, {0: 0.30, 250: 0.25, 500: 0.26})
assert best == 250 and names(sym) == ["overfit"], (best, sym)        # 0 단계(베이스)는 고르지 않는다
# 750 단계는 움직임을 못 쟀다 (NaN): 잰 단계보다 좋아 보이면 안 된다
sym, best, ov, _, _ = run("움직임만 (문자열 단계, NaN)", {"250": {"motion_ratio": 0.5}, "500": {"motion_ratio": 0.4, "flicker_rel": None},
                                                       "750": {"motion_ratio": float("nan")}}, {0: float("nan"), 250: 0.2, 500: 0.19, 750: 0.185})
assert best == 500 and names(sym) == ["motion_loss"] and "image_share" in ov, (best, sym, ov)
# 정체성을 안 쟀으면 '가장 이른 통과 단계' 가 아니라 품질 통과 단계 가운데 검증 손실 최저
assert RM.pick_best_step({s: {"motion_ratio": 1.0} for s in STEPS}, VAL_DOWN) == 1000
assert RM.pick_best_step({s: {"motion_ratio": 1.0} for s in STEPS}, None) == 1000
# 같은 검증 손실이면 늦은 단계 (A14B: 저노이즈 마지막 손실을 고노이즈 단계에 이어 붙이면 두 전문가를 다 학습한 짝을 고른다)
assert RM.pick_best_step({s: {"motion_ratio": 1.0} for s in STEPS}, {0: 0.3, 250: 0.28, 500: 0.27, 750: 0.27, 1000: 0.27}) == 1000
assert RM.pick_best_step({s: {"motion_ratio": 1.0} for s in STEPS}, {0: 0.3, 250: 0.26, 500: 0.27, 750: 0.27, 1000: 0.27}) == 250
# 과적합 근거는 이어 붙인 손실을 처음 잰 단계로 적는다 (A14B: 저노이즈 1500 의 값이 1750·2000 에 복사됨)
_ev = RM.diagnose({s: {"motion_ratio": 1.0} for s in (500, 1000, 1500, 1750, 2000)},
                  {0: 0.30, 500: 0.29, 1000: 0.30, 1500: 0.32, 1750: 0.32, 2000: 0.32})
assert [s["symptom"] for s in _ev] == ["overfit"] and "1500단계" in _ev[0]["evidence"] and "2000단계" not in _ev[0]["evidence"], _ev
# A14B (after = 첫 전문가의 마지막 단계): 정체성을 쟀어도, 두 전문가를 다 학습한 뒤 단계에 정체성·품질 통과 단계가 있으면 그 가운데
# 가장 이른 단계 (저노이즈가 먼저 통과해도 고노이즈 LoRA 를 빼지 않는다)
A14 = {s: dict(OK) for s in (250, 500, 750, 1000, 1250, 1500, 1750, 2000)}
assert RM.pick_best_step(A14, VAL_DOWN) == 250 and RM.pick_best_step(A14, VAL_DOWN, after=1500) == 1750
# 뒤 단계가 품질에 떨어지면 (고노이즈 LoRA 가 움직임을 망침) 앞 구간의 가장 이른 통과 단계 그대로
assert RM.pick_best_step({**A14, 1750: {**OK, "motion_ratio": 2.0}, 2000: {**OK, "motion_ratio": 2.0}}, None, after=1500) == 250
assert RM.pick_best_step({**A14, 1750: {**OK, "id_pass": 0.5}}, None, after=1500) == 2000
# diagnose 도 같은 단계에서 점 지표를 본다 (1750 의 암기는 after 를 줄 때만 고른 단계의 근거)
assert RM.diagnose({**A14, 1750: {**OK, "memo": 0.97}}, None) == []
assert names(RM.diagnose({**A14, 1750: {**OK, "memo": 0.97}}, None, after=1500)) == ["overfit"]

# 4b. 홀드아웃 검증 손실이 없음 (학습 샘플 손실이면 노트북이 val_loss=None 을 넘긴다): 손실 근거는 쓰지 않는다
RISE = {0: 0.300, 250: 0.250, 500: 0.260, 750: 0.270, 1000: 0.280}
assert names(RM.diagnose(metrics(), RISE)) == ["overfit"] and RM.diagnose(metrics(), None) == []
# 과적합 2 와 같은 지표: 손실 없이도 암기 (고른 단계 뒤 악화) 로 과적합, 정도는 손실 근거가 없어 medium
sm = metrics(id_pass=[0.80, 0.92, 0.95, 0.96], motion_ratio=[0.95, 0.90, 0.75, 0.62], memo=[0.90, 0.92, 0.95, 0.97])
sym, best, ov, _, _ = run("과적합, 홀드아웃 손실 없음 (암기)", sm, None)
assert names(sym) == ["overfit"] and best == 500 and sym[0]["severity"] == "medium", sym
assert "검증 손실" not in sym[0]["evidence"] and "암기" in sym[0]["evidence"], sym
assert ov == {"max_train_steps": 1200, "learning_rate": 5e-05}, ov
# 그림체 드리프트가 고른 단계 뒤에 기준을 넘음 → 과적합 (손실 없이)
sym, best, _, _, _ = run("과적합, 홀드아웃 손실 없음 (드리프트)", metrics(drift_rel=[1.0, 1.2, 2.5, 3.0]), None)
assert names(sym) == ["overfit"] and best == 250 and "그림체" in sym[0]["evidence"], (sym, best)
# 추천 단계: 품질 통과 단계 가운데 가장 늦은 단계, 품질 지표가 없으면 마지막 단계 (손실 최저를 보지 않음)
q = {250: {"motion_ratio": 1.0}, 500: {"motion_ratio": 1.1}, 750: {"motion_ratio": 1.5}, 1000: {"motion_ratio": 1.8}}
assert RM.pick_best_step(q, None) == 500 and RM.pick_best_step({s: {} for s in STEPS}, None) == 1000
assert RM.pick_best_step({s: {"memo": 0.5} for s in STEPS}, None) == 1000

# 5. 과소학습 (통과율이 아직 오르는 중) / 정체성 약함 (멈춤)
sym, best, ov, _, _ = run("과소학습", metrics(id_pass=[0.50, 0.65, 0.78, 0.85]), VAL_DOWN)
assert names(sym) == ["underfit"] and ov == {"max_train_steps": 3000, "rank": 32, "alpha": 32} and best == 1000, (sym, ov, best)
sym, _, ov, _, _ = run("정체성 약함", metrics(id_pass=[0.70, 0.78, 0.80, 0.80]), VAL_DOWN)
assert names(sym) == ["identity_weak"] and ov == {"low_steps_share": 0.85, "rank": 32, "alpha": 32}, (sym, ov)
_, _, ov, _, _ = run("정체성 약함, 전문가 하나", metrics(id_pass=[0.70, 0.78, 0.80, 0.80]), VAL_DOWN, {**SINGLE, "rank": 32, "alpha": 32})
assert ov == {"shift": 2.0}, ov                                      # rank 는 이미 32 (상한) → shift 만
# 정체성이 오르는 중이면 검증 손실 상승만으로 과적합이라 하지 않는다
sym, _, _, _, _ = run("과소학습 + 검증 손실 상승", metrics(id_pass=[0.50, 0.65, 0.78, 0.85]), {0: 0.3, 250: 0.25, 500: 0.26, 1000: 0.28})
assert names(sym) == ["underfit"], sym

# 6. 범위 자르기·그대로인 값 건너뛰기
_, _, ov, _, _ = run("과적합, 하한 근처", sm := metrics(memo=[0.96] * 4), None, {**FIXED, "max_train_steps": 300, "learning_rate": 1e-5})
assert ov == {"max_train_steps": 250, "caption_dropout": 0.05}, ov  # 1e-5 × 0.5 → 하한 1e-5 (그대로) → 건너뜀
ov, ch = RM.propose(MappingProxyType(FIXED), [dict(symptom="flicker", evidence="x", severity="low")])
assert ov == {"learning_rate": 5e-05, "min_frames": 33} and ch[0]["old"] == 1e-4, ov
ov, _ = RM.propose({**FIXED, "learning_rate": 1e-5, "min_frames": 81}, [dict(symptom="flicker", evidence="x", severity="low")])
assert ov == {}, ov
ov, _ = RM.propose({"learning_rate": 1e-4}, [dict(symptom="color_shift", evidence="x", severity="low")])
assert ov == {"learning_rate": 7e-05}, ov                            # preservation 키가 없으면 건너뜀
# 범위 때문에 방향이 거꾸로 되는 변경은 하지 않는다 (노트북 시험 모드 4단계·학습률이 이미 상한 위)
ov, _ = RM.propose({**FIXED, "max_train_steps": 4}, [dict(symptom="overfit", evidence="x", severity="low")])
assert ov == {"learning_rate": 5e-05, "caption_dropout": 0.05}, ov
ov, _ = RM.propose({**FIXED, "learning_rate": 5e-4, "rank": 64, "alpha": 64}, [dict(symptom="underfit", evidence="x", severity="low")])
assert ov == {"max_train_steps": 3000}, ov                           # rank·학습률은 이미 올림 상한(32, 2e-4) 위 → 그대로
ov, _ = RM.propose({**FIXED, "low_steps_share": 0.95}, [dict(symptom="identity_weak", evidence="x", severity="low")])
assert ov == {"rank": 32, "alpha": 32}, ov                           # 0.95 → 0.9 로 내리지 않는다
# None·빠진 키·이상한 증상: 죽지 않고 건너뛴다
assert RM.propose(FIXED, None) == ({}, []) and RM.propose(None, [dict(symptom="flicker")]) == ({}, [])
ov, ch = RM.propose(FIXED, [dict(evidence="x"), "flicker", None, dict(symptom="prompt_following"), dict(symptom="flicker")])
assert ov == {"learning_rate": 5e-05, "min_frames": 33} and ch[0]["evidence"] == "", (ov, ch)
ov, _ = RM.propose({"max_train_steps": "2000", "learning_rate": 1e-4}, [dict(symptom="overfit", evidence="x", severity="low")])
assert ov == {"learning_rate": 5e-05}, ov                            # 문자열 단계 수는 건너뜀, 없는 키도 건너뜀
assert RM.diagnose(metrics(motion_ratio=[0.5] * 4), None, thr={"motion_low": 0.4}) == []   # 일부만 준 기준값은 THR 로 채움
assert RM.pick_best_step(metrics(), None, thr={}) == 250
sym = RM.diagnose({250: 0.5, 500: MappingProxyType({"motion_ratio": 0.3}), "x": {"motion_ratio": 0.1}}, None)   # dict 아닌 값·이상한 키
assert names(sym) == ["motion_loss"] and "500단계" in sym[0]["evidence"], sym

# 7. 우선순위와 최대 2개
sm = metrics(flicker_rel=[2.0] * 4, color_shift=[0.2] * 4, drift_rel=[3.0] * 4, motion_ratio=[1.8] * 4)
sym, best, ov, ch, _ = run("여러 증상", sm, VAL_DOWN)
assert names(sym) == ["flicker", "color_shift", "style_drift", "motion_excess"], sym
assert [c["symptom"] for c in ch] == ["flicker", "flicker"] and ov == {"learning_rate": 5e-05, "min_frames": 33}, ch
sym, _, ov, _, _ = run("클립 안 정체성 흔들림", metrics(id_slope=[0.004] * 4), VAL_DOWN)
assert names(sym) == ["identity_drift"] and sym[0]["severity"] == "high" and ov == {"min_frames": 33, "low_steps_share": 0.85}, (sym, ov)
sym, _, ov, _, _ = run("움직임 과다", metrics(motion_ratio=[1.2, 1.5, 1.8, 2.0]), VAL_DOWN)
assert "overfit" in names(sym) and ov["max_train_steps"] == 1200, (sym, ov)   # 250 단계 뒤 움직임이 과해짐 → 단계 줄이기
sym, _, ov, _, _ = run("움직임 과다 (처음부터)", metrics(motion_ratio=[1.8, 1.9, 2.0, 2.0]), VAL_DOWN)
assert names(sym) == ["motion_excess"] and ov == {"learning_rate": 7e-05, "low_steps_share": 0.85}, (sym, ov)

# 8. 실행 이름 (자릿수 유지)
for a, b in [("niki_wan_v1", "niki_wan_v2"), ("niki", "niki_v2"), ("niki_v9", "niki_v10"), ("nkgreen_wan22-v3", "nkgreen_wan22-v4"),
             ("niki_wan22", "niki_wan22_v2"), ("v1", "v1_v2"), ("niki_v01", "niki_v02"), ("niki_v09", "niki_v10"),
             ("niki_v99", "niki_v100"), ("a-V007", "a-V008")]:
    assert RM.next_run_name(a) == b, (a, RM.next_run_name(a), b)

# 9. 다음 실행 계획: None 입력, 절대 경로, run_many 가 이번 실행 폴더의 next_run.json 을 읽는지
r = RM.next_run_plan(USER, None, None, None)["runs"][0]
assert r["fixed_overrides"] == {} and r["why"] == [] and r["previous"] == {"run_name": "niki_wan_v1", "best_step": None}, r
old_user = {k: v for k, v in USER.items() if k not in ("TARGET_SCOPE", "CAPTION_FIXED_WORDS", "EXPERT")}   # 예전 USER: 빈 값
r = RM.next_run_plan(old_user, {}, [], None)["runs"][0]
assert r["target_scope"] == r["caption_fixed_words"] == r["expert"] == "", r
r = RM.next_run_plan(USER, {"rank": 8, "alpha": 8}, [], 250, base_overrides=MappingProxyType({"rank": 32, "alpha": 32, "learning_rate": 5e-5}))
assert r["runs"][0]["fixed_overrides"] == {"rank": 8, "alpha": 8, "learning_rate": 5e-5}, r   # 이번 제안이 앞선 값을 덮는다
assert RM.next_run_plan(USER, {}, [], None, base_overrides={"rank": 32})["runs"][0]["fixed_overrides"] == {"rank": 32, "alpha": 32}
tmp = Path(tempfile.mkdtemp(prefix="rm_plan_"))
cwd = os.getcwd()
try:
    os.chdir(tmp)                                   # = 노트북 작업 폴더
    (tmp / "models" / "wan").mkdir(parents=True)
    (tmp / "models" / "wan" / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp / "dataset" / "niki").mkdir(parents=True)
    (tmp / "val.json").write_text(json.dumps({"prompts": ["she waves"]}), encoding="utf-8")
    u = {**USER, "MODEL_PATH": "models/wan", "VALIDATION_FILE": "val.json", "TARGET_SCOPE": "kvo"}
    plan = RM.next_run_plan(u, {"learning_rate": 5e-05}, [], 250, base_overrides={"image_share": 0.3})
    r = plan["runs"][0]
    assert r["model_path"] == str((tmp / "models" / "wan").resolve()) and r["dataset_dir"] == str((tmp / "dataset" / "niki").resolve())
    assert r["validation_file"] == str((tmp / "val.json").resolve()) and r["output_dir"] == str((tmp / "outputs_video").resolve()), r
    assert RM.next_run_plan({**u, "OUTPUT_DIR": ""}, {}, [], 250)["runs"][0]["output_dir"] == str(tmp.resolve())   # 빈 값 = 작업 폴더
    hf = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    assert RM.next_run_plan({**u, "MODEL_PATH": hf}, {}, [], 250)["runs"][0]["model_path"] == hf      # 저장소 이름은 그대로
    try:
        sys.path.insert(0, str(HERE.parents[1]))    # notebooks/run_many.py
        import run_many
    except ImportError as e:
        run_many = None
        print("SKIP run_many 확인 (불러오지 못함):", e)
    if run_many is not None:
        rd = tmp / u["OUTPUT_DIR"] / u["RUN_NAME"]  # 노트북 6-3: RUN_DIR / "next_run.json"
        rd.mkdir(parents=True)
        (rd / "next_run.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        env_names = {f"LORA_{k}" for k in ("MODEL_FAMILY", "MODEL_PATH", "DATASET_DIR", "OUTPUT_DIR", "TRIGGER", "RUN_NAME",
                                           "VALIDATION_FILE", "TARGET_SCOPE", "EXPERT", "CLASS_WORD", "FIXED_OVERRIDES",
                                           "CAPTION_FIXED_WORDS")}
        os.chdir(cwd)                               # run_many 는 다른 폴더에서 실행해도 같아야 한다
        video = "video" in inspect.signature(run_many.load_plan).parameters    # 영상 노트북의 target_scope (all·attn·kvo)
        if not video:
            r["target_scope"] = "all"               # 예전 run_many 는 kvo 를 모른다
            (rd / "next_run.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        runs = run_many.load_plan(rd / "next_run.json", {"wan22_t2v"}, False, env_names, **({"video": True} if video else {}))
        env = runs[0]["env"]
        assert env["LORA_TARGET_SCOPE"] == r["target_scope"], env
        assert runs[0]["out_dir"] == Path(r["output_dir"]) and runs[0]["name"] == "niki_wan_v2", runs[0]
        assert env["LORA_DATASET_DIR"] == r["dataset_dir"] and env["LORA_MODEL_PATH"] == r["model_path"], env
        assert env["LORA_VALIDATION_FILE"] == r["validation_file"] and env["LORA_CAPTION_FIXED_WORDS"] == USER["CAPTION_FIXED_WORDS"]
        assert json.loads(env["LORA_FIXED_OVERRIDES"]) == {"image_share": 0.3, "learning_rate": 5e-05}, env
        print("run_many.load_plan: 이번 실행 폴더의 next_run.json → 같은 데이터셋·출력 폴더, 범위", env["LORA_TARGET_SCOPE"],
              "덮어쓰기", env["LORA_FIXED_OVERRIDES"])
finally:
    os.chdir(cwd)
    shutil.rmtree(tmp, ignore_errors=True)

# 10. v1 → v2 → v3 → v4: 앞선 보완을 잃지 않고, 실행마다 바뀌는 값은 이번 제안 (2개까지, rank 면 alpha 도) 뿐
def next_run(fixed_now, user_now, symptom):
    """노트북 6-3 처럼: 이번 FIXED 로 제안 → 이번 FIXED_OVERRIDES 위에 얹은 계획 → 다음 실행의 USER·FIXED"""
    ov, ch = RM.propose(fixed_now, [dict(symptom=symptom, evidence=symptom, severity="medium")])
    rr = RM.next_run_plan(user_now, ov, ch, 500, base_overrides=json.loads(user_now["FIXED_OVERRIDES"] or "{}"))["runs"][0]
    nxt_fixed = {**FIXED, **rr["fixed_overrides"]}  # 노트북: FIXED = COMMON_FIXED + 계열 값 + OVERRIDES
    changed = {k for k in nxt_fixed if nxt_fixed[k] != fixed_now.get(k)}
    assert changed == set(ov) and len(ch) <= RM.MAX_CHANGES, (changed, ov)
    print(f"  {user_now['RUN_NAME']} → {rr['run_name']} ({symptom}): 바꿈 {ov} | 누적 {rr['fixed_overrides']}")
    return {**user_now, "RUN_NAME": rr["run_name"], "FIXED_OVERRIDES": json.dumps(rr["fixed_overrides"])}, nxt_fixed, rr


print("[v1 → v4 이어 가기]")
u2, f2, r2 = next_run(dict(FIXED), USER, "motion_loss")
assert r2["run_name"] == "niki_wan_v2" and r2["fixed_overrides"] == {"image_share": 0.3, "target_scope": "kvo"}, r2
u3, f3, r3 = next_run(f2, u2, "underfit")
assert r3["run_name"] == "niki_wan_v3" and r3["fixed_overrides"] == {"image_share": 0.3, "target_scope": "kvo", "max_train_steps": 3000,
                                                                     "rank": 32, "alpha": 32}, r3
u4, f4, r4 = next_run(f3, u3, "motion_loss")        # 같은 증상이 남으면 같은 첫 제안을 되풀이하지 않고 다음 항목으로
assert r4["run_name"] == "niki_wan_v4" and r4["fixed_overrides"] == {"image_share": 0.1, "target_scope": "kvo", "max_train_steps": 3000,
                                                                     "rank": 32, "alpha": 32, "min_motion": 0.05}, r4
assert f4["target_scope"] == "kvo" and f4["rank"] == f4["alpha"] == 32
print("REMEDY OK")
