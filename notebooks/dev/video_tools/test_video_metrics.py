"""video_metrics 시험: python test_video_metrics.py [같은_캐릭터_폴더 다른_캐릭터_폴더 ...]

- 합성 클립 (부드러운·느린·빠른 움직임, 정지, 정지+잡음, 정지+거친 입자 잡음, 카메라 팬 (잡음 유무), 밝기 깜빡임,
  위치 떨림, 채도 낮춤) 이 지표에서 갈리는지. 입자 잡음만 있는 정지 클립은 움직임이 아니다 (움직임 손실을 가리면 안 됨)
- clip_metrics 의 키와 베이스 대비 값, 노트북 시험 모드 크기, 빈·None 입력 (죽지 않고 None·NaN), 실제 크기 (33x480x832) 시간
- 재검증에서 찾은 결함의 회귀 시험 (832x480, 17프레임): 화면 일부 (띠 10·40%, 가운데 타원 25%) 에만 입자가 낀 정지
  클립은 움직임 < 0.2px 이고 3px 베이스 대비 < 0.6, 실제 선화 (niki_start.webp, 없으면 SKIP) 팬 1~8px/프레임이 속도에
  단조이고 LoRA 4px vs 베이스 2px 의 motion_ratio > 1.3 (뒤바꾸면 < 0.75), 대비·노출 깜빡임은 움직임이 아니라
  깜빡임 (flicker_rel > 1.5). 깨끗한 클립의 대비 성분은 FLICKER_FLOOR 아래 (밝기 깜빡임만 보던 판정이 그대로)
- 두 번째 재검증 결함의 회귀 시험: 어둡거나 흐린 (x0.15·대비 x0.12) 장면·선화 팬이 같은 속도 원본과 같은 값 (클립
  대비 이득), 어두운 정지 클립 + 입자는 움직임이 아님, 망점 (스크린톤, 화면 주기 4·5px) 선화 팬이 부풀지 않고 속도에
  단조 (앤티에일리어스 축소), 검은 프레임·검게 페이드는 움직임이 아님 (대비가 크게 바뀐 전이는 재지 않음), 장면 전환
  한 번은 대비 깜빡임이 아님 (대비 항은 큰 고역 표본 2개를 뺌), 1~5프레임 클립의 contrast_flicker
- 세 번째 재검증 결함의 회귀 시험: 프레임마다 번갈아 오는 강한 대비·노출 깜빡임 (±30·40%) 이 낀 팬은 움직임이 0 이
  아니고 (motion_ratio > 0.6) 깜빡임으로 걸림, 흐린 (낮은 주파수) 시간 잡음이 낀 어두운 정지 클립은 움직임이 아님
- 네 번째 재검증 결함의 회귀 시험: 무늬가 화면 전체에 있는 (1/f) 어둡거나 흐린 팬이 같은 속도 원본과 같은 값 (잡음 추정에서
  움직임을 맞춰 뺌), 3프레임마다 검은 프레임·뒤 절반 검은 클립은 남은 전이만 재서 정지는 0·팬은 원본과 같음, 한 장 건너
  검은 프레임은 같은 위상 두 프레임으로 재서 정지는 0·팬은 원본과 같음 (예전 판은 7 배)
- DINOv2: 못 불러오면 SKIP. letterbox 가 프레임 전체를 남기는지, 시간 일관성, 그림체 드리프트
- CCIP: onnxruntime 이 없으면 (ImportError) 또는 깨져서 불러오다 다른 예외 (OSError 등) 가 나면 load() 가 이유를 출력하고
  None 인지 (있어도 없는 척해서 확인). 있으면 실제 모델로 같은 그림은 통과하는지,
  폴더를 주면 첫 폴더 그림 2장을 참조로 같은 캐릭터와 다른 캐릭터가 갈리는지
- 모델은 HF_HOME 아래에 내려받는다. 마지막에 VIDEO METRICS OK
"""
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_metrics as VM

KEYS = {"motion", "motion_ratio", "luma_flicker", "contrast_flicker", "flicker_rel", "flicker_ratio", "sat", "color_shift",
        "consistency", "drift", "id_pass", "id_mean", "id_slope"}


def scene(x, W=416, H=240, bright=0, palette=0, desat=0.0, noise=0.0, seed=0):
    """그라데이션+무늬 배경에 노란 사각형 피사체 (x: 왼쪽 위치 px)"""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 255 / (W - 1), yy * 255 / (H - 1), np.full((H, W), 80.0 + 120 * palette)], -1)
    img += 20 * (np.sin(xx / 4.0) * np.sin(yy / 5.0))[..., None]
    s, y0, x0 = H // 4, H // 2 - H // 8, int(round(x))
    img[y0:y0 + s, x0:x0 + s] = (255, 220, 40)
    img[y0 + s // 3:y0 + 2 * s // 3, x0 + s // 3:x0 + 2 * s // 3] = (40, 60, 200)
    if desat:
        img = img * (1 - desat) + img.mean(-1, keepdims=True) * desat
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img + bright, 0, 255).round().astype(np.uint8)


def pan(v, noise=0.0, W=416, H=240):
    """카메라 팬: 넓게 그린 장면 (피사체 고정) 에서 W 폭 창을 프레임마다 v px (정수) 씩 옮긴다. noise: 프레임마다 다른 잡음"""
    big = scene(300, W=W + v * (T - 1), H=H).astype(np.float64)
    out = [big[:, v * t:v * t + W] + (np.random.default_rng(100 + t).normal(0, noise, (H, W, 3)) if noise else 0) for t in range(T)]
    return np.clip(np.stack(out), 0, 255).round().astype(np.uint8)


T = 33
C = {
    "smooth": np.stack([scene(40 + 3 * t) for t in range(T)]),                       # 3px/프레임
    "slow": np.stack([scene(40 + t) for t in range(T)]),                             # 1px/프레임
    "fast": np.stack([scene(40 + 6 * t) for t in range(T)]),                         # 6px/프레임
    "static": np.stack([scene(40)] * T),
    "static_noise": np.stack([scene(40, noise=3, seed=t) for t in range(T)]),        # 프레임마다 다른 잡음
    "static_grain": np.stack([scene(40, noise=12, seed=t) for t in range(T)]),       # 거친 입자 잡음 (σ 12)
    "pan": pan(3),                                                                   # 화면 전체 3px/프레임
    "pan_grain": pan(3, noise=12),
    "flicker": np.stack([scene(40 + 3 * t, bright=12 if t % 2 else -12) for t in range(T)]),
    "jitter": np.stack([scene(40 + 3 * t + (4 if t % 2 else -4)) for t in range(T)]),
    "desat": np.stack([scene(40 + 3 * t, desat=0.4) for t in range(T)]),
    "switch": np.stack([scene(40 + 3 * t, palette=t > T // 2) for t in range(T)]),   # 중간에 배경색이 바뀜
}
mo = {k: VM.block_motion_p95(v) for k, v in C.items()}
fl = {k: VM.flicker(v) for k, v in C.items()}
mad = {k: VM.motion_amount(v) for k, v in C.items()}
for k in C:
    print(f"{k:12s} motion {mo[k]:6.3f}px  mad {mad[k]:.4f}  flicker_ratio {fl[k]['flicker_ratio']:.3f}  "
          f"luma_flicker {fl[k]['luma_flicker']:7.3f}  sat {VM.color_stats(C[k])['sat']:.1f}")

# 움직임: 밝기 깜빡임·잡음은 움직임이 아니다 (프레임 차 평균은 깜빡임에 속는다), 속도 비율이 맞다
assert mo["static"] == 0 and mo["static_noise"] < 0.2, (mo["static"], mo["static_noise"])
# 거친 입자 잡음만 있는 정지 클립도 움직임 0 근처 (고정 문턱 noise=0 이면 1px 넘게 나와 움직임 손실을 가렸다)
Yg = VM._luma(C["static_grain"][:3], VM.MOTION_SIDE)
old_gate = VM._top5(VM._block_disp(Yg[0], Yg[1], noise=0)) * 416 / Yg.shape[2]
assert mo["static_grain"] < 0.2 and old_gate > 1.0, (mo["static_grain"], old_gate)
assert VM.clip_metrics(C["static_grain"], base=C["smooth"])["motion_ratio"] < 0.6, "입자 잡음이 낀 정지 LoRA 클립 = 움직임 손실"
# 팬은 그대로: 깨끗한 팬은 잡음 문턱이 없을 때와 블록 변위까지 같고, 입자 잡음이 낀 팬도 깨끗한 팬과 비슷하다
Yp = VM._luma(C["pan"][:4], VM.MOTION_SIDE)
assert all(np.array_equal(VM._block_disp(Yp[i], Yp[i + 1]), VM._block_disp(Yp[i], Yp[i + 1], noise=0)) for i in range(3))
assert abs(mo["pan"] / 3 - 1) < 0.25 and 0.8 < mo["pan_grain"] / mo["pan"] < 1.2, (mo["pan"], mo["pan_grain"])
assert abs(mo["flicker"] / mo["smooth"] - 1) < 0.15, (mo["flicker"], mo["smooth"])
assert mad["flicker"] > 10 * mad["smooth"], "motion_amount 는 깜빡임에 속는 지표 (참고용)"
assert 1.6 < mo["fast"] / mo["smooth"] < 2.4 and 0.2 < mo["slow"] / mo["smooth"] < 0.5, (mo["fast"], mo["slow"], mo["smooth"])
assert mo["jitter"] > 2 * mo["smooth"], (mo["jitter"], mo["smooth"])
# 깜빡임: 부드러운 움직임 < 1, 깜빡임·떨림 ≈ 2. 밝기 깜빡임만 luma_flicker 가 크다
assert fl["smooth"]["flicker_ratio"] < 1.0 and fl["slow"]["flicker_ratio"] < 1.0
assert fl["flicker"]["flicker_ratio"] > 1.8 and fl["jitter"]["flicker_ratio"] > 1.8
assert fl["static"] == dict(flicker_ratio=0.0, luma_flicker=0.0)
assert fl["flicker"]["luma_flicker"] > 15 and max(fl[k]["luma_flicker"] for k in ("smooth", "jitter", "static_noise")) < 1.0

# clip_metrics: 키는 늘 같고, 못 잰 값은 None
m0 = VM.clip_metrics(C["smooth"])
assert set(m0) == KEYS and all(m0[k] is None for k in ("motion_ratio", "flicker_rel", "color_shift", "consistency", "drift",
                                                        "id_pass", "id_mean", "id_slope")), m0
same = VM.clip_metrics(C["smooth"], base=C["smooth"])
assert abs(same["motion_ratio"] - 1) < 1e-9 and same["color_shift"] == 0 and same["flicker_rel"] <= 1.0, same
cm = {k: VM.clip_metrics(C[k], base=C["smooth"]) for k in ("slow", "fast", "flicker", "desat")}
assert cm["slow"]["motion_ratio"] < 0.6 and cm["fast"]["motion_ratio"] > 1.6, (cm["slow"], cm["fast"])
assert cm["flicker"]["flicker_rel"] > 1.5 and abs(cm["flicker"]["motion_ratio"] - 1) < 0.15, cm["flicker"]
assert cm["desat"]["color_shift"] > 0.10 and cm["slow"]["color_shift"] < 0.02, (cm["desat"], cm["slow"])
assert VM.clip_metrics(C["smooth"], base=C["static"])["motion_ratio"] is None, "정지한 베이스와는 비율을 내지 않는다"
print("base 대비:", {k: {a: round(b, 3) for a, b in v.items() if a in ("motion_ratio", "flicker_rel", "color_shift")} for k, v in cm.items()})

# 노트북 시험 모드처럼 아주 작은 클립, 1프레임
rng = np.random.default_rng(0)
tiny = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
mt = VM.clip_metrics(tiny, base=rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8))
assert all(v is None or np.isfinite(v) for v in mt.values()), mt
assert VM.block_motion_p95(tiny[:1]) == 0.0 and VM.flicker(tiny[:2])["flicker_ratio"] == 0.0

# 빈·None 입력 (생성 실패 등): 죽지 않고, 못 잰 값은 None (color_stats 는 NaN)
empty = np.zeros((0, 64, 64, 3), np.uint8)
NONE_METRICS = {k: (0.0 if k == "motion" else None) for k in KEYS}
for e in (empty, None, []):
    assert VM.motion_amount(e) == 0.0 and VM.block_motion_p95(e) == 0.0, e
    assert VM.flicker(e) == dict(flicker_ratio=0.0, luma_flicker=0.0), e
    assert all(np.isnan(v) for v in VM.color_stats(e).values()), VM.color_stats(e)
    assert VM.clip_metrics(e) == NONE_METRICS and VM.clip_metrics(e, base=C["smooth"]) == NONE_METRICS, VM.clip_metrics(e)
    mb_ = VM.clip_metrics(C["smooth"], base=e)                  # 빈 베이스 = 베이스 없음
    assert set(mb_) == KEYS and all(mb_[k] is None for k in ("motion_ratio", "flicker_rel", "color_shift", "drift")), mb_
    assert VM.subject_consistency(e) == 1.0 and np.isnan(VM.style_drift(e, np.ones((3, 4)))), e
assert VM.clip_metrics(C["smooth"][:1], base=C["smooth"][:1])["motion_ratio"] is None   # 1프레임: 움직임을 못 잼

# 실제 크기 시간 (CPU)
big = np.stack([np.asarray(Image.fromarray(f).resize((832, 480), Image.Resampling.BILINEAR)) for f in C["smooth"]])
t0 = time.time()
mb = VM.clip_metrics(big, base=big[::-1].copy())
print(f"33x480x832 clip_metrics (base 포함, 모델 없이): {time.time() - t0:.2f}s, motion {mb['motion']:.2f}px")


# ---------------------------------------------------------------- 재검증 결함 회귀 (미리보기 크기 832x480, 17프레임)
def part_grain(frames, sigma, frac=None, blob=None, seed=0):
    """프레임마다 새 입자 (채널마다 따로) 를 화면 일부에만: 왼쪽 frac 폭 띠, 또는 화면 비율과 같은 가운데 타원 (넓이 비율 blob)"""
    h, w = frames.shape[1:3]
    if blob:
        k = np.sqrt(4 * blob / np.pi)
        yy, xx = np.mgrid[0:h, 0:w] + 0.5
        m = np.hypot((xx - w / 2) / (k * w / 2), (yy - h / 2) / (k * h / 2)) <= 1
    else:
        m = np.zeros((h, w), bool)
        m[:, :int(round(w * frac))] = True
    out = frames.astype(np.float64) + np.random.default_rng(seed).normal(0, sigma, frames.shape) * m[..., None]
    return np.clip(out, 0, 255).round().astype(np.uint8)


def pump(frames, contrast=0.0, exposure=0.0):
    """프레임마다 번갈아 대비 (128 기준 x(1±contrast)) 와 노출 (x(1±exposure)) 을 바꾼다"""
    out = frames.astype(np.float64)
    for t in range(len(out)):
        sg = 1 if t % 2 else -1
        out[t] = ((out[t] - 128) * (1 + sg * contrast) + 128) * (1 + sg * exposure)
    return np.clip(out, 0, 255).round().astype(np.uint8)


T2, W2, H2 = 17, 832, 480
still = np.stack([scene(200, W=W2, H=H2)] * T2)
base3 = np.stack([scene(100 + 3 * t, W=W2, H=H2) for t in range(T2)])          # 깨끗한 3px/프레임 베이스
# 1) 화면 일부에만 낀 입자: 잡음 바닥을 블록마다 (다듬은 뒤 남은 SAD) 재므로 움직임이 아니다
#    (예전 판은 화면 전체 중앙값 바닥이라 띠 40% σ12 를 1.67px, 3px 베이스 대비 0.65 로 읽어 움직임 손실을 가렸다)
for name, clip in (("띠 10% σ12", part_grain(still, 12, frac=0.10)), ("띠 40% σ12", part_grain(still, 12, frac=0.40)),
                   ("띠 40% σ16", part_grain(still, 16, frac=0.40)), ("가운데 타원 25% σ12", part_grain(still, 12, blob=0.25))):
    mg, rg = VM.block_motion_p95(clip), VM.clip_metrics(clip, base=base3)["motion_ratio"]
    print(f"부분 입자 정지 {name}: motion {mg:.3f}px, 3px 베이스 대비 motion_ratio {rg:.3f}")
    assert mg < 0.2 and rg is not None and rg < 0.6, (name, mg, rg)

# 2) 실제 선화 팬: 1px 아래 다듬기·개구 문제 처리로 속도에 단조 (예전 판은 선을 따라 가는 먼 정수 변위로 맞춰
#    1/2/3/4/5/6/8 px 을 5.48/10.24/7.01/5.29/9.01/11.76/9.50 으로 읽고, 4px vs 2px 가 0.52 = 거짓 움직임 손실)
START = Path(__file__).resolve().parents[4] / "niki_video" / "niki_start.webp"     # 니키 전신 928x1664, 밋밋한 배경
if not START.is_file():
    print(f"SKIP 실제 선화 팬 시험: 그림이 없습니다 ({START})")
else:
    src = Image.open(START).convert("RGB")
    sc = (W2 + 8 * (T2 - 1)) / src.width               # 8px/프레임 x 16전이가 들어가는 배율 (모든 속도에 같은 배율)
    band = np.asarray(src.resize((round(src.width * sc), round(src.height * sc)), Image.Resampling.LANCZOS))
    band = band[band.shape[0] // 2 - H2 // 2:][:H2]     # 가운데 띠 (망토·반바지의 선)
    line_pan = lambda v: np.stack([band[:, v * t:v * t + W2] for t in range(T2)])   # noqa: E731
    sp = (1, 2, 3, 4, 5, 6, 8)
    lv = [VM.block_motion_p95(line_pan(v)) for v in sp]
    print("실제 선화 팬 1/2/3/4/5/6/8 px/프레임 →", "/".join(f"{v:.2f}" for v in lv))
    assert all(b > a for a, b in zip(lv, lv[1:])) and all(0.6 < v / s < 1.6 for v, s in zip(lv, sp)), lv
    r42 = VM.clip_metrics(line_pan(4), base=line_pan(2))["motion_ratio"]
    r24 = VM.clip_metrics(line_pan(2), base=line_pan(4))["motion_ratio"]
    print(f"실제 선화 팬 LoRA 4px vs 베이스 2px motion_ratio {r42:.2f}, 뒤바꾸면 {r24:.2f}")
    assert r42 > 1.3 and r24 < 0.75, (r42, r24)

# 3) 대비·노출 깜빡임은 움직임이 아니라 깜빡임: 대비 성분 (contrast_flicker) 을 베이스 대비로 flicker_rel 에 넣는다
#    (예전 판은 정지 클립의 대비 ±10% 를 0.88px 움직임으로 읽고 flicker_rel 0.69 라 깜빡임으로도 못 잡았다)
for name, kw in (("대비 ±5%", dict(contrast=0.05)), ("대비 ±10%", dict(contrast=0.10)), ("대비 ±15%", dict(contrast=0.15)),
                 ("노출 ±5%", dict(exposure=0.05))):
    ms, mm = VM.clip_metrics(pump(still, **kw), base=still), VM.clip_metrics(pump(base3, **kw), base=base3)
    print(f"{name}: 정지 motion {ms['motion']:.3f}px flicker_rel {ms['flicker_rel']:.2f} | 3px 움직임 + 깜빡임 "
          f"motion_ratio {mm['motion_ratio']:.3f} flicker_rel {mm['flicker_rel']:.2f}")
    assert ms["motion"] < 0.2 and ms["flicker_rel"] > 1.5, (name, ms)
    assert abs(mm["motion_ratio"] - 1) < 0.15 and mm["flicker_rel"] > 1.5, (name, mm)
cfl = {k: VM.contrast_flicker(C[k]) for k in ("smooth", "slow", "fast", "static", "static_noise", "static_grain", "pan",
                                               "pan_grain", "jitter", "desat")}
print("contrast_flicker (깨끗한 클립):", {k: round(v, 3) for k, v in cfl.items()})
assert max(cfl.values()) < VM.FLICKER_FLOOR and VM.contrast_flicker(pump(C["static"], contrast=0.05)) > 2, cfl
assert VM.contrast_flicker(None) == 0.0 and VM.contrast_flicker(C["smooth"][:2]) == 0.0
for k, v in cm.items():     # 밝기 성분은 예전 flicker_rel 그대로 (대비 성분은 더 큰 쪽으로만 들어간다)
    assert v["flicker_rel"] >= fl[k]["luma_flicker"] / max(fl["smooth"]["luma_flicker"], VM.FLICKER_FLOOR) - 1e-12, (k, v)
    assert k == "flicker" or v["flicker_rel"] < 1.0, (k, v)
    assert v["contrast_flicker"] == VM.contrast_flicker(C[k]), (k, v)


# ---------------------------------------------------------------- 두 번째 재검증 결함 회귀 (832x480, 17프레임)
def dark(frames, k):
    """밝기 x k (어두운 밤 장면)"""
    return np.clip(np.rint(frames.astype(np.float32) * k), 0, 255).astype(np.uint8)


def lowcon(frames, k):
    """첫 프레임 평균색 둘레로 대비 x k (안개·흐린 대비. 평균 밝기는 그대로)"""
    m = frames[0].reshape(-1, 3).mean(0)
    return np.clip(np.rint(m + (frames.astype(np.float32) - m) * k), 0, 255).astype(np.uint8)


def tone(img, p):
    """망점 스크린톤: 화면 주기 p px 의 날카로운 점을 곱한다 (점 자리 x0.5)"""
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    s = (np.cos(2 * np.pi * xx / p) * np.cos(2 * np.pi * yy / p) > 0.3).astype(np.float32)
    return np.clip(np.rint(img.astype(np.float32) * (1 - 0.5 * s[..., None])), 0, 255).astype(np.uint8)


# 4) 어둡거나 흐린 클립도 같은 속도면 같은 값: 분석 휘도에 클립 대비 이득 (CONTRAST_REF / 블록 대비, 1~CONTRAST_GMAX) 을
#    곱해 문턱 tau·변위 벌점 lam 이 대비에 맞는다 (예전 판은 절대 휘도라 이 3px 장면 x0.15 를 0.67 배, 대비 x0.12 를
#    0.45 배로 읽었다 = 거짓 움직임 손실). 어두운 정지 클립의 입자는 이득을 받아도 움직임이 아니다
m3 = VM.block_motion_p95(base3)
rd = {name: VM.block_motion_p95(f(base3)) / m3 for name, f in (("x0.15", lambda x: dark(x, 0.15)),
                                                                ("대비 x0.12", lambda x: lowcon(x, 0.12)))}
dg = VM.block_motion_p95(part_grain(dark(still, 0.15), 4, frac=1.0))
print("어두운·흐린 3px 장면 / 원본:", {k: round(v, 3) for k, v in rd.items()}, f"| 어두운 정지 + 입자 σ4: {dg:.3f}px")
assert all(0.8 < v < 1.25 for v in rd.values()) and dg < 0.2, (rd, dg)
if START.is_file():   # 선화 x0.15·대비 x0.12 (예전 판: 1px/프레임을 0 으로 읽음)
    rl = {name: [VM.block_motion_p95(f(line_pan(v))) / lv[sp.index(v)] for v in (1, 2, 8)]
          for name, f in (("x0.15", lambda x: dark(x, 0.15)), ("대비 x0.12", lambda x: lowcon(x, 0.12)))}
    print("어두운·흐린 선화 팬 1/2/8 px / 원본:", {k: [round(x, 3) for x in v] for k, v in rl.items()})
    assert all(0.8 < x < 1.25 for v in rl.values() for x in v), rl

# 5) 망점 (스크린톤) 선화 팬: 움직임 분석 축소가 앤티에일리어스 (MOTION_RESAMPLE) 라 무아레가 움직임으로 읽히지 않는다
#    (예전 BOX 축소는 주기 4px 망점의 1/2/3/4 px/프레임을 3.85/2.47/4.97/4.56 으로 읽었다 = 속도에 단조가 아님)
if START.is_file():
    for p in (4, 5):
        tb = tone(band, p)
        tv = [VM.block_motion_p95(np.stack([tb[:, v * t:v * t + W2] for t in range(T2)])) for v in (1, 2, 3, 4)]
        print(f"망점 {p}px 선화 팬 1/2/3/4 px/프레임 →", "/".join(f"{x:.2f}" for x in tv),
              "(망점 없이", "/".join(f"{x:.2f}" for x in lv[:4]) + ")")
        assert all(b > a for a, b in zip(tv, tv[1:])) and all(0.8 < x / y < 1.25 for x, y in zip(tv, lv[:4])), (p, tv, lv[:4])

# 6) 검은·흰 프레임과 페이드 끝: 화면 대비가 한 전이에서 GAIN_RANGE 밖으로 바뀌면 그 전이는 재지 않는다 (예전 판은 정지
#    클립 가운데 검은 프레임 2장을 1.66px 움직임으로 읽었다). 움직이는 클립은 나머지 전이로 잰다
blk = [0.0 if t in (7, 8) else 1.0 for t in range(T2)]
fade = [1 - t / (T2 - 1) for t in range(T2)]
mk = lambda fr, k: np.stack([np.clip(np.rint(f.astype(np.float32) * kk), 0, 255).astype(np.uint8)   # noqa: E731
                             for f, kk in zip(fr, k)])
mb0, mf0 = VM.block_motion_p95(mk(still, blk)), VM.block_motion_p95(mk(still, fade))
mb3 = VM.block_motion_p95(mk(base3, blk)) / m3
print(f"정지 + 검은 프레임 {mb0:.3f}px, 정지가 검게 페이드 {mf0:.3f}px, 3px + 검은 프레임 / 깨끗한 3px {mb3:.3f}")
assert mb0 < 0.2 and mf0 < 0.2 and abs(mb3 - 1) < 0.1, (mb0, mf0, mb3)

# 7) 장면 전환 한 번은 대비 깜빡임이 아니다: contrast_flicker 는 제곱이 가장 큰 고역 표본 2개 (전환 = 프레임 표준편차의
#    계단 하나 = ±계단/2 두 개) 를 빼고 잰다. 평균 밝기가 같고 대비만 다른 장면으로 넘어가는 클립을 예전 판은
#    flicker_rel 4.5 로 읽었다 (밝기 항만이면 0.04). 대비가 4프레임 주기로 오가는 깜빡임은 그대로 걸린다
cut = np.concatenate([base3[:T2 // 2], lowcon(base3, 0.5)[T2 // 2:]])
mc = VM.clip_metrics(cut, base=base3)
luma_only = mc["luma_flicker"] / max(VM.flicker(base3)["luma_flicker"], VM.FLICKER_FLOOR)
contrast_term = mc["contrast_flicker"] / max(VM.contrast_flicker(base3), VM.FLICKER_FLOOR)
s_cut = (VM._luma(cut, 32) / 255.0).std(axis=(1, 2)) * 255
untrimmed = VM._rms_hp(s_cut) / max(VM._rms_hp((VM._luma(base3, 32) / 255.0).std(axis=(1, 2)) * 255), VM.FLICKER_FLOOR)
p4 = np.stack([((f.astype(np.float64) - 128) * (1 + 0.10 * np.sin(np.pi * t / 2)) + 128).clip(0, 255).round().astype(np.uint8)
               for t, f in enumerate(still)])
f4 = VM.clip_metrics(p4, base=still)["flicker_rel"]
print(f"장면 전환 (대비만 x0.5): flicker_rel {mc['flicker_rel']:.3f} (밝기 항 {luma_only:.3f}, 빼지 않은 대비 항 {untrimmed:.2f}),"
      f" 대비 ±10% 4프레임 주기 flicker_rel {f4:.2f}")
assert untrimmed > 1.5 and mc["flicker_rel"] < 1.0 and contrast_term <= max(luma_only, 1.0), (mc, luma_only, contrast_term)
assert f4 > 1.5, f4


# 8) 짧은 클립의 contrast_flicker: 2프레임 이하 0, 고역 표본 1·2개 (3·4프레임) 는 빼지 않고, 3개부터 큰 2개를 뺀다
def two_level(stds):
    """왼쪽 절반 128-a, 오른쪽 128+a 인 64x64 프레임들 (32px 휘도의 표준편차 = a)"""
    out = np.full((len(stds), 64, 64, 3), 128.0)
    for t, a in enumerate(stds):
        out[t, :, :32] -= a
        out[t, :, 32:] += a
    return out.round().astype(np.uint8)


for seq, want in (((10,), 0.0), ((10, 30), 0.0), ((10, 30, 10), 20.0), ((10, 30, 10, 12), float(np.sqrt((20 ** 2 + 11 ** 2) / 2))),
                  ((10, 30, 10, 12, 10), 2.0)):
    got = VM.contrast_flicker(two_level(seq))
    mcs = VM.clip_metrics(two_level(seq), base=two_level(seq))
    assert abs(got - want) < 1e-3 and mcs["contrast_flicker"] == got and np.isfinite(mcs["flicker_rel"]), (seq, got, want, mcs)
print("짧은 클립 contrast_flicker 1~5프레임 확인")

# 9) 프레임마다 번갈아 오는 강한 대비·노출 깜빡임: 전이가 모두 GAIN_RANGE 밖이어도 움직임을 잰다 (빼면 움직임 0 → 거짓
#    motion_loss 가 깜빡임 보완을 밀어냈다. 이 3px 팬의 대비 ±30% 를 0.0 으로 읽었다). 이제 같은 위상 두 프레임으로 재서 1.2 근처
for kw in (dict(contrast=0.3), dict(exposure=0.4)):
    mp = VM.clip_metrics(pump(base3, **kw), base=base3)
    print(f"3px 팬 + 번갈아 {kw}: motion_ratio {mp['motion_ratio']:.3f}, flicker_rel {mp['flicker_rel']:.1f}")
    assert mp["motion_ratio"] is not None and mp["motion_ratio"] > 0.6 and mp["flicker_rel"] > 1.5, (kw, mp)


# 10) 흐린 (낮은 주파수) 시간 잡음이 낀 어두운 정지 클립: 대비 이득이 잡음도 키우므로 문턱 tau·변위 벌점 lam 을 그 전이의
#     잡음만큼 올린다 (이 규칙 전에는 x0.15 정지 + 잡음 (격자 8·16px, σ3) 을 9·23px 로 읽었다)
def smooth_noise(frames, amp, cell, seed=0):
    """프레임마다 새 매끈한 휘도 잡음: cell px 격자의 정규 잡음 (σ amp) 을 BICUBIC 으로 키워 모든 채널에 더한다"""
    h, w = frames.shape[1:3]
    rng = np.random.default_rng(seed)
    out = frames.astype(np.float32)
    for t in range(len(out)):
        g = rng.normal(0, amp, (h // cell + 1, w // cell + 1)).astype(np.float32)
        out[t] += np.asarray(Image.fromarray(g, "F").resize((w, h), Image.Resampling.BICUBIC))[..., None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


sn = {cell: VM.block_motion_p95(smooth_noise(dark(still, 0.15), 3.0, cell)) for cell in (8, 16)}
print("어두운 정지 + 흐린 시간 잡음 σ3 (격자 8/16px):", {k: round(v, 3) for k, v in sn.items()})
assert all(v < 0.3 for v in sn.values()), sn


# 11) 무늬가 화면 전체에 있는 (1/f) 어둡거나 흐린 팬: 밋밋한 하위 25% 블록도 무늬라 움직이면 바뀐다. 잡음 n 을 화면 전체 정수
#     변위 + 블록 LK 로 맞춘 뒤에도 재서 작은 쪽을 쓴다 (안 맞추면 n 이 속도만큼 올라 1~3px/프레임을 0 으로 읽었다)
def pink_pan(v, amp=38.0, seed=3):
    """1/f 무늬 세계 (밋밋한 곳이 없는 그림) 를 v px/프레임 팬"""
    wb = W2 + v * (T2 - 1)
    rng = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(H2)[:, None], np.fft.rfftfreq(wb)[None, :]
    f = np.sqrt(fy ** 2 + fx ** 2)
    f[0, 0] = 1.0
    img = np.fft.irfft2((rng.normal(size=(H2, wb // 2 + 1)) + 1j * rng.normal(size=(H2, wb // 2 + 1))) / f, s=(H2, wb))
    img = (img - img.mean()) / img.std() * amp + 128
    big = np.clip(np.stack([img, img * 0.9 + 10, img * 0.8 + 20], -1), 0, 255)
    return np.stack([big[:, v * t:v * t + W2] for t in range(T2)]).round().astype(np.uint8)


pr = {}
for v in (1, 2):
    pb = pink_pan(v)
    mb = VM.block_motion_p95(pb)
    pr[v] = [round(VM.block_motion_p95(f(pb)) / mb, 3) for f in (lambda x: dark(x, 0.15), lambda x: lowcon(x, 0.12))]
print("무늬 빽빽한 (1/f) 팬 1/2 px, 어둡게 x0.15·대비 x0.12 / 원본:", pr)
assert all(0.85 < x < 1.15 for v in pr.values() for x in v), pr

# 12) 3프레임마다 검은 프레임·뒤 절반 검은 클립: 전이가 하나라도 남으면 남은 전이만 잰다 (전이 절반 규칙이던 판은 그림→검정
#     전이까지 재서 정지를 5px, 3px 팬을 4 배로 읽었다)
every3 = [0.0 if t % 3 == 2 else 1.0 for t in range(T2)]
last9 = [0.0 if t >= 8 else 1.0 for t in range(T2)]
for name, k in (("3프레임마다 검정", every3), ("뒤 9장 검정", last9)):
    ms, mr = VM.block_motion_p95(mk(still, k)), VM.block_motion_p95(mk(base3, k)) / m3
    print(f"{name}: 정지 {ms:.3f}px, 3px 팬 / 깨끗한 3px {mr:.3f}")
    assert ms < 0.2 and abs(mr - 1) < 0.15, (name, ms, mr)

# 13) 한 장 건너 검은 프레임 (전이가 모두 밖): 한 칸 건너 같은 위상의 두 프레임을 견줘 2 로 나눈다 (모두 재던 판은 3px 팬을
#     7 배, 정지를 11px 로 읽었다)
alt = [0.0 if t % 2 else 1.0 for t in range(T2)]
ma0, mar = VM.block_motion_p95(mk(still, alt)), VM.block_motion_p95(mk(base3, alt)) / m3
print(f"한 장 건너 검정: 정지 {ma0:.3f}px, 3px 팬 / 깨끗한 3px {mar:.3f}")
assert ma0 < 0.2 and 0.8 < mar < 1.25, (ma0, mar)

# ---------------------------------------------------------------- DINOv2
saved = sys.modules.get("transformers")
sys.modules["transformers"] = None          # 설치 안 된 것처럼
try:
    assert VM.Dino.load("cpu") is None
finally:
    if saved is None:
        del sys.modules["transformers"]
    else:
        sys.modules["transformers"] = saved
lb = VM.Dino.letterbox(np.concatenate([np.full((480, 16, 3), (255, 0, 0), np.uint8), np.full((480, 800, 3), 128, np.uint8),
                                       np.full((480, 16, 3), (0, 0, 255), np.uint8)], 1))
assert lb.shape == (224, 224, 3) and tuple(lb[0, 112]) == VM.Dino.PAD, "위아래는 평균색으로 채운다"
assert lb[112, 1, 0] > 200 and lb[112, 222, 2] > 200, "letterbox 는 양옆을 자르지 않는다"
dino = VM.Dino.load("cpu")
if dino is None:
    print("SKIP DINOv2 (불러오지 못함)")
else:
    t0 = time.time()
    E = {k: dino.embed(C[k]) for k in ("smooth", "static", "slow", "switch", "desat")}
    print(f"DINOv2 {sum(len(e) for e in E.values())}프레임 {time.time() - t0:.1f}s")
    assert E["smooth"].shape == (T, 384) and np.allclose(np.linalg.norm(E["smooth"], axis=1), 1, atol=1e-4)
    sc = {k: VM.subject_consistency(e) for k, e in E.items()}
    dr = {k: VM.style_drift(E[k], E["smooth"]) for k in ("smooth", "slow", "switch", "desat")}
    print("subject_consistency", {k: round(v, 4) for k, v in sc.items()}, "| style_drift vs smooth", {k: round(v, 4) for k, v in dr.items()})
    assert sc["static"] > 0.999 and sc["switch"] < sc["smooth"], sc
    assert dr["smooth"] < 1e-5 and dr["switch"] > dr["slow"] and dr["desat"] > dr["slow"], dr
    md = VM.clip_metrics(C["desat"], base=C["smooth"], dino=dino)
    assert md["consistency"] is not None and abs(md["drift"] - dr["desat"]) < 1e-5, md
    assert dino.embed([]).shape == (0, 384) and dino.embed(None).shape == (0, 384)
    assert VM.clip_metrics(empty, base=C["smooth"], dino=dino) == NONE_METRICS
    sizes = [np.zeros((480, 832, 3), np.uint8), np.zeros((832, 480, 3), np.uint8)]     # 크기가 다른 그림 목록 (학습 썸네일)
    assert dino.embed(sizes).shape == (2, 384)

# ---------------------------------------------------------------- CCIP
saved = sys.modules.get("onnxruntime")
sys.modules["onnxruntime"] = None           # 설치 안 된 것처럼
try:
    assert VM.CCIP.load() is None
finally:
    if saved is None:
        del sys.modules["onnxruntime"]
    else:
        sys.modules["onnxruntime"] = saved
print("onnxruntime 이 없으면 CCIP.load() → None 확인")


class BrokenOrt:
    """깨진 설치처럼 import onnxruntime 이 OSError (DLL 을 못 찾음) 를 내게 한다"""
    def find_spec(self, name, path=None, target=None):
        if name == "onnxruntime":
            raise OSError("[WinError 126] 지정된 모듈을 찾을 수 없습니다 (시험)")
        return None


saved = sys.modules.pop("onnxruntime", None)
sys.meta_path.insert(0, BrokenOrt())
try:
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        assert VM.CCIP.load() is None
    assert "OSError" in buf.getvalue() and "WinError 126" in buf.getvalue(), buf.getvalue()   # 이유를 알려 준다
finally:
    sys.meta_path.pop(0)
    if saved is not None:
        sys.modules["onnxruntime"] = saved
print("onnxruntime 을 불러오다 OSError 가 나도 CCIP.load() → None 확인:", buf.getvalue().strip()[:80])
try:
    import onnxruntime  # noqa: F401
    have_ort = True
except ImportError:
    have_ort = False
ccip = VM.CCIP.load() if have_ort else None
if ccip is None:
    print("SKIP CCIP 실제 모델 (onnxruntime 없음 또는 내려받기 실패)")
else:
    assert abs(ccip.threshold - 0.1785) < 1e-3, ccip.threshold
    ref = scene(100, W=384, H=384)
    rf = ccip.features([ref, Image.fromarray(scene(10, W=384, H=384)).convert("RGBA")])
    D = ccip.differences(rf)
    assert rf.shape == (2, 768) and np.allclose(D, D.T, atol=1e-4) and abs(D[0, 0]) < 0.05, D
    ide = ccip.identity(np.stack([ref] * 9), rf[:1])
    assert ide["id_pass"] == 1.0 and ide["id_mean"] < ccip.threshold and abs(ide["id_slope"]) < 1e-3, ide
    mc = VM.clip_metrics(np.stack([ref] * 9), ccip=ccip, ref_feats=rf[:1])
    assert mc["id_pass"] == 1.0, mc
    none_id = dict(id_pass=None, id_mean=None, id_slope=None)
    assert ccip.identity(empty, rf[:1]) == none_id and ccip.identity(np.stack([ref] * 3), rf[:0]) == none_id
    assert ccip.features(None).shape == (0, 768) and ccip.features([]).shape == (0, 768)
    assert VM.clip_metrics(empty, ccip=ccip, ref_feats=rf[:1]) == NONE_METRICS
    dirs = [Path(p) for p in sys.argv[1:]]
    files = [sorted(q for q in d.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")) for d in dirs]
    if len(dirs) >= 2 and (len(files[0]) < 3 or not any(files[1:])):
        print("SKIP CCIP 폴더 시험 (첫 폴더 3장 이상, 다른 폴더 1장 이상 필요):", [len(f) for f in files])
    elif len(dirs) >= 2:
        load = lambda p: np.asarray(Image.open(p).convert("RGB").resize((384, 384), Image.Resampling.LANCZOS))
        refs = ccip.features([Image.open(p) for p in files[0][:2]])
        same_clip = np.stack([load(p) for p in files[0][2:]])
        other_clip = np.stack([load(p) for d in files[1:] for p in d])
        a, b = ccip.identity(same_clip, refs), ccip.identity(other_clip, refs)
        print(f"CCIP 같은 캐릭터 {len(same_clip)}장 {a} | 다른 캐릭터 {len(other_clip)}장 {b}")
        assert a["id_mean"] < b["id_mean"] and a["id_pass"] > b["id_pass"], (a, b)
print("VIDEO METRICS OK")
