"""video_io.py 시험. 합성 데이터셋(make_test_dataset)으로 읽기·EDA·전처리·저장을 모두 확인하고 'VIDEO IO OK' 를 찍는다.

python test_video_io.py [작업 폴더]      # Windows 는 짧은 경로 권장 (기본: 임시 폴더)
작업 폴더를 주면 그 안에 vio_test 하위 폴더를 만들어 쓰고, 다시 실행할 때 그 하위 폴더만 지운다 (주어진 폴더 자체와 다른
파일은 건드리지 않으며, 이 시험이 만들지 않은 vio_test 폴더는 지우지 않고 멈춘다). 결과는 남겨 둔다.
기본 임시 폴더는 통과하면 지운다 (실패하면 남겨 둠).
"""
import itertools
import math
import re
import shutil
import sys
import tempfile
import time
import warnings
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import video_io as V  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
T_START = time.time()
MARK = ".video_io_test"                           # 이 시험이 만든 폴더라는 표식
if len(sys.argv) > 1:
    WORK = Path(sys.argv[1]) / "vio_test"
    if WORK.exists() and any(WORK.iterdir()) and not (WORK / MARK).exists():
        sys.exit(f"{WORK} 는 이 시험이 만든 폴더가 아니어서 지우지 않습니다. 다른 작업 폴더를 주세요.")
    shutil.rmtree(WORK, ignore_errors=True)
else:
    WORK = Path(tempfile.mkdtemp(prefix="vio_"))
WORK.mkdir(parents=True, exist_ok=True)
(WORK / MARK).write_text("test_video_io.py 작업 폴더 (다시 실행하면 지워짐)\n", encoding="utf-8")
TRIG = "nkgreen"
TRIGGER_RE = re.compile(r"(?<!\w)" + re.escape(TRIG) + r"(?![A-Za-z0-9_])")    # 이미지 노트북과 같은 트리거 경계


def decoded_count(p):
    with av.open(str(p)) as c:
        return sum(1 for _ in c.decode(video=0))


def region(a, box, inset=3):
    y0, y1, x0, x1 = box
    return a[y0 + inset:y1 - inset, x0 + inset:x1 - inset].astype(np.float64)


def is_red(a, box):
    m = region(a, box).mean((0, 1))
    return m[0] > 170 and m[1] < 90 and m[2] < 90


def corners(a, s=16):
    H, W = a.shape[:2]
    return {"TL": (0, s, 0, s), "TR": (0, s, W - s, W), "BL": (H - s, H, 0, s), "BR": (H - s, H, W - s, W)}


def same_summary(a, b):
    return a.keys() == b.keys() and all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)    # nan == nan


# ---------------------------------------------------------------- 1. 시험 데이터 만들기
assert not (V.IMG_EXT & V.VID_EXT) and ".gif" in V.VID_EXT and ".webm" in V.VID_EXT and ".png" in V.IMG_EXT
t0 = time.time()
exp = V.make_test_dataset(WORK / "data", TRIG)
t_make = time.time() - t0
D = Path(exp["root"])
names = exp["images"] + exp["clips"]
assert t_make < 10, f"시험 데이터가 너무 느림 {t_make:.1f}s"
assert len(exp["images"]) == 4 and len(exp["clips"]) == 14 and all((D / n).exists() for n in names)
assert sorted(p.name for p in D.iterdir() if p.suffix != ".txt") == sorted(names), "기대값에 없는 파일이 있음"
print(f"dataset: {len(exp['images'])} images + {len(exp['clips'])} clips in {t_make:.2f}s -> {D}")

# ---------------------------------------------------------------- 2. probe / read_frames
KEYS = ("kind", "frames", "fps", "duration", "vfr", "rotation", "width", "height", "codec", "pix_fmt", "header_frames",
        "avg_rate", "has_alpha", "hdr")
frames, stamps, infos = {}, {}, {}
for n in names:
    p = D / n
    still = n in exp["images"]
    assert (p.suffix.lower() in V.IMG_EXT and not V.is_animated_image(p)) == still, n     # 이미지 = 확장자 + 움직이지 않음
    assert p.suffix.lower() in (V.IMG_EXT if still or n in exp["animated"] else V.VID_EXT), n
    info = V.probe(p)
    fr, ts = V.read_frames(p)
    frames[n], stamps[n], infos[n] = fr, ts, info
    assert all(k in info for k in KEYS), (n, [k for k in KEYS if k not in info])
    h, w = exp["size"][n]
    assert (info["height"], info["width"]) == (h, w), (n, info["width"], info["height"], exp["size"][n])
    assert fr.dtype == np.uint8 and fr.ndim == 4 and fr.shape[1:] == (h, w, 3), (n, fr.shape)
    assert ts.shape == (len(fr),) and ts[0] == 0.0 and np.all(np.diff(ts) > 0), (n, ts[:5])
    assert info["has_alpha"] == (n in exp["alpha"]), (n, info["has_alpha"], info["pix_fmt"])
    assert info["hdr"] is False, n
    if n in exp["portrait"]:
        assert h > w, n
    if still:
        assert info["kind"] == "image" and info["frames"] == 1 and len(fr) == 1 and not info["vfr"]
        continue
    assert info["kind"] == "video", n
    k = exp["frames"][n]
    assert info["frames"] == k == len(fr) == decoded_count(p), (n, info["frames"], k, len(fr))      # 패킷 수 = 디코딩 수
    assert abs(info["fps"] - exp["fps"][n]) / exp["fps"][n] < 0.03, (n, info["fps"], exp["fps"][n])
    assert info["vfr"] == exp["vfr"][n], (n, info["vfr"])
    assert (info["header_frames"] == 0) == (n in exp["header_zero"]), (n, info["header_frames"])
    assert info["rotation"] == exp["rotation"].get(n, 0), (n, info["rotation"])
    assert abs(info["duration"] - (ts[-1] + np.median(np.diff(ts)))) < 2e-3, (n, info["duration"], ts[-1])
print("probe/read_frames: frames == packets == decoded, fps, VFR, header 0, rotation, display size OK")

# 편집 목록: ffmpeg -ss … -c copy 로 자른 mp4 는 앞 패킷이 is_discard (디코더가 버림) → frames 에서 뺀다
for n, k in exp["discard"].items():
    with av.open(str(D / n)) as c:
        pk = [x for x in c.demux(video=0) if x.size]
    assert len(pk) == exp["frames"][n] + k and sum(x.is_discard for x in pk) == k, (n, len(pk))
# 움직이는 이미지: 확장자가 이미지여도 클립 (Pillow 로만 읽으면 첫 프레임 1장)
assert [n for n in names if (D / n).suffix.lower() in V.IMG_EXT and V.is_animated_image(D / n)] == exp["animated"]
assert V.is_animated_image(D / "anim.gif") and V.is_animated_image(D / "hold.gif") and not V.is_animated_image(D / "cfr24.mp4")
assert V.load_image(D / "anim_loop.webp").shape == (96, 160, 3) and len(frames["anim_loop.webp"]) == 24
more = WORK / "more"
more.mkdir()
apng = [Image.fromarray(V._scene(t, 2, speed=5)) for t in range(10)]
apng[0].save(more / "apng.png", save_all=True, append_images=apng[1:], duration=[100, 50] * 5, loop=0)
apng[0].save(more / "one.webp")
apng[0].save(more / "pages.tif", save_all=True, append_images=apng[1:3])                  # 여러 쪽 TIFF 는 애니가 아님
assert V.is_animated_image(more / "apng.png") and not V.is_animated_image(more / "one.webp")
assert not V.is_animated_image(more / "pages.tif") and not V.is_animated_image(more / "없는 파일.png")
info = V.probe(more / "apng.png")
fr, ts = V.read_frames(more / "apng.png")
assert (info["kind"], info["frames"], info["vfr"], len(fr)) == ("video", 10, True, 10), info
assert np.allclose(ts, np.cumsum([0] + [0.1, 0.05] * 4 + [0.1])), ts
assert max(np.abs(fr[i].astype(int) - np.asarray(apng[i])).max() for i in range(10)) <= 1     # Pillow 프레임과 같은 화소
assert V.probe(more / "one.webp")["kind"] == "image" and V.read_frames(more / "one.webp")[0].shape == (1, 96, 160, 3)
print("edit-list discard, animated WebP/APNG/GIF as clips (is_animated_image), multi-page TIFF stays an image OK")

# 회전 방향: 저장 기준 왼쪽 위 빨간 표시가 네 모서리 가운데 기대한 곳에만 있어야 한다 (반대로 돌리면 다른 모서리)
for n, box in exp["red"].items():
    a = frames[n][0]
    assert is_red(a, box), (n, region(a, box).mean((0, 1)))
    red = [k for k, b in corners(a).items() if is_red(a, b)]
    assert red == [k for k, b in corners(a).items() if b == box], (n, red, box)
with av.open(str(D / "rot90 세로.mp4")) as c:                  # 따로 연 디코더(libvpx)용: 스트림 표시 행렬에서 읽은 회전
    assert V._stream_rotation(c.streams.video[0]) == 90 == infos["rot90 세로.mp4"]["rotation"]
with av.open(str(D / "cfr24.mp4")) as c:
    assert V._stream_rotation(c.streams.video[0]) == 0
for n, (y0, y1, x0, x1) in exp["white"].items():  # 투명 → 흰색 (png 는 이미지 경로, gif·webp 는 bgra·argb, webm 은 libvpx)
    assert frames[n][:, y0:y1, x0:x1].min() >= 250, n
assert frames["anim.gif"][:, 20:, 20:].mean() < 240, "gif 의 불투명한 부분까지 하얘짐"
assert frames["vp9_alpha.webm"][:, :, 56:].mean() < 240, "알파 webm 의 불투명한 부분까지 하얘짐"
lo, hi = exp["gray16_mean"]
assert lo < frames["img_gray16.png"].mean() < hi, frames["img_gray16.png"].mean()

# 알파 VP8·VP9 webm: 기본 디코더는 알파를 버려 숨은 빨간색이 보인다 → libvpx 로 읽어 흰색. libvpx 가 없으면 경고하고 알파는 있다고 보고
with av.open(str(D / "vp9_alpha.webm")) as c:
    hidden = next(c.decode(video=0)).to_ndarray(format="rgb24")[:, :40].reshape(-1, 3).mean(0)
assert hidden[0] > 150 and hidden[1] < 90, hidden
assert infos["vp9_alpha.webm"]["pix_fmt"] == "yuva420p"
_codec_ok = V._codec_ok
V._codec_ok = lambda name, mode="r": False
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = V.probe(D / "vp9_alpha.webm")
        fr, _ = V.read_frames(D / "vp9_alpha.webm")
finally:
    V._codec_ok = _codec_ok
assert info["has_alpha"] and sum("libvpx" in str(x.message) for x in caught) == 2, [str(x.message) for x in caught]
assert fr[:, :, :40, 0].mean() > 150 and fr[:, :, :40, 1].mean() < 90 and len(fr) == 24
vp8 = [np.dstack([V._scene(t, 3), np.full((96, 160), 255, np.uint8)]) for t in range(8)]
for a in vp8:
    a[:, :48] = (200, 40, 40, 0)
V._encode(more / "vp8_alpha.webm", vp8, 16, "libvpx", "yuva420p", {"auto-alt-ref": "0", "b": "1M"})
info = V.probe(more / "vp8_alpha.webm")
fr, _ = V.read_frames(more / "vp8_alpha.webm")
assert info["has_alpha"] and info["codec"] == "vp8" and len(fr) == info["frames"] == 8, info
assert fr[:, :, :40].min() >= 250 and fr[:, :, 56:].mean() < 240

# max_side: 회전 전에 줄이고, 비율 유지
fr, _ = V.read_frames(D / "cfr24.mp4", max_side=80)
assert fr.shape == (48, 48, 80, 3), fr.shape
fr, _ = V.read_frames(D / "rot90 세로.mp4", max_side=80)
assert fr.shape == (36, 80, 48, 3), fr.shape
fr, ts = V.read_frames(D / "img_plain.png", max_side=64)
assert fr.shape == (1, 48, 64, 3) and ts.tolist() == [0.0], fr.shape
fr, _ = V.read_frames(D / "anim_loop.webp", max_side=80)
assert fr.shape == (24, 48, 80, 3), fr.shape

# SAR (애너모픽): 저장 80x96, SAR 2:1 → 표시 160x96
sar_path = WORK / "sar.mp4"
with av.open(str(sar_path), "w") as c:
    st = c.add_stream("libx264", rate=16)
    st.width, st.height, st.pix_fmt = 80, 96, "yuv420p"
    st.codec_context.sample_aspect_ratio = Fraction(2, 1)
    src = [V._scene(t, 0)[:, ::2] for t in range(9)]
    for a in src:
        for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(a), format="rgb24")):
            c.mux(pk)
    for pk in st.encode():
        c.mux(pk)
info = V.probe(sar_path)
assert (info["width"], info["height"], info["sar"]) == (160, 96, 2.0), info
fr, _ = V.read_frames(sar_path)
assert fr.shape == (9, 96, 160, 3), fr.shape
assert np.abs(fr.astype(int) - np.stack([V._scene(t, 0) for t in range(9)])).mean() < 12      # 가로로 늘어난 원래 그림
fr, _ = V.read_frames(sar_path, max_side=80)
assert fr.shape == (9, 48, 80, 3), fr.shape
print("rotation direction, alpha on white (png, gif, webp, VP8/VP9 webm via libvpx), 16-bit, max_side, SAR OK")

# ---------------------------------------------------------------- 3. load_image = 이미지 노트북과 같은 결과
src = (HERE.parent / "build_notebook.py").read_text(encoding="utf-8")
snippet = src[src.index("def to_8bit(im):"):src.index("def read_caption(path):")]
nb = {"np": np, "Image": Image}
exec(snippet, nb)                                  # 이미지 노트북의 to_8bit·flatten_alpha 그대로
extra = WORK / "extra"
extra.mkdir()
rng = np.random.default_rng(1)
Image.fromarray(rng.random((30, 40)).astype("float32"), "F").save(extra / "f01.tif")
Image.fromarray((rng.random((30, 40)) * 70000).astype("int32"), "I").save(extra / "i32.tif")
pal = Image.fromarray(rng.integers(0, 255, (30, 40, 3), dtype=np.uint8)).convert("P", palette=Image.Palette.ADAPTIVE)
pal.info["transparency"] = 0
pal.save(extra / "p_trans.png", transparency=0)
la = np.dstack([rng.integers(0, 255, (30, 40), dtype=np.uint8), np.tile(np.linspace(0, 255, 40), (30, 1)).astype(np.uint8)])
Image.fromarray(la, "LA").save(extra / "la.png")
for p in [D / n for n in exp["images"]] + sorted(extra.iterdir()):
    with Image.open(p) as im:
        ref = np.asarray(nb["flatten_alpha"](nb["to_8bit"](ImageOps.exif_transpose(im))))
    got = V.load_image(p)
    assert got.dtype == np.uint8 and got.shape == ref.shape and np.array_equal(got, ref), p.name
    assert np.array_equal(V.read_frames(p)[0][0], got), p.name
    assert got.flags.writeable, p.name
# EXIF 방향 1~8: probe 의 표시 크기 = load_image 크기, rotation 은 회전 부분 (2·4·5·7 은 좌우 뒤집힘도 포함)
for o, rot in {1: 0, 2: 0, 3: 180, 4: 180, 5: 90, 6: -90, 7: -90, 8: 90}.items():
    ex = Image.Exif()
    ex[0x0112] = o
    Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)).save(more / f"exif{o}.jpg", exif=ex.tobytes())
    info = V.probe(more / f"exif{o}.jpg")
    assert (info["height"], info["width"]) == V.load_image(more / f"exif{o}.jpg").shape[:2] and info["rotation"] == rot, (o, info)
print("load_image == image notebook to_8bit/flatten_alpha (EXIF 1-8, I;16, I, F, P+transparency, LA, RGBA) OK")

# ---------------------------------------------------------------- 4. 신호·사건·요약
summaries, events, events_plain = {}, {}, {}
for n in names:
    fr, fps, ts = frames[n], infos[n]["fps"] or 16.0, stamps[n]
    sig = V.frame_signals(fr, fps)
    T = len(fr)
    for key in ("mean_luma", "black", "blur", "sat"):
        assert sig[key].shape == (T,), (n, key)
    assert sig["rgb"].shape == (T, 3) and sig["side"] == min(96, max(fr.shape[1:3])), (n, sig["side"])
    if T > 1:
        for key in ("mad", "mad_ref", "content", "hist_corr", "motion_med", "motion_p95"):
            assert sig[key].shape == (T - 1,), (n, key)
        assert sig["motion_k"] == max(2, round(fps * V.MOTION_SPAN)), (n, sig["motion_k"])
        assert sig["motion_s"] == max(1, math.ceil(fps * V.MOTION_MID - 1e-6)), (n, sig["motion_s"])
        for sfx, key in (("_s", "motion_s"), ("_k", "motion_k")):
            assert sig["motion_p95" + sfx].shape == sig["motion_med" + sfx].shape == (max(T - sig[key], 0),), (n, sfx)
    else:
        assert "mad" not in sig
    ev = V.detect_events(sig, fps, timestamps=ts)      # 노트북처럼 read_frames 시각을 준다
    ev0 = V.detect_events(sig, fps)
    sm = V.summarize_clip(fr, fps, sig, ev)
    summaries[n], events[n], events_plain[n] = sm, ev, ev0
    if n in ("img_plain.png", "events.mp4", "hold.gif", "anim_loop.webp", "cfr24_dup.mp4"):   # sig·events 를 안 줘도 같은 값
        assert same_summary(sm, V.summarize_clip(fr, fps, timestamps=ts)), n
    for key in ("subject_motion", "camera_motion", "hold_ratio", "drawing_fps", "blur", "n_cuts", "n_black", "n_freeze",
                "sat", "mean_rgb", "static"):
        assert key in sm, (n, key)
    if n in exp["events"]:
        e = exp["events"][n]
        assert (ev["black"], ev["freeze"], ev["cuts"]) == (e["black"], e["freeze"], e["cuts"]), (n, ev)
        assert V.clean_segments(T, ev) == e["segments"], V.clean_segments(T, ev)
        assert (sm["n_cuts"], sm["n_black"], sm["n_freeze"]) == (
            len(e["cuts"]), sum(b - a for a, b in e["black"]), sum(b - a for a, b in e["freeze"])), sm
    else:
        assert ev["black"] == ev["freeze"] == ev["cuts"] == [], (n, ev)
        assert V.clean_segments(T, ev) == [(0, T)], n
    if n == "hold.gif":                            # 원본에서는 1프레임이라 시각 없이는 못 잡는다
        assert ev0["freeze"] == [] and ev["freeze"] == [(8, 9)], (ev0, ev)
    else:                                          # CFR·짧은 VFR 은 시각을 줘도 같다 (ms 로 반올림한 시각 포함)
        assert all(ev0[k] == ev[k] for k in ("black", "freeze", "cuts")) and same_summary(
            {k: ev0[k] for k in ("hold_ratio", "drawing_fps")}, {k: ev[k] for k in ("hold_ratio", "drawing_fps")}), n
    if n in exp["clips"]:
        assert sm["static"] == (n in exp["static"]), (n, sm["subject_motion"])
        assert (sm["subject_motion"] < V.STATIC_MOTION) == (n in exp["static"]), (n, sm["subject_motion"])
        assert sm["camera_motion"] < V.CAMERA_MOVING, (n, sm["camera_motion"])          # 배경은 모두 고정
        assert 0 <= sm["hold_ratio"] <= 1 and abs(sm["drawing_fps"] - fps * (1 - sm["hold_ratio"])) < 1e-9, n
    else:
        assert not sm["static"] and np.isnan(sm["hold_ratio"]) and sm["subject_motion"] == 0.0, n
    print(f"  {n:16s} subj={sm['subject_motion']:.3f}/s cam={sm['camera_motion']:.3f}/s hold={sm['hold_ratio']:.2f} "
          f"events={ {k: ev[k] for k in ('black', 'freeze', 'cuts')} }")
assert summaries["cfr24.mp4"]["subject_motion"] > 4 * V.STATIC_MOTION

# 이미지(1프레임): 노트북 1-2 처럼 summarize_clip(frames, 1.0) 으로 blur·sat 를 잰다
for n in exp["images"]:
    sm = V.summarize_clip(frames[n], 1.0)
    assert np.isfinite(sm["blur"]) and sm["blur"] >= 0 and 0 <= sm["sat"] <= 255 and len(sm["mean_rgb"]) == 3, (n, sm)
    assert sm["subject_motion"] == sm["camera_motion"] == 0.0 and not sm["static"] and np.isnan(sm["hold_ratio"]), (n, sm)
    assert sm["n_cuts"] == sm["n_black"] == sm["n_freeze"] == 0, (n, sm)
assert V.summarize_clip(frames["img_plain.png"], 1.0)["blur"] > 100 and V.summarize_clip(frames["img_plain.png"], 1.0)["sat"] > 30
tiny = V.summarize_clip(np.full((1, 2, 2, 3), 7, np.uint8), 1.0)                  # 3px 미만도 죽지 않는다
assert tiny["blur"] == 0.0 and tiny["sat"] == 0.0, tiny

# 카메라 이동 (화면 전체가 프레임당 v px, 160px 기준 → 초당 긴 변의 v*16/160). 1px 미만은 0.5초 간격 비교가 잡는다
tex = Image.fromarray(np.random.default_rng(3).integers(0, 256, (24, 80, 3), dtype=np.uint8)).resize((1920, 576))


def pan(u, fps, n):
    """화면 전체가 초당 긴 변의 u 만큼 움직이는 팬 (160x96, 4배 크기 무늬를 BOX 로 줄임)"""
    return np.stack([np.asarray(tex.resize((160, 96), Image.Resampling.BOX, box=(8 + 640 * u * t / fps, 40,
                                                                                648 + 640 * u * t / fps, 424)))
                     for t in range(n)])


for v, lo, hi in [(0.2, 0.0, V.STATIC_MOTION), (0.75, 0.06, 0.12), (1.0, 0.06, 0.25), (3.0, 0.2, 0.42), (5.0, 0.35, 0.6)]:
    sm = V.summarize_clip(pan(v * 16 / 160, 16, 32), 16)
    assert lo <= sm["camera_motion"] < hi and sm["subject_motion"] >= sm["camera_motion"], (v, sm)
    assert sm["static"] == (v * 16 / 160 < V.STATIC_MOTION), (v, sm)
    flagged = any("카메라가 움직임" in s for s in V.caption_issues(f"{TRIG}, she walks, static camera", TRIGGER_RE, True, (), sm))
    assert flagged == (sm["camera_motion"] > V.CAMERA_MOVING), (v, sm)
# 30·60fps: 이웃 비교는 1px 미만, 0.5초 비교는 ±4px 를 넘는 속도를 1/16초 간격이 잰다 (예전에는 60fps 초당 30% 팬이 0.08)
ratios = {}
for fps in (30, 60):
    for u in (0.08, 0.3, 0.6):
        sm = V.summarize_clip(pan(u, fps, int(0.8 * fps)), fps)
        ratios[fps, u] = round(sm["camera_motion"] / u, 2)
        assert 0.6 * u <= sm["camera_motion"] <= 1.7 * u and sm["subject_motion"] >= sm["camera_motion"], (fps, u, sm)
        flagged = any("카메라가 움직임" in s for s in V.caption_issues(f"{TRIG}, she walks, static camera", TRIGGER_RE, True, (), sm))
        assert flagged == (u >= 0.3), (fps, u, sm)                # 초당 30% 이상 팬은 'static camera' 와 모순
print(f"camera pans 16/30/60 fps OK (measured/true at 30·60fps: {ratios})")

# 셀 애니 홀드 (on twos/threes)와 프레임 단위 사건 (메모리 배열로 정확히)
base = [V._scene(t, 2, speed=4) for t in range(12)]
for rep in (2, 3):
    clip = np.stack([f for f in base for _ in range(rep)])
    ev = V.detect_events(V.frame_signals(clip, 24), 24)
    n_tr, n_same = len(clip) - 1, len(base) * (rep - 1)
    assert ev["freeze"] == [] and abs(ev["hold_ratio"] - n_same / n_tr) < 1e-9, (rep, ev)
    assert abs(ev["drawing_fps"] - 24 * (1 - n_same / n_tr)) < 1e-9
clip = np.stack(base[:4] + [base[4]] * 8 + base[5:] + [np.zeros_like(base[0])] * 5)   # 4~11 멈춤, 끝의 검은 5장
ev = V.detect_events(V.frame_signals(clip, 16), 16)
assert ev["freeze"] == [(4, 12)] and ev["black"] == [(19, 24)] and ev["cuts"] == [], ev
assert V.clean_segments(len(clip), ev) == [(0, 4), (12, 19)]
assert V.detect_events(V.frame_signals(clip, 16), 16, timestamps=np.arange(len(clip)) / 16) == ev    # CFR 시각 = 시각 없음
assert V.clean_segments(10, dict(cuts=[0, 3, 10], black=[(8, 10)], freeze=[])) == [(0, 3), (3, 8)]
cut = np.stack([V._scene(t, 2) for t in range(10)] + [V._scene(t, 5) for t in range(10, 20)])
ev = V.detect_events(V.frame_signals(cut, 16), 16)
assert ev["cuts"] == [10] and np.isclose(ev["hold_ratio"], 0.0), ev
ev = V.detect_events(V.frame_signals(cut, 16), 16, min_scene_s=2.0)       # 첫 컷은 항상 남는다
assert ev["cuts"] == [10], ev
ev = V.detect_events(V.frame_signals(frames["img_plain.png"], 16), 16)
assert ev["black"] == ev["freeze"] == ev["cuts"] == [] and np.isnan(ev["hold_ratio"]) and np.isnan(ev["drawing_fps"]), ev


# 느린 움직임은 멈춤이 아니다 (freezedetect 처럼 멈춤 후보의 첫 프레임과 비교): 1280x720 셀화의 프레임당 1.2px 이동을
# 480px 로 옮긴 것. 이웃 프레임끼리는 차이가 FREEZE_MAD 보다 작아 예전 방식은 클립 전체를 멈춤으로 뺐다
def cel(dx):
    im = Image.new("RGB", (960, 540), (200, 220, 240))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 378, 960, 540), fill=(120, 170, 90))
    d.ellipse((225, 90, 390, 195), fill=(250, 250, 255), outline=(60, 60, 80), width=2)
    cx = 480 + 2 * dx
    d.ellipse((cx - 82, 150, cx + 82, 450), fill=(240, 200, 170), outline=(30, 20, 20), width=3)
    d.ellipse((cx - 38, 210, cx - 15, 232), fill=(30, 20, 20))
    d.ellipse((cx + 15, 210, cx + 38, 232), fill=(30, 20, 20))
    return np.asarray(im.resize((480, 270), Image.Resampling.BOX))


drift = np.stack([cel(0.45 * t) for t in range(48)])
sig = V.frame_signals(drift, 24)
assert np.all(sig["mad"] < V.FREEZE_MAD), sig["mad"].max()                 # 이웃 비교로는 모두 '같은 프레임'
assert V.detect_events(sig, 24)["freeze"] == [], V.detect_events(sig, 24)
held = np.concatenate([drift[:20], drift[20:21].repeat(12, 0), drift[21:]])
fz = V.detect_events(V.frame_signals(held, 24), 24)["freeze"]
assert len(fz) == 1 and fz[0][0] == 20 and 32 <= fz[0][1] <= 36, fz          # 진짜 정지는 잡는다 (느린 이동이 이어지면 조금 길게)

# VFR 긴 정지: 원본에서 1프레임이라도 표시 시간이 길면 시각을 줄 때 멈춤 (표시 시간을 프레임 수로 반올림 - 1 ≥ round(0.25*fps))
clip = np.stack(base)
sig = V.frame_signals(clip, 12.5)
for extra_s, want in ((1.0, [(5, 6)]), (3 / 12.5, [(5, 6)]), (2 / 12.5, [])):
    ts = np.arange(12) / 12.5
    ts[6:] += extra_s                              # 5번 프레임이 1/12.5 + extra_s 초 떠 있음
    ev = V.detect_events(sig, 12.5, timestamps=ts)
    assert ev["freeze"] == want, (extra_s, ev)
    assert V.clean_segments(12, ev) == ([(0, 5), (6, 12)] if want else [(0, 12)])
assert V.detect_events(sig, 12.5)["freeze"] == []
assert V.detect_events(sig, 4.0, timestamps=np.arange(12) / 4.0)["freeze"] == []   # 4fps 클립의 0.25초 프레임은 정상
blk = np.stack(base[:3] + [np.zeros_like(base[0])] + base[3:6])               # 검은 프레임이 오래 떠 있는 것은 black 이 맡는다
ev = V.detect_events(V.frame_signals(blk, 12.5), 12.5, timestamps=np.array([0, 0.08, 0.16, 0.24, 2.0, 2.08, 2.16]))
assert ev["black"] == [(3, 4)] and ev["freeze"] == [], ev
try:
    V.detect_events(sig, 12.5, timestamps=np.arange(5) / 12.5)
    raise AssertionError("길이가 다른 timestamps 가 통과함")
except ValueError:
    pass
print("events: black/freeze (first-frame reference, VFR long holds), cut, clean segments, hold ratio on twos/threes OK")

# ---------------------------------------------------------------- 5. 중복 클립
sigs = {n: V.clip_signature(frames[n]) for n in exp["clips"]}
assert all(s.shape == (8, 64) and s.dtype == bool for s in sigs.values())
assert V.clip_signature(frames["cfr24.mp4"], skip_first=True).shape == (7, 64)
dist = {(a, b): V.signature_distance(sigs[a], sigs[b]) for a, b in itertools.combinations(exp["clips"], 2)}
found = {pair for pair, d in dist.items() if d <= V.DUP_BITS}
assert found == {tuple(x) for x in exp["duplicates"]}, (found, sorted(dist.items(), key=lambda kv: kv[1])[:3])
assert V.signature_distance(sigs["cfr24.mp4"], sigs["cfr24.mp4"]) == 0.0
print(f"duplicates: {sorted(found)} (dup {dist[('cfr24.mp4', 'cfr24_dup.mp4')]:.2f} bits, "
      f"nearest other {min(d for pr, d in dist.items() if pr not in found):.2f} bits) OK")

# ---------------------------------------------------------------- 6. 캡션
for n in names:
    cp = (D / n).with_suffix(".txt")
    assert cp.exists() == (n not in exp["no_caption"]), n
    if not cp.exists():
        continue
    issues = V.caption_issues(cp.read_text(encoding="utf-8"), TRIGGER_RE, n in exp["clips"], exp["fixed_words"], summaries[n])
    want = exp["caption_issues"].get(n, [])
    assert all(any(w in s for s in issues) for w in want), (n, issues)
    assert len(issues) == (2 if n == "odd_444.mp4" else len(want)), (n, issues)      # 고정 특징 단어 2개
good = "nkgreen, a girl turns around and waves, static camera, medium shot, park background"
assert V.caption_issues(good, TRIGGER_RE, True) == []
iss = V.caption_issues("nkgreen with a black bob and wine-red cape standing", TRIGGER_RE, True, ("black bob", "wine-red"))
assert sum("고정 특징" in s for s in iss) == 2 and any("카메라" in s for s in iss) and any("동작" in s for s in iss), iss  # standing 은 정지
assert any("트리거" in s for s in V.caption_issues("nkgreen_x walks, static camera, medium shot", TRIGGER_RE, True))
moving_cam = dict(subject_motion=0.4, camera_motion=0.5)
still_sum = dict(subject_motion=0.0, camera_motion=0.0)
assert any("카메라가 움직임" in s for s in V.caption_issues(good, TRIGGER_RE, True, summary=moving_cam))
assert V.caption_issues("nkgreen, standing in a park, full shot", TRIGGER_RE, False) == []       # 이미지는 동작·카메라 검사 없음
NO_MOTION, NO_CAMERA = "동작 표현 없음", "카메라·샷 표현 없음"
for cap, summary, want in [                       # 검증 보고서의 사례: 정지 상태 낱말은 동작이 아니고, 샷·카메라 표현은 넓게
        ("nkgreen, close-up, static camera, a quiet park at dusk", still_sum, [NO_MOTION]),
        ("nkgreen, close-up, static camera, eyes closed, standing still", still_sum, [NO_MOTION]),
        ("nkgreen with pointed ears holding a bowl, medium shot, static camera", still_sum, [NO_MOTION]),
        ("nkgreen with wavy hair and a bow tie, lower body, open field, static camera", still_sum, [NO_MOTION]),
        ("nkgreen, closeup, static camera, she waves", moving_cam | dict(camera_motion=0.0), []),
        ("nkgreen, close up, static camera, she stands up and waves", moving_cam | dict(camera_motion=0.0), []),
        ("nkgreen waves while the camera slowly pans left", moving_cam, []),
        ("nkgreen waves, the camera is panning to the right", moving_cam, []),
        ("nkgreen waves, locked-off camera, medium shot", moving_cam, ["카메라가 움직임"]),
        ("nkgreen, long shot, she walks toward the camera", moving_cam, []),
        ("nkgreen, upper body, she looks around and smiles", moving_cam, []),
        ("nkgreen, she sits down on a bench in the park", moving_cam, [NO_CAMERA]),
        ("nkgreen looks at the viewer in a sunny park", moving_cam, [NO_MOTION, NO_CAMERA]),
        ("nkgreen sits on the stone steps by the swings, medium shot, static camera", still_sum, [NO_MOTION]),   # 명사
        ("nkgreen waves while the camera is still panning left", moving_cam, [])]:          # 'still panning' 은 고정 아님
    got = V.caption_issues(cap, TRIGGER_RE, True, (), summary)
    assert len(got) == len(want) and all(any(w in s for s in got) for w in want), (cap, got)
kr = re.compile(r"(?<!\w)" + re.escape("니키") + r"(?![A-Za-z0-9_])")
assert V.caption_issues("니키가 공원에서 손을 흔든다, walks, static camera", kr, True, ("단발",)) == []
assert any("단발" in s for s in V.caption_issues("니키가 단발머리로 walks, static camera, medium shot", kr, True, ("단발머리",)))
assert V.caption_issues("니키가 손을 흔든다, 고정 카메라, 미디엄 샷", kr, True, (), still_sum | dict(subject_motion=0.3)) == []
assert V.caption_issues("니키가 걸어간다, 카메라가 천천히 오른쪽으로 이동한다", kr, True) == []
iss = V.caption_issues("니키가 공원 벤치에 앉아 있다, 클로즈업", kr, True)                      # 앉아 있다 = 정지 상태
assert len(iss) == 1 and NO_MOTION in iss[0], iss
assert V.caption_issues("니키가 카메라를 향해 다가온다, 미디엄 샷", kr, True, (), moving_cam) == []   # 카메라'를' = 피사체 동작
iss = V.caption_issues("니키가 공원 벤치에서 카메라를 가만히 바라본다", kr, True)
assert len(iss) == 2 and NO_MOTION in iss[0] and NO_CAMERA in iss[1], iss
assert any("카메라가 움직임" in s for s in V.caption_issues("니키가 손을 흔든다, 고정 카메라, 미디엄 샷", kr, True, (), moving_cam))
print("caption issues (static states are not motion, close-up/closeup/close up, camera ... pans, Korean) OK")

# 노트북 TEST 모드의 대조 함수 (build_video_notebook.py 의 df·EVENTS·DUP_PAIRS 와 같은 모양으로)
import pandas as pd  # noqa: E402

rows = []
for n in names:
    cp = (D / n).with_suffix(".txt")
    cap = cp.read_text(encoding="utf-8") if cp.exists() else ""
    is_clip = infos[n]["kind"] == "video"          # 노트북처럼: 이미지 확장자 + is_animated_image → probe 의 kind
    iss = V.caption_issues(cap, TRIGGER_RE, is_clip, exp["fixed_words"], summaries[n] if is_clip else None) if cap else ["캡션 없음"]
    rows.append(dict(file=n, kind=infos[n]["kind"], width=infos[n]["width"], height=infos[n]["height"],
                     frames=infos[n]["frames"], vfr=infos[n]["vfr"], rotation=infos[n]["rotation"] if is_clip else 0,
                     has_alpha=infos[n]["has_alpha"] if is_clip else False, decoded=len(frames[n]),
                     subject_motion=summaries[n]["subject_motion"] if is_clip else float("nan"),
                     has_caption=bool(cap), caption_issues="; ".join(iss)))
table = pd.DataFrame(rows)
clip_events = {n: events[n] for n in exp["clips"]}
pairs = [(a, b, d) for (a, b), d in dist.items() if d <= 8]
V.check_test_expectations(table, clip_events, pairs, exp)
bad_cases = [(table, dict(clip_events, **{"events.mp4": dict(events["events.mp4"], cuts=[])}), pairs),
             (table.assign(subject_motion=table.subject_motion.where(table.file != "static.mp4", 0.4)), clip_events, pairs),
             (table.assign(rotation=0), clip_events, pairs), (table, clip_events, []),
             (table.assign(caption_issues=""), clip_events, pairs), (table.assign(decoded=1), clip_events, pairs),
             (table.assign(kind=table.kind.where(table.file != "anim_loop.webp", "image")), clip_events, pairs),   # 정지 이미지로 읽음
             (table, dict(clip_events, **{"hold.gif": events_plain["hold.gif"]}), pairs),                         # 시각 없이 사건
             (table.assign(frames=table.frames.where(table.file != "trim_copy.mp4", 36)), clip_events, pairs),     # 편집 목록 무시
             (table.assign(has_alpha=table.has_alpha.where(table.file != "vp9_alpha.webm", False)), clip_events, pairs)]
for t, e, p in bad_cases:
    try:
        V.check_test_expectations(t, e, p, exp)
        raise RuntimeError("틀린 결과가 통과함")
    except AssertionError:
        pass
print(f"check_test_expectations (notebook TEST mode) passes on real results and catches {len(bad_cases)} broken cases OK")


# ---------------------------------------------------------------- 7. fps 리샘플·4k+1·창
def fps_filter_ref(ts, fps, n):
    """ffmpeg vf_fps.c 를 흉내 낸 기준 (정확한 분수): 시작 시각을 출력 단위로 반올림(round=near, 반은 위로)하고,
    다음 프레임의 반올림 시각이 지금 출력 번호 이하이면 앞 프레임을 버리고, 아니면 앞 프레임을 내보낸다."""
    order = sorted(range(len(ts)), key=lambda i: (float(ts[i]), i))
    q = [Fraction(float(ts[i])).limit_denominator(10 ** 6) for i in order]
    pts = [math.floor((x - q[0]) * Fraction(fps) + Fraction(1, 2)) for x in q]
    out, cur = [], 0
    for k in range(n):
        while cur + 1 < len(pts) and pts[cur + 1] <= k:
            cur += 1
        out.append(order[cur])
    return out


for ts, fps, n in [(np.arange(49) / 24, 16, 33), (np.arange(61) / 30, 16, 33), (stamps["vfr_mp4.mp4"], 16, 31),
                   (stamps["vfr_mkv.mkv"], 16, 31), (stamps["anim.gif"], 16, 15), (stamps["hold.gif"], 16, 38),
                   (np.arange(81) / 16, 16, 81), (np.arange(49) / 24, 24, 49), (stamps["cfr24.mp4"], 30, 60),
                   (np.arange(33) / 32, 16, 17), (np.arange(24) / 12, 16, 32), (np.arange(40) / 48, 24, 20),
                   (stamps["vp9.webm"], 16, 22), (stamps["anim_loop.webp"], 24, 36)]:
    idx = V.resample_to_fps(ts, fps)
    assert len(idx) == n and idx.tolist() == fps_filter_ref(ts, fps, n) and np.all(np.diff(idx) >= 0), (n, len(idx), idx[:12])
assert V.resample_to_fps(np.arange(49) / 24, 24).tolist() == list(range(49))
assert V.resample_to_fps(np.arange(33) / 32, 16).tolist() == list(range(0, 33, 2))          # 반은 올림: ffmpeg 처럼 짝수
assert V.resample_to_fps(np.arange(24) / 12, 16)[:8].tolist() == [0, 1, 1, 2, 3, 4, 4, 5]    # docstring 예시
gif = V.resample_to_fps(stamps["anim.gif"], 16)
assert len(gif) > len(set(gif.tolist())), gif                               # 원본이 더 느리면 같은 프레임 반복
held = V.resample_to_fps(stamps["hold.gif"], 16)
assert (held == 8).sum() == 19 and held[10:29].tolist() == [8] * 19, held   # 1.2초 떠 있는 프레임 = 16fps 에서 19칸 그대로
perm = np.random.default_rng(0).permutation(49)
shuf = (np.arange(49) / 24)[perm]
idx = V.resample_to_fps(shuf, 16)
assert np.all(np.diff(shuf[idx]) > 0) and idx.tolist() == fps_filter_ref(shuf, 16, len(idx))  # 섞인 시각도 원래 번호로
assert V.resample_to_fps([3.0], 16).tolist() == [0]
assert [V.round_4k1(n) for n in (-3, 0, 1, 2, 4, 5, 33, 80, 81, 83, 120, 121)] == [0, 0, 1, 1, 1, 5, 33, 77, 81, 81, 117, 121]
assert V.round_4k1(20, temporal=8) == 17 and V.round_4k1(81.0) == 81
assert V.windows(8, 9) == [] and V.windows(9, 0) == []
assert V.windows(21, 9, "head") == [0] and V.windows(21, 9, "uniform") == [0, 6, 12] and V.windows(21, 9, "uniform", n=1) == [0]
assert V.windows(27, 9, "chunk") == [0, 9, 18] and V.windows(21, 9, "slide") == [0, 4, 8, 12] and V.windows(9, 9, "uniform") == [0]
r = V.windows(40, 9, "random", n=3, rng=np.random.default_rng(0))
assert len(r) == 3 == len(set(r)) and r == sorted(r) and all(0 <= s <= 31 for s in r)
assert V.windows(10, 9, "random", n=5, rng=np.random.default_rng(0)) == [0, 1]
assert V.windows(40, 9, "random", rng=np.random.default_rng(5)) == V.windows(40, 9, "random", rng=np.random.default_rng(5))
for mode in ("head", "uniform", "chunk", "slide", "random"):
    assert all(s + 9 <= 30 for s in V.windows(30, 9, mode, n=4, rng=np.random.default_rng(1)))
try:
    V.windows(30, 9, "bogus")
    raise AssertionError("모르는 창 방식이 통과함")
except ValueError:
    pass
seg = exp["events"]["events.mp4"]["segments"]                                # 깨끗한 구간 안에서만 창
starts = [(a, s) for a, b in seg for s in V.windows(b - a, 9, "uniform")]
assert starts == [(3, 0), (3, 6), (3, 12)], starts                         # 7·8 프레임 구간에는 9프레임 창이 없음
print("resample_to_fps (== ffmpeg fps filter: CFR, 2x ties, VFR, GIF holds, shuffled), round_4k1, windows OK")

# ---------------------------------------------------------------- 8. 버킷·자르기
B = V.make_buckets(32760, [1, 17, 33, 49, 81], [832 / 480, 480 / 832, 1.0], 832 * 480, 16, 4, 16)
assert (81, 480, 832, 32760) in B and (81, 832, 480, 32760) in B, B                 # 표준 비율 → 정확히 480x832
for F, H, W, tok in B:
    assert (F - 1) % 4 == 0 and H % 16 == 0 and W % 16 == 0 and H * W <= 832 * 480 and tok <= 32760
    assert tok == (1 + (F - 1) // 4) * (H // 16) * (W // 16)
assert V.make_buckets(32760, [81], [16 / 9], 832 * 480, 16, 4, 16) == [(81, 464, 832, 21 * 29 * 52)]
B5 = V.make_buckets(27280, [1, 121], [1280 / 704, 704 / 1280], 1280 * 704, 32, 4, 32)
assert (121, 704, 1280, 27280) in B5 and (121, 1280, 704, 27280) in B5 and all(h % 32 == 0 == w % 32 for _, h, w, _ in B5)
try:
    V.make_buckets(32760, [80], [1.0], 832 * 480, 16, 4, 16)
    raise AssertionError("4k+1 이 아닌 프레임 수가 통과함")
except ValueError:
    pass
assert V.pick_bucket(B, 60, 832, 480) == (49, 480, 832, 13 * 30 * 52)
assert V.pick_bucket(B, 100, 832, 480) == (81, 480, 832, 32760)
assert V.pick_bucket(B, 100, 480, 832) == (81, 832, 480, 32760)
assert V.pick_bucket(B, 100, 640, 360) is None                                     # 업스케일 금지
assert V.pick_bucket(B, 100, 640, 360, allow_upscale=1.4) == (81, 480, 832, 32760)
assert V.pick_bucket(B, 100, 1000, 400) is None                                    # 비율이 달라 세로를 키워야 함
assert V.pick_bucket(B, 1, 1024, 1024) == (1, 624, 624, 39 * 39)
assert V.pick_bucket(B, 0, 832, 480) is None
B2 = V.make_buckets(14040, [1, 33, 81], [832 / 480], 832 * 480, 16, 4, 16)          # 81프레임은 예산 때문에 줄어듦
assert (81, 304, 544, 21 * 19 * 34) in B2                                           # 배수 맞춤으로 비율 1.789 (요청 1.733)
assert V.pick_bucket(B2, 100, 832, 480) == (81, 304, 544, 21 * 19 * 34)
assert V.pick_bucket(B2, 100, 832, 480, prefer="area") == (33, 480, 832, 9 * 30 * 52)
B3 = V.make_buckets(32760, [81], [832 / 480, 1280 / 704], 832 * 480, 16, 4, 16)     # 가까운 두 비율: 정확한 쪽
assert V.pick_bucket(B3, 81, 832, 480) == (81, 480, 832, 32760) and V.pick_bucket(B3, 81, 1280, 704)[1:3] == (464, 848)

src = np.stack([V._scene(t, 4) for t in range(3)])
src[:, 44:52, 76:84] = (255, 0, 0)                                                 # 가운데 표시
out = V.resize_crop_frames(src, 48, 48)
assert out.shape == (3, 48, 48, 3) and out.dtype == np.uint8
assert is_red(out[0], (20, 28, 20, 28)), out[0, 20:28, 20:28].mean((0, 1))
manual = np.asarray(Image.fromarray(src[1]).crop((32, 0, 128, 96)).resize((48, 48), Image.Resampling.LANCZOS))
assert np.abs(out[1].astype(int) - manual)[4:-4, 4:-4].max() <= 2               # 가장자리는 box 밖 화소도 필터에 씀
same = V.resize_crop_frames(np.stack([src[0]] * 4), 64, 96)
assert all(np.array_equal(same[0], f) for f in same)                              # 모든 프레임에 같은 자르기
up = V.resize_crop_frames(src[:, :20, :30], 40, 48)
assert up.shape == (3, 40, 48, 3)
rng8 = np.random.default_rng(8)                   # w/(w/w0) 가 부동소수 오차로 w0 를 넘어 상자가 음수가 되던 크기 (약 5% 버킷)
for (w0, h0), (w, h) in [((47, 38), (96, 64)), ((47, 52), (48, 48)), ((1611, 3699), (464, 848))]:
    a = rng8.integers(0, 256, (2, h0, w0, 3), dtype=np.uint8)
    assert V.resize_crop_frames(a, h, w).shape == (2, h, w, 3), (w0, h0, w, h)
print("make_buckets (exact 480x832, 704x1280, token budget), pick_bucket (no upscale), resize_crop_frames (box clamp) OK")

# ---------------------------------------------------------------- 9. 컨택트 시트·저장
c1, c2 = frames["cfr24.mp4"][:9], np.full((9, 96, 160, 3), (200, 30, 30), np.uint8)
img = np.full((64, 64, 3), 90, np.uint8)
sheet = V.contact_sheet([c1, c2, img], labels=["step 0", "단계 500", "image"], n=8, thumb_h=60, pad=4)
assert isinstance(sheet, Image.Image) and sheet.mode == "RGB"
lab_w = sheet.width - 4 - 8 * (100 + 4)
assert sheet.height == 4 + 3 * (60 + 4) and lab_w > 20, sheet.size
a = np.asarray(sheet)
for r in range(3):                                                                  # 행마다 글자가 그려짐
    assert a[4 + r * 64:4 + r * 64 + 60, :lab_w].min() < 128, r
x0 = lab_w + 4
assert np.abs(a[4 + 64 + 10:4 + 64 + 50, x0 + 10:x0 + 90].astype(int) - (200, 30, 30)).max() <= 2   # 두 번째 행 썸네일
plain = V.contact_sheet([c1], n=4, thumb_h=48, pad=2)
assert plain.size == (2 + 4 * (80 + 2), 2 + 48 + 2), plain.size

odd = np.stack([V._scene(t, 1, h=97, w=161) for t in range(10)])
path = V.save_mp4(WORK / "out_odd.mp4", odd, 16)
assert isinstance(path, str) and Path(path).exists()
info = V.probe(path)
assert (info["width"], info["height"], info["frames"], info["codec"], info["pix_fmt"]) == (162, 98, 10, "h264", "yuv420p"), info
assert abs(info["fps"] - 16) < 1e-6 and abs(info["avg_rate"] - 16) < 1e-6
back, _ = V.read_frames(path)
ref = np.pad(odd, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="edge").astype(np.float64)
psnr = 10 * np.log10(255 ** 2 / ((back.astype(np.float64) - ref) ** 2).mean())
assert back.shape == (10, 98, 162, 3) and psnr > 30, psnr
info = V.probe(V.save_mp4(WORK / "out_ntsc.mp4", odd[:, :96, :160], 24000 / 1001))
assert abs(info["fps"] - 23.976) < 0.01 and info["frames"] == 10, info
assert V.probe(V.save_mp4(WORK / "out_one.mp4", odd[0, :96, :160], 16))["frames"] == 1
info = V.probe(V._encode(WORK / "out_mpeg4.mp4", odd[:, :96, :160], 16, "mpeg4", "yuv420p", {"b": "8M"}))   # libx264 없을 때
assert (info["codec"], info["frames"], info["width"]) == ("mpeg4", 10, 160), info
for bad in (odd.astype(np.float32) / 255, odd.astype(np.int64)):                  # 0~1 실수를 uint8 로 자르면 검은 영상
    try:
        V.save_mp4(WORK / "bad.mp4", bad, 16)
        raise AssertionError("uint8 이 아닌 프레임이 통과함")
    except TypeError:
        pass


def webp_durations(path):
    with Image.open(path) as im:
        assert im.format == "WEBP"
        out = []
        for i in range(im.n_frames):
            im.seek(i)
            im.load()                                                                # Pillow 는 load 때 duration 을 채움
            out.append(im.info["duration"])
        return out, im.size


held = np.stack([f for f in odd[:6, :64, :96] for _ in range(3)])                  # 같은 프레임 3장씩 → Pillow 가 합침
path = V.save_webp(WORK / "out.webp", held, 16)
durs, size = webp_durations(path)
assert isinstance(path, str) and size == (96, 64)
assert len(durs) <= len(held) and abs(sum(durs) - len(held) * 1000 / 16) <= 1, durs
moving = np.stack([V._scene(t, 1, h=64, w=96, speed=5) for t in range(16)])
path = V.save_webp(WORK / "out2.webp", moving, 16)
durs, _ = webp_durations(path)
assert len(durs) == 16 and durs[:4] == [62, 63, 63, 62] and sum(durs) == 1000, durs          # 16fps = 62.5ms 누적 반올림
info = V.probe(path)                                                               # 미리보기 WebP 도 클립으로 다시 읽힌다
assert V.is_animated_image(path) and (info["kind"], info["frames"]) == ("video", 16) and abs(info["fps"] - 16) < 0.6, info
print(f"contact_sheet, save_mp4 (odd -> even, PSNR {psnr:.1f} dB, 23.976 fps), save_webp (merged frames keep time) OK")

assert "imageio_ffmpeg" not in sys.modules and "cv2" not in sys.modules
if len(sys.argv) == 1:                            # 기본 임시 폴더는 통과하면 지운다 (실패하면 남겨 둠)
    shutil.rmtree(WORK, ignore_errors=True)
print(f"total {time.time() - T_START:.1f}s")
print("VIDEO IO OK")
