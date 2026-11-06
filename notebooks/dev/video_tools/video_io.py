"""영상 LoRA 노트북의 영상 입출력·EDA·전처리 도구 (numpy, Pillow, PyAV 만 사용).

따르는 기준과 계산식:
  * 메타데이터: 헤더 값(stream.frames, average_rate)은 mkv·webm·VFR 에서 틀린다. 디코딩 없이 패킷 PTS 를 모아
    frames = 패킷 수, fps = 1/median(dt), duration = (마지막 - 처음) + median(dt), VFR = std(dt)/mean(dt) > 0.05
    (webm 은 1ms 단위 시각이라 CFR 이어도 0.01 안팎이 나온다). 재인코딩 없이 자른 mp4·mov(ffmpeg -ss … -c copy)는 편집 목록
    (edit list) 앞의 패킷을 디코더가 버리므로(is_discard) 세지 않는다. 회전·HDR 만 첫 프레임 1장을 디코딩해 읽는다.
  * 디코딩: PyAV 는 회전을 적용하지 않으므로 frame.rotation(DISPLAYMATRIX, 반시계 각도)만큼 np.rot90(k=rotation/90)
    (ffmpeg 자동 회전과 화소까지 같음). SAR≠1 이면 폭을 coded_w*SAR 로 늘리고, 알파는 흰 배경에 합성한다.
    FFmpeg 기본 vp8·vp9 디코더는 webm 의 알파(alpha_mode=1, BlockAdditional)를 버리므로 그런 클립은 libvpx 로 디코딩한다.
    움직이는 GIF·PNG(APNG)·WebP 는 확장자가 이미지여도 클립이다 (is_animated_image, 노트북도 같은 판정으로 나눈다).
    imageio-ffmpeg 는 VFR·GIF 를 CFR 로 복제해 내보내므로(30장 → 60장) 쓰지 않는다.
  * 이미지: 이미지 노트북과 같게 EXIF 회전, 16비트·실수 → 8비트, 투명 → 흰 배경.
  * 검은 프레임 = ffmpeg blackdetect (휘도 ≤ 0.10*255 인 화소 비율 ≥ 0.98). 멈춤 = ffmpeg freezedetect: 멈춤 후보의 첫 프레임과의
    평균 |dY|/255 < 0.001 이 0.25초 이상 (첫 프레임과 비교하므로 느린 움직임은 차이가 쌓여 멈춤이 풀린다). 시각을 주면 VFR·GIF 에서
    한 그림이 화면에 떠 있는 시간으로 잰다. 컷 = PySceneDetect ContentDetector 와 같은 식: HSV 평균 차 ≥ 27 (단 색상은 PIL 0~255
    척도라 OpenCV 0~179 보다 색상 차가 약 1.42배 크게 들어가 조금 더 민감하다), 또는 차 ≥ 15 이면서 (Y 히스토그램 상관 ≤ 0.80
    이거나 앞뒤 4전환 평균의 3배 이상). 히스토그램 기준에 15 조건을 붙인 것은 평평한 셀화·GIF 팔레트의 오검출을 막기 위함.
    셀 애니의 2·3프레임 홀드는 결함이 아니므로 hold_ratio·drawing_fps = fps*(1-hold_ratio) 로 보고만 한다.
  * 움직임 = 긴 변 96px 휘도에서 8x8 블록 정합(±4px). 블록 변위의 중앙값 = 카메라, 상위 5% = 피사체 (VBench dynamic degree).
    이웃 프레임(빠른 움직임), 1/16초(30·60fps 에서 이웃은 1px 미만, 0.5초는 ±4px 를 넘는 중간 속도), 0.5초(1px 미만의 느린
    움직임) 떨어진 프레임을 비교해 큰 값을 쓴다. 단위는 '초당 화면 긴 변의 비율'(0.5 = 1초에 긴 변의 절반을 움직임)이라
    해상도·fps 가 달라도 비교할 수 있다 (16·24·30·60fps 화면 전체 팬에서 초당 긴 변의 0.8 까지는 참값의 약 0.8~1.7배,
    그보다 빠르면 이웃 비교의 ±4px 한계로 작게 읽는다: 16fps 에서 최대 약 0.67).
  * 중복 = 같은 상대 위치 8프레임의 pHash (imagehash.phash 와 같은 계산) 평균 해밍 거리, DUP_BITS(8) 이하면 중복 의심.
  * 리샘플 = ffmpeg fps 필터(round=near)와 같다: 격자 시각 t_k = k/target_fps 에 화면에 떠 있는 원본 프레임 (시작 시각을
    출력 단위로 반올림한 값이 k 이하인 마지막 프레임). 시각 기준이라 VFR·GIF 의 긴 프레임은 떠 있는 시간만큼 반복된다.
  * Wan VAE: 잠재 프레임 = 1 + (F-1)//4 이고 4k+1 을 넘는 프레임은 경고 없이 버려진다 → round_4k1 로 미리 자른다.
  * 버킷: 토큰 = (1+(F-1)//4) * (H/TOKEN_DIV) * (W/TOKEN_DIV) ≤ 예산, H*W ≤ max_area, 업스케일 없음.
    표준 비율(832/480, 1280/704)을 그대로 넣으면 정확히 480x832·704x1280 이 나온다 (16/9 를 넣으면 464x832).
  * 저장: libx264 yuv420p 는 짝수 크기만 받으므로 가장자리를 복제해 맞춘다. Pillow 는 GIF·WebP 에서 같은 프레임을 합치고
    GIF 는 10ms 단위라 재생 속도가 바뀌므로, 지표는 메모리 배열로 재고 미리보기는 mp4·WebP 로 저장한다.
  * 시험: make_test_dataset 이 알려진 문제(회전·VFR·홀수 크기·검은/멈춘 구간·컷·중복·정지·편집 목록·알파 webm·움직이는 WebP·
    VFR 긴 정지)를 넣은 합성 데이터와 기대값을 만들고, check_test_expectations 가 노트북 EDA 결과와 대조한다.
    플랫폼·FFmpeg 빌드 차이를 실행할 때 잡기 위함.
"""
import math
import os
import re
import warnings
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
from av.video.format import VideoFormat
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts",
           ".wmv", ".flv", ".3gp"}

VFR_CV = 0.05                       # 프레임 간격의 변동계수가 이보다 크면 VFR
BLACK_PIX, BLACK_PIC = 0.10, 0.98   # blackdetect pix_th, pic_th
FREEZE_MAD = 1e-3                   # freezedetect noise (-60dB)
CUT_CONTENT, CUT_HIST, CUT_RATIO, CUT_MIN = 27.0, 0.20, 3.0, 15.0
STATIC_MOTION = 0.05                # subject_motion(초당 화면 긴 변의 비율)이 이보다 작으면 '거의 정지' (초당 긴 변의 5%)
MOTION_SPAN = 0.5                   # 느린 움직임을 잴 프레임 간격(초): 96px 에서 0.5초에 1px = 초당 긴 변의 2%
MOTION_MID = 1 / 16                 # 중간 간격(초): 30·60fps 에서 이웃 비교(1px 미만은 0)와 0.5초 비교(±4px 에서 포화) 사이
CAMERA_MOVING = 0.10                # camera_motion(초당 화면 긴 변의 비율)이 이보다 크면 'static camera' 캡션과 모순
DUP_BITS = 8.0                      # 서명 거리(64비트 중)가 이 이하이면 중복 의심 (자동 삭제는 하지 않음)
BLUR_SIDE = 256                     # 흐림(라플라시안 분산)을 재는 긴 변
CAPTION_WORDS = (6, 120)
ASPECT_TOL = math.log(1.1)          # pick_bucket: 가장 가까운 비율보다 10% 안쪽으로 어긋난 버킷은 같은 비율로 본다


# ------------------------------------------------------------------ 읽기
def _has_alpha(pix_fmt):
    """화소 형식에 알파가 있는지 (bgra·rgba·yuva420p·ya8·pal8). 이름에 'a' 가 있는지로 보면 gray 도 걸린다."""
    try:
        f = VideoFormat(pix_fmt)
    except Exception:
        return False
    return bool(f.has_palette or any(c.is_alpha for c in f.components))


def _sar(vs):
    s = vs.sample_aspect_ratio or vs.codec_context.sample_aspect_ratio
    return float(s) if s and float(s) > 0 else 1.0


@lru_cache(None)
def _codec_ok(name, mode="r"):
    """이 PyAV(FFmpeg) 빌드에 name 디코더(r)·인코더(w)가 있는지"""
    try:
        av.Codec(name, mode)
        return True
    except Exception:
        return False


def _alpha_decoder(vs):
    """알파가 있는 VP8·VP9 webm(alpha_mode=1)이면 알파째 디코딩할 libvpx 디코더 이름, 아니면 None.
    FFmpeg 기본 vp8·vp9 디코더는 BlockAdditional 에 담긴 알파를 버려 투명한 곳에 숨은 색이 나온다.
    libvpx 디코더가 없는 빌드면 '' (부르는 쪽이 경고하고 알파 없이 읽는다)."""
    meta = {str(k).lower(): str(v).strip() for k, v in (vs.metadata or {}).items()}
    name = vs.codec_context.name
    if meta.get("alpha_mode") != "1" or name not in ("vp8", "vp9", "libvpx", "libvpx-vp9"):
        return None
    lib = "libvpx-vp9" if name in ("vp9", "libvpx-vp9") else "libvpx"
    return lib if _codec_ok(lib, "r") else ""


def _warn_no_alpha(p):
    warnings.warn(f"{Path(p).name}: 알파가 있는 VP8·VP9 webm 인데 이 PyAV(FFmpeg)에 libvpx 디코더가 없어 알파 없이 읽음 "
                  "(투명한 곳이 흰색이 아니라 숨은 색으로 나옴)", stacklevel=3)


def _stream_rotation(vs):
    """스트림 표시 행렬(display_matrix)의 회전, 반시계 각도. 따로 연 디코더(libvpx)의 프레임에는 이 값이 붙지 않는다."""
    try:
        m = np.frombuffer(vs.codec_context.coded_side_data["display_matrix"], np.int32)[:5] / 65536.0
        return int(round(-math.degrees(math.atan2(m[1] / math.hypot(m[1], m[4]), m[0] / math.hypot(m[0], m[3])))))
    except Exception:
        return 0


def is_animated_image(path):
    """움직이는 GIF·PNG(APNG)·WebP 이면 True (확장자가 아니라 내용으로 본다). probe·read_frames 는 이런 파일을 확장자가
    이미지여도 클립으로 읽는다: Pillow 로 열면 첫 프레임만 나와 정지 이미지로 학습되기 때문이다 (ComfyUI 'Save Animated
    WEBP'·save_webp 출력이 이 형식). 열 수 없는 파일, 1프레임 파일, 여러 쪽 TIFF·MPO(카메라 JPEG)는 False."""
    try:
        with Image.open(path) as im:
            return im.format in ("GIF", "PNG", "WEBP") and bool(getattr(im, "is_animated", False))
    except Exception:
        return False


def _is_still_image(p):
    return p.suffix.lower() in IMG_EXT and not is_animated_image(p)


def _decoded(c, vs, decoder=None):
    """스트림 vs 의 프레임. decoder(예: 'libvpx-vp9')를 주면 기본 디코더 대신 그것으로 디코딩한다 (시각은 패킷에서)."""
    if not decoder:
        yield from c.decode(vs)
        return
    cc = av.CodecContext.create(decoder, "r")
    for pkt in c.demux(vs):
        if pkt.size:                             # 빈 패킷(끝)은 아래에서 한 번만 보낸다
            yield from cc.decode(pkt)
    yield from cc.decode(None)


def probe(path):
    """메타데이터 dict: kind('image'·'video'), frames, fps, duration, vfr, rotation, width, height(회전·SAR 적용한 표시 크기),
    codec, pix_fmt, header_frames(헤더의 프레임 수, mkv·webm 은 0), avg_rate(헤더 fps), has_alpha, hdr.
    frames 는 디코딩되는 프레임 수다 (편집 목록 밖 패킷은 세지 않음). has_alpha 는 디코딩한 화소에 알파가 있어 흰 배경에
    합성된다는 뜻이다 (GIF·움직이는 WebP 디코더는 늘 알파 형식, 알파 VP8·VP9 webm 은 libvpx 로 읽음).
    이미지는 1프레임으로 돌려준다: rotation 은 EXIF 방향의 회전 부분이고 EXIF 2·4·5·7 은 좌우 뒤집힘도 함께 적용된다.
    움직이는 GIF·PNG·WebP 는 확장자가 이미지여도 클립(kind='video')으로 읽는다 (is_animated_image)."""
    p = Path(path)
    if _is_still_image(p):
        with Image.open(p) as im:
            o = im.getexif().get(0x0112, 1)
            w, h = (im.height, im.width) if o in (5, 6, 7, 8) else im.size
            return dict(kind="image", frames=1, fps=0.0, duration=0.0, vfr=False,
                        rotation={3: 180, 4: 180, 5: 90, 6: -90, 7: -90, 8: 90}.get(o, 0),
                        width=w, height=h, codec=(im.format or "").lower(), pix_fmt=im.mode, header_frames=1, avg_rate=0.0,
                        has_alpha=im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info, hdr=False)
    with av.open(str(p)) as c:
        if not c.streams.video:
            raise ValueError(f"영상 스트림이 없음: {p.name}")
        vs = c.streams.video[0]
        cc = vs.codec_context
        dec = _alpha_decoder(vs)
        if dec == "":
            _warn_no_alpha(p)
        info = dict(kind="video", container=c.format.name, codec=cc.name, pix_fmt="yuva420p" if dec is not None else cc.pix_fmt,
                    coded_w=cc.width, coded_h=cc.height, sar=_sar(vs), header_frames=int(vs.frames or 0),
                    avg_rate=float(vs.average_rate or 0), has_alpha=dec is not None or _has_alpha(cc.pix_fmt),
                    has_audio=bool(c.streams.audio))
        rate, tb, pts, n = float(vs.guessed_rate or vs.average_rate or 0), float(vs.time_base or 0), [], 0
        for pkt in c.demux(vs):                  # 디코딩 없이 패킷만 (1 패킷 = 1 프레임)
            if pkt.size == 0 or getattr(pkt, "is_discard", False):   # 끝을 알리는 빈 패킷, 편집 목록 밖 (디코더가 버림)
                continue
            n += 1
            t = pkt.pts if pkt.pts is not None else pkt.dts
            if t is not None:
                pts.append(t)
    with av.open(str(p)) as c:                   # 회전·HDR 은 프레임 정보라 첫 프레임만 디코딩
        f = next(c.decode(video=0), None)
    info["rotation"] = int(f.rotation) if f is not None else 0
    info["hdr"] = f is not None and int(f.color_trc) in (16, 18)    # PQ·HLG: rgb24 로 바꾸면 톤 매핑 없이 색이 바램
    ts = np.sort(np.asarray(pts, np.float64)) * tb
    dt = np.diff(ts)
    dt = dt[dt > 0]
    if len(dt):
        med = float(np.median(dt))
        info.update(fps=1.0 / med, duration=float(ts[-1] - ts[0]) + med, vfr=bool(dt.std() / dt.mean() > VFR_CV))
    else:                                        # 1프레임이거나 시각이 없는 스트림
        info.update(fps=rate, duration=n / rate if rate else 0.0, vfr=False)
    w, h = info["coded_w"], info["coded_h"]
    if info["sar"] != 1.0:                       # 애너모픽: 표시 폭 = coded_w * SAR
        w = max(1, round(w * info["sar"]))
    if info["rotation"] % 180:
        w, h = h, w
    info.update(frames=n, width=w, height=h)
    return info


def _over_white(rgba):
    """알파를 흰 배경에 합성 (이미지 노트북 flatten_alpha 와 같은 결과)"""
    a = rgba[..., 3:4].astype(np.uint16)
    return ((rgba[..., :3].astype(np.uint16) * a + 255 * (255 - a) + 127) // 255).astype(np.uint8)


def read_frames(path, max_side=None):
    """((T,H,W,3) uint8, (T,) 초) — 회전·SAR 적용, 알파는 흰 배경. 시각은 첫 프레임이 0.
    max_side 를 주면 긴 변을 그 이하로 줄여 읽는다 (EDA 용). 이미지는 1프레임 클립으로 돌려준다.
    움직이는 GIF·PNG·WebP 는 모든 프레임을 읽는다. 알파 VP8·VP9 webm 은 libvpx 로 디코딩한다 (없으면 경고).
    모든 프레임을 메모리에 올리므로 긴 영상은 max_side 를 주거나 미리 잘라 넣는다."""
    p = Path(path)
    if _is_still_image(p):
        a = load_image(p)
        if max_side and max(a.shape[:2]) > max_side:
            s = max_side / max(a.shape[:2])
            a = np.array(Image.fromarray(a).resize((max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))),
                                                   Image.Resampling.LANCZOS))
        return a[None], np.zeros(1)
    frames, ts, size = [], [], None
    with av.open(str(p)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        dec = _alpha_decoder(vs)
        if dec == "":
            _warn_no_alpha(p)
        fmt = "rgba" if dec or _has_alpha(vs.codec_context.pix_fmt) else "rgb24"
        rot0 = _stream_rotation(vs) if dec else 0    # 따로 연 디코더의 프레임에는 회전 정보가 없다
        sar, rate = _sar(vs), float(vs.guessed_rate or vs.average_rate or 0) or 16.0
        for i, f in enumerate(_decoded(c, vs, dec)):
            if size is None:                     # 첫 프레임 기준 크기 (중간에 크기가 바뀌는 스트림도 같은 크기로)
                w, h = f.width * sar, f.height
                s = min(1.0, max_side / max(w, h)) if max_side else 1.0
                size = (max(1, round(w * s)), max(1, round(h * s)))
                interp = "AREA" if size[0] <= f.width and size[1] <= f.height else "BICUBIC"
            a = f.to_ndarray(format=fmt, width=size[0], height=size[1], interpolation=interp)
            if fmt == "rgba":
                a = _over_white(a)
            k = int(round((f.rotation or rot0) / 90)) % 4  # 반시계 각도
            frames.append(np.ascontiguousarray(np.rot90(a, k)) if k else a)
            ts.append(float(f.time) if f.time is not None else i / rate)    # raw 스트림은 시각이 없음
    if not frames:
        raise ValueError(f"프레임을 읽지 못함: {p.name}")
    ts = np.asarray(ts, np.float64)
    order = np.argsort(ts, kind="stable")
    return np.stack([frames[i] for i in order]), ts[order] - ts[order[0]]


def to_8bit(im):
    """16비트 흑백(I;16)·정수(I)·실수(F) 이미지를 0~255 로 줄인다. 그대로 RGB 로 바꾸면 255 에서 잘려 하얗게 된다."""
    if im.mode.startswith("I;16"):
        a = np.asarray(im, dtype=np.float32) / 257.0
    elif im.mode in ("I", "F"):
        a = np.asarray(im, dtype=np.float32)
        hi = float(np.nanmax(a)) if a.size else 0.0
        if im.mode == "F" and hi <= 1.0:
            a = a * 255.0                  # 0~1 실수 이미지
        elif hi > 255.0:
            a = a / hi * 255.0             # 범위를 모르는 큰 값은 최댓값 기준
    else:
        return im
    return Image.fromarray(np.nan_to_num(a).clip(0, 255).astype(np.uint8), "L")


def flatten_alpha(im):
    """투명한 부분을 흰 배경에 합성한 RGB. EDA 와 학습이 같은 이미지를 보게 한다."""
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im.convert("RGBA"))
    return im.convert("RGB")


def load_image(path):
    """(H,W,3) uint8. 이미지 노트북 load_rgb 와 같은 처리: EXIF 회전 → 8비트 → 투명은 흰 배경."""
    with Image.open(path) as im:
        return np.array(flatten_alpha(to_8bit(ImageOps.exif_transpose(im))))


# ------------------------------------------------------------------ 프레임 신호와 사건
def _u8(frames):
    """uint8 배열로 (0~1 실수를 그냥 uint8 로 바꾸면 검게 잘리므로 막는다)"""
    a = np.asarray(frames)
    if a.dtype != np.uint8:
        raise TypeError(f"uint8(0~255) 프레임이 필요함: {a.dtype} (0~1 실수는 (x * 255).round().astype(np.uint8))")
    return a


def _luma(rgb):
    """BT.601 휘도 (0~255, float32)"""
    rgb = rgb.astype(np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _shrink(frames, side):
    """긴 변이 side 보다 크면 모든 프레임을 같은 크기로 줄인다 (BOX 평균)"""
    h, w = frames.shape[1:3]
    s = side / max(h, w)
    if s >= 1:
        return frames
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.stack([np.asarray(Image.fromarray(f).resize(size, Image.Resampling.BOX)) for f in frames])


def _hist_corr(a, b, bins=128):
    """Y 히스토그램 상관 (cv2.compareHist HISTCMP_CORREL 과 같음)"""
    ha = np.histogram(a, bins, (0, 256))[0].astype(np.float64)
    hb = np.histogram(b, bins, (0, 256))[0].astype(np.float64)
    ha -= ha.mean()
    hb -= hb.mean()
    d = math.sqrt(float((ha * ha).sum() * (hb * hb).sum()))
    return float((ha * hb).sum() / d) if d > 0 else 1.0


_OFFSETS = sorted(((dy, dx) for dy in range(-4, 5) for dx in range(-4, 5)), key=lambda o: math.hypot(*o))


def _block_motion(g0, g1, block=8, bias=0.5):
    """8x8 블록 정합: g1 의 각 블록이 g0 의 ±4px 안 어디서 왔는지 찾아 변위 크기(px)를 돌려준다.
    평평하거나 잡음뿐인 블록이 엉뚱한 변위를 고르지 않게 변위 1px 마다 화소당 bias 만큼 비용을 더한다."""
    r = 4
    bh, bw = (g0.shape[0] - 2 * r) // block, (g0.shape[1] - 2 * r) // block
    if bh < 1 or bw < 1:
        return np.zeros(1, np.float32)
    ref = g1[r:r + bh * block, r:r + bw * block]
    best = np.full((bh, bw), np.inf, np.float32)
    best_d = np.zeros((bh, bw), np.float32)
    for dy, dx in _OFFSETS:                     # 가까운 변위부터 → 비용이 같으면 작은 변위
        d = math.hypot(dy, dx)
        cand = g0[r + dy:r + dy + bh * block, r + dx:r + dx + bw * block]
        cost = np.abs(ref - cand).reshape(bh, block, bw, block).sum((1, 3)) + bias * block * block * d
        better = cost < best
        best[better], best_d[better] = cost[better], d
    return best_d.ravel()


def frame_signals(frames, fps, analysis_side=96):
    """프레임별·전환별 신호 (detect_events·summarize_clip 의 입력). blur 를 뺀 값은 긴 변 analysis_side 로 줄여 잰다.
    프레임별 (T,): mean_luma(0~1), black(검은 화소 비율), blur(라플라시안 분산, 긴 변 BLUR_SIDE), sat(HSV 채도 0~255), rgb (T,3)
    전환별 (T-1,): mad(이웃 프레임과의 평균 |dY|/255, 홀드용), mad_ref(멈춤 후보의 첫 프레임과의 평균 |dY|/255, freezedetect 처럼
      FREEZE_MAD 이상 달라지면 그 프레임이 새 기준. 느린 움직임이 묻히지 않게 긴 변 BLUR_SIDE 에서 잰다),
      content(HSV 평균 차), hist_corr(Y 히스토그램 상관),
      motion_med·motion_p95(블록 변위 px)
    motion_med_s·motion_p95_s (T-s,), motion_med_k·motion_p95_k (T-k,): 프레임 i 와 i+s (s = motion_s, 약 MOTION_MID 초),
      i+k (k = motion_k, 약 MOTION_SPAN 초) 비교를 간격으로 나눈 프레임당 변위. 이웃 비교는 96px 에서 1px 미만을 0 으로 읽고
      (16fps 에서 초당 긴 변의 17%, 60fps 에서 62% 미만), k 간격은 ±4px 를 넘으면 포화하므로 30·60fps 의 중간 속도는 s 간격이 잰다.
    side = 분석 프레임의 긴 변 (움직임을 '긴 변 비율' 로 바꿀 때 씀). 1프레임(이미지)이면 전환별 값은 없다."""
    frames = _u8(frames)
    T = len(frames)
    small = _shrink(frames, analysis_side)
    Y = _luma(small)
    hsv = np.stack([np.asarray(Image.fromarray(f).convert("HSV"), np.int16) for f in small])
    yb = _luma(_shrink(frames, BLUR_SIDE))
    lap = yb[:, 1:-1, 1:-1] * -4 + yb[:, :-2, 1:-1] + yb[:, 2:, 1:-1] + yb[:, 1:-1, :-2] + yb[:, 1:-1, 2:]
    sig = dict(fps=float(fps), side=int(max(Y.shape[1:])), mean_luma=Y.mean((1, 2)) / 255.0,
               black=(Y <= BLACK_PIX * 255).mean((1, 2)), blur=lap.reshape(T, -1).var(1) if lap[0].size else np.zeros(T),
               sat=hsv[..., 1].mean((1, 2)), rgb=small.reshape(T, -1, 3).mean(1))
    if T > 1:
        sig["mad"] = np.abs(np.diff(Y, axis=0)).mean((1, 2)) / 255.0
        ref, mr = 0, np.empty(T - 1)             # freezedetect: 멈춤 후보의 첫 프레임과 비교 (느린 움직임은 쌓여서 풀림)
        for i in range(1, T):                    # 96px 에서는 0.3px/프레임 같은 느린 팬이 3배 느리게 쌓여 멈춤으로 읽힌다
            mr[i - 1] = np.abs(yb[i] - yb[ref]).mean() / 255.0
            if mr[i - 1] >= FREEZE_MAD:
                ref = i
        sig["mad_ref"] = mr
        sig["content"] = np.abs(np.diff(hsv, axis=0)).mean((1, 2, 3))   # PIL 색상은 0~255 (OpenCV 0~179)
        sig["hist_corr"] = np.array([_hist_corr(Y[i], Y[i + 1]) for i in range(T - 1)])
        k = sig["motion_k"] = max(2, round(fps * MOTION_SPAN))
        s = sig["motion_s"] = max(1, math.ceil(fps * MOTION_MID - 1e-6))   # 23.976fps 도 2 (round 면 1)
        done = {}
        for suffix, step in (("", 1), ("_s", s), ("_k", k)):
            if step not in done:                 # 16fps 이하는 s = 1 이라 이웃 비교와 같다
                mv = [_block_motion(Y[i], Y[i + step]) / step for i in range(T - step)]
                done[step] = (np.array([np.median(m) for m in mv]), np.array([np.percentile(m, 95) for m in mv]))
            sig["motion_med" + suffix], sig["motion_p95" + suffix] = done[step]
    return sig


def _runs(mask, min_len=1):
    """True 가 이어지는 구간 [(시작, 끝)) 목록"""
    m = np.concatenate([[0], np.asarray(mask, np.int8), [0]])
    e = np.flatnonzero(np.diff(m))
    return [(int(a), int(b)) for a, b in zip(e[::2], e[1::2]) if b - a >= min_len]


def detect_events(sig, fps, min_scene_s=0.5, freeze_min_s=0.25, timestamps=None):
    """black·freeze: 프레임 구간 [(시작, 끝)), cuts: 프레임 번호 i (i-1 → i 에서 장면이 바뀜),
    hold_ratio: 멈춤·검은 구간 밖에서 바로 앞 프레임과 같은 전환의 비율 (셀 애니 on twos ≈ 0.5, 24→30fps 변환 ≈ 0.2),
    drawing_fps = fps * (1 - hold_ratio) (실제로 새 그림이 바뀌는 횟수/초). 검은 화면과의 경계는 컷으로 세지 않는다.
    멈춤 = 첫 프레임과 같은 그림(mad_ref < FREEZE_MAD)이 이어지는 묶음이 freeze_min_s 이상 (ffmpeg freezedetect 처럼 첫 프레임의
    시작부터 마지막 같은 프레임의 시작까지, 곧 같은 전환 round(freeze_min_s*fps) 개 이상).
    timestamps(read_frames 의 시각)를 주면 묶음 길이를 화면에 떠 있는 시간으로 잰다: 표시 시간(다음 다른 프레임까지,
    마지막 프레임은 1/fps)을 프레임 수로 반올림한 값 - 1. 그래서 VFR·GIF 에서 한 프레임이 오래 떠 있는 정지 화면
    (리샘플하면 같은 프레임이 연달아 들어감)도 멈춤이 되고 clean_segments 가 뺀다. CFR 시각이면 주지 않았을 때와 같다."""
    blk = np.asarray(sig["black"]) >= BLACK_PIC
    T = len(blk)
    ev = dict(black=_runs(blk), freeze=[], cuts=[], hold_ratio=float("nan"), drawing_fps=float("nan"))
    if "mad" not in sig:
        return ev
    still = sig["mad"] < FREEZE_MAD               # 홀드: 바로 앞 프레임과 같음
    valid = ~(blk[:-1] & blk[1:])                 # 검은 프레임끼리의 전환은 멈춤·홀드로 세지 않는다
    same = (np.asarray(sig.get("mad_ref", sig["mad"])) < FREEZE_MAD) & valid
    g0 = np.flatnonzero(np.concatenate([[True], ~same]))             # 같은 그림 묶음 [g0, g1) 의 시작 프레임
    g1 = np.append(g0[1:], T)
    length = (g1 - g0 - 1).astype(np.float64)                        # 묶음 안 전환 수
    if timestamps is not None and fps > 0:
        t = np.asarray(timestamps, np.float64)
        if t.shape != (T,):
            raise ValueError(f"timestamps 길이 {t.shape} 가 프레임 수 {T} 와 다름")
        end = np.append(t[1:], t[-1] + 1.0 / fps)                    # 프레임이 화면에서 내려가는 시각
        length = np.floor((end[g1 - 1] - t[g0]) * fps + 0.5) - 1     # 표시 시간을 프레임 수로 (CFR 이면 위와 같음)
    need = max(1, round(freeze_min_s * fps))
    for a, b, n in zip(g0, g1, length):
        if n >= need and not blk[a:b].all():
            ev["freeze"].append((int(a), int(b)))
            valid[a:b - 1] = False                # 멈춘 묶음 안의 전환은 홀드 비율에서 뺀다
    if valid.any():
        ev["hold_ratio"] = float(still[valid].mean())
        ev["drawing_fps"] = float(fps) * (1.0 - ev["hold_ratio"])
    c = sig["content"]
    pad = np.pad(c, 2, mode="edge")
    neigh = (pad[:-4] + pad[1:-3] + pad[3:-1] + pad[4:]) / 4.0
    # 히스토그램·이웃 대비 기준은 HSV 차가 CUT_MIN 이상일 때만 쓴다: 평평한 셀화·그라데이션·GIF 팔레트 변화는
    # 히스토그램이 뾰족해 움직임만으로 상관이 0.8 아래로 떨어진다 (합성 GIF 0.44, yuv444 첫 전환 0.75 에서 확인)
    assist = (sig["hist_corr"] <= 1.0 - CUT_HIST) | (c / np.maximum(neigh, 1e-6) >= CUT_RATIO)
    cut = (c >= CUT_CONTENT) | ((c >= CUT_MIN) & assist)
    last, gap = -10 ** 9, max(1, round(min_scene_s * fps))
    for i in np.flatnonzero(cut & ~(blk[:-1] | blk[1:])):     # 검은 화면과의 경계는 black 구간이 맡는다
        if i + 1 - last >= gap:                   # 장면은 min_scene_s 이상
            ev["cuts"].append(int(i + 1))
            last = i + 1
    return ev


def clean_segments(n_frames, events):
    """컷으로 나누고 검은·멈춘 프레임을 뺀 깨끗한 구간 [(시작, 끝)) 목록. 학습 창은 이 안에서만 고른다."""
    keep = np.ones(n_frames, bool)
    for a, b in list(events.get("black", [])) + list(events.get("freeze", [])):
        keep[a:b] = False
    bounds = [0] + sorted(c for c in events.get("cuts", []) if 0 < c < n_frames) + [n_frames]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out += [(a + s, a + e) for s, e in _runs(keep[a:b])]
    return out


def summarize_clip(frames, fps, sig=None, events=None, timestamps=None):
    """EDA 표의 클립 요약값. subject_motion·camera_motion 은 깨끗한 구간 안에서 잰 평균 움직임 (초당 화면 긴 변의 비율):
    이웃 프레임, motion_s(약 1/16초), motion_k(약 0.5초) 간격 비교 평균 가운데 가장 큰 값 (빠른 움직임은 이웃, 30·60fps 의
    중간 속도는 1/16초, 1px 미만의 느린 움직임은 0.5초 간격이 맞다). static = 클립인데 subject_motion < STATIC_MOTION.
    blur 는 프레임 중앙값, sat(0~255)·mean_rgb(0~255) 는 평균. 1프레임(이미지)도 받는다: 움직임 0, static False,
    hold_ratio·drawing_fps 는 nan. timestamps 는 events 를 주지 않을 때 detect_events 에 넘긴다."""
    sig = frame_signals(frames, fps) if sig is None else sig
    ev = detect_events(sig, fps, timestamps=timestamps) if events is None else events
    T = len(sig["black"])
    out = dict(subject_motion=0.0, camera_motion=0.0, hold_ratio=ev["hold_ratio"], drawing_fps=ev["drawing_fps"],
               blur=float(np.median(sig["blur"])), n_cuts=len(ev["cuts"]), n_black=sum(b - a for a, b in ev["black"]),
               n_freeze=sum(b - a for a, b in ev["freeze"]), sat=float(np.mean(sig["sat"])),
               mean_rgb=tuple(round(float(v), 1) for v in np.mean(sig["rgb"], 0)))
    if T > 1:
        steps = [("", 1)] + [(sfx, sig[key]) for sfx, key in (("_s", "motion_s"), ("_k", "motion_k")) if key in sig]
        ok = {sfx: np.zeros(max(T - st, 0), bool) for sfx, st in steps}
        for a, b in clean_segments(T, ev):        # 비교하는 두 프레임이 같은 깨끗한 구간 안에 있을 때만
            for sfx, st in steps:
                ok[sfx][a:max(a, b - st)] = True
        for key, name in (("subject_motion", "motion_p95"), ("camera_motion", "motion_med")):
            m = [sig[name + sfx][ok[sfx]].mean() for sfx, _ in steps if ok[sfx].any()]
            out[key] = float(max(m, default=0.0) * fps / sig["side"])
    out["static"] = bool(T > 1 and out["subject_motion"] < STATIC_MOTION)
    return out


# ------------------------------------------------------------------ 중복·캡션
_DCT = {}


def _phash(rgb, hs=8, hf=4):
    """imagehash.phash 와 같은 64비트 (32x32 회색 → 정규화 안 한 DCT-II → 왼쪽 위 8x8 이 중앙값보다 큰지)"""
    n = hs * hf
    if n not in _DCT:
        k, i = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        _DCT[n] = 2.0 * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    g = np.asarray(Image.fromarray(rgb).convert("L").resize((n, n), Image.Resampling.LANCZOS), np.float64)
    low = (_DCT[n] @ g @ _DCT[n].T)[:hs, :hs]
    return (low > np.median(low)).ravel()


def clip_signature(frames, k=8, skip_first=False):
    """같은 상대 위치(0~1) k 프레임의 pHash (k,64). fps·길이·재인코딩이 달라도 같은 클립이면 가깝다.
    같은 시작 이미지로 만든 I2V 클립끼리는 0번 프레임이 같으므로 skip_first=True 로 비교한다."""
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return np.stack([_phash(_u8(frames[i])) for i in (idx[1:] if skip_first else idx)])


def signature_distance(a, b):
    """같은 위치끼리의 평균 해밍 거리 (64비트 중). 같은 클립 0, 재인코딩본 ~4, 다른 클립 20 이상, DUP_BITS 이하면 중복 의심."""
    n = min(len(a), len(b))
    return float((a[:n] != b[:n]).sum(1).mean())


# 동작 = 움직이는 동사 형태만. 정지 상태(standing, sitting, holding, looking at, pointed ears, eyes closed, open field,
# close-up, wavy hair, bow tie, lower body)는 동작이 아니다. 영어는 단어 경계, 한국어는 어간(뒤에 어미가 붙음)으로 찾는다.
_MOTION_EN = (
    r"walk\w*|run(?:s|ning)?|ran|jog\w*|sprint\w*|hurr(?:y|ies|ying|ied)|rush(?:es|ed|ing)|wander\w*|"
    r"turn\w*|spin(?:s|ning)?|spun|twirl\w*|rotat\w*|wav(?:e|es|ed|ing)|nod(?:s|ded|ding)?|shak(?:e|es|ing)|shook|"
    r"blink\w*|wink\w*|smil(?:es|ed)|laugh\w*|giggl\w*|cr(?:y|ies|ied|ying)|yawn\w*|sigh(?:s|ed|ing)|"
    r"glanc\w*|look(?:s|ed)? (?:around|up|down|back|away|over)|looking around|"
    r"rais(?:e|es|ing)|lower(?:s|ing)|lift(?:s|ed|ing)|reach(?:es|ed|ing)|point(?:s|ing)|gestur(?:e|es|ed|ing)|clap\w*|"
    r"(?:sits?|sitting|sat) (?:down|up)|(?:stands?|standing|stood|gets?|getting|got) up|kneels(?: down)?|knelt|kneeling down|"
    r"jump\w*|hop(?:s|ped|ping)?|skip(?:s|ped|ping)|leans|leaning (?:forward|in|over|back)|"
    r"(?:steps?|stepped|stepping) (?:forward|back|backward|aside|closer|in|out|up|down|onto|off|toward|towards|into|over)|"
    r"bowing|bows down|tilt(?:s|ing)|sway\w*|bounc\w*|open(?:s|ing)|clos(?:es|ing)|grab\w*|throw\w*|threw|toss\w*|"
    r"catch(?:es|ing)|caught|push(?:es|ed|ing)|pull(?:s|ed|ing)|kick\w*|punch\w*|hug(?:s|ged|ging)|stretch(?:es|ed|ing)|"
    r"danc\w*|climb\w*|fall(?:s|ing)|fell|slid(?:e|es|ing)|swinging|swung|swings (?:her|his|its|their|back|forward|around)|"
    r"swim(?:s|ming)?|swam|crawl\w*|"
    r"fl(?:y|ies|ying|ew)|float\w*|drift\w*|roll(?:s|ing)|flutter\w*|blow(?:s|ing)|ripple\w*|flap\w*|wag(?:s|ged|ging)|"
    r"wiggl\w*|twitch\w*|mov(?:e|es|ed|ing)|approach\w*|enter(?:s|ed|ing)|exit(?:s|ed|ing)|leav(?:e|ing)|"
    r"eat(?:s|ing)|drink(?:s|ing)|chew\w*|talk(?:s|ed|ing)|speak(?:s|ing)|sing(?:s|ing)|transform\w*")
_MOTION_KO = (
    r"걷는|걷고|걷다|걸어가|걸어오|걸어간|걸어온|걸음을|달린다|달리는|달리고|달려가|달려오|달려간|달려온|뛴다|뛰는|뛰고|뛰며|"
    r"뛰어(?!난)|점프|춤|움직|흔들|흔든|끄덕|깜빡|깜박|윙크|돌아보|돌아서|돌아선|뒤돌|돌린|돌며|빙글|회전|웃는다|웃으며|미소 ?짓|"
    r"일어나|일어서|일어선|앉는다|앉으며|날아|날갯짓|떨어지|떨어진다|기울이|기울인|다가오|다가가|다가서|다가온|다가간|"
    r"들어 ?올리|들어 ?올린|내려놓|던지|던진|잡는다|잡으며|펄럭|흩날|휘날|나부끼|미끄러|굴러|구른다|튀어 ?오르|튄다|흘러|"
    r"떠다니|손짓|하품|박수|손을 흔|고개를 (?:끄덕|돌|숙이|젓|기울|든다|들어)")
# 카메라·샷 = 고정 카메라, 카메라 움직임('the camera slowly pans', 'camera is panning'), 샷 크기·앵글(close-up·closeup·close up,
# long shot, full body, from above …)
_STATIC_CAM_EN = (r"(?:static|fixed|locked(?:[- ]off)?|stationary|still|tripod)[- ](?:camera|shot)"
                  r"|camera (?:is|stays|remains|holds) (?:\w+ )?(?:static|still|fixed|locked|stationary|steady)(?! \w+ing\b)"
                  r"|camera does(?: not|n['’]t) move|no camera (?:movement|motion)")
_STATIC_CAM_KO = r"(?:고정(?:된)?|정지(?:된)?) ?카메라|카메라(?:는|가|도)? ?(?:고정|정지|움직이지 않)"
_CAMERA_EN = (
    _STATIC_CAM_EN
    + r"|camera (?:is |then |also |still |now |\w+ly ){0,2}(?:pan|tilt|zoom|push|pull|track|follow|mov|orbit|rotat|dolly|truck|crane|"
      r"circl|ris|descend|ascend|shak|sway|drift|glid|swing|swoop|hold|stay|remain)\w*"
    + r"|(?:pan(?:s|ning)?|tilt(?:s|ing)?|zoom(?:s|ing)?|dolly(?:ing)?) (?:in|out|left|right|up|down|forward|back|away)"
    + r"|(?:extreme |medium |big )?close[- ]?ups?"
    + r"|(?:wide|full|long|medium|establishing|tracking|aerial|overhead|cowboy|bust|two|pov|point[- ]of[- ]view|low[- ]angle|"
      r"high[- ]angle|dutch[- ]angle|bird'?s[- ]eye|over[- ]the[- ]shoulder|first[- ]person|drone|insert|static)[- ]shots?"
    + r"|(?:upper|full|lower)[- ]body|full[- ]length|hand[- ]?held|(?:low|high|dutch|eye[- ]level)[- ]angle"
    + r"|(?:bird'?s[- ]eye|aerial|overhead|top[- ]down) view|from (?:above|below|behind)|pov")
_CAMERA_KO = (
    _STATIC_CAM_KO
    + r"|카메라(?:가|는|도) [^,.;]{0,20}?(?:이동|움직|따라|돌|회전|다가|멀어|올라|내려|흔들|패닝|팬|줌|틸트|당기|밀|비추|비춘)"
    + r"|카메라 ?(?:이동|워크|무빙|움직임|패닝|줌|틸트)"
    + r"|클로즈 ?업|근접 ?(?:촬영|샷)|(?:미디엄|바스트|풀|전신|와이드|롱|트래킹|오버 ?더 ?숄더|시점|부감|항공|드론) ?샷"
    + r"|상반신|하반신|전신|패닝|줌 ?(?:인|아웃)|틸트 ?(?:업|다운)|부감|앙각|하이 ?앵글|로우 ?앵글|핸드 ?헬드")
MOTION_RE = re.compile(r"\b(?:" + _MOTION_EN + r")\b|" + _MOTION_KO, re.I)
# 부정한 움직임('does not move', 'without moving', '움직이지 않는다')은 동작이 아니다
_NEG_MOTION = re.compile(r"\b(?:does(?: not|n['’]t)|do(?: not|n['’]t)|did(?: not|n['’]t)|not|never|without|no) "
                         r"(?:\w+ly |really |even )?mov(?:e|es|ed|ing)\b"
                         r"|움직이지 ?(?:않|마)|움직임(?:이|은)? ?없", re.I)
CAMERA_RE = re.compile(r"\b(?:" + _CAMERA_EN + r")\b|" + _CAMERA_KO, re.I)
_STATIC_CAM = re.compile(r"\b(?:" + _STATIC_CAM_EN + r")\b|" + _STATIC_CAM_KO, re.I)


def caption_issues(caption, trigger_re, is_clip, fixed_words=(), summary=None):
    """캡션 문제 목록 (비면 문제 없음). 영상 캡션은 트리거 + 동작(시간 순서) + 카메라·샷 표현, 고정 외형은 쓰지 않는다.
    동작·카메라 표현은 영어와 한국어의 흔한 표현을 찾는다 (MOTION_RE·CAMERA_RE, 정지 상태 낱말은 동작으로 보지 않음).
    summary(summarize_clip 결과)를 주면 EDA 값과의 모순도 본다."""
    cap, out = caption or "", []
    if not trigger_re.search(cap):
        out.append("트리거 없음")
    n = len(cap.split())
    if not CAPTION_WORDS[0] <= n <= CAPTION_WORDS[1]:
        out.append(f"단어 수 {n} (권장 {CAPTION_WORDS[0]}~{CAPTION_WORDS[1]})")
    for w in fixed_words:                      # 한글 뒤 조사는 허용 (이미지 노트북 TRIGGER_RE 와 같은 경계)
        if re.search(r"(?<!\w)" + re.escape(w) + r"(?![A-Za-z0-9_])", cap, re.I):
            out.append(f"고정 특징 단어: {w} (외형은 트리거가 배우므로 빼기)")
    if is_clip:
        moving = MOTION_RE.search(_NEG_MOTION.sub(" ", _STATIC_CAM.sub(" ", cap)))
        if not moving:
            out.append("동작 표현 없음 (예: turns, walks, waves)")
        if not CAMERA_RE.search(cap):
            out.append("카메라·샷 표현 없음 (예: static camera, camera pans left, medium shot)")
        if summary is not None:
            if _STATIC_CAM.search(cap) and summary["camera_motion"] > CAMERA_MOVING:
                out.append(f"static camera 라고 했지만 카메라가 움직임 ({summary['camera_motion']:.2f})")
            if moving and summary["subject_motion"] < STATIC_MOTION:
                out.append(f"동작을 적었지만 거의 정지된 클립 ({summary['subject_motion']:.3f})")
    return out


# ------------------------------------------------------------------ 전처리
def resample_to_fps(timestamps, target_fps):
    """격자 시각 t_k = k/target_fps 에 화면에 떠 있는 원본 프레임 번호 (ffmpeg fps 필터 round=near 와 같음): 시작 시각을
    출력 단위로 반올림한 값(반은 올림, 그래서 32fps → 16fps 는 [0,2,4,...])이 k 이하인 마지막 프레임.
    길이 = round((마지막 시각 + 원본 프레임 길이(dt 중앙값)) * target_fps) (ffmpeg 처럼 끝도 반올림).
    VFR·GIF 의 긴 프레임은 떠 있는 시간만큼, 원본 fps 가 더 낮으면 같은 프레임이 반복된다
    (예: 12fps → 16fps 는 [0,1,1,2,3,4,4,5,...]). 섞인 순서로 줘도 원래 번호를 돌려준다."""
    ts = np.asarray(timestamps, np.float64)
    order = np.argsort(ts, kind="stable")
    t = ts[order] - ts[order[0]]
    dt = np.diff(t)
    dt = dt[dt > 0]
    dur = float(np.median(dt)) if len(dt) else 1.0 / target_fps
    n = max(1, int(math.floor((t[-1] + dur) * target_fps + 0.5 + 1e-9)))
    r = np.searchsorted(t, (np.arange(n) + 0.5) / target_fps - 1e-9, side="left") - 1   # 반올림 시각 < k + 0.5 인 마지막
    return order[np.clip(r, 0, len(t) - 1)]


def round_4k1(n, temporal=4):
    """n 이하에서 가장 큰 temporal*k+1 (80 → 77, 83 → 81). Wan VAE 는 이를 넘는 프레임을 경고 없이 버린다."""
    return 0 if n < 1 else (int(n) - 1) // temporal * temporal + 1


def windows(seg_len, F, mode="uniform", n=3, rng=None):
    """길이 seg_len 구간 안 F 프레임 창의 시작 위치. head: 처음 1개, uniform: 고르게 n 개, chunk: 겹치지 않게 전부,
    slide: F//2 간격, random: 무작위 n 개 (에폭마다 다시 부르면 다른 창 → I2V 는 다른 시작 이미지 증강)."""
    if F < 1 or seg_len < F:
        return []
    last = seg_len - F
    if mode == "head":
        return [0]
    if mode == "uniform":
        return sorted(set(np.linspace(0, last, n).round().astype(int).tolist()))
    if mode == "chunk":
        return list(range(0, last + 1, F))
    if mode == "slide":
        return list(range(0, last + 1, max(1, F // 2)))
    if mode == "random":
        rng = np.random.default_rng() if rng is None else rng
        return sorted(int(x) for x in rng.choice(last + 1, size=min(n, last + 1), replace=False))
    raise ValueError(f"모르는 창 방식: {mode}")


def make_buckets(token_budget, frame_options, aspects, max_area, token_div, temporal, multiple):
    """(F, H, W, tokens) 목록. tokens = (1+(F-1)//temporal) * (H//token_div) * (W//token_div) ≤ token_budget,
    H*W ≤ max_area, H·W 는 multiple 의 배수. aspects 는 W/H (예: 832/480, 480/832)."""
    out = set()
    for F in frame_options:
        if F < 1 or (F - 1) % temporal:
            raise ValueError(f"프레임 수는 {temporal}k+1 이어야 함: {F}")
        lat_f = 1 + (F - 1) // temporal
        area = min(max_area, token_budget / lat_f * token_div * token_div)
        for a in aspects:
            w = int(math.sqrt(area * a) + 1e-6) // multiple * multiple
            h = int(math.sqrt(area / a) + 1e-6) // multiple * multiple
            if w >= multiple and h >= multiple:
                out.add((F, h, w, lat_f * (h // token_div) * (w // token_div)))
    return sorted(out)


def pick_bucket(buckets, avail_frames, w, h, prefer="frames", allow_upscale=1.0):
    """원본 w x h, 쓸 수 있는 프레임 avail_frames 에 맞는 (F, H, W, tokens). 없으면 None.
    덮도록 키우는 배율 max(W/w, H/h) 가 allow_upscale 이하인 것 가운데 비율이 가장 가까운 것과 ASPECT_TOL 안의 것을 같은
    비율로 보고 (토큰 예산 때문에 줄어든 버킷은 배수 맞춤으로 비율이 몇 % 어긋남), prefer='frames' 면 F, 'area' 면 넓이가 큰 것."""
    la = math.log(w / h)
    cand = [b for b in buckets if b[0] <= avail_frames and max(b[2] / w, b[1] / h) <= allow_upscale + 1e-9]
    if not cand:
        return None
    err = {b: round(abs(math.log(b[2] / b[1]) - la), 9) for b in cand}
    best = min(err.values())
    cand = [b for b in cand if err[b] <= best + ASPECT_TOL]
    if prefer == "frames":
        return max(cand, key=lambda b: (b[0], -err[b], b[1] * b[2]))
    return max(cand, key=lambda b: (b[1] * b[2], -err[b], b[0]))


def resize_crop_frames(frames, h, w):
    """모든 프레임에 같은 배율·같은 가운데 자르기 (프레임마다 다르면 흔들림을 배운다). 줄일 때 LANCZOS, 키울 때 BICUBIC."""
    frames = _u8(frames)
    h0, w0 = frames.shape[1:3]
    s = max(w / w0, h / h0)
    bw, bh = min(w0, w / s), min(h0, h / s)       # 원본에서 잘라 쓸 크기 (w/(w/w0) 가 부동소수 오차로 w0 를 넘지 않게)
    box = ((w0 - bw) / 2, (h0 - bh) / 2, (w0 + bw) / 2, (h0 + bh) / 2)
    rs = Image.Resampling.LANCZOS if s < 1 else Image.Resampling.BICUBIC
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), rs, box=box)) for f in frames])


# ------------------------------------------------------------------ 보기·저장
@lru_cache(None)
def _font(size):
    """한글이 보이는 글꼴: 노트북 0-1 셀이 찾은 글꼴 파일(환경변수 LORA_KO_FONT), Windows 맑은 고딕, Linux 나눔·Noto CJK
    (Pillow 는 ~/.local/share/fonts 도 파일 이름으로 찾는다), 없으면 기본 글꼴"""
    for name in (os.environ.get("LORA_KO_FONT", ""), "malgun.ttf", "NanumGothic.ttf", "NotoSansCJK-Regular.ttc",
                 "NotoSansKR-Regular.ttf", "AppleSDGothicNeo.ttc", "DejaVuSans.ttf"):
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(rows, labels=None, n=8, thumb_h=120, pad=4):
    """행마다 클립 1개(예: 체크포인트별 같은 시드 결과)에서 고르게 n 장을 뽑아 붙인 PIL.Image. labels 는 행 이름."""
    thumbs = []
    for clip in rows:
        clip = _u8(clip)
        clip = clip[None] if clip.ndim == 3 else clip
        tw = max(1, round(clip.shape[2] * thumb_h / clip.shape[1]))
        idx = np.linspace(0, len(clip) - 1, min(n, len(clip))).round().astype(int)
        thumbs.append([Image.fromarray(clip[i]).resize((tw, thumb_h), Image.Resampling.LANCZOS) for i in idx])
    font = _font(max(10, min(18, thumb_h // 6)))
    lab = [str(x)[:40] for x in labels] if labels is not None else []
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lab_w = int(max(measure.textlength(s, font=font) for s in lab)) + 2 * pad if lab else 0
    tw, cols = max(t[0].width for t in thumbs), max(len(t) for t in thumbs)
    sheet = Image.new("RGB", (lab_w + pad + cols * (tw + pad), pad + len(thumbs) * (thumb_h + pad)), "white")
    d = ImageDraw.Draw(sheet)
    for r, row in enumerate(thumbs):
        y = pad + r * (thumb_h + pad)
        if lab:
            d.text((pad, y + (thumb_h - getattr(font, "size", 10)) // 2), lab[r], fill="black", font=font)
        for k, im in enumerate(row):
            sheet.paste(im, (lab_w + pad + k * (tw + pad), y))
    return sheet


def _encode(path, frames, fps, codec, pix_fmt, options=None, pts_ms=None, rotation=0):
    """PyAV 로 영상 쓰기. 프레임은 (H,W,3) RGB 또는 (H,W,4) RGBA. pts_ms 를 주면 1ms 단위 시각(VFR, 음수면 mp4 편집 목록이
    생겨 그 프레임은 재생·디코딩에서 빠진다), rotation 은 표시 회전(반시계)."""
    h, w = frames[0].shape[:2]
    with av.open(str(path), "w") as c:
        st = c.add_stream(codec, rate=Fraction(fps).limit_denominator(1001))
        st.width, st.height, st.pix_fmt = w, h, pix_fmt
        st.options = dict(options or {})
        if pts_ms is not None:
            st.codec_context.time_base = st.time_base = Fraction(1, 1000)
        if rotation:
            st.set_display_rotation(rotation)
        for i, a in enumerate(frames):
            a = np.ascontiguousarray(_u8(a))
            f = av.VideoFrame.from_ndarray(a, format="rgba" if a.shape[-1] == 4 else "rgb24")
            if pts_ms is not None:
                f.pts, f.time_base = int(pts_ms[i]), Fraction(1, 1000)
            for pk in st.encode(f):
                c.mux(pk)
        for pk in st.encode():
            c.mux(pk)
    return str(path)


def save_mp4(path, frames, fps, crf=18):
    """H.264(yuv420p) mp4 (노트북에서는 IPython.display.Video(path, embed=True) 로 본다).
    홀수 크기는 가장자리를 복제해 짝수로 맞춘다 (libx264 yuv420p 는 홀수에서 'Generic error')."""
    frames = _u8(frames)
    frames = frames[None] if frames.ndim == 3 else frames
    H, W = frames.shape[1:3]
    if H % 2 or W % 2:
        frames = np.pad(frames, ((0, 0), (0, H % 2), (0, W % 2), (0, 0)), mode="edge")
    if "libx264" in av.codecs_available:
        return _encode(path, frames, fps, "libx264", "yuv420p", {"crf": str(crf), "preset": "medium"})
    return _encode(path, frames, fps, "mpeg4", "yuv420p", {"b": "8M"})       # libx264 가 없는 FFmpeg 빌드


def save_webp(path, frames, fps):
    """움직이는 WebP 미리보기 (GIF 보다 작고 색이 온전함). 프레임 길이는 누적 반올림(16fps → 62,63,63,62ms)이라
    전체 재생 시간이 정확하다. 같은 프레임이 이어지면 Pillow 가 합치지만 재생 시간은 그대로다."""
    frames = _u8(frames)
    ims = [Image.fromarray(f) for f in (frames[None] if frames.ndim == 3 else frames)]
    ms = np.diff(np.round(np.arange(len(ims) + 1) * 1000.0 / fps)).astype(int).tolist()
    ims[0].save(path, format="WEBP", save_all=True, append_images=ims[1:], duration=ms, loop=0, quality=80, method=4)
    return str(path)


# ------------------------------------------------------------------ 시험 데이터
def _is_red(a, box, inset=3):
    y0, y1, x0, x1 = box
    m = a[y0 + inset:y1 - inset, x0 + inset:x1 - inset].reshape(-1, 3).mean(0)
    return bool(m[0] > 170 and m[1] < 90 and m[2] < 90)


def check_test_expectations(df, events, dup_pairs, exp):
    """노트북 TEST 모드: make_test_dataset 기대값과 EDA 결과를 대조한다 (틀리면 AssertionError).
    df 열: file, kind, width, height, frames, vfr, rotation, subject_motion (+ 있으면 decoded, has_alpha, has_caption,
    caption_issues). 움직이는 WebP 처럼 확장자가 이미지인 클립도 kind='video' 행이어야 한다 (is_animated_image 로 나눔).
    events = {클립: detect_events(sig, fps, timestamps=read_frames 시각) 결과} (VFR 긴 정지는 시각을 줘야 멈춤으로 잡힘),
    dup_pairs = [(a, b, 거리), ...].
    회전 방향(빨간 표시가 네 모서리 중 기대한 곳에만)·투명 → 흰색·16비트 밝기는 exp["root"] 의 파일을 다시 읽어 화소로 본다."""
    r, root = df.set_index("file"), Path(exp["root"])
    for n in exp["images"] + exp["clips"]:
        assert n in r.index, f"{n}: EDA 표에 없음"
        assert (int(r.loc[n, "height"]), int(r.loc[n, "width"])) == tuple(exp["size"][n]), f"{n}: 표시 크기가 기대와 다름"
        want = "image" if n in exp["images"] else "video"
        assert r.loc[n, "kind"] == want, (f"{n}: {r.loc[n, 'kind']} 로 읽음 (기대 {want}"
                                          + (", 움직이는 이미지는 is_animated_image 로 클립 경로에)" if n in exp.get("animated", []) else ")"))
    for n in exp["clips"]:
        x, ev = r.loc[n], events[n]
        want = exp["events"].get(n, dict(black=[], freeze=[], cuts=[]))
        hint = " (편집 목록 밖 패킷까지 셈)" if n in exp.get("discard", {}) else ""
        assert int(x["frames"]) == exp["frames"][n], f"{n}: 프레임 수 {x['frames']} != {exp['frames'][n]}{hint}"
        assert "decoded" not in r.columns or int(x["decoded"]) == int(x["frames"]), f"{n}: 패킷 수와 디코딩 수가 다름"
        assert bool(x["vfr"]) == exp["vfr"][n] and int(x["rotation"]) == exp["rotation"].get(n, 0), f"{n}: VFR·회전 판정 오류"
        assert "has_alpha" not in r.columns or bool(x["has_alpha"]) == (n in exp["alpha"]), f"{n}: 알파 판정 오류"
        assert [ev[k] for k in ("black", "freeze", "cuts")] == [want[k] for k in ("black", "freeze", "cuts")], f"{n}: 사건 {ev}"
        assert (float(x["subject_motion"]) < STATIC_MOTION) == (n in exp["static"]), f"{n}: 정지 판정 오류"
    assert {frozenset(p[:2]) for p in dup_pairs} == {frozenset(p) for p in exp["duplicates"]}, f"중복 쌍 {dup_pairs}"
    for n in exp["portrait"]:
        assert r.loc[n, "height"] > r.loc[n, "width"], f"{n}: 세로로 보이지 않음"
    for n, box in exp["red"].items():
        a = read_frames(root / n)[0][0]
        H, W = a.shape[:2]
        corner = [(0, 16, 0, 16), (0, 16, W - 16, W), (H - 16, H, 0, 16), (H - 16, H, W - 16, W)]
        assert [c for c in corner if _is_red(a, c)] == [tuple(box)], f"{n}: 회전 방향이 틀림"
    for n, (y0, y1, x0, x1) in exp["white"].items():
        assert read_frames(root / n)[0][:, y0:y1, x0:x1].min() >= 250, f"{n}: 투명한 부분이 흰색이 아님"
    lo, hi = exp["gray16_mean"]
    assert lo < load_image(root / "img_gray16.png").mean() < hi, "16비트 이미지가 하얗거나 검게 잘림"
    for n in exp["no_caption"] if "has_caption" in r.columns else []:
        assert not r.loc[n, "has_caption"], f"{n}: 캡션이 없어야 함"
    for n, words in exp["caption_issues"].items() if "caption_issues" in r.columns else []:
        assert all(w in str(r.loc[n, "caption_issues"]) for w in words), f"{n}: 캡션 문제 {words} 를 못 찾음"


def _scene(t, kind, h=96, w=160, speed=3.0, color=(245, 245, 235)):
    """kind 마다 다른 부드러운 무늬 배경 위를 좌우로 오가는 검은 테두리 사각형 (t 번째 프레임, 끝에서 튀지 않게 왕복).
    1차원 그라데이션은 pHash 계수가 0 근처에 몰려 서명이 불안정하므로 대비가 큰 2차원 무늬를 쓴다."""
    g = np.random.default_rng(100 + kind).random((3, 5)).astype(np.float32)
    g = np.asarray(Image.fromarray(g, "F").resize((w, h), Image.Resampling.BICUBIC)).clip(0, 1)[..., None]
    img = (np.array([10, 20, 60]) * (1 - g) + np.array([240, 230, 200]) * g).astype(np.uint8)
    s = h // 3
    p = (t * speed) % (2 * (w - s))
    x0, y0 = int(min(p, 2 * (w - s) - p)), h // 2 - s // 2
    img[y0:y0 + s, x0:x0 + s] = 0                 # 테두리가 있어 배경 밝기와 상관없이 움직임이 보인다
    img[y0 + 3:y0 + s - 3, x0 + 3:x0 + s - 3] = color
    return img


def make_test_dataset(root, trigger):
    """알려진 문제를 넣은 작은 시험 데이터 (이미지 4장 + 클립 14개, 161px·60프레임 이하, PyAV·Pillow 만, 1초 안팎).
    돌려주는 기대값으로 TEST 모드가 probe·read_frames·사건 검출·중복·정지·캡션 검사를 assert 한다 (예: test_video_io.py).
      images·clips: 파일 이름 (클립에는 확장자가 이미지인 움직이는 WebP 도 있다) / frames·fps·vfr: 클립별 디코딩되는 프레임 수
      (= 편집 목록 밖을 뺀 패킷 수)·중앙값 fps·VFR 여부 / header_zero: 헤더 프레임 수 0 / discard: 편집 목록 밖이라 디코더가 버리는
      패킷 수 (ffmpeg -ss … -c copy 로 자른 mp4) / animated: 확장자는 이미지지만 움직이는 파일 (클립으로 읽어야 함)
      size: 표시 크기 (h, w) / rotation·portrait: 회전 클립·세로로 보여야 할 파일 / red: 빨간 표시가 보여야 할 영역
      (y0, y1, x0, x1), 다른 모서리에 있으면 회전 방향이 틀림 / alpha: 디코딩한 화소에 알파가 있는 파일 (GIF·움직이는 WebP 는
      늘 알파 형식) / white: 투명해서 흰색이어야 할 영역 (알파 VP9 webm 은 그 아래 빨간색이 숨어 있음)
      gray16_mean: 16비트 이미지 평균 밝기 범위 / events: black·freeze·cuts·segments (read_frames 시각을 detect_events 에 준
      결과, hold.gif 는 1.2초 떠 있는 프레임이 멈춤) / duplicates: 중복 쌍 / static: 거의 정지
      no_caption: 캡션 없는 파일 / caption_issues: 파일 → caption_issues 결과에 들어 있어야 할 문구 / fixed_words, trigger
    크기·영역은 회전을 적용한 뒤 기준."""
    need = [("libx264", "w"), ("libvpx-vp9", "w"), ("libvpx-vp9", "r")]
    missing = [f"{n} {'인코더' if m == 'w' else '디코더'}" for n, m in need if not _codec_ok(n, m)]
    if missing:
        raise RuntimeError(f"시험 데이터에 필요한 코덱이 PyAV 에 없음: {missing} (PyPI 의 av 휠에는 들어 있음)")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fixed = ["black bob", "wine-red cape"]
    caps = {}
    # 이미지: 보통 png, EXIF 회전 jpg (저장은 가로, 보이는 것은 세로), 16비트 흑백 png, 왼쪽 절반이 투명한 png
    blocks = Image.fromarray(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8))
    blocks.resize((128, 96), Image.Resampling.NEAREST).save(root / "img_plain.png")
    a = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    a[:16, :16] = (255, 0, 0)                     # 저장 기준 왼쪽 위 → 보이는 기준 오른쪽 위
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.fromarray(a).save(root / "img_exif.jpg", exif=exif.tobytes(), quality=95)
    Image.fromarray((np.linspace(0, 65535, 64)[None, :].repeat(80, 0)).astype(np.uint16)).save(root / "img_gray16.png")
    a = rng.integers(0, 256, (96, 96, 4), dtype=np.uint8)
    a[..., 3] = 255
    a[:, :48, 3] = 0
    Image.fromarray(a, "RGBA").save(root / "img_alpha.png")
    caps.update({"img_plain.png": f"{trigger}, standing in a sunny park, full shot",
                 "img_exif.jpg": f"{trigger}, seen from the side, close-up portrait",
                 "img_gray16.png": f"{trigger}, grayscale sketch of the character, full shot",
                 "img_alpha.png": f"{trigger}, character cutout on a transparent background"})
    # 클립
    base = [_scene(t, 0) for t in range(48)]
    _encode(root / "cfr24.mp4", base, 24, "libx264", "yuv420p")
    fr, ts = read_frames(root / "cfr24.mp4")      # 거의 같은 클립: 축소·30fps·저화질 재인코딩
    idx = resample_to_fps(ts, 30)
    _encode(root / "cfr24_dup.mp4", resize_crop_frames(fr[idx], 76, 128), 30, "libx264", "yuv420p", {"crf": "30"})
    pts = np.concatenate([[0], np.cumsum([33, 66, 100] * 10)[:-1]])       # 간격 33/66/100ms 반복
    yel = (250, 230, 60)
    _encode(root / "vfr_mp4.mp4", [_scene(t, 5) for t in range(30)], 30, "libx264", "yuv420p", pts_ms=pts)
    _encode(root / "vfr_mkv.mkv", [_scene(t, 6, color=yel) for t in range(30)], 30, "libx264", "yuv420p", pts_ms=pts)
    rot = [_scene(t, 7) for t in range(36)]
    for f in rot:
        f[:16, :16] = (255, 0, 0)                 # 저장 기준 왼쪽 위 → 반시계 90° 표시 기준 왼쪽 아래
    _encode(root / "rot90 세로.mp4", rot, 24, "libx264", "yuv420p", rotation=90)
    _encode(root / "vp9.webm", [_scene(t, 4) for t in range(33)], 24, "libvpx-vp9", "yuv420p",
            {"deadline": "realtime", "cpu-used": "8"})
    _encode(root / "odd_444.mp4", [_scene(t, 1, h=97, w=161) for t in range(25)], 16, "libx264", "yuv444p")
    gif = []
    for t in range(12):
        g = np.dstack([_scene(t, 8, speed=4), np.full((96, 160), 255, np.uint8)])
        g[:16, :16, 3] = 0                        # 투명 → 흰색으로 읽혀야 함
        gif.append(Image.fromarray(g, "RGBA"))
    gif[0].save(root / "anim.gif", save_all=True, append_images=gif[1:], duration=[40, 40, 80, 80, 120, 120] * 2,
                loop=0, disposal=2)
    b = [_scene(t, 3, speed=4, color=yel) for t in range(24, 40)]   # 장면 B (t = 24~39)
    ev = [np.zeros((96, 160, 3), np.uint8)] * 3 + [_scene(t, 2) for t in range(3, 24)]   # 검은 0~2, 장면 A 3~23
    ev += b[:8] + [b[7]] * 9 + b[8:]              # 컷 24, B 24~31, 31~40 멈춤, 41~48 이어서 움직임
    _encode(root / "events.mp4", ev, 16, "libx264", "yuv420p")
    still = _scene(0, 9, speed=0).astype(np.float32)   # 거의 정지: 그림은 그대로, 약한 잡음만 (멈춤으로 잡히지 않게)
    noisy = [np.clip(still + rng.normal(0, 3.0, still.shape), 0, 255).astype(np.uint8) for _ in range(40)]
    _encode(root / "static.mp4", noisy, 24, "libx264", "yuv420p", {"crf": "10"})
    # 재인코딩 없이 자른 클립(ffmpeg -ss … -c copy)과 같은 mp4 편집 목록: 앞 4프레임은 시각이 음수라 디코더가 버린다
    _encode(root / "trim_copy.mp4", [_scene(t, 10, speed=2) for t in range(36)], 24, "libx264", "yuv420p",
            pts_ms=(np.arange(36) - 4) * 1000 / 24)
    va = [np.dstack([_scene(t + 8, 11, speed=4), np.full((96, 160), 255, np.uint8)]) for t in range(24)]
    for a in va:
        a[:, :48] = (200, 40, 40, 0)              # 투명한 왼쪽 48px 아래 숨은 빨간색 (기본 vp9 디코더는 알파를 버려 빨갛게 나옴)
    _encode(root / "vp9_alpha.webm", va, 24, "libvpx-vp9", "yuva420p", {"deadline": "realtime", "cpu-used": "8"})
    wp = []                                       # 움직이는 WebP (ComfyUI 'Save Animated WEBP'·save_webp 처럼 손실 압축, 16fps)
    for t in range(24):
        a = np.dstack([_scene(t, 12, speed=5), np.full((96, 160), 255, np.uint8)])
        a[:16, :16] = (200, 40, 40, 0)            # 왼쪽 위 투명
        wp.append(Image.fromarray(a, "RGBA"))
    wp[0].save(root / "anim_loop.webp", format="WEBP", save_all=True, append_images=wp[1:], loop=0, quality=80, method=4,
               duration=np.diff(np.round(np.arange(25) * 1000 / 16)).astype(int).tolist())
    hold = [Image.fromarray(_scene(t, 13, speed=5)) for t in range(16)]   # VFR 긴 정지: 8번 프레임이 1.2초 떠 있음
    hold[0].save(root / "hold.gif", save_all=True, append_images=hold[1:], duration=[80] * 8 + [1200] + [80] * 7, loop=0)
    shot = f"{trigger}, medium shot, static camera,"
    caps.update({"cfr24.mp4": f"{shot} a white box slides left and right",
                 "cfr24_dup.mp4": f"{shot} a white box slides left and right",
                 "vfr_mp4.mp4": f"{trigger}, wide shot, static camera, a white box moves across a blotchy wall",
                 "rot90 세로.mp4": f"{trigger}, full shot, static camera, a white box slides down the tall frame",
                 "vp9.webm": "a white box bounces across the frame, static camera, wide shot",                 # 트리거 없음
                 "odd_444.mp4": f"{trigger} with a black bob and wine-red cape walks to the left, static camera, medium shot",
                 "anim.gif": f"{trigger}, close-up, static camera, a white box jumps in uneven steps",
                 "events.mp4": f"{shot} a white box slides right, then the scene changes to a yellow box",
                 "static.mp4": f"{trigger}, close-up, static camera, she turns her head and waves",          # 동작 vs 정지
                 "trim_copy.mp4": f"{trigger}, long shot, locked-off camera, a white box drifts to the right",
                 "vp9_alpha.webm": f"{trigger}, full body, the camera stays still while a white box bounces",
                 "anim_loop.webp": f"{trigger}, 클로즈업, 고정 카메라, 흰 상자가 오른쪽으로 미끄러진다",             # 한국어 캡션
                 "hold.gif": f"{trigger}, medium close-up, the camera is static, a white box stops and then moves on"})
    for name, text in caps.items():               # vfr_mkv.mkv 는 캡션 없음 (캡션은 같은 이름 .txt 라 이름 앞부분이 겹치면 안 됨)
        (root / name).with_suffix(".txt").write_text(text, encoding="utf-8")
    clips = ["cfr24.mp4", "cfr24_dup.mp4", "vfr_mp4.mp4", "vfr_mkv.mkv", "rot90 세로.mp4", "vp9.webm", "odd_444.mp4", "anim.gif",
             "events.mp4", "static.mp4", "trim_copy.mp4", "vp9_alpha.webm", "anim_loop.webp", "hold.gif"]
    return dict(
        root=str(root), trigger=trigger, fixed_words=fixed,
        images=["img_plain.png", "img_exif.jpg", "img_gray16.png", "img_alpha.png"], clips=clips,
        frames={"cfr24.mp4": 48, "cfr24_dup.mp4": len(idx), "vfr_mp4.mp4": 30, "vfr_mkv.mkv": 30, "rot90 세로.mp4": 36,
                "vp9.webm": 33, "odd_444.mp4": 25, "anim.gif": 12, "events.mp4": 49, "static.mp4": 40, "trim_copy.mp4": 32,
                "vp9_alpha.webm": 24, "anim_loop.webp": 24, "hold.gif": 16},
        fps={"cfr24.mp4": 24, "cfr24_dup.mp4": 30, "vfr_mp4.mp4": 1000 / 66, "vfr_mkv.mkv": 1000 / 66, "rot90 세로.mp4": 24,
             "vp9.webm": 24, "odd_444.mp4": 16, "anim.gif": 12.5, "events.mp4": 16, "static.mp4": 24, "trim_copy.mp4": 24,
             "vp9_alpha.webm": 24, "anim_loop.webp": 16, "hold.gif": 12.5},
        vfr={c: c in ("vfr_mp4.mp4", "vfr_mkv.mkv", "anim.gif", "hold.gif") for c in clips},
        header_zero=["vfr_mkv.mkv", "vp9.webm", "vp9_alpha.webm", "anim_loop.webp"],   # 헤더 프레임 수 0 (패킷으로 세야 함)
        discard={"trim_copy.mp4": 4}, animated=["anim_loop.webp"],
        size={"img_plain.png": (96, 128), "img_exif.jpg": (96, 64), "img_gray16.png": (80, 64), "img_alpha.png": (96, 96),
              "cfr24.mp4": (96, 160), "cfr24_dup.mp4": (76, 128), "vfr_mp4.mp4": (96, 160), "vfr_mkv.mkv": (96, 160),
              "rot90 세로.mp4": (160, 96), "vp9.webm": (96, 160), "odd_444.mp4": (97, 161), "anim.gif": (96, 160),
              "events.mp4": (96, 160), "static.mp4": (96, 160), "trim_copy.mp4": (96, 160), "vp9_alpha.webm": (96, 160),
              "anim_loop.webp": (96, 160), "hold.gif": (96, 160)},
        rotation={"rot90 세로.mp4": 90}, portrait=["img_exif.jpg", "rot90 세로.mp4"],
        red={"img_exif.jpg": (0, 16, 48, 64), "rot90 세로.mp4": (144, 160, 0, 16)},       # 회전 방향 확인용 빨간 표시
        alpha=["img_alpha.png", "anim.gif", "vp9_alpha.webm", "anim_loop.webp", "hold.gif"],
        white={"img_alpha.png": (0, 96, 0, 48), "anim.gif": (0, 16, 0, 16), "vp9_alpha.webm": (0, 96, 0, 40),
               "anim_loop.webp": (0, 16, 0, 16)},
        gray16_mean=(60, 200),                    # 16비트가 하얗게 잘리지 않았는지 (평균 밝기 범위)
        events={"events.mp4": dict(black=[(0, 3)], freeze=[(31, 41)], cuts=[24], segments=[(3, 24), (24, 31), (41, 49)]),
                "hold.gif": dict(black=[], freeze=[(8, 9)], cuts=[], segments=[(0, 8), (9, 16)])},
        duplicates=[("cfr24.mp4", "cfr24_dup.mp4")], static=["static.mp4"], no_caption=["vfr_mkv.mkv"],
        caption_issues={"vp9.webm": ["트리거 없음"], "odd_444.mp4": ["고정 특징"], "static.mp4": ["거의 정지"]},
    )
