"""영상 검증 지표. 체크포인트 클립을 같은 시드·프롬프트의 베이스(0 단계) 클립과 비교한다 (numpy·PIL).

선택 모델 두 가지는 없으면 load() 가 이유를 출력하고 None 을 돌려준다: Dino (transformers), CCIP (onnxruntime).
  * motion_amount    : 긴 변 64px 휘도 (BOX 축소) 의 프레임 차 평균 mean|Y[t+1]-Y[t]| (0~1). 밝기 깜빡임에도 커진다 (참고용).
  * block_motion_p95 : 움직임. 단위: 원본 해상도 px/프레임. clip_metrics 의 motion. 전이마다 8x8 블록 변위의 상위 5% 평균
                       (VBench dynamic_degree 와 같은 통계) 을 내고, 클립 값은 잴 수 있는 전이의 평균.
                       0) 분석 휘도: 긴 변 224px 로 LANCZOS 축소 (MOTION_RESAMPLE, 줄이는 배율만큼 넓힌 앤티에일리어스 필터.
                          BOX 는 화면 주기 3~8px 망점을 접어 느린 팬을 몇 배로 읽었고, BILINEAR 는 주기 5~10px 망점에서
                          그랬다). 클립 대비 c (8x8 블록 표준편차의 98 백분위, 프레임 중앙값) 가 CONTRAST_REF 60 보다 낮으면
                          휘도에 60/c 배 (최대 CONTRAST_GMAX 4) 를 곱한다. 아래 문턱 tau·변위 벌점 lam 이 절대 휘도라 어둡거나
                          흐린 클립은 같은 속도가 느리게 읽혔다 (선화 x0.15 의 1px/프레임이 0). 보통 그림은 이득 1 이다
                          (실제 그림 46장의 가로·세로 화면 92개 가운데 89개, c 중앙값 78).
                       1) 정수 탐색: ±6px 과 그 바깥 ±8px 의 짝수 변위 (분석 px). 비용은 블록 평균을 뺀 SAD (밝기 변화를
                          움직임으로 보지 않음) + 변위 1px 마다 화소당 lam 1 휘도 (반복 무늬·가장자리의 먼 오정합 방지).
                          대비·노출 깜빡임은 화면 전체 표준편차 비 (GAIN_RANGE 0.6~1.6 안) 로 앞 프레임의 대비를 맞춰 지운다.
                       2) 1px 아래: 흐린 휘도 ([1,2,1] 이항) 에서 대칭 기울기 Lucas-Kanade 두 번. 시작점은 정수 최선 변위와 0
                          이고, 다듬은 뒤 남은 SAD + 변위 벌점이 작은 쪽을 쓴다.
                       3) 개구 문제: 블록 기울기 구조 텐서의 고윳값 비 λ2/λ1 이 작은 (선 하나뿐인) 블록은 변위의 선 방향 성분을
                          그 블록에서 정할 수 없으므로 법선 성분만 쓰고, 선 방향 성분은 이웃 3x3 의 모서리·무늬 블록 변위로 채운다.
                       4) 움직인 블록 = 0 변위 SAD 가 화소당 tau 2 휘도를 넘고, 다듬은 뒤 남은 SAD (그 블록의 잡음 바닥) 의
                          2 배를 넘는 블록. 어떤 변위로도 줄지 않는 변화 (프레임마다 다른 입자 잡음 — 화면 일부만이어도 —,
                          대비 깜빡임) 는 움직임이 아니다. 대비 이득 g 는 매끈한 (낮은 주파수) 시간 잡음까지 키우므로, 전이마다
                          밋밋한 블록 (두 프레임 8x8 표준편차의 작은 쪽이 하위 25%) 에서 블록 평균을 뺀 |프레임 차| 의 중앙값
                          n 으로 tau = max(2, min(2g, 2n)), lam = max(1, min(g, n)) 을 쓴다 (상한 2g·g 는 이득 전 휘도로 tau 2·
                          lam 1. 이득 1 클립은 n 을 재지 않고 그대로. 이 규칙 전에는 흐린 시간 잡음이 낀 어두운 정지 클립을 20px
                          넘게 읽었다). 무늬가 화면 전체에 있는 그림이 움직이면 밋밋한 블록도 바뀌므로, n 은 화면 전체 정수
                          변위 (위상 상관) 로 맞춘 뒤 블록마다 LK 한 번으로 설명하고 남은 차이로도 재서 작은 쪽을 쓴다 (잡음은
                          남고 움직임은 설명된다. 안 맞추면 무늬 빽빽한 어두운 팬의 1~3px/프레임을 0 으로 읽었다). 밋밋한 하위
                          25% 블록이 잡음 없는 고른 영역 (완전히 검은 레터박스 띠 등, 화면의 약 25% 넘게) 이면 n 이 0 이라 이
                          규칙이 꺼진다 (흐린 잡음 σ2 의 어두운 정지 클립을 14px 로 읽음).
                       5) 화면 표준편차가 한 전이에서 GAIN_RANGE 밖으로 바뀌면 (검은·흰 프레임, 페이드 끝, 대비가 크게 다른
                          장면 전환) 두 프레임을 견줄 수 없어 그 전이는 평균에서 뺀다 (예전 판: 정지 클립 가운데 검은 프레임
                          2장을 1.7px 움직임으로 읽음). 남은 전이가 하나도 없으면 (프레임마다 번갈아 오는 ±30% 넘는 대비·노출
                          깜빡임, 한 장 건너 검은 프레임) 대비 비가 GAIN_RANGE 안인 한 칸 건너 같은 위상의 두 프레임 (t, t+2)
                          을 견줘 변위를 2 로 나누고, 짝·홀 위상별 평균의 큰 쪽을 쓴다 (그런 짝도 없으면 표준편차가 0 이 아닌
                          전이를 모두 잰다). 모두 빼면 움직임이 0 이 되어 거짓 motion_loss 가, 대비가 안 맞는 두 프레임을 그대로
                          견주면 2~19 배로 부푼 값이 나왔다. 전이가 하나라도 남으면 남은 전이만 잰다.
                       합성 벤치 (vmh/bench 2,168 사례: 실제 선화 4장·합성 무늬 팬, 정지 선화 위 스프라이트 (화면의 5~40%),
                       832x480·480x832, 17·33프레임): 1~16px/프레임에서 참값의 0.9~1.4 배 (1px 쪽이 큼), 계열마다 속도에
                       단조, 정지·입자 (전체·일부·어두운 곳) 는 0.06px 아래, 24~32px/프레임 피사체는 0.85 배 이상. 벤치 밖
                       재검증: 어둡게 (x0.15·x0.3)·흐리게 (대비 x0.12·x0.25) 한 같은 팬은 원본의 0.8~1.0 배 (예전 판 0~1.0),
                       화면 주기 3~6.5px 망점 팬은 망점 없는 팬의 1.0~1.3 배 (예전 판 최대 5 배), 어두운 정지 클립 + 입자·x264
                       왕복 (기본 GOP)·페이드·검은 프레임은 0.1px 아래, 흐린 시간 잡음이 낀 어두운 정지 클립은 1px 아래.
                       한계: 화면의 5% 보다 작은 피사체는 상위 5% 에 움직이지 않는 블록이 섞여 작게 나온다 (속도에 단조는
                       그대로). 평평한 배경 위 회전·줌은 무늬가 있는 곳의 흐름만 잰다. 빠른 쪽은 분석 크기에 묶인다 (탐색 ±8
                       분석 px + 다듬기): 832x480 은 약 32~36px/프레임, 624x624 는 약 24px/프레임 위에서 포화한다. 화면
                       전체가 규칙 무늬이면 한 주기 어긋난 정수 해를 고를 수 있다 (주기 11~22px 줄무늬의 1px/프레임이 11px
                       로, 화면 주기 13~16px 망점의 6~8px/프레임이 약 22px 로. 날카로운 망점은 화면 주기 약 7~10px (분석
                       해상도의 나이퀴스트 근처) 에서도 생긴다. 화면 일부의 무늬는 괜찮다).
                       밋밋하고 큰 실루엣이 1px/프레임으로 들어오면 0.3~0.55 배로 작게 읽는다. 0 이나 255 로 잘린 곳이 많은
                       그림 (흰 종이 선화) 의 대비·노출 깜빡임은 전체 이득으로 다 지우지 못해, ±20% 까지는 정지 클립 0.4px
                       아래지만 번갈아 노출 ±26~30% 는 정지 1.8~2.4px·움직이는 클립 1.8~2.4 배로 읽고, 255 로 잘리는
                       번쩍임 프레임 (노출 x2) 은 한 장에 1~3px, 여러 장이면 9px 까지 움직임으로 읽는다 (둘 다 flicker_rel
                       로는 걸린다). 대비 이득은 클립 하나에 하나 (c 는 98 백분위의 프레임 중앙값) 다: 화면 블록의 약 2% 넘는
                       밝은 정지 요소 (달·등불·창·자막) 가 있으면 이득이 1 이 되어 어두운 부분의 느린 움직임을 예전처럼 작게
                       읽고, 어두운 프레임과 밝은 프레임이 반씩인 클립 (불 켜기·끄기) 은 어느 쪽이 많은지에 따라 값이 달라지며,
                       c 가 15 (= 60/4) 아래인 무늬 없는 부드러운 피사체는 상한에 걸려 0~0.6 배로 읽는다. 시간 잡음이 낀
                       어둡거나 흐린 클립은 문턱이 잡음만큼 (이득 전 앞 판 수준까지) 올라가 움직임을 작게 읽는다: 흐린 잡음
                       σ1 휘도 정도면 대비가 아주 낮은 (x0.12) 실제 영상은 8~17px/프레임에서도 0.5~0.6 배, 어두운 선화는
                       2px/프레임에서 0.2~0.4 배다 (motion_loss 기준 0.6 근처·아래). 강하게 압축한 (x264
                       crf 35) 어두운 클립은 압축 잡음까지 이득을 받아 최대 1.4 배로 크게 읽고 (crf 23 은 그대로), 키프레임
                       간격이 짧은 (GOP 4) crf 40 어두운 정지 클립은 3px 까지 읽을 수 있다 (노트북은 압축 전 프레임을 잰다).
                       1px/프레임 아래는 들쭉날쭉하다 (베이스가 MOTION_FLOOR 아래면 motion_ratio 는 None).
  * flicker          : 긴 변 32px 휘도 (BOX 축소). flicker_ratio = mean|Y[t+2]-2Y[t+1]+Y[t]| / mean|Y[t+1]-Y[t]|
                       (부드러운 움직임 < 1, 프레임마다 오가는 떨림·깜빡임 ≈ 2),
                       luma_flicker = 프레임 평균 휘도 m (0~255) 의 고역 성분 m[t] - (m[t-1]+m[t+1])/2 의 RMS (밝기 깜빡임).
  * contrast_flicker : 같은 32px 휘도의 프레임 표준편차 s (대비, 0~255) 의 고역 성분 s[t] - (s[t-1]+s[t+1])/2 의 RMS 인데,
                       제곱이 가장 큰 고역 표본 2개를 빼고 잰다 (표본 3개 이상일 때. 2프레임 이하는 0). 장면 전환 한 번은 s 의
                       계단 하나라 큰 표본이 2개 (±계단/2) 뿐이어서 지워지고, 2·4프레임 주기로 오가는 대비 깜빡임은 표본
                       대부분이 커서 남는다. 2~4프레임 디졸브나 두 번 장면 전환은 큰 표본이 4개 넘게 생겨 대비 깜빡임으로
                       걸릴 수 있다. 128 둘레로 대비만 오가는 깜빡임 (±5~15%) 은 평균 휘도가 거의 그대로라
                       luma_flicker 로는 안 보인다. clip_metrics 는 contrast_flicker 키로 내고, flicker_rel = max(luma_flicker·
                       contrast_flicker 를 각각 베이스 값 (하한 FLICKER_FLOOR) 으로 나눈 값). 밝기 쪽은 예전 flicker_rel 그대로,
                       깨끗한 클립의 대비 성분은 FLICKER_FLOOR 아래. 8프레임 주기처럼 느린 대비 변화는 약하게 잡고, 대비가
                       낮은 (어둡거나 흐린) 클립의 ±5~10% 깜빡임은 절대값이 FLICKER_FLOOR (1 휘도) 근처라 놓칠 수 있다.
  * color_stats      : 고르게 고른 8프레임의 평균 R·G·B 와 HSV 채도·명도 (0~255).
  * Dino             : facebook/dinov2-small pooler_output (layernorm 한 CLS), L2 정규화. letterbox 224 (기본 전처리의
                       가운데 자르기는 832x480 프레임의 양옆을 잘라 낸다), ImageNet 평균·표준편차. torchvision 없이.
  * subject_consistency: VBench subject_consistency 식. t>=1 평균 (max(0,cos(e_t,e_t-1)) + max(0,cos(e_t,e_0))) / 2.
  * style_drift      : 1 - 같은 번호 프레임 cos 평균. 트리거 없는(class word) 프롬프트를 베이스와 비교하면 그림체 변화.
  * CCIP             : deepghs/ccip_onnx ccip-caformer-24-randaug-pruned. imgutils.metrics.ccip 와 같은 계산 (흰 배경 RGB,
                       384x384 bilinear, CLIP 평균·표준편차, model_feat → model_metrics 차이 행렬). 차이 <= threshold
                       (metrics.json, 0.1785) 면 같은 캐릭터. 정체성은 DINO 가 아니라 CCIP 로 잰다 (DINO 는 구도 변화와 못 가름).
출처: VBench subject_consistency.py·temporal_flickering.py, imgutils metrics/ccip.py. 기준값은 remedy.THR.
"""
import json

import numpy as np
from PIL import Image

MOTION_SIDE = 224     # block_motion_p95 분석 해상도 (긴 변 px)
RING = 2              # 블록 정합 정수 탐색 ±6px 바깥에 ±8px 까지 짝수 변위 고리 (빠른 피사체, 1px 아래 다듬기가 사이를 메움)
MOTION_RESAMPLE = Image.Resampling.LANCZOS    # block_motion_p95 축소 필터 (줄이는 배율만큼 넓힌 앤티에일리어스 필터)
CONTRAST_REF = 60.0   # 클립 대비 기준 (분석 휘도 8x8 블록 표준편차의 98 백분위, 프레임 중앙값). 이보다 낮으면 휘도를 키운다
CONTRAST_GMAX = 4.0   # 대비 이득 상한 (아주 어두운 클립의 잡음·8비트 계단을 너무 키우지 않게)
GAIN_RANGE = (0.6, 1.6)   # 전이마다 화면 대비 비 (뒤 / 앞 표준편차): 이 안이면 앞 프레임에 곱해 맞추고, 밖이면 재지 않는다
MOTION_FLOOR = 1.0    # 베이스 움직임이 이보다 작으면 (px/프레임) motion_ratio 를 내지 않는다 (None)
FLICKER_FLOOR = 1.0   # flicker_rel 분모 하한 (휘도 0~255). 베이스가 매끄러우면 LoRA 가 1.5 를 넘어야 깜빡임
SAT_FLOOR = 10.0      # color_shift 채도 분모 하한 (회색에 가까운 베이스)


_W601 = np.array([0.299, 0.587, 0.114], np.float32)   # PIL 'L' 과 같은 휘도 가중치


def _empty(frames):
    """프레임이 없음 (None·빈 배열·빈 목록)"""
    return frames is None or len(frames) == 0


def _luma(frames, side, resample=Image.Resampling.BOX):
    """(T,H,W,3) uint8 → 긴 변이 side 이하인 휘도 (T,h,w) float32 0~255. 실수 그대로 축소 (8비트 반올림 잡음 없음).
    resample: BOX (기본, 깜빡임·motion_amount) 또는 MOTION_RESAMPLE (block_motion_p95). 프레임이 없으면 (0,1,1)."""
    if _empty(frames):
        return np.zeros((0, 1, 1), np.float32)
    out = []
    for f in frames:
        im = Image.fromarray(np.asarray(f, np.float32) @ _W601)          # mode 'F'
        s = side / max(im.size)
        if s < 1:
            im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), resample)
        out.append(np.asarray(im, np.float32))
    return np.stack(out)


def _pick(n, k):
    """0..n-1 에서 고르게 k 개 (중복 없음)"""
    return np.unique(np.linspace(0, n - 1, min(n, k)).round().astype(int))


def motion_amount(frames):
    Y = _luma(frames, 64) / 255.0
    return float(np.abs(np.diff(Y, axis=0)).mean()) if len(Y) > 1 else 0.0


def _fine(g):
    """1px 아래 다듬기용 (흐린 휘도, x 기울기, y 기울기) float32. [1,2,1]/4 분리형 흐림 (가장자리 복제) 은 선화의 날카로운
    선과 축소 계단 (aliasing) 을 눅여 선형 보간·기울기 근사가 1px 아래 변위에서도 맞게 한다."""
    p = np.pad(np.asarray(g, np.float32), 1, mode="edge")
    f = (p[:-2] + 2 * p[1:-1] + p[2:]) * 0.25
    f = (f[:, :-2] + 2 * f[:, 1:-1] + f[:, 2:]) * 0.25
    gy, gx = np.gradient(f)
    return f, gx, gy


def _bilinear(flat, W, H, yy, xx):
    """평평하게 편 (H, W) 영상 flat 의 실수 좌표 (yy, xx) 선형 보간 (밖은 가장자리)"""
    yy = np.clip(yy, 0, H - 1.001)
    xx = np.clip(xx, 0, W - 1.001)
    y0, x0 = yy.astype(np.int64), xx.astype(np.int64)
    fy, fx = (yy - y0).astype(np.float32), (xx - x0).astype(np.float32)
    i = y0 * W + x0
    a, b, c, d = flat[i], flat[i + 1], flat[i + W], flat[i + W + 1]
    top = a + (b - a) * fx
    return top + (c + (d - c) * fx - top) * fy


def _block_disp(g0, g1, block=8, radius=6, tau=2.0, lam=1.0, noise=2.0, fine0=None, fine1=None):
    """g1 의 블록마다 g0 에서 온 변위 크기 (분석 해상도 px, 블록 수 길이). fine0·fine1 = _fine(g0)·_fine(g1) (없으면 만든다).

    0) 대비 이득: g0 에 화면 전체 표준편차 비 std(g1)/std(g0) (0.6~1.6) 를 곱해 맞춘다. 밝기는 블록 평균을 빼서, 대비·노출
       깜빡임은 이 이득으로 지운다 (움직임만 있으면 이득은 1 근처).
    1) 정수 탐색 ±radius 와 그 바깥 ±(radius+RING) 의 짝수 변위 (g0 은 가장자리를 복제해 늘림, 블록 격자는 그대로):
       비용 = 블록 평균을 뺀 SAD + lam·|변위|·화소 수 (비슷한 SAD 면 짧은 변위: 반복 무늬·가장자리의 먼 오정합 방지).
       최선 SAD 가 0 변위 SAD 의 0.8 배보다 크면 0.
    2) 1px 아래 다듬기 (흐린 휘도): 정수 최선 변위와 0 두 곳에서 대칭 기울기 Lucas-Kanade 두 번 (둘째는 다듬은 자리에서
       선형 보간). 다듬은 뒤 남은 SAD + 같은 변위 벌점이 작은 쪽을 쓴다. 1px 아래로 느린 선은 정수 변위 어디에도 맞지 않아
       선을 따라 먼 정수 변위가 뽑히곤 했다 (선화 팬 2px/프레임이 10px 로 읽힘). 0 에서 다듬으면 실제 변위가 남은 SAD 도
       작고 더 짧다.
    3) 개구 문제: 블록 기울기 구조 텐서 J 의 고윳값 비 q = λ2/λ1 이 0.15 아래 (선 하나뿐인 블록) 면 변위의 선 방향 성분은
       그 블록에서 정해지지 않으므로 버리고 (법선 성분만 씀), 0.4 위 (모서리·무늬) 면 모두 쓴다. 사이는 선형으로 섞는다.
       버린 선 방향 성분은 이웃 3x3 블록 가운데 믿을 만한 블록 (q 가중) 의 평균 변위에서 채운다: 세로 선이 많은 옷이
       세로로 움직일 때 (작은 피사체·세로 화면)·회전·줌에서 법선 성분만으로는 작게 나오던 것.
    4) 움직인 블록 = 0 변위 SAD > tau·화소 수 이고, 0 변위 SAD > noise × 다듬은 뒤 남은 SAD (흐린 휘도, 블록마다).
       남은 SAD 가 그 블록의 잡음 바닥이다: 프레임마다 다른 입자 잡음 (화면 일부만이어도)·대비 깜빡임은 어떤 변위로도
       줄지 않아 움직임이 아니고, 깨끗한 움직임은 남은 SAD 가 0 근처라 그대로다. noise=0 이면 이 문턱을 끈다."""
    r, b = radius, block
    H, W = g0.shape
    nby, nbx = (H - 2 * r) // b, (W - 2 * r) // b
    if nby < 1 or nbx < 1:
        return np.zeros(1, np.float32)
    ga = np.float32(np.clip(np.std(g1) / max(float(np.std(g0)), 1e-6), *GAIN_RANGE))   # 화면 전체 대비 이득

    def zb(a):   # 블록 평균을 뺀 (nby, b, nbx, b)
        a = a.reshape(nby, b, nbx, b)
        return a - a.mean(axis=(1, 3), keepdims=True)

    rr = r + RING                                             # 바깥 고리: ±(r+2) 의 짝수 변위 (빠른 피사체)
    cand = np.array([(y, x) for y in range(-rr, rr + 1) for x in range(-rr, rr + 1)
                     if max(abs(y), abs(x)) <= r or (y % 2 == 0 and x % 2 == 0)])
    ref, g0a = zb(g1[r:r + nby * b, r:r + nbx * b]), np.pad(g0 * ga, RING, mode="edge")
    S = np.stack([np.abs(ref - zb(g0a[rr + y:rr + y + nby * b, rr + x:rr + x + nbx * b])).sum(axis=(1, 3))
                  for y, x in cand])                                                  # (후보 수, nby, nbx)
    D = np.hypot(cand[:, 0], cand[:, 1])[:, None, None]                               # 후보 변위 크기
    C = S + lam * b * b * D
    best = np.where(C <= C.min(0) + 1e-3, D, np.inf).argmin(0)                       # 비용이 같으면 작은 변위
    i0 = len(cand) // 2                                                               # (0, 0) 후보
    s00 = S[i0]
    best = np.where(np.take_along_axis(S, best[None], 0)[0] > 0.8 * s00, i0, best)   # 별로 안 나아지면 0
    out = np.zeros(nby * nbx, np.float32)
    sel = np.flatnonzero(s00.ravel() > tau * b * b)          # 0 변위에서도 거의 같은 블록은 움직이지 않은 것
    if sel.size == 0:
        return out
    iy, ix = cand[best.ravel()[sel]].T + r
    f0, gx0, gy0 = _fine(g0) if fine0 is None else fine0
    f1, gx1, gy1 = _fine(g1) if fine1 is None else fine1
    F0, GX0, GY0 = f0.ravel() * ga, gx0.ravel() * ga, gy0.ravel() * ga
    byi, bxi = np.divmod(sel, nbx)
    oy, ox = np.mgrid[0:b, 0:b]
    Py, Px = (r + byi * b)[:, None, None] + oy, (r + bxi * b)[:, None, None] + ox      # (K, b, b) 블록 화소 좌표
    lin = Py * W + Px
    R = f1.ravel()[lin]
    R = R - R.mean((1, 2), keepdims=True)
    Rgx, Rgy = gx1.ravel()[lin], gy1.ravel()[lin]
    A = F0[lin]
    s00f = np.abs(A - A.mean((1, 2), keepdims=True) - R).sum((1, 2))                   # 흐린 휘도의 0 변위 SAD

    def refine(sy, sx):
        """정수 시작 변위 (sy, sx) 에서 대칭 기울기 LK 두 번 → (dy, dx, 남은 SAD, Jxx, Jxy, Jyy)"""
        li = np.clip(Py + sy[:, None, None], 0, H - 1) * W + np.clip(Px + sx[:, None, None], 0, W - 1)
        A = F0[li]
        e = A - A.mean((1, 2), keepdims=True) - R
        Gx, Gy = 0.5 * (GX0[li] + Rgx), 0.5 * (GY0[li] + Rgy)     # 두 프레임 기울기 평균: 1차 근사가 2차까지 맞음
        Gx -= Gx.mean((1, 2), keepdims=True)
        Gy -= Gy.mean((1, 2), keepdims=True)
        Jxx, Jxy, Jyy = (Gx * Gx).sum((1, 2)), (Gx * Gy).sum((1, 2)), (Gy * Gy).sum((1, 2))
        reg = 1e-3 * (Jxx + Jyy) + 1e-6                           # 선 하나뿐인 블록: 선 방향으로는 거의 움직이지 않음
        a, c = Jxx + reg, Jyy + reg
        det = a * c - Jxy * Jxy

        def step(e):
            bx, by = (Gx * e).sum((1, 2)), (Gy * e).sum((1, 2))
            return np.clip((Jxy * bx - a * by) / det, -1, 1), np.clip((Jxy * by - c * bx) / det, -1, 1)

        ddy, ddx = step(e)
        dy, dx = sy + ddy, sx + ddx
        A = _bilinear(F0, W, H, Py + dy[:, None, None], Px + dx[:, None, None])
        e = A - A.mean((1, 2), keepdims=True) - R
        ddy, ddx = step(e)
        res = np.abs(e + Gx * ddx[:, None, None] + Gy * ddy[:, None, None]).sum((1, 2))   # 선형 근사로 다듬은 뒤 SAD
        return dy + ddy, dx + ddx, res, Jxx, Jxy, Jyy

    dy, dx, res, Jxx, Jxy, Jyy = refine(iy - r, ix - r)
    z = np.zeros(sel.size, np.int64)
    zy, zx, zres, zJxx, zJxy, zJyy = refine(z, z)
    use = zres + lam * b * b * np.hypot(zy, zx) <= res + lam * b * b * np.hypot(dy, dx)
    dy, dx, res = np.where(use, zy, dy), np.where(use, zx, dx), np.where(use, zres, res)
    Jxx, Jxy, Jyy = np.where(use, zJxx, Jxx), np.where(use, zJxy, Jxy), np.where(use, zJyy, Jyy)
    tr, disc = Jxx + Jyy, np.sqrt((Jxx - Jyy) ** 2 * 0.25 + Jxy ** 2)
    q = (0.5 * tr - disc) / np.maximum(0.5 * tr + disc, 1e-9)                         # λ2 / λ1
    th = 0.5 * np.arctan2(2 * Jxy, Jxx - Jyy)                                          # 기울기 (법선) 방향
    pn, pt = dx * np.cos(th) + dy * np.sin(th), dy * np.cos(th) - dx * np.sin(th)     # 법선·선 방향 성분
    w = np.clip((q - 0.15) / 0.25, 0, 1)                                              # 선 방향 성분을 믿는 정도
    moving = s00f > noise * res if noise > 0 else np.ones(sel.size, bool)
    # 선 블록의 선 방향 성분은 이웃 3x3 블록 가운데 믿을 만한 블록 (w 가중, 정지로 판정된 블록은 변위 0) 의 평균 변위에서
    V = np.zeros((3, nby * nbx))
    V[:, sel] = w, w * np.where(moving, dx, 0.0), w * np.where(moving, dy, 0.0)
    V = np.pad(V.reshape(3, nby, nbx), ((0, 0), (1, 1), (1, 1)))
    V = sum(V[:, i:i + nby, j:j + nbx] for i in range(3) for j in range(3)).reshape(3, -1)[:, sel]
    vx, vy = V[1] / np.maximum(V[0], 1e-9), V[2] / np.maximum(V[0], 1e-9)
    tn = vy * np.cos(th) - vx * np.sin(th)
    mag = np.sqrt(pn * pn + w * pt * pt + (1 - w) * np.minimum(V[0], 1.0) * tn * tn)
    out[sel] = np.where(moving, mag, 0.0)
    return out


def _top5(d):
    """상위 5% 평균 (VBench dynamic_degree 의 흐름 크기 통계). 움직이는 블록이 5% 근처여도 값이 무너지지 않는다."""
    k = max(1, int(np.ceil(0.05 * d.size)))
    return float(np.sort(d)[-k:].mean())


def _contrast_gain(Y):
    """클립 대비 이득 (스칼라). 대비 c = 프레임마다 8x8 블록 표준편차의 98 백분위 (가장 무늬가 짙은 블록들), 그 프레임
    중앙값. c 가 CONTRAST_REF 보다 낮으면 CONTRAST_REF / c 배 (최대 CONTRAST_GMAX), 아니면 1 (보통 그림은 그대로).
    _block_disp 의 문턱 tau·변위 벌점 lam 은 절대 휘도라, 어둡거나 흐린 클립에서는 같은 속도가 느리게 읽혔다."""
    T, h, w = Y.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    sd = Y[:, :h8, :w8].reshape(T, h8 // 8, 8, w8 // 8, 8).std(axis=(2, 4)).reshape(T, -1)
    c = float(np.median(np.percentile(sd, 98, axis=1)))
    return np.float32(np.clip(CONTRAST_REF / max(c, 1e-6), 1.0, CONTRAST_GMAX))


def _motion_luma(frames, side=MOTION_SIDE, with_gain=False):
    """block_motion_p95 의 분석 휘도: MOTION_RESAMPLE 축소 (스크린톤 무아레 방지) 에 클립 대비 이득을 곱한 것"""
    Y = _luma(frames, side, MOTION_RESAMPLE)
    g = _contrast_gain(Y) if len(Y) and min(Y.shape[1:]) >= 8 else np.float32(1.0)
    return (Y * g, float(g)) if with_gain else Y * g


def _int_shift(a, b):
    """b 가 a 를 화면 전체로 옮긴 정수 변위 (dy, dx): 위상 상관 피크"""
    h, w = a.shape
    A, B = np.fft.rfft2(a - a.mean()), np.fft.rfft2(b - b.mean())
    R = B * np.conj(A)
    c = np.fft.irfft2(R / np.maximum(np.abs(R), 1e-9), s=(h, w))
    iy, ix = np.unravel_index(int(np.argmax(c)), c.shape)
    return (iy if iy <= h // 2 else iy - h), (ix if ix <= w // 2 else ix - w)


def _lk_noise(a, b):
    """같은 크기 두 프레임: 블록마다 대칭 기울기 LK 한 번으로 설명하고 남은 프레임 차 (블록 평균 뺌) 평균 |e| 를, 두 프레임
    8x8 표준편차의 작은 쪽이 하위 25% 인 블록에서 중앙값"""
    h, w = a.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    if h8 < 16 or w8 < 16:
        return np.inf
    gya, gxa = np.gradient(a)
    gyb, gxb = np.gradient(b)
    blk = lambda x: x[:h8, :w8].reshape(h8 // 8, 8, w8 // 8, 8)
    zm = lambda x: x - x.mean(axis=(1, 3), keepdims=True)
    A, B = blk(a), blk(b)
    D = zm(B - A)
    Gx, Gy = zm(blk(0.5 * (gxa + gxb))), zm(blk(0.5 * (gya + gyb)))
    jxx, jxy, jyy = (Gx * Gx).sum((1, 3)), (Gx * Gy).sum((1, 3)), (Gy * Gy).sum((1, 3))
    bx, by = -(Gx * D).sum((1, 3)), -(Gy * D).sum((1, 3))
    reg = 1e-3 * (jxx + jyy) + 1e-6
    p, r = jxx + reg, jyy + reg
    det = p * r - jxy * jxy
    dx, dy = (r * bx - jxy * by) / det, (p * by - jxy * bx) / det
    e = D + Gx * dx[:, None, :, None] + Gy * dy[:, None, :, None]
    md = np.abs(e).mean(axis=(1, 3)).ravel()
    s = np.minimum(A.std(axis=(1, 3)), B.std(axis=(1, 3))).ravel()
    return float(np.median(md[s <= np.percentile(s, 25)]))


def _flat_noise(Y):
    """전이마다 시간 잡음 (분석 휘도): 두 프레임 8x8 표준편차의 작은 쪽이 하위 25% 인 블록에서 블록 평균을 뺀 프레임 차
    평균 |dY| 의 중앙값. 무늬가 있는 그림이 움직이면 밋밋한 블록도 바뀌므로, 화면 전체 정수 변위로 맞춘 뒤 블록마다
    LK 한 번으로 설명하고 남은 차이로도 재서 작은 쪽을 쓴다 (잡음은 남고, 움직임은 설명된다)."""
    T, h, w = Y.shape
    h8, w8 = h // 8 * 8, w // 8 * 8
    Z = Y[:, :h8, :w8].reshape(T, h8 // 8, 8, w8 // 8, 8)
    sd = Z.std(axis=(2, 4)).reshape(T, -1)
    D = np.diff(Z, axis=0)
    md = np.abs(D - D.mean(axis=(2, 4), keepdims=True)).mean(axis=(2, 4)).reshape(T - 1, -1)
    s = np.minimum(sd[:-1], sd[1:])
    q = np.percentile(s, 25, axis=1, keepdims=True)
    out = np.array([np.median(md[i][s[i] <= q[i]]) for i in range(T - 1)])
    for i in range(T - 1):
        dy, dx = _int_shift(Y[i], Y[i + 1])
        a = Y[i, max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]       # b[y, x] ~ a[y - dy, x - dx]
        b = Y[i + 1, max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)]
        out[i] = min(out[i], _lk_noise(a, b))
    return out


def block_motion_p95(frames, side=MOTION_SIDE):
    if _empty(frames) or len(frames) < 2:
        return 0.0
    frames = np.asarray(frames)
    Y, g = _motion_luma(frames, side, with_gain=True)
    scale = frames.shape[2] / Y.shape[2]                      # 분석 px → 원본 px
    if min(Y.shape[1:]) < 20:                                 # 8x8 블록 ±6px 이 들어가지 않음
        return 0.0
    # 화면 대비 (표준편차) 가 한 전이에서 GAIN_RANGE 밖으로 바뀌면 (검은·흰 프레임, 페이드 끝, 대비가 다른 장면 전환)
    # 두 프레임을 견줄 수 없어 움직임을 재지 않고 평균에서 뺀다
    sd = Y.reshape(len(Y), -1).astype(np.float64).std(axis=1)
    ok = [i for i in range(len(Y) - 1) if sd[i] > 0 and GAIN_RANGE[0] <= sd[i + 1] / sd[i] <= GAIN_RANGE[1]]
    pairs = [(i, i + 1) for i in ok]
    if not ok:
        pairs = [(i, i + 2) for i in range(len(Y) - 2) if sd[i] > 0 and GAIN_RANGE[0] <= sd[i + 2] / sd[i] <= GAIN_RANGE[1]]
        pairs = pairs or [(i, i + 1) for i in range(len(Y) - 1) if sd[i] > 0]
    if not pairs:
        return 0.0
    F = {i: _fine(Y[i]) for i in sorted({j for p in pairs for j in p})}
    nz = _flat_noise(Y) if g > 1 else np.zeros(len(Y) - 1)
    out = {}
    for a, b in pairs:
        n = float(max(nz[a:b]))
        tau, lam = max(2.0, min(2.0 * g, 2.0 * n)), max(1.0, min(g, n))
        v = _top5(_block_disp(Y[a], Y[b], tau=tau, lam=lam, fine0=F[a], fine1=F[b])) / (b - a)
        out.setdefault(a % 2 if b - a == 2 else 0, []).append(v)
    return float(max(np.mean(v) for v in out.values()) * scale)


def _rms_hp(v, trim=0):
    """고역 성분 h[t] = v[t] - (v[t-1]+v[t+1])/2 의 RMS. trim: 제곱이 가장 큰 표본 trim 개를 빼고 잰다 (표본이 3개
    아래면 빼지 않음). 표본이 없으면 0."""
    h2 = (v[1:-1] - (v[:-2] + v[2:]) / 2) ** 2
    if trim and h2.size >= 3:
        h2 = np.sort(h2)[:-trim]
    return float(np.sqrt(h2.mean())) if h2.size else 0.0


def _flicker_y(Y):
    """긴 변 32px 휘도 (T,h,w) 0~1 → (flicker_ratio, luma_flicker, contrast_flicker). 3프레임 아래면 모두 0.
    luma_flicker = 프레임 평균 휘도 (밝기, 0~255) 의 고역 성분 RMS (매끈한 변화는 0 근처, 프레임마다 오가면 커짐).
    contrast_flicker = 프레임 휘도 표준편차 (대비, 0~255) 의 고역 성분 RMS 인데, 제곱이 가장 큰 표본 2개를 빼고 잰다
    (고역 표본 3개 이상일 때): 장면 전환 한 번은 계단이라 고역 표본 2개 (±계단/2) 만 크고, 대비가 프레임마다 오가는
    깜빡임은 모든 표본이 크다."""
    if len(Y) < 3:
        return 0.0, 0.0, 0.0
    d1 = np.abs(np.diff(Y, axis=0)).mean()
    d2 = np.abs(Y[2:] - 2 * Y[1:-1] + Y[:-2]).mean()
    return float(d2 / (d1 + 1e-6)), _rms_hp(Y.mean(axis=(1, 2)) * 255), _rms_hp(Y.std(axis=(1, 2)) * 255, trim=2)


def flicker(frames):
    r, lf, _ = _flicker_y(_luma(frames, 32) / 255.0)
    return dict(flicker_ratio=r, luma_flicker=lf)


def contrast_flicker(frames):
    """대비 깜빡임 (휘도 0~255): 프레임 휘도 표준편차의 고역 성분 RMS (가장 큰 2개를 뺌, _flicker_y). 128 둘레로 대비만
    오가는 깜빡임은 평균 휘도가 거의 그대로라 luma_flicker 에 안 보인다. clip_metrics 는 이 값을 contrast_flicker 키로
    내고, 베이스 값으로 나눠 flicker_rel 에 넣는다."""
    return _flicker_y(_luma(frames, 32) / 255.0)[2]


def color_stats(frames, k=8):
    if _empty(frames):
        return dict.fromkeys(("r", "g", "b", "sat", "val"), float("nan"))
    sub = np.asarray(frames)[_pick(len(frames), k)]
    rgb = sub.reshape(-1, 3).mean(0)
    hsv = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f)).convert("HSV"), np.float32) for f in sub])
    return dict(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), sat=float(hsv[..., 1].mean()), val=float(hsv[..., 2].mean()))


def _color_shift(c, cb):
    """채도 상대 변화와 '평균색 거리 / 베이스 평균색 크기' 가운데 큰 값 (0.10 = 10%)."""
    sat = abs(c["sat"] - cb["sat"]) / max(cb["sat"], SAT_FLOOR)
    a, b = (np.array([d["r"], d["g"], d["b"]]) for d in (c, cb))
    return float(max(sat, np.linalg.norm(a - b) / max(np.linalg.norm(b), 1.0)))


class Dino:
    """프레임 임베딩 (T, 384). Dino.load(device) 가 None 이면 (설치·내려받기 실패) 시간 일관성·그림체 지표 없이 진행한다."""
    NAME = "facebook/dinov2-small"
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)
    PAD = (124, 116, 104)     # ImageNet 평균색: 정규화하면 0

    def __init__(self, model, device):
        self.model, self.device = model, device

    @classmethod
    def load(cls, device="cpu", name=NAME):
        try:
            import torch
            from transformers import Dinov2Model
            model = Dinov2Model.from_pretrained(name, dtype=torch.float32).to(device).eval()
        except Exception as e:      # transformers·torch 없음, 내려받기 실패 등
            print(f"DINOv2 지표를 쓰지 않습니다 ({name}): {type(e).__name__}: {str(e)[:300]}")
            return None
        return cls(model, device)

    @classmethod
    def letterbox(cls, img, size=224):
        """비율을 지키며 긴 변을 size 로 줄이고 남는 곳을 평균색으로 채운다 (프레임 전체가 남는다)."""
        im = img if isinstance(img, Image.Image) else Image.fromarray(np.ascontiguousarray(img))
        im = im.convert("RGB")
        s = size / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), cls.PAD)
        canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
        return np.asarray(canvas)

    def embed(self, frames, batch=32):
        """(T,H,W,3) uint8 또는 그림 목록 (크기가 달라도 됨) → (T, 384) float32, 행마다 L2 정규화. 없으면 (0, 384)."""
        if _empty(frames):
            return np.zeros((0, getattr(getattr(self.model, "config", None), "hidden_size", 384)), np.float32)
        import torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(frames), batch):
                x = np.stack([(self.letterbox(f) / 255.0 - self.MEAN) / self.STD for f in frames[i:i + batch]])
                x = torch.from_numpy(x.transpose(0, 3, 1, 2).astype(np.float32)).to(self.device)
                e = self.model(pixel_values=x).pooler_output.float()
                out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out)


def _unit(E):
    """(T, D) 행마다 L2 정규화. None·빈 목록은 (0, 1), 벡터 하나 (D,) 는 (1, D)."""
    E = np.asarray([] if E is None else E, np.float32)
    if E.ndim == 1:
        E = E[None] if E.size else np.zeros((0, 1), np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def subject_consistency(E):
    E = _unit(E)
    if len(E) < 2:
        return 1.0
    prev = np.clip((E[1:] * E[:-1]).sum(1), 0, None)
    first = np.clip(E[1:] @ E[0], 0, None)
    return float(((prev + first) / 2).mean())


def style_drift(E_a, E_b):
    """한쪽이라도 비면 NaN"""
    a, b = _unit(E_a), _unit(E_b)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    return float(1.0 - (a[:n] * b[:n]).sum(1).mean())


class CCIP:
    """애니 캐릭터 정체성. CCIP.load() 가 None 이면 (onnxruntime 없음·내려받기 실패) 정체성 지표 없이 진행한다."""
    REPO = "deepghs/ccip_onnx"
    MODEL = "ccip-caformer-24-randaug-pruned"
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)[:, None, None]
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)[:, None, None]

    def __init__(self, feat, metric, threshold):
        self.feat, self.metric, self.threshold = feat, metric, threshold

    @classmethod
    def load(cls, model=MODEL, providers=("CPUExecutionProvider",)):
        try:
            import onnxruntime as ort
        except ImportError as e:
            print(f"CCIP 정체성 지표를 쓰지 않습니다: onnxruntime 이 없습니다 ({e}). 쓰려면 pip install onnxruntime")
            return None
        except Exception as e:      # 깨진 설치: DLL 을 못 찾는 OSError 등
            print(f"CCIP 정체성 지표를 쓰지 않습니다: onnxruntime 을 불러오지 못했습니다 ({type(e).__name__}: {str(e)[:300]})."
                  " 다시 설치하려면 pip install --force-reinstall onnxruntime")
            return None
        try:
            from huggingface_hub import hf_hub_download
            path = lambda f: hf_hub_download(cls.REPO, f"{model}/{f}")
            feat = ort.InferenceSession(path("model_feat.onnx"), providers=list(providers))
            metric = ort.InferenceSession(path("model_metrics.onnx"), providers=list(providers))
            with open(path("metrics.json"), encoding="utf-8") as f:
                threshold = float(json.load(f)["threshold"])
        except Exception as e:
            print(f"CCIP 정체성 지표를 쓰지 않습니다 ({cls.REPO}/{model}): {type(e).__name__}: {str(e)[:300]}")
            return None
        return cls(feat, metric, threshold)

    @classmethod
    def _prep(cls, img):
        im = img if isinstance(img, Image.Image) else Image.fromarray(np.ascontiguousarray(img))
        if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):   # 투명 → 흰 배경
            im = im.convert("RGBA")
            im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
        x = np.asarray(im.convert("RGB").resize((384, 384), Image.Resampling.BILINEAR), np.float32)
        return (x.transpose(2, 0, 1) / 255.0 - cls.MEAN) / cls.STD

    def features(self, images, batch=16):
        """그림 목록 (PIL 또는 uint8 배열, (N,H,W,3) 배열도 됨) → (N, 768). 없으면 (0, 768)."""
        images = [] if images is None else images
        out = [self.feat.run(["output"], {"input": np.stack([self._prep(im) for im in images[i:i + batch]]).astype(np.float32)})[0]
               for i in range(0, len(images), batch)]
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    def differences(self, feats):
        """쌍별 차이 행렬 (작을수록 닮음)"""
        return self.metric.run(["output"], {"input": np.asarray(feats, np.float32)})[0]

    def identity(self, frames, ref_feats, max_frames=16):
        """프레임마다 가장 가까운 참조 그림과의 차이 d. id_pass = mean(d <= threshold), id_mean = mean(d),
        id_slope = 프레임 번호에 대한 d 의 기울기 (+ 면 클립이 진행될수록 덜 닮아 감). 프레임·참조가 없으면 모두 None."""
        if _empty(frames) or ref_feats is None or len(ref_feats) == 0:
            return dict(id_pass=None, id_mean=None, id_slope=None)
        idx = _pick(len(frames), max_frames)
        f = self.features([frames[i] for i in idx])
        d = self.differences(np.concatenate([f, np.asarray(ref_feats, np.float32)]))[:len(idx), len(idx):].min(1)
        slope = float(np.polyfit(idx, d, 1)[0]) if len(idx) > 2 else 0.0
        return dict(id_pass=float((d <= self.threshold).mean()), id_mean=float(d.mean()), id_slope=slope)


def clip_metrics(frames, base=None, dino=None, ccip=None, ref_feats=None):
    """클립 하나의 지표 (평평한 dict, 못 잰 값은 None).

    base: 같은 시드·프롬프트(·시작 이미지)로 만든 베이스 클립 프레임. *_ratio·*_rel·color_shift·drift 는 base 가 있어야 나온다.
    luma_flicker·contrast_flicker: 이 클립의 밝기·대비 깜빡임 (휘도 0~255, _flicker_y). 어느 쪽이 깜빡였는지 표에서 본다.
    flicker_rel = max(밝기: luma_flicker / max(베이스 luma_flicker, FLICKER_FLOOR),
                      대비: contrast_flicker / max(베이스 contrast_flicker, FLICKER_FLOOR)). 밝기 쪽은 예전 flicker_rel 그대로.
    drift·color_shift 는 트리거 없는(class word) 프롬프트에서 그림체 변화로 읽는다 (트리거 프롬프트는 캐릭터가 바뀌는 게 정상).
    ref_feats: ccip.features(참조 캐릭터 그림들).
    frames 가 None·빈 것이면 motion 0.0 에 나머지는 모두 None (생성 실패 등). 빈 base 는 None 과 같다.
    """
    if _empty(frames):
        return dict(motion=0.0, motion_ratio=None, luma_flicker=None, contrast_flicker=None, flicker_rel=None,
                    flicker_ratio=None, sat=None, color_shift=None, consistency=None, drift=None, id_pass=None,
                    id_mean=None, id_slope=None)
    if base is not None and len(base) == 0:
        base = None
    frames = np.asarray(frames)
    fr, lf, cf = _flicker_y(_luma(frames, 32) / 255.0)
    col = color_stats(frames)
    E = dino.embed(frames) if dino is not None else None
    out = dict(motion=block_motion_p95(frames), motion_ratio=None, luma_flicker=lf, contrast_flicker=cf, flicker_rel=None,
               flicker_ratio=fr, sat=col["sat"], color_shift=None,
               consistency=subject_consistency(E) if E is not None else None, drift=None,
               id_pass=None, id_mean=None, id_slope=None)
    if base is not None:
        base = np.asarray(base)
        bm = block_motion_p95(base)
        out["motion_ratio"] = out["motion"] / bm if bm >= MOTION_FLOOR else None
        _, blf, bcf = _flicker_y(_luma(base, 32) / 255.0)
        out["flicker_rel"] = max(lf / max(blf, FLICKER_FLOOR), cf / max(bcf, FLICKER_FLOOR))
        out["color_shift"] = _color_shift(col, color_stats(base))
        if E is not None:
            out["drift"] = style_drift(E, dino.embed(base))
    if ccip is not None and ref_feats is not None and len(ref_feats) and len(frames):
        out.update(ccip.identity(frames, ref_feats))
    return out
