"""Build lora_training_video.ipynb (video character LoRA, Wan family) from cell sources + module files.

  python build_video_notebook.py <out.ipynb>

Embeds video_adapters/<family>.py (-> lora_video_adapters/) and video_adapters/wan_common.py,
video_tools/{video_io,video_metrics,remedy}.py (-> lora_video_tools/) with %%writefile. See VIDEO_INTERFACE.md.
"""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
FAMILIES = ["wan22_t2v", "wan22_i2v", "wan22_ti2v_5b", "wan21"]
TOOLS = [("video_adapters", "wan_common"), ("video_tools", "video_io"), ("video_tools", "video_metrics"),
         ("video_tools", "remedy")]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 캐릭터 LoRA 학습 노트북 — 영상 모델 (Wan 2.1·2.2)

이 노트북은 영상 생성 모델에 캐릭터 LoRA 를 학습한다. 베이스 모델은 얼리고 LoRA 어댑터만 학습한다.

| 단계 | 하는 일 |
|---|---|
| 0. 준비 | 라이브러리 확인, 설정(고정값), 도구·어댑터, 모델 불러오기 |
| 1. EDA | 이미지·클립의 프레임 수·fps·길이·회전·VFR, 검은·멈춘 구간, 장면 전환, 움직임, 흐림, 중복, 캡션 검사 |
| 2. 시각화 | 분포 그래프, 컨택트 시트, 클립별 신호 곡선 |
| 3. 전처리 | 모델 fps 로 리샘플, 깨끗한 구간에서 창 고르기, (프레임·높이·너비) 버킷, 잠재값·텍스트 캐시 |
| 4. 학습 | 전문가별 LoRA 학습, 정해진 단계마다 저장·검증 손실·미리보기 클립 |
| 5. 검증 | 손실 곡선, 단계별 미리보기, 저장 파일 다시 불러오기, 최종 검증 세트와 지표(움직임·깜빡임·색·정체성), 새 프롬프트 시험 |
| 6. 보완 | 지표로 증상 진단, 체크포인트 추천, 다음 실행 설정(`next_run.json`), 결과 요약 |

- 모델을 바꿀 때 고치는 값은 `0-2` 의 `USER` 처음 5개뿐이다: `TRIGGER`, `MODEL_PATH`, `RUN_NAME`, `DATASET_DIR`, `TARGET_SCOPE` (환경변수 `LORA_*` 로도 줄 수 있다).
  - 모델 계열은 `MODEL_PATH` 폴더의 설정 파일로 자동으로 정하고, 계열별 값(fps·버킷·검증 step·전문가 수)도 따라 바뀐다.
- 학습 조건은 `FIXED` 에 고정되어 있다.
- `FIXED` 를 바꾸는 방법은 하나다. `LORA_FIXED_OVERRIDES` 에 `{키: 값}` 을 준다 (6 보완 단계가 만든 `next_run.json` 의 값을 그대로 넣거나, 예: `{"token_budget": 20280}`).
  - `FIXED` 에 있는 키만 받고, 보완 규칙(`remedy.ALLOWED`)에 범위·선택지가 있는 값은 그 안에서만 받는다.
- 모델은 diffusers 형식 폴더(`model_index.json` 이 있는 폴더) 또는 Hugging Face 저장소 이름으로 불러온다.

### 지원 모델 (2026-11-05 조사, 모두 Apache-2.0)

| 계열 (자동 인식) | 학습할 모델 | 학습 데이터 | LoRA 파일 | 쓰임 |
|---|---|---|---|---|
| `wan22_t2v` | Wan2.2-T2V-A14B-Diffusers | 이미지(1프레임)·클립 | 고노이즈·저노이즈 2개 | 지금 가진 이미지로 학습하는 1단계. 블록 모양이 같아 I2V-A14B 에도 걸 수 있다 (효과는 실측 필요) |
| `wan22_i2v` | Wan2.2-I2V-A14B-Diffusers | 클립만 | 2개 | 회사 기본 영상 모델. 클립이 생기면 학습 |
| `wan22_ti2v_5b` | Wan2.2-TI2V-5B-Diffusers | 이미지·클립 | 1개 | 파일럿 (가볍다). 5B LoRA 는 A14B 에 쓸 수 없다 |
| `wan21` | Wan2.1-T2V-1.3B·14B-Diffusers | 이미지·클립 | 1개 | 코드 점검(1.3B)·대안(14B) |

- 다른 후보를 뺀 이유 (라이선스 원문 확인)
  - HunyuanVideo·HunyuanVideo-1.5·MiniMax H3: 라이선스가 대한민국을 제외한다.
  - LTX-2.x: 연매출 US$10M 이상이면 유료 라이선스, 상업적으로 출력물을 다른 모델 학습에 쓸 수 없다.
  - Kandinsky 5·6: 모델은 MIT 지만 Tencent HunyuanVideo VAE 를 포함한다 (법률 검토 필요).
  - CogVideoX·Mochi 는 오래되었거나 사용 조건이 있고, LongCat-Video·MAGI 는 diffusers·ComfyUI 지원이 없다.
- 애니 특화 Index-AniSora V3.2 (Bilibili, Wan2.2-I2V-A14B 구조, Apache-2.0 + 추가 조건) 는 diffusers 형식으로 변환하면 `wan22_i2v` 로 학습할 수 있다 [미검증].

### A14B 의 두 전문가

- Wan2.2 A14B 는 고노이즈 전문가(`transformer`)와 저노이즈 전문가(`transformer_2`)로 나뉜다. 노이즈 단계 t = 1000·σ 가 경계(T2V 875, I2V 900) 이상이면 고노이즈 전문가가 그린다.
- 전문가마다 LoRA 를 따로 학습하고 파일도 따로 저장한다. 각 전문가는 자기 구간의 σ 로만 학습한다. 구간이 어긋나면 LoRA 가 효과가 없다.
- 기본은 저노이즈(정체성·세부)에 단계의 75%, 고노이즈(구도·움직임)에 25% 를 쓴다 [제안, 실무자 경험]. 고노이즈는 기본으로 클립으로만 학습한다. 정지 이미지가 움직임을 굳힐 위험 때문이다.

### 데이터셋 폴더

```
dataset/niki_video/
  niki_01.png   niki_01.txt     # 이미지 = 1프레임 영상
  walk_01.mp4   walk_01.txt     # 클립 (mp4·mov·mkv·webm·avi·gif)
```

- 캡션: 트리거 + 바뀌는 것(샷·카메라·시간 순서의 동작·배경). 머리·옷 같은 고정 외형은 적지 않는다.
  - 예: `nkq7, medium shot, static camera, she turns from the side to face the camera, then smiles, autumn park`
- 클립은 모델 fps(Wan 16, 5B 24)로 바꾼 뒤 4n+1 프레임 창으로 잘라 쓴다. 2~5초 클립이 알맞다. 늘이거나 줄이지 않는다.
- 이미지를 반복해 만든 가짜 클립은 넣지 않는다 (움직임을 잃는 원인으로 알려져 있다).
- `wan22_i2v` 는 클립이 있어야 한다. 1프레임 이미지는 조건 채널에 정답이 그대로 들어가 배우는 것이 없다.

### 메인 서버에서 실행하기

1. 학습용 가상환경에 영상 라이브러리를 더한다: `pip install av ftfy` (선택: 애니 캐릭터 정체성 지표용 `pip install onnxruntime`)
2. 모델과 검증 지표용 작은 모델을 받는다.
   - 예: `hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir /home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers`
   - 지표: `hf download facebook/dinov2-small config.json model.safetensors`
   - 선택(CCIP): `hf download deepghs/ccip_onnx ccip-caformer-24-randaug-pruned/model_feat.onnx ccip-caformer-24-randaug-pruned/model_metrics.onnx ccip-caformer-24-randaug-pruned/metrics.json`
   - 파일 이름을 적어 받는다. `--include` 는 값을 하나만 받는다: 뒤에 이어 쓴 값은 파일 이름이 되어 `--include` 가 무시되고 첫 값이 빠진다 (글롭이면 오류). 패턴을 여러 개 쓰려면 값마다 `--include` 를 붙인다.
   - 실행 공간에서 인터넷이 막혀 있으면 미리 받아 두고 `HF_HUB_OFFLINE=1` 로 실행한다. 아니면 4-2 셀이 내려받기를 몇 분 기다린 뒤 그 지표를 빼고 넘어간다. 지표가 필요 없으면 `LORA_FIXED_OVERRIDES='{"use_dino": false, "use_ccip": false}'` 로 끈다.
3. 처음 한 번은 아래 '시험 실행' 으로 돌려, 리눅스에서 영상 읽기(회전·VFR·프레임 수)가 맞는지 확인한다. EDA 셀이 합성 데이터의 정답과 비교한다.
4. 실제 학습 전에 ComfyUI 대기열을 멈춰 GPU 를 비운다 (계획서 P2).
- A14B 는 전문가 2개(bf16 각 약 28.6GB)를 번갈아 GPU 에 올리고, 쉬는 쪽은 CPU 메모리에 둔다. 미리보기·최종 생성 때는 모든 구성요소(약 69GB)가 CPU 메모리에 있다.
- 0-4 셀이 모델을 불러오기 전에 GPU 가 보이는지와 메모리를 확인하고, 모자라면 필요한 양을 보여 주고 멈춘다. 메모리 확인을 건너뛰려면 `LORA_SKIP_RAM_CHECK=1` 을 준다.
  - 시스템 메모리: 가중치 + 8GB 가 비어 있어야 한다. 컨테이너 메모리 한도나 서버 RAM 이 그보다 작으면 다른 프로그램을 멈춰도 안 되므로, 원인(한도·RAM)과 들어가는 더 작은 모델을 알린다.
  - GPU 메모리: 가장 큰 구성요소 + 4GB, 학습 단계 추정(LoRA·AdamW 포함), 생성(미리보기·최종) 추정 가운데 가장 큰 값이 비어 있어야 한다. 생성 추정은 전문가를 번갈아 올리는 A14B 면 전문가 하나 + 생성 토큰, 아니면 디노이저 모두 + VAE 디코드다.
- 시간과 GPU 메모리 [추정: 실제 폭의 블록을 노트북 GPU 에서 재고 연산량·층 수로 늘려 잡음. 첫 실행의 `summary.json` 으로 확인한다]
  - A14B·이미지만·기본값이면 연산의 약 27% 가 학습, 30% 가 미리보기(9번 × 3클립), 43% 가 최종 생성(10클립)이다. 서버 GPU 의 bf16 실효 성능을 200 TFLOPS 로 잡으면 약 2시간(학습만 약 33분)이다.
  - 미리보기를 줄이려면 `LORA_FIXED_OVERRIDES='{"validate_every": 500}'` 를 준다 (9번 → 5번).
  - A14B 학습 단계의 GPU 메모리는 약 30GB + 토큰당 약 1MB 다. 클립이 있으면 기본 예산 14,040토큰에서 약 44GB 다.
  - A14B 미리보기·최종 생성은 전문가 하나 + 생성 토큰(기본 33프레임 480x832 = 14,040토큰, 토큰당 약 0.25MB)으로 잡아, 이미지만 학습해도 0-4 셀은 약 35GB 가 비어 있기를 요구한다.
  - 토큰 예산(`token_budget`)은 주로 클립에 걸린다. 이미지는 `max_area`(480x832)에 묶여 한 장이 최대 1,560토큰(약 32GB)이라, 이미지만 학습하면 `token_budget` 을 늘려도 학습이 같다. 1,560 아래로 내리면 이미지도 작아진다 (예: 1,000 이면 512x768 → 400x608).
  - 첫 실행 뒤 `summary.json` 의 `train_minutes`(학습 단계만), `preview_minutes`(학습 중 미리보기·검증·저장), `train_peak_gpu_gb`(학습 단계 최대), `max_train_tokens` 를 본다. 클립이 있으면 0-4 셀이 보여 주는 '지금 들어가는 클립 token_budget' 안에서 늘린다.

### 시험 실행 (오류 검사)

환경변수 `LORA_NB_TEST=<모델 계열>` 을 주면, 같은 구조의 아주 작은 무작위 모델과 문제를 일부러 넣은 합성 데이터셋(회전·VFR·검은 프레임·장면 전환·중복 등)으로 모든 셀을 끝까지 돌리고 EDA 결과를 확인한다. 작은 토크나이저를 내려받으므로 인터넷이 필요하다. `LORA_NB_TEST_DEVICE=cuda` 를 더하면 GPU·bf16 으로 시험한다.

```
LORA_NB_TEST=wan22_t2v jupyter nbconvert --to notebook --execute lora_training_video.ipynb --output test_run.ipynb
```
""")

md("## 0. 준비\n\n### 0-1. 라이브러리 버전 확인과 불러오기")
code(r'''
# 고정 버전: 이 조합에서 4개 계열을 작은 무작위 모델로 시험했다.
#   pip install torch==2.15.0 diffusers==0.42.1 peft==0.21.2 transformers==5.22.1 accelerate==1.15.0 \
#               av safetensors sentencepiece protobuf ftfy pillow numpy pandas matplotlib tqdm
#   (선택) pip install onnxruntime   # 애니 캐릭터 정체성 지표(CCIP)
import importlib.metadata as _md

PINNED = {"torch": "2.15.0", "diffusers": "0.42.1", "peft": "0.21.2", "transformers": "5.22.1", "accelerate": "1.15.0"}
_rows = []
for _pkg, _want in PINNED.items():
    try:
        _have = _md.version(_pkg)
    except _md.PackageNotFoundError:
        raise ImportError(f"{_pkg} 가 설치되어 있지 않습니다. 위 pip 명령으로 {_pkg}=={_want} 를 설치하세요.")
    _ok = _have.split("+")[0] == _want
    _rows.append((_pkg, _want, _have, "OK" if _ok else "버전 다름"))
for _r in _rows:
    print(f"{_r[0]:<13} 고정 {_r[1]:<8} 설치 {_r[2]:<14} {_r[3]}")
if any(r[3] != "OK" for r in _rows):
    print("\n경고: 고정 버전과 다릅니다. 시험하지 않은 조합이라 오류가 날 수 있습니다.")
for _pkg, _why in [("av", "영상 읽기·쓰기 (PyAV)"), ("ftfy", "Wan 프롬프트 정리 (diffusers 가 사용)")]:
    try:
        print(f"{_pkg:<13} {_md.version(_pkg):<23} {_why}")
    except _md.PackageNotFoundError:
        raise ImportError(f"{_pkg} 가 필요합니다: pip install {_pkg}")

import contextlib, copy, gc, hashlib, importlib, json, math, os, random, re, shutil, sys, tempfile, time
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import diffusers
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model_state_dict
from tqdm.auto import tqdm

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
# 그래프의 한글: 설치된 한글 글꼴을 찾아 쓴다. 없으면 한글이 네모로 보이므로 경고만 끈다.
from matplotlib import font_manager
_KO_FONTS = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "Noto Sans KR", "Noto Sans CJK KR", "Noto Sans CJK JP",
             "Noto Sans CJK SC", "AppleGothic", "UnDotum")
_ko_font = next((f for f in _KO_FONTS if f in {x.name for x in font_manager.fontManager.ttflist}), None)
if _ko_font is None:   # matplotlib 글꼴 캐시를 만든 뒤에 설치한 글꼴: 시스템 글꼴 파일에서 직접 찾아 넣는다
    for _f in font_manager.findSystemFonts():
        if re.search(r"nanum|notosanscjk|notosanskr|malgun|applegothic|undotum", Path(_f).name, re.I):
            try:
                font_manager.fontManager.addfont(_f)
            except Exception:
                pass
    _ko_font = next((f for f in _KO_FONTS if f in {x.name for x in font_manager.fontManager.ttflist}), None)
if _ko_font:
    plt.rcParams["font.family"] = _ko_font
    try:   # 컨택트 시트(PIL)도 같은 글꼴 파일을 쓴다
        os.environ["LORA_KO_FONT"] = font_manager.findfont(font_manager.FontProperties(family=_ko_font), fallback_to_default=False)
    except Exception:
        pass
else:
    import warnings
    warnings.filterwarnings("ignore", message=r"Glyph .* missing from")
    print("참고: 한글 글꼴이 없어 그래프·컨택트 시트의 한글이 네모로 보입니다. 리눅스에서 관리자 권한이 없으면 NanumGothic.ttf 를 "
          "~/.local/share/fonts 에 복사하고 커널을 다시 시작합니다 (관리자: sudo apt install fonts-nanum).")
plt.rcParams["axes.unicode_minus"] = False
try:
    from IPython.display import Video, display
except ImportError:
    Video = display = None
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
''')

md(r"""
### 0-2. 설정

- 모델을 바꿀 때 고치는 값은 `USER` 의 처음 5개뿐이다: `TRIGGER`, `MODEL_PATH`, `RUN_NAME`, `DATASET_DIR`, `TARGET_SCOPE`.
  - 환경변수 `LORA_TRIGGER`, `LORA_MODEL_PATH`, `LORA_RUN_NAME`, `LORA_DATASET_DIR`, `LORA_TARGET_SCOPE` 로도 줄 수 있다 (`run_many.py` 가 이 방식을 쓴다).
  - `MODEL_PATH`: `model_index.json` 이 있는 diffusers 형식 폴더(`~` 가능) 또는 Hugging Face 저장소 이름. 모델 계열은 이 폴더의 설정 파일로 자동으로 정하고, 계열별 값(fps·버킷·검증 step·전문가 수)도 따라 바뀐다.
  - `DATASET_DIR`: 이미지·클립과 같은 이름의 캡션(.txt)이 든 폴더 (`~` 가능, 상대 경로는 이 노트북 폴더 기준).
  - `TARGET_SCOPE`: LoRA 를 붙일 층. `all`(어텐션 + FFN = 블록 안 선형층 전부, 기본), `attn`(어텐션만), `kvo`(어텐션의 k·v·o 만, 움직임 보존용). 이미지 v2 의 `attn_mlp` 는 `all`, `attn_all` 은 `attn` 으로 받는다.
  - `RUN_NAME`: 결과 폴더 이름 (`OUTPUT_DIR/RUN_NAME`). 모델·설정을 바꿀 때마다 바꾼다. 그 폴더에 저장된 LoRA 가 있으면 덮어쓰지 않게 여기서 멈춘다 (첫 저장 전에 멈춘 실행의 폴더는 이어서 쓴다).
- 아래 값은 보통 그대로 둔다.
  - `MODEL_FAMILY`: `auto` 면 위처럼 자동으로 정한다. 계열 이름을 적으면 모델 폴더와 맞는지 확인한다.
  - `EXPERT`: `auto`(A14B 는 저노이즈 → 고노이즈 순서로 둘 다, 클립이 없으면 저노이즈만. `both` 도 같다), `low`, `high`(클립 필요).
  - `CLASS_WORD`: 트리거 대신 넣는 일반 낱말 (그림체 변화 확인·보존 학습). `auto` 면 캡션에서 girl·boy·woman 등을 찾는다.
  - `FIXED_OVERRIDES`: 6 보완 단계가 만든 `next_run.json` 의 `fixed_overrides` (JSON 문자열이나 파일 경로). `FIXED` 에 있는 키만 받는다.
  - `VALIDATION_FILE`: 검증 프롬프트 JSON. 비우면 움직임이 있는 기본 4개를 쓴다.
    - 예: `{"prompts": ["full body, static camera, she waves"], "negative": "...", "seeds": [1001], "images": ["start_01.png"]}`
    - `images` 는 I2V 의 시작 이미지다 (`wan22_i2v` 에서 비우면 데이터셋 이미지나 클립 첫 프레임을 쓴다).
- `FIXED` 는 공통 시작값에 계열별 값을 덮어쓴 고정값이다. 대부분 근거가 약한 시작값이라 [제안] 이다. 출처는 주석에 있다.
- 랭크와 알파는 늘 같다. ComfyUI 는 alpha 값이 없는 LoRA 를 강도 1(알파 = 랭크)로 적용한다.

| 학습할 모델 | `MODEL_PATH` 예 (메인 서버) | `RUN_NAME` 예 | 학습 데이터 |
|---|---|---|---|
| Wan2.2-T2V-A14B | `/home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers` | `nkq7_wan22t2v_v1` | 이미지·클립 (이미지만 있으면 저노이즈 LoRA 만) |
| Wan2.2-I2V-A14B | `/home/mgs/models/diffusers/Wan2.2-I2V-A14B-Diffusers` | `nkq7_wan22i2v_v1` | 클립 필수 |
| Wan2.2-TI2V-5B | `/home/mgs/models/diffusers/Wan2.2-TI2V-5B-Diffusers` | `nkq7_wan22ti2v5b_v1` | 이미지·클립 |
| Wan2.1-T2V-14B·1.3B | `/home/mgs/models/diffusers/Wan2.1-T2V-14B-Diffusers` | `nkq7_wan21_14b_v1` | 이미지·클립 |
""")
code(r'''
TEST_FAMILY = os.environ.get("LORA_NB_TEST", "").strip()     # 시험 실행이면 모델 계열 이름
SMOKE = os.environ.get("LORA_NB_SMOKE", "").strip() == "1"   # 개발용: 실제 경로 그대로 두고 학습 크기만 시험 실행처럼 줄임

USER = MappingProxyType(dict(
    # ── 모델·실행마다 바꾸는 값: 이 5개만 고친다 ──
    TRIGGER=os.environ.get("LORA_TRIGGER", "nkq7"),
    MODEL_PATH=os.environ.get("LORA_MODEL_PATH", "/home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers"),
    RUN_NAME=os.environ.get("LORA_RUN_NAME", "nkq7_wan22t2v_v1"),
    DATASET_DIR=os.environ.get("LORA_DATASET_DIR", "~/lora_datasets/nkq7_final"),
    TARGET_SCOPE=os.environ.get("LORA_TARGET_SCOPE", "all"),                 # all / attn / kvo
    # ── 보통 그대로 둔다 ──
    MODEL_FAMILY=os.environ.get("LORA_MODEL_FAMILY", TEST_FAMILY or "auto"),   # auto = MODEL_PATH 의 설정 파일로 정함
    OUTPUT_DIR=os.environ.get("LORA_OUTPUT_DIR", "outputs_video"),
    EXPERT=os.environ.get("LORA_EXPERT", "auto"),                            # auto / both / low / high (A14B 만 의미 있음)
    CLASS_WORD=os.environ.get("LORA_CLASS_WORD", "auto"),                    # auto = 캡션에서 찾음 (girl·boy·woman …)
    VALIDATION_FILE=os.environ.get("LORA_VALIDATION_FILE", ""),
    FIXED_OVERRIDES=os.environ.get("LORA_FIXED_OVERRIDES", ""),              # 6 보완 단계의 fixed_overrides (JSON 문자열 또는 파일)
    CAPTION_FIXED_WORDS=os.environ.get("LORA_CAPTION_FIXED_WORDS", ""),      # 캡션에 쓰지 않을 고정 외형 낱말 (쉼표로 구분)
))
for _k in ("TRIGGER", "MODEL_PATH", "RUN_NAME", "DATASET_DIR", "TARGET_SCOPE"):
    if not USER[_k].strip():
        raise ValueError(f"{_k} 가 비어 있습니다")
if USER["TRIGGER"] != USER["TRIGGER"].strip():
    raise ValueError(f"TRIGGER='{USER['TRIGGER']}' 앞뒤에 공백이 있습니다")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", USER["RUN_NAME"]):   # run_many.py 와 같은 규칙
    raise ValueError(f"RUN_NAME='{USER['RUN_NAME']}' 는 영문·숫자로 시작하고 영문·숫자와 . _ - 만 씁니다 (예: nkq7_wan22t2v_v1)")
if USER["EXPERT"] not in ("auto", "both", "low", "high"):
    raise ValueError(f"EXPERT='{USER['EXPERT']}' 는 쓸 수 없습니다. auto / both / low / high")

# 학습 범위. 이미지 v2 이름도 받는다 (Wan 블록의 선형층은 어텐션 q·k·v·o 와 FFN 이 전부라 attn_mlp = all)
SCOPE_ALIASES = {"attn_mlp": "all", "attn_all": "attn"}
SCOPE = SCOPE_ALIASES.get(USER["TARGET_SCOPE"].strip(), USER["TARGET_SCOPE"].strip())
if SCOPE not in ("all", "attn", "kvo"):
    raise ValueError(f"TARGET_SCOPE='{USER['TARGET_SCOPE']}' 는 쓸 수 없습니다. all / attn / kvo (이미지 v2 의 attn_mlp·attn_all 도 받음)")

# 모델 계열 자동 인식: 설정 파일(JSON)만 읽는다. 0-4 에서 불러올 때 어댑터가 같은 값을 한 번 더 확인한다.
WAN_MODELS = {"wan22_t2v": "Wan2.2-T2V-A14B", "wan22_i2v": "Wan2.2-I2V-A14B", "wan22_ti2v_5b": "Wan2.2-TI2V-5B",
              "wan21": "Wan2.1-T2V-1.3B·14B"}

def norm_model_path(mp):
    """MODEL_PATH 정리: 앞뒤 공백, Hugging Face 주소(https://huggingface.co/org/name/...), 저장소 이름 끝의 '/' 를 뗀다."""
    s = str(mp).strip()
    if re.match(r"https?://(?:www\.)?huggingface\.co/", s):
        s = re.sub(r"/(?:tree|blob|resolve)/.*$", "", re.sub(r"^https?://(?:www\.)?huggingface\.co/", "", s))
    if not Path(s).expanduser().exists():
        s = s.rstrip("/\\")
    return s

def model_config_reader(mp):
    """MODEL_PATH 의 설정 파일을 읽는 함수를 돌려준다. 로컬 폴더 또는 Hugging Face 저장소 이름 (설정 파일만 받는다)."""
    mp = norm_model_path(mp)
    p = Path(mp).expanduser()
    if p.is_dir():
        return lambda rel: json.loads((p / rel).read_text(encoding="utf-8")) if (p / rel).is_file() else None
    if p.exists():
        raise ValueError(f"MODEL_PATH 가 파일입니다: {p}\n  model_index.json 이 있는 diffusers 형식 폴더를 적습니다 "
                         "(ComfyUI 용 .safetensors 한 파일은 읽지 못합니다).")
    repo = str(mp).strip()
    if p.is_absolute() or repo.startswith((".", "~")) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        raise FileNotFoundError(f"MODEL_PATH 폴더가 없습니다: {mp}\n  diffusers 형식 모델 폴더(model_index.json 이 있는 폴더)의 경로나 "
                                "Hugging Face 저장소 이름(org/name)을 적습니다. 예: /home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers")
    from huggingface_hub import hf_hub_download

    def read(rel):
        try:
            return json.loads(Path(hf_hub_download(repo, rel)).read_text(encoding="utf-8"))
        except Exception as e:
            if rel == "model_index.json":
                raise FileNotFoundError(f"Hugging Face 저장소 {repo} 의 model_index.json 을 받지 못했습니다 ({type(e).__name__}). "
                                        "저장소 이름(…-Diffusers)·인터넷·로그인을 확인합니다.") from e
            return None
    return read

def detect_family(mp):
    """model_index.json 과 transformer·vae 설정으로 Wan 계열을 정한다 (가중치는 읽지 않는다)."""
    read = model_config_reader(mp)
    idx = read("model_index.json")
    if idx is None:
        raise FileNotFoundError(f"{mp} 에 model_index.json 이 없습니다. diffusers 형식 모델(저장소 이름이 -Diffusers 로 끝남)이 필요합니다.")
    cls = idx.get("_class_name", "")
    two = bool((idx.get("transformer_2") or [None])[0]) and idx.get("boundary_ratio") is not None
    cin = (read("transformer/config.json") or {}).get("in_channels")
    zd = (read("vae/config.json") or {}).get("z_dim")
    expand, clip = bool(idx.get("expand_timesteps") or False), bool((idx.get("image_encoder") or [None])[0])
    if cls == "WanPipeline" and two and (cin, zd) == (16, 16) and not expand:
        return "wan22_t2v"
    if cls == "WanImageToVideoPipeline" and two and (cin, zd) == (36, 16) and not expand and not clip:
        return "wan22_i2v"
    if cls == "WanPipeline" and not two and (cin, zd) == (48, 48) and expand:
        return "wan22_ti2v_5b"
    if cls == "WanPipeline" and not two and (cin, zd) == (16, 16) and not expand:
        return "wan21"
    raise ValueError(f"{mp} 는 이 노트북이 학습하는 Wan 모델이 아닙니다 (파이프라인 {cls or '?'}, 전문가 {2 if two else 1}개, "
                     f"in_channels {cin}, VAE z_dim {zd}, expand_timesteps {expand}{', CLIP 이미지 인코더 있음' if clip else ''}).\n"
                     "  지원: " + ", ".join(f"{v}-Diffusers" for v in WAN_MODELS.values())
                     + " (Wan2.1 I2V·FLF2V·VACE 등은 지원하지 않음)")

if TEST_FAMILY:
    FAMILY = TEST_FAMILY
else:
    FAMILY = detect_family(USER["MODEL_PATH"])
    if USER["MODEL_FAMILY"].strip() not in ("", "auto") and USER["MODEL_FAMILY"].strip() != FAMILY:
        raise ValueError(f"MODEL_FAMILY='{USER['MODEL_FAMILY']}' 인데 MODEL_PATH 는 {FAMILY} ({WAN_MODELS[FAMILY]}) 모델입니다. "
                         "MODEL_FAMILY 를 auto 로 두면 모델 폴더에 맞춰 정합니다.")

COMMON_FIXED = dict(
    seed=42,
    # LoRA: 블록 안 선형층만 (musubi-tuner·DiffSynth·ai-toolkit 과 같은 범위).
    #   target_scope 는 USER 의 TARGET_SCOPE: all = 어텐션 q·k·v·o + FFN, attn = 어텐션만, kvo = k·v·o 만 (움직임 보존 연구 V-Warper)
    rank=16, alpha=16, dropout=0.0, init_lora_weights="gaussian", target_scope=SCOPE,
    # AdamW, 전문가마다 새로 만든다. 학습률 1e-4 (alpha = rank 기준 ai-toolkit·DiffSynth·finetrainers 예시)
    learning_rate=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
    lr_scheduler="constant_with_warmup", warmup_steps=50,
    # 학습 길이. A14B 는 저노이즈 몫(low_steps_share) → 고노이즈 나머지 순서 [제안, 실무자 경험 70~80%]
    max_train_steps=2000, low_steps_share=0.75, high_uses_images=False,
    max_grad_norm=1.0,
    # 노이즈 단계: logit-normal(0,1) → shift → 전문가 구간 안으로 거절 샘플링 (musubi preserve_distribution_shape)
    #   shift_low·shift_high 는 A14B 전문가별, shift 는 전문가가 하나인 계열 [제안, musubi 토론 #455·예시]
    logit_mean=0.0, logit_std=1.0, shift_low=1.0, shift_high=5.0, shift=3.0,
    # 데이터: 이미지·클립이 다 있을 때 이미지를 뽑을 확률, 캡션 비우기 확률, 움직임이 이보다 작은 클립 제외
    image_share=0.5, caption_dropout=0.0, min_motion=0.0, min_frames=17,
    # 버킷: 토큰 = 잠재 프레임 × (H/TOKEN_DIV) × (W/TOKEN_DIV). 14,040 = 33프레임 480x832 [제안, 파일럿 시작점]
    frame_options=(1, 17, 33, 49, 81), token_budget=14040, max_area=480 * 832,
    aspects=(480 / 832, 832 / 480, 2 / 3, 3 / 2, 1.0, 704 / 1280, 1280 / 704), allow_upscale=1.0,
    window_mode="uniform", windows_per_segment=3, analysis_side=96, ti2v_i2v_share=0.5,
    # 검증 손실용으로 학습에서 뺄 파일 수. 이미지·클립 종류마다 10개당 1개까지만 빼서, 종류마다 학습 파일이 남는다
    holdout_files=2,
    # 그림체 보존 학습(DOP, v2 노트북과 같은 방식). 기본 끔, 켜면 해당 단계마다 디노이저를 3번 계산
    preservation="off", preservation_weight=1.0, preservation_every=2,
    # 정밀도·메모리. offload_sampling: 미리보기 생성 때 두 전문가를 번갈아 GPU 에 올림
    mixed_precision="bf16", gradient_checkpointing=True, offload_sampling=True,
    # 저장·검증
    save_every=250, validate_every=250, val_loss_items=4,
    preview_prompts=2, preview_frames=17, preview_steps=20, final_frames=33, final_steps=0,   # 0 = 공식 step 수
    use_dino=True, use_ccip=True, max_nan_steps=10,
)
# 계열별 값 (여기 없는 값은 공통값). TI2V-5B 는 24fps, 32 배수, 공식 해상도 1280x704 면적까지.
#   토큰 예산 12,090 = 121프레임 480p 또는 49프레임 720p (긴 클립은 해상도를 낮춰 프레임을 지킨다)
FAMILY_FIXED = {
    "wan22_ti2v_5b": dict(frame_options=(1, 25, 49, 73, 97, 121), token_budget=12090, max_area=704 * 1280,
                          min_frames=25, preview_frames=25, final_frames=49),
}

def parse_overrides(text):
    """FIXED_OVERRIDES: JSON 문자열 또는 JSON 파일 경로. FIXED 에 있는 키만, 같은 종류의 값만 받는다."""
    if not text.strip():
        return {}
    p = Path(text).expanduser()
    raw = json.loads(p.read_text(encoding="utf-8-sig") if p.suffix == ".json" and p.is_file() else text)
    if isinstance(raw, dict) and "runs" in raw:          # next_run.json 을 통째로 준 경우
        raw = raw["runs"][0].get("fixed_overrides", {})
    if not isinstance(raw, dict):
        raise ValueError("FIXED_OVERRIDES 는 {키: 값} 이어야 합니다")
    out = {}
    for k, v in raw.items():
        if k not in COMMON_FIXED:
            raise ValueError(f"FIXED_OVERRIDES: 모르는 키 '{k}'")
        d = COMMON_FIXED[k]
        if isinstance(d, tuple):
            v = tuple(v)
        elif isinstance(d, bool) or isinstance(v, bool):
            if not isinstance(v, bool) or not isinstance(d, bool):
                raise ValueError(f"FIXED_OVERRIDES: '{k}' 값의 종류가 다릅니다 ({v!r})")
        elif isinstance(d, (int, float)):
            if not isinstance(v, (int, float)):
                raise ValueError(f"FIXED_OVERRIDES: '{k}' 는 숫자여야 합니다 ({v!r})")
            v = type(d)(v) if isinstance(d, float) or float(v).is_integer() else v
            # 0 이 뜻 있는 값만 0 을 받는다: 시드 0, 워밍업 없음, 검증용 파일 없음, 공식 step 수, 첫 NaN 에서 멈춤
            _lo = 0 if k in ("seed", "warmup_steps", "holdout_files", "final_steps", "max_nan_steps") else 1
            if isinstance(d, int) and (not isinstance(v, int) or v < _lo):
                raise ValueError(f"FIXED_OVERRIDES: '{k}' 는 {_lo} 이상의 정수여야 합니다 ({v!r})")
        elif not isinstance(v, type(d)):
            raise ValueError(f"FIXED_OVERRIDES: '{k}' 값의 종류가 다릅니다 ({v!r})")
        out[k] = v
    if "rank" in out:
        out["alpha"] = out["rank"]                       # 랭크를 바꾸면 알파도 같이
    return out

OVERRIDES = parse_overrides(USER["FIXED_OVERRIDES"])
FIXED = MappingProxyType({**COMMON_FIXED, **FAMILY_FIXED.get(FAMILY, {}), **OVERRIDES})
if FIXED["alpha"] != FIXED["rank"]:
    raise ValueError(f"alpha({FIXED['alpha']}) 와 rank({FIXED['rank']}) 가 다릅니다. 둘을 같게 둡니다.")

TEST_DEVICE = os.environ.get("LORA_NB_TEST_DEVICE", "cpu")   # 시험 실행 장치: cpu(fp32) 또는 cuda(bf16)
if TEST_FAMILY or SMOKE:   # 시험 실행: 같은 코드 경로를 아주 작은 값으로
    _mult = 32 if FAMILY == "wan22_ti2v_5b" else 16
    # 6단계: A14B 는 저노이즈 4 + 고노이즈 2 (워밍업 1단계의 학습률은 0 이라 전문가마다 2단계 이상 필요)
    FIXED = MappingProxyType({**FIXED, "max_train_steps": 6, "warmup_steps": 1, "save_every": 2, "validate_every": 2,
                              "val_loss_items": 2, "frame_options": (1, 5, 9), "min_frames": 5,
                              "max_area": (2 * _mult) ** 2, "token_budget": 3 * 4, "analysis_side": 48,
                              "preview_prompts": 1, "preview_frames": 5, "preview_steps": 2, "final_frames": 5,
                              "final_steps": 2, "windows_per_segment": 2,
                              "mixed_precision": "bf16" if TEST_DEVICE == "cuda" else "fp32"})
    if TEST_FAMILY:   # 시험 모드는 DINO·CCIP 를 받지 않는다 (작은 T5 는 받음). SMOKE 는 실제처럼 켠다 (없으면 이유를 보여 주고 끔)
        FIXED = MappingProxyType({**FIXED, "use_dino": False, "use_ccip": False})

def model_notes(family, path):
    """계열·모델 경로 이름으로 알려 주는 주의 사항."""
    p, notes = str(path).lower(), []
    if family == "wan22_i2v":
        notes.append("I2V 는 클립이 있어야 학습합니다 (이미지만 있으면 0-3 에서 멈춥니다).")
    if family == "wan22_ti2v_5b":
        notes.append("TI2V-5B 로 만든 LoRA 는 A14B 에 쓸 수 없습니다 (파일럿용).")
    if family == "wan21" and "1.3b" in p:
        notes.append("Wan2.1 1.3B 는 코드 점검용입니다. 품질 비교에는 14B 이상을 씁니다.")
    if "anisora" in p:
        notes.append("Index-AniSora 는 Apache-2.0 에 추가 조건(배상·하위 사용자 구속)이 있어 법무 확인이 필요합니다.")
    if "turbo" in p or "distill" in p or "lightx2v" in p:
        notes.append("증류(적은 step) 모델로 보입니다. 증류하지 않은 원본으로 학습하고 LoRA 는 증류판에서 씁니다.")
    return notes

OUTPUT_PATH = Path(USER["OUTPUT_DIR"]).expanduser()
DATASET_PATH = Path(USER["DATASET_DIR"]).expanduser()
if not TEST_FAMILY:
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(f"DATASET_DIR 폴더가 없습니다: {DATASET_PATH.resolve()}\n  이미지·클립과 캡션(.txt)이 든 폴더의 "
                                "경로를 적습니다 (상대 경로는 이 노트북 폴더 기준, ~ 가능).")
    # 저장된 LoRA·결과가 있는 폴더는 덮어쓰지 않는다. 첫 저장 전에 멈춘 실행의 폴더는 이어서 쓴다.
    _run_dir = OUTPUT_PATH / USER["RUN_NAME"]
    _done = sorted(_run_dir.glob("checkpoint-*/*.safetensors")) + sorted(_run_dir.glob("summary.json"))
    if _done:
        raise FileExistsError(f"{_run_dir.resolve()} 에 이전 실행의 결과({_done[0].name} 등)가 있습니다. 덮어쓰지 않도록 RUN_NAME 을 "
                              "바꿉니다 (예: _v1 → _v2). 지워도 되는 결과면 폴더를 지운 뒤 다시 실행합니다.")
    if _run_dir.is_dir() and any(_run_dir.iterdir()):
        print(f"참고: {_run_dir} 에 첫 저장 전에 멈춘 실행의 파일이 있습니다. 같은 폴더에 이어서 씁니다.")

# 검증 프롬프트: 움직임이 있는 기본 4개, 또는 VALIDATION_FILE(JSON)
_TRIG = re.compile(r"(?<!\w)" + re.escape(USER["TRIGGER"]) + r"(?![A-Za-z0-9_])")
_T = USER["TRIGGER"]
VALIDATION_PROMPTS = [
    f"{_T}, medium shot, static camera, she turns from the side to face the camera, then smiles, plain light gray background, clean anime illustration",
    f"{_T}, full body, static camera, she walks forward along a forest path, a light breeze moves the leaves, clean anime illustration",
    f"{_T}, close-up, static camera, she blinks and tilts her head, soft daylight, clean anime illustration",
    f"{_T}, full body, camera slowly pans right, she raises one arm and waves, summer meadow, clean anime illustration",
]
VALIDATION_NEGATIVE = None          # None = 어댑터의 공식 Wan 부정 프롬프트
VALIDATION_SEEDS, VALIDATION_IMAGES = None, []
if USER["VALIDATION_FILE"]:
    _vf = Path(USER["VALIDATION_FILE"]).expanduser()
    if not _vf.is_file():
        raise FileNotFoundError(f"VALIDATION_FILE 이 없습니다: {_vf}")
    _v = json.loads(_vf.read_text(encoding="utf-8-sig"))
    _p = _v.get("prompts") if isinstance(_v, dict) else None
    if not isinstance(_p, list) or not _p or not all(isinstance(x, str) and x.strip() for x in _p):
        raise ValueError(f"{_vf.name}: prompts 는 비어 있지 않은 문자열 목록이어야 합니다")
    VALIDATION_PROMPTS = [x.strip() if _TRIG.search(x) else f"{_T}, {x.strip()}" for x in _p]
    VALIDATION_NEGATIVE = _v.get("negative")
    VALIDATION_SEEDS = _v.get("seeds")
    VALIDATION_IMAGES = [str((_vf.parent / x) if not Path(x).is_absolute() else x) for x in _v.get("images", [])]
    for _im in VALIDATION_IMAGES:
        if not Path(_im).is_file():
            raise FileNotFoundError(f"검증 시작 이미지가 없습니다: {_im}")
if VALIDATION_SEEDS is None:
    VALIDATION_SEEDS = [1001 + i for i in range(len(VALIDATION_PROMPTS))]
if not isinstance(VALIDATION_SEEDS, list) or len(VALIDATION_SEEDS) != len(VALIDATION_PROMPTS) \
        or not all(isinstance(s, int) and not isinstance(s, bool) for s in VALIDATION_SEEDS):
    raise ValueError(f"검증 seeds 는 prompts 와 개수가 같은 정수 목록이어야 합니다 (prompts {len(VALIDATION_PROMPTS)}개)")
if VALIDATION_NEGATIVE is not None and not isinstance(VALIDATION_NEGATIVE, str):
    raise ValueError("검증 negative 는 문자열이어야 합니다")
FIXED_WORDS = [w.strip() for w in USER["CAPTION_FIXED_WORDS"].split(",") if w.strip()]

print(f"모델 계열: {FAMILY} ({WAN_MODELS.get(FAMILY, '?')})"
      + (" ← MODEL_PATH 에서 자동 인식" if not TEST_FAMILY and USER["MODEL_FAMILY"].strip() in ("", "auto") else ""))
if not TEST_FAMILY:
    for _note in model_notes(FAMILY, USER["MODEL_PATH"]):
        print("주의:", _note)
print(json.dumps(dict(USER), ensure_ascii=False, indent=1))
print("계열별 값:", FAMILY_FIXED.get(FAMILY) or "없음 (공통값)", "| 덮어쓴 값:", OVERRIDES or "없음")
if SCOPE != USER["TARGET_SCOPE"].strip():
    print(f"TARGET_SCOPE {USER['TARGET_SCOPE']} → {SCOPE} (영상 모델에서 같은 범위)")
if FIXED["target_scope"] != SCOPE:
    print(f"참고: FIXED_OVERRIDES 의 target_scope={FIXED['target_scope']} 를 씁니다 (TARGET_SCOPE={SCOPE} 대신)")
print(f"랭크 {FIXED['rank']} | 학습률 {FIXED['learning_rate']} | 단계 {FIXED['max_train_steps']} | 범위 {FIXED['target_scope']}"
      f" | 토큰 예산 {FIXED['token_budget']} | 검증 프롬프트 {len(VALIDATION_PROMPTS)}개")
''')

md(r"""
### 0-3. 영상 도구와 모델 계열별 어댑터 (수정하지 않는다)

- `lora_video_tools/`: Wan 공통 코드(`wan_common`), 영상 읽기·분석·전처리(`video_io`), 검증 지표(`video_metrics`), 보완 규칙(`remedy`).
- `lora_video_adapters/`: 계열마다 같은 함수 이름(`encode_prompts`, `encode_video`, `forward_loss`, `save_lora`, `generate`)을 가진 어댑터. 선택한 계열 하나만 불러온다.
- Wan 학습 공식 스크립트는 diffusers 에 없어서, 각 어댑터는 diffusers v0.42.1 파이프라인의 계산과 대조했고 학습 방식은 musubi-tuner·DiffSynth-Studio·ai-toolkit 과 맞췄다. 작은 무작위 모델로 CPU·GPU 에서 처음부터 끝까지 시험했고, 실제 가중치로는 아직 학습해 보지 않았다.
- LoRA 는 ComfyUI 가 읽는 Wan 원래 이름(`diffusion_model.blocks.N.self_attn.q.lora_A.weight` 등)으로 저장한다. diffusers 도 같은 파일을 그대로 불러온다.
""")
code('os.makedirs("lora_video_tools", exist_ok=True)\nos.makedirs("lora_video_adapters", exist_ok=True)')
for sub, name in TOOLS:
    src = (HERE / sub / f"{name}.py").read_text(encoding="utf-8").rstrip() + "\n"
    code(f"%%writefile lora_video_tools/{name}.py\n" + src)
for fam in FAMILIES:
    src = (HERE / "video_adapters" / f"{fam}.py").read_text(encoding="utf-8").rstrip() + "\n"
    code(f"%%writefile lora_video_adapters/{fam}.py\n" + src)
code(r'''
for _d in (Path("lora_video_tools").resolve(), Path("lora_video_adapters").resolve()):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))
for _m in ["wan_common", "video_io", "video_metrics", "remedy"] + sorted(p.stem for p in Path("lora_video_adapters").glob("*.py")):
    if _m in sys.modules:                     # 셀을 다시 실행하면 새 파일로 다시 불러온다
        importlib.reload(sys.modules[_m])
import video_io as V
import video_metrics as VM
import remedy as RM
SUPPORTED = sorted(p.stem for p in Path("lora_video_adapters").glob("*.py"))
if FAMILY not in SUPPORTED:
    raise ValueError(f"{'LORA_NB_TEST' if TEST_FAMILY else 'MODEL_FAMILY'}='{FAMILY}' 는 지원하지 않는 계열입니다. "
                     f"고를 수 있는 값: {SUPPORTED}")
A = importlib.import_module(FAMILY)
for _name in ["FAMILY", "PIPELINE_CLASS", "EXPERTS", "EXPERT_TAGS", "CONDITIONING", "FPS", "SPATIAL_FACTOR",
              "SPATIAL_MULTIPLE", "TEMPORAL_FACTOR", "TOKEN_DIV", "VAL_STEPS", "VAL_SHIFT", "VAL_GUIDANCE",
              "NEGATIVE_PROMPT", "LORA_TARGETS", "sigma_range", "load_pipeline", "tiny_pipeline", "encode_prompts",
              "encode_video", "forward_loss", "lora_to_comfy", "save_lora", "load_lora", "generate"]:
    if not hasattr(A, _name):
        raise AttributeError(f"어댑터 {FAMILY} 에 {_name} 가 없습니다")
if FIXED["target_scope"] not in A.LORA_TARGETS:
    raise ValueError(f"target_scope='{FIXED['target_scope']}' 는 쓸 수 없습니다: {list(A.LORA_TARGETS)}")
for _k, _v in OVERRIDES.items():                  # 보완 규칙(remedy.ALLOWED)에 범위·선택지가 있는 값은 그 안인지
    _al = RM.ALLOWED.get(_k)
    if isinstance(_al, tuple) and len(_al) == 2 and all(isinstance(x, (int, float)) for x in _al) \
            and isinstance(_v, (int, float)):
        if not _al[0] <= _v <= _al[1]:
            raise ValueError(f"FIXED_OVERRIDES: {_k}={_v} 가 허용 범위 [{_al[0]}, {_al[1]}] 밖입니다")
    elif isinstance(_al, tuple) and _al and all(isinstance(x, str) for x in _al) and _v not in _al:
        raise ValueError(f"FIXED_OVERRIDES: {_k}='{_v}' 는 {', '.join(_al)} 중 하나여야 합니다")
# 학습 셀(3-1·4-3)이 쓰는 값으로 같은 객체를 한 번 만들어 본다: 값이 틀리면 모델(수십 GB)을 불러오기 전에 멈춘다
try:
    _buckets = V.make_buckets(FIXED["token_budget"], FIXED["frame_options"], FIXED["aspects"], FIXED["max_area"],
                              A.TOKEN_DIV, A.TEMPORAL_FACTOR, A.SPATIAL_MULTIPLE)
    _clip_buckets = [b for b in _buckets if b[0] >= FIXED["min_frames"]]          # 3-1 의 CLIP_BUCKETS 와 같음
    V.windows(FIXED["min_frames"], FIXED["min_frames"], FIXED["window_mode"], FIXED["windows_per_segment"],
              np.random.default_rng(0))
    _o = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=FIXED["learning_rate"], betas=FIXED["betas"],
                           weight_decay=FIXED["weight_decay"], eps=FIXED["eps"])
    get_scheduler(FIXED["lr_scheduler"], optimizer=_o, num_warmup_steps=FIXED["warmup_steps"],
                  num_training_steps=FIXED["max_train_steps"])
except Exception as _e:
    raise ValueError(f"FIXED_OVERRIDES {OVERRIDES} 로는 학습을 준비할 수 없습니다: {type(_e).__name__}: {_e}") from None
# 오류 없이 학습을 망치는 값: clip_grad_norm_(…, 0) 은 '자르지 않음' 이 아니라 기울기를 0 으로 만들어 LoRA 가 0 으로 남는다
_bad = [k for k, good in (("max_grad_norm", FIXED["max_grad_norm"] > 0), ("logit_std", FIXED["logit_std"] > 0),
                           ("dropout", 0 <= FIXED["dropout"] < 1), ("caption_dropout", 0 <= FIXED["caption_dropout"] < 1),
                           ("image_share", 0 <= FIXED["image_share"] <= 1),
                           ("init_lora_weights", FIXED["init_lora_weights"] == "gaussian"),
                           ("mixed_precision", FIXED["mixed_precision"] in ("bf16", "fp16", "fp32")),
                           ("frame_options", max(FIXED["frame_options"]) >= FIXED["min_frames"]),
                           ("allow_upscale", FIXED["allow_upscale"] >= 1), ("aspects·max_area", bool(_buckets))) if not good]
if _bad:
    raise ValueError(f"FIXED_OVERRIDES 의 {', '.join(_bad)} 값을 쓸 수 없습니다: max_grad_norm·logit_std 는 0 보다 크고, "
                     "dropout·caption_dropout 은 0 이상 1 미만, image_share 는 0~1, init_lora_weights 는 gaussian, "
                     "mixed_precision 은 bf16·fp16·fp32 중 하나, frame_options 에는 min_frames 이상인 값이 있어야 하고, "
                     "allow_upscale 은 1 이상 (1.0 = 업스케일 없음), aspects·max_area 로 버킷이 하나 이상 나와야 합니다")
NEGATIVE = VALIDATION_NEGATIVE if VALIDATION_NEGATIVE is not None else A.NEGATIVE_PROMPT

# 3-1 의 버킷 고르기. 아래 데이터셋 확인(모델을 불러오기 전)과 3-1 이 같은 규칙을 쓰도록 여기서 정의한다
def own_buckets(w, h):
    """원본이 기본 버킷보다 작을 때 쓸 버킷: 비율마다 원본 안에 들어가는 가장 큰 넓이를 상한으로 만든다 (업스케일 없음).
    그 넓이가 max_area 의 1/8 보다 작으면 만들지 않는다 (너무 작은 원본은 학습에서 뺀다)."""
    out = set()
    for a in FIXED["aspects"]:
        cap = min(FIXED["max_area"], min(w, h * a) * min(h, w / a))
        if cap >= FIXED["max_area"] / 8:
            out.update(V.make_buckets(FIXED["token_budget"], FIXED["frame_options"], (a,), cap,
                                      A.TOKEN_DIV, A.TEMPORAL_FACTOR, A.SPATIAL_MULTIPLE))
    return sorted(out)

def fit_bucket(buckets, n, w, h, image):
    """기본 버킷에서 고르고, 없으면 원본 크기로 만든 버킷에서 고른다 (기본 버킷이 맞는 파일은 결과가 그대로다)."""
    b = V.pick_bucket(buckets, n, w, h, prefer="frames", allow_upscale=FIXED["allow_upscale"])
    if b is None:
        own = [x for x in own_buckets(w, h) if (x[0] == 1 if image else x[0] >= FIXED["min_frames"])]
        b = V.pick_bucket(own, n, w, h, prefer="frames", allow_upscale=FIXED["allow_upscale"])
    return b

# 데이터셋을 모델(수십 GB)보다 먼저 확인한다
if not TEST_FAMILY:
    _media = [f for f in DATASET_PATH.iterdir() if f.is_file() and f.suffix.lower() in V.IMG_EXT | V.VID_EXT]
    if not _media:
        raise FileNotFoundError(f"{DATASET_PATH} 에 이미지·영상 파일이 없습니다 (확장자 {sorted(V.IMG_EXT | V.VID_EXT)})")
    _videos = [f for f in _media if f.suffix.lower() in V.VID_EXT or V.is_animated_image(f)]
    print(f"데이터셋 {DATASET_PATH.resolve()}: 파일 {len(_media)}개 (영상·움직이는 그림 {len(_videos)}개)")

    def _is_clip(f):
        """3-1 이 학습 창을 고를 수 있는 영상: 모델 fps 로 리샘플한 프레임 수(V.resample_to_fps 와 같은 식)와 원본 크기로
        3-1 의 fit_bucket 이 클립 버킷을 고른다. 1프레임 영상은 정지 그림으로 다룬다.
        (장면 전환·멈춤으로 줄어드는 깨끗한 구간과 min_motion 은 1-2 분석 뒤에야 알아서 여기서는 보지 않는다)"""
        try:
            m = V.probe(f)
            n = math.floor(m["duration"] * A.FPS + 0.5 + 1e-9)
            return m["frames"] > 1 and fit_bucket(_clip_buckets, n, m["width"], m["height"], False) is not None
        except Exception:
            return False

    if A.CONDITIONING == "i2v_concat" and not any(_is_clip(f) for f in _videos):
        _fm = min((b[0] for b in _clip_buckets), default=FIXED["min_frames"])   # 가장 짧은 학습 창 (3-1 과 같음)
        raise ValueError(f"{A.FAMILY} 는 클립이 있어야 학습합니다 (1프레임 이미지는 조건 채널에 정답이 들어가 배우는 것이 없음). "
                         f"모델 fps({A.FPS})로 {_fm}프레임(약 {_fm / A.FPS:.1f}초, min_frames {FIXED['min_frames']} 이상인 가장 "
                         f"짧은 학습 창) 이상이고, 쓸 수 있는 넓이가 max_area 의 1/8({FIXED['max_area'] // 8:,}화소) 이상인 클립이 "
                         "필요합니다. 이미지만 있으면 T2V-A14B·TI2V-5B·Wan2.1 모델 경로로 먼저 학습하고, 클립이 생기면 I2V 로 학습합니다.")
print("어댑터:", A.FAMILY, "| 파이프라인:", A.PIPELINE_CLASS.__name__, "| 전문가:", A.EXPERTS, "| 조건:", A.CONDITIONING,
      f"| {A.FPS}fps, H·W 는 {A.SPATIAL_MULTIPLE} 배수, 프레임 4n+1")
print("LoRA 대상:", A.LORA_TARGETS[FIXED["target_scope"]])
''')

md("### 0-4. 장치·정밀도 결정, 시드 고정, 모델 불러오기")
code(r'''
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(FIXED["seed"])
if TEST_FAMILY or SMOKE:
    DEVICE = torch.device("cuda" if TEST_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU 가 보이지 않습니다 (torch.cuda.is_available() = False). 실제 모델은 GPU 에서만 학습합니다.\n"
                           "컨테이너를 GPU 와 함께 띄웠는지(docker run --gpus all, 안에서 nvidia-smi), torch 가 CUDA 판"
                           "(torch.__version__ 끝이 +cu…)인지 확인하고 커널을 다시 시작합니다.")
    DEVICE = torch.device("cuda")
if FIXED["mixed_precision"] == "bf16" and DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    WEIGHT_DTYPE = torch.bfloat16
elif FIXED["mixed_precision"] in ("bf16", "fp16") and DEVICE.type == "cuda":
    WEIGHT_DTYPE = torch.float16
else:
    WEIGHT_DTYPE = torch.float32
print("장치:", DEVICE, "| 가중치 dtype:", WEIGHT_DTYPE)

def check_model_path(mp):
    """0-2 의 detect_family 가 확인한 경로. 로컬 폴더면 ~ 를 펼친 절대 경로, 아니면 Hugging Face 저장소 이름."""
    mp = norm_model_path(mp)
    p = Path(mp).expanduser()
    return str(p.resolve()) if p.is_dir() else mp

# 같은 커널에서 다시 실행하면 앞 실행의 모델·학습 상태(A14B 면 수십 GB)를 먼저 놓아 준다.
# 셀 오류·중지 뒤에는 IPython 이 마지막 traceback 의 프레임(지역 변수에 모델)을 쥐고 있으므로 그것부터 비운다.
# 프레임이 쥔 함수의 클로저(체크포인트 재계산, DOP 의 run)도 모델을 쥐므로 traceback 자체도 놓는다 (sys.last_*, %debug 용 사본,
# 이 셀을 오류 셀 바로 다음에 실행하면 IPython 이 이 셀이 끝날 때까지 쥐는 앞 셀의 실행 결과 last_execution_result).
# 학습 셀(4-3)이 도중에 멈추면 전역 loss·pl 이 계산 그래프로 앞 모델의 가중치를 쥐고 있으므로 함께 지운다.
import traceback
try:
    traceback.clear_frames(getattr(sys, "last_traceback", None))
except Exception:
    pass
for _a in ("last_type", "last_value", "last_traceback", "last_exc"):
    if hasattr(sys, _a):
        setattr(sys, _a, None)
try:
    get_ipython().InteractiveTB.tb = None
    _r = get_ipython().last_execution_result
    if _r is not None:
        _r.error_in_exec = _r.error_before_exec = None
    del _r
except Exception:
    pass
for _name in ("pipe", "den", "optimizer", "params", "lr_scheduler", "_c", "p", "SAMPLES", "VAL_NOISE", "PREVIEW_HISTORY",
              "DINO", "CCIP", "TEXT", "VAL_TEXT", "DRIFT_TEXT", "PRES_TEXT", "ckpt_paths", "loss", "pl", "_captured"):
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 불러오기 전에 메모리를 확인한다. A14B 는 미리보기·최종 생성 때 두 전문가와 텍스트 인코더가 모두 CPU 메모리에 있다(약 69GB).
# 모자라면 불러오는 도중이나 첫 미리보기에서 커널이 이유를 남기지 않고 죽는다 ('kernel died').
def model_weights_gb(folder, dtype):
    """구성요소별로 불러올 가중치 크기(GB)와, 가중치 파일이 다 있지 않은 구성요소 목록.
    safetensors 머리말만 읽어 원소 수를 세고, load_pipeline 처럼 VAE 는 fp32, 나머지는 dtype 으로 센다."""
    folder = Path(folder)
    nbytes = torch.tensor([], dtype=dtype).element_size()
    sizes, missing = {}, []
    for k, v in json.loads((folder / "model_index.json").read_text(encoding="utf-8")).items():
        sub = folder / k
        if not (isinstance(v, list) and v and v[0] is not None):
            continue
        if not sub.is_dir():
            missing.append(k)
            continue
        if not (sub / "config.json").is_file():        # 토크나이저·스케줄러 (가중치 없음)
            continue
        files = None
        for stem in ("diffusion_pytorch_model", "model"):
            ix = sub / f"{stem}.safetensors.index.json"
            if ix.is_file():   # 나뉜 파일: 목록의 조각이 모두 있어야 한다
                try:
                    wm = json.loads(ix.read_text(encoding="utf-8"))["weight_map"]
                    files = [sub / f for f in sorted(set(wm.values()))]
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
                    raise OSError(f"가중치 목록 파일을 읽을 수 없습니다: {ix} ({type(e).__name__}: {e}). 복사가 끊기거나 깨진 "
                                  "파일이면 hf download <저장소> --local-dir <모델 폴더> 로 다시 받습니다.") from None
            elif (sub / f"{stem}.safetensors").is_file():
                files = [sub / f"{stem}.safetensors"]
            if files:
                break
        if not files or not all(f.is_file() for f in files):
            missing.append(k)
            continue
        n = 0
        for f in files:
            size = f.stat().st_size
            try:
                with open(f, "rb") as fh:
                    hl = int.from_bytes(fh.read(8), "little")
                    if not 2 <= hl <= min(size - 8, 100_000_000):     # safetensors 머리말 길이 한도
                        raise ValueError(f"머리말 길이 {hl}")
                    head = json.loads(fh.read(hl))
                ts = [t for key, t in head.items() if key != "__metadata__"]
                end = max((t["data_offsets"][1] for t in ts), default=0)
                if size != 8 + hl + end:          # safetensors 도 크기가 정확히 맞아야 읽는다 (데이터가 잘리거나 덧붙은 파일)
                    raise ValueError(f"파일 크기가 머리말 기준 {8 + hl + end:,}바이트와 다름")
                n += sum(math.prod(t["shape"]) for t in ts)
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
                raise OSError(f"가중치 파일을 읽을 수 없습니다: {f} ({size:,}바이트, {type(e).__name__}: {e}). git-lfs 포인터·"
                              "0바이트·데이터가 잘린 파일이면 hf download <저장소> --local-dir <모델 폴더> 로 다시 받습니다.") from None
        sizes[k] = n * (4 if k == "vae" else nbytes) / 1e9
    return sizes, missing

def free_ram_gb():
    """(지금 쓸 수 있는 시스템 메모리, 다른 프로그램을 모두 멈췄을 때 이 커널이 더 쓸 수 있는 메모리, 상한을 정한 컨테이너
    메모리 한도 또는 None) GB. psutil 이 없으면 (None, None, None).
    리눅스 컨테이너에 메모리 한도(cgroup)가 있으면 그 한도 안에서 잰다. cgroup v2 는 /proc/self/cgroup 의 경로와 그 위 단계의
    memory.max 를 모두 본다 (--cgroupns=host 컨테이너는 /sys/fs/cgroup 이 호스트 루트다). v1 은 hierarchical_memory_limit 도 본다.
    커널이 회수하는 파일 캐시(inactive·active file)는 빈 것으로 센다 ('file'·'cache' 는 공유 메모리까지 들어가 쓰지 않는다)."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        free, cap, rss = vm.available, vm.total, psutil.Process().memory_info().rss
    except Exception:
        return None, None, None
    lim_gb, rel = None, ""
    try:
        rel = next((l[3:].strip().strip("/") for l in Path("/proc/self/cgroup").read_text().splitlines()
                    if l.startswith("0::")), "")
    except OSError:
        pass
    parts = [] if ".." in rel.split("/") else [x for x in rel.split("/") if x]
    v2 = [("/sys/fs/cgroup/" + "/".join(parts[:i])).rstrip("/") for i in range(len(parts) + 1)]
    for d, lim, cur in [(x, "memory.max", "memory.current") for x in v2] + \
                       [("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes")]:
        try:
            raw = (Path(d) / lim).read_text().strip()
            stat = dict(l.split()[:2] for l in (Path(d) / "memory.stat").read_text().splitlines() if len(l.split()) >= 2)
            limit = min(1 << 62 if raw == "max" else int(raw), int(stat.get("hierarchical_memory_limit", 1 << 62)))
            if limit >= 1 << 60:
                continue
            reclaim = sum(int(stat.get("total_" + k, stat.get(k, 0))) for k in ("inactive_file", "active_file"))
            free = min(free, limit - int((Path(d) / cur).read_text()) + reclaim)
            if limit < cap:
                cap, lim_gb = limit, limit / 1e9
        except (OSError, ValueError):
            pass
    return free / 1e9, (cap - rss) / 1e9, lim_gb

def gpu_need_gb(folder, w, tokens, offload):
    """GPU 에 비어 있어야 할 메모리(GB) 추정 [추정: 실제 폭의 Wan 블록을 RTX 5060 에서 1~6층으로 재어 층 수로 늘림].
    - LoRA: 대상 선형층마다 rank × (입력 + 출력) 파라미터, fp32 파라미터·기울기·AdamW 상태 2개로 파라미터당 16바이트.
    - 학습 단계: 가장 큰 디노이저 + LoRA + 토큰 수 × 토큰당 바이트 + 여유 2GB.
      토큰당 = 체크포인트로 남기는 층마다의 블록 입력(dim) + 다시 계산·역전파하는 한 블록의 작업분(16 × (dim + ffn)), 가중치 dtype 기준.
    - 생성(미리보기·최종): 오프로드(A14B 기본)면 전문가 하나 + LoRA + 생성 토큰 × 토큰당 바이트의 1/4 (추론은 활성값을 남기지
      않음, 실측 약 0.2배) + 여유 2GB. 생성 토큰은 max(preview_frames, final_frames) 의 잠재 프레임 × max_area 토큰.
      오프로드가 없으면 디노이저 모두 + LoRA + VAE + VAE 디코드(출력 화소당 약 23KB, fp32) + 여유 2GB.
    (디노이저 이름, 학습 몫, 생성 몫, 토큰당 바이트, LoRA 몫) 를 돌려준다."""
    nb = torch.tensor([], dtype=WEIGHT_DTYPE).element_size()
    dens = [k for k in w if k.startswith("transformer")]
    den = max(dens, key=w.get)
    cfg = json.loads((Path(folder) / den / "config.json").read_text(encoding="utf-8"))
    dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    per_tok = nb * dim * cfg["num_layers"] + 16 * nb * (dim + cfg["ffn_dim"])
    lin = sum(2 * dim if t.startswith("attn") else dim + cfg["ffn_dim"] for t in A.LORA_TARGETS[FIXED["target_scope"]])
    lora = 16 * FIXED["rank"] * lin * cfg["num_layers"] / 1e9
    train = w[den] + lora + per_tok * tokens / 1e9 + 2
    if offload:
        gen_tok = ((max(FIXED["preview_frames"], FIXED["final_frames"]) - 1) // A.TEMPORAL_FACTOR + 1) \
            * (FIXED["max_area"] // A.TOKEN_DIV ** 2)
        gen = w[den] + lora + per_tok / 4 * gen_tok / 1e9 + 2
    else:
        gen = sum(w[k] for k in dens) + lora + w.get("vae", 0.0) + 23e-6 * FIXED["max_area"] + 2
    return den, train, gen, per_tok, lora

t0 = time.time()
_mdir = None if TEST_FAMILY else check_model_path(USER["MODEL_PATH"])
_wdir = _mdir if _mdir is not None and Path(_mdir).is_dir() else None
if os.environ.get("LORA_SKIP_RAM_CHECK") == "1":
    if _mdir is not None:
        print("참고: LORA_SKIP_RAM_CHECK=1 이라 불러오기 전 메모리 확인을 건너뜁니다.")
    _wdir = None
elif _mdir is not None and _wdir is None:   # Hugging Face 저장소 이름: 이미 받아 둔 사본이 있으면 그것으로 센다
    try:
        from huggingface_hub import snapshot_download
        _wdir = snapshot_download(_mdir, local_files_only=True)
    except Exception:
        print("참고: 아직 받지 않은 저장소라 메모리 확인을 건너뜁니다 (불러오기 전에 내려받습니다).")
_w, _miss = model_weights_gb(_wdir, WEIGHT_DTYPE) if _wdir is not None else ({}, [])
if _miss:
    print("참고: 가중치 파일이 다 있지 않아 메모리 확인을 건너뜁니다:", _miss, "(저장소 이름을 주었으면 불러오기 전에 내려받습니다)")
_up = lambda x: math.ceil(x * 10) / 10          # 오류 문구: 필요량은 올리고 남은 양은 내려, 같은 숫자로 보이지 않게
_dn = lambda x: math.floor(x * 10) / 10         # (안내 줄은 소수 둘째 자리라 오류 숫자는 늘 그 값의 보수적인 한 자리)
if _w and not _miss:
    _need, (_free, _room, _lim) = sum(_w.values()), free_ram_gb()
    print("불러올 가중치(GB):", {k: round(v, 2) for k, v in _w.items()}, f"| 합 {_need:.2f}GB |",
          f"쓸 수 있는 시스템 메모리 {_free:.2f}GB" if _free is not None else "시스템 메모리 확인 못 함 (psutil 없음)")
    if _free is not None and _room < _need + 8:
        _nbw = torch.tensor([], dtype=WEIGHT_DTYPE).element_size()
        # 더 작은 계열 [공식 저장소 머리말로 잰 bf16 GB: 텍스트 인코더 + 디노이저, VAE(fp32)] 가운데 이 커널에 들어가는 것만 권한다
        _small = [n for n, g, v in (("Wan2.1-T2V-14B", 39.94, 0.51), ("Wan2.2-TI2V-5B", 21.36, 2.82),
                                    ("Wan2.1-T2V-1.3B", 14.2, 0.51))
                  if g * _nbw / 2 + v + 8 <= _room and g * _nbw / 2 + v < _need - 0.1]
        raise MemoryError(
            f"이 {'컨테이너' if _lim else '서버'}의 메모리로는 이 모델을 불러올 수 없습니다: 가중치 {_up(_need):.1f}GB + 여유 8GB 가 "
            f"필요한데, 다른 프로그램을 모두 멈춰도 이 커널이 더 쓸 수 있는 메모리는 약 {_dn(_room):.1f}GB 입니다.\n"
            + (f"컨테이너 메모리 한도(docker run --memory, 지금 약 {_lim:.0f}GB)를 늘립니다." if _lim else
               "컨테이너 메모리 한도가 아니라 서버 RAM 이 모자랍니다. RAM 이 더 큰 서버를 씁니다.")
            + (f" 또는 더 작은 모델({', '.join(_small)})을 씁니다." if _small else ""))
    if _free is not None and _free < _need + 8:
        raise MemoryError(
            f"시스템 메모리가 모자랍니다: 가중치 {_up(_need):.1f}GB + 여유 8GB 가 필요한데 {_dn(_free):.1f}GB 만 비어 있습니다. "
            "미리보기·최종 생성 때는 모든 구성요소가 CPU 메모리에 올라가므로 이대로면 도중에 커널이 죽습니다.\n"
            "다른 프로그램(ComfyUI 실행 공간, 다른 노트북 커널)을 멈추고 이 셀부터 다시 실행합니다. "
            "확인을 건너뛰려면 환경변수 LORA_SKIP_RAM_CHECK=1")
    if DEVICE.type == "cuda":
        _gfree, _gtotal = (x / 1e9 for x in torch.cuda.mem_get_info(DEVICE))
        _bk = max(_w, key=_w.get)   # 한 번에 GPU 에 올리는 가장 큰 구성요소 (A14B 는 전문가 하나, 5B·1.3B 는 텍스트 인코더)
        _clip = any(_is_clip(f) for f in _videos)
        # 학습 토큰: 클립은 토큰 예산까지, 이미지(1프레임)는 max_area 에 묶인다
        _tok = FIXED["token_budget"] if _clip else min(FIXED["max_area"] // A.TOKEN_DIV ** 2, FIXED["token_budget"])
        _offl = FIXED["offload_sampling"] and len(A.EXPERTS) > 1
        try:
            _den, _gtrain, _gprev, _ptok, _lora = gpu_need_gb(_wdir, _w, _tok, _offl)
        except (KeyError, ValueError, OSError) as _e:
            print("참고: 학습 GPU 메모리를 추정하지 못해 가장 큰 구성요소로만 봅니다:", repr(_e))
            _den, _gtrain, _gprev, _ptok, _lora = _bk, 0.0, 0.0, None, 0.0
        _gneed = max(_w[_bk] + 4, _gtrain, _gprev)
        print(f"GPU 메모리: 비어 있음 {_gfree:.2f}GB / 전체 {_gtotal:.1f}GB | 필요 추정 {_gneed:.2f}GB = 가장 큰 값 "
              f"[{_bk} 올리기 {_w[_bk] + 4:.2f}, 학습 {_gtrain:.2f} ({_tok:,}토큰, {'클립 예산' if _clip else '이미지 한 장 최대'})"
              + (f", 생성 {_gprev:.2f} ({'전문가 하나씩' if _offl else '디노이저 모두 + VAE 디코드'})" if _gprev else "") + "]")
        # 지금 빈 메모리로 학습 단계에 들어가는 토큰 수. 생성 몫이 이미 들어가지 않으면 토큰 예산을 줄여도 소용없으므로 0
        _fit = int((_gfree - _w[_den] - _lora - 2) * 1e9 / _ptok) if _ptok and max(_w[_bk] + 4, _gprev) <= _gfree else 0
        if _clip and _ptok:
            print(f"  지금 비어 있는 GPU 메모리로 들어가는 클립 token_budget 약 {max(_fit, 0):,} [추정]")
        if _gfree < _gneed:
            raise MemoryError(
                f"GPU 메모리가 모자랍니다: {_up(_gneed):.1f}GB 이상 비어 있어야 하는데 {_dn(_gfree):.1f}GB 입니다 [추정]. "
                "ComfyUI 대기열 등 GPU 를 쓰는 작업을 멈추고 이 셀부터 다시 실행합니다."
                + ("\n클립 토큰 예산을 줄여도 됩니다 (지금 덮어쓴 값은 그대로 둠): LORA_FIXED_OVERRIDES='"
                   + json.dumps({**OVERRIDES, "token_budget": _fit // 1000 * 1000}) + "'"
                   if _clip and _gneed == _gtrain and _fit >= 1000 else "")
                + "\n추정이 맞지 않는다고 보면 환경변수 LORA_SKIP_RAM_CHECK=1 로 확인을 건너뜁니다.")
        if _gtotal - _gfree > 2:
            print(f"참고: 다른 프로그램이 GPU 메모리 약 {_gtotal - _gfree:.1f}GB 를 쓰고 있습니다. 학습 중에 그 프로그램이 "
                  "메모리를 더 쓰면 도중에 멈출 수 있습니다.")
pipe = A.tiny_pipeline(WEIGHT_DTYPE) if TEST_FAMILY else A.load_pipeline(_mdir, WEIGHT_DTYPE)
if not isinstance(pipe, A.PIPELINE_CLASS):
    raise TypeError(f"불러온 파이프라인 {type(pipe).__name__} 이 {A.PIPELINE_CLASS.__name__} 가 아닙니다.")
if TEST_FAMILY:   # 자동 인식 확인: 작은 모델의 설정 파일만 폴더에 써 두고 계열을 맞히는지 본다
    _cfg = Path(tempfile.mkdtemp()) / "cfg"
    pipe.save_config(_cfg)
    for _n in ("transformer", "transformer_2", "vae"):
        if getattr(pipe, _n, None) is not None:
            getattr(pipe, _n).save_config(_cfg / _n)
    assert detect_family(str(_cfg)) == TEST_FAMILY, (detect_family(str(_cfg)), TEST_FAMILY)
    print("모델 계열 자동 인식 확인:", TEST_FAMILY)
for _c in pipe.components.values():
    if isinstance(_c, torch.nn.Module):
        _c.requires_grad_(False)
        _c.eval()
MODULES = [k for k, v in pipe.components.items() if isinstance(v, torch.nn.Module)]

def place(on_device=()):
    """on_device 에 든 구성요소만 장치에 올리고 나머지는 CPU 로 내린다."""
    for k in MODULES:
        m = getattr(pipe, k, None)
        if isinstance(m, torch.nn.Module):
            m.to(DEVICE if k in on_device else "cpu")
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

place([])
print(pd.DataFrame([(k, type(getattr(pipe, k)).__name__, sum(p.numel() for p in getattr(pipe, k).parameters()) / 1e6)
                    for k in MODULES], columns=["구성요소", "클래스", "파라미터(백만)"]).to_string(index=False))
print(f"불러온 가중치 {sum(p.numel() * p.element_size() for k in MODULES for p in getattr(pipe, k).parameters()) / 1e9:.2f}GB")
if len(A.EXPERTS) > 1:
    print("전문가 σ 구간:", {A.EXPERT_TAGS[e]: tuple(round(x, 4) for x in A.sigma_range(pipe, e)) for e in A.EXPERTS})
print(f"불러오기 {time.time() - t0:.1f}초")
''')

md(r"""
## 1. EDA (데이터 살펴보기)

### 1-1. 파일 목록과 메타데이터

- 영상은 헤더 값이 틀리는 경우가 많아(VFR, webm·mkv 의 프레임 수 0 등) PyAV 로 패킷 시각을 모아 프레임 수·fps·길이를 계산한다.
- 휴대폰 영상의 회전 정보와 화소 비율(SAR)을 반영한 '보이는 크기'를 쓴다.
""")
code(r'''
IMG_EXT, VID_EXT = V.IMG_EXT, V.VID_EXT
if TEST_FAMILY:
    DATA_DIR = Path(tempfile.mkdtemp()) / "data"
    TEST_EXPECT = V.make_test_dataset(DATA_DIR, USER["TRIGGER"])
    FIXED_WORDS = list(TEST_EXPECT.get("fixed_words") or FIXED_WORDS)   # 시험 데이터에 넣어 둔 고정 외형 낱말
else:
    DATA_DIR, TEST_EXPECT = DATASET_PATH, None
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"DATASET_DIR 가 없습니다: {DATA_DIR}")

def read_caption(path):
    """UTF-8(BOM 포함)·UTF-16·CP949 캡션을 읽는다 (이미지 노트북과 같음)."""
    b = path.read_bytes()
    if b.startswith((b"\xff\xfe", b"\xfe\xff")):
        return b.decode("utf-16").strip()
    if len(b) >= 4 and b"\x00" in b:
        even, odd = b[0::2], b[1::2]
        if odd.count(0) > len(odd) * 0.6:
            return b.decode("utf-16-le", errors="replace").strip()
        if even.count(0) > len(even) * 0.6:
            return b.decode("utf-16-be", errors="replace").strip()
    try:
        return b.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        pass
    try:
        text = b.decode("cp949").strip()
        problems.append((path.name, "캡션을 CP949 로 읽음 (UTF-8 로 저장 권장)"))
        return text
    except UnicodeDecodeError:
        problems.append((path.name, "캡션 인코딩을 알 수 없어 일부 글자를 바꿈"))
        return b.decode("utf-8", errors="replace").strip()

def load_still(path):
    """정지 그림 한 장 (np.uint8 H×W×3). 1프레임짜리 영상 파일(정지 GIF·1프레임 mp4 등)은 PyAV 로 첫 프레임을 읽는다."""
    path = Path(path)
    if path.suffix.lower() in IMG_EXT and not V.is_animated_image(path):
        return V.load_image(path)
    fr, _ = V.read_frames(path)
    return fr[0]

TRIGGER_RE = re.compile(r"(?<!\w)" + re.escape(USER["TRIGGER"]) + r"(?![A-Za-z0-9_])")
rows, problems = [], []
for p in sorted(DATA_DIR.iterdir()):
    ext = p.suffix.lower()
    if ext not in IMG_EXT and ext not in VID_EXT:
        continue
    try:
        if ext in IMG_EXT and not V.is_animated_image(p):   # 움직이는 webp·png·gif 는 클립으로 읽는다
            arr = V.load_image(p)
            meta = dict(kind="image", width=arr.shape[1], height=arr.shape[0], frames=1, fps=float("nan"),
                        duration=0.0, vfr=False, rotation=0, codec="image", has_alpha=False, hdr=False)
        else:
            m = V.probe(p)
            meta = dict(kind="video", width=m["width"], height=m["height"], frames=m["frames"], fps=m.get("fps", float("nan")),
                        duration=m.get("duration", 0.0), vfr=bool(m.get("vfr", False)), rotation=int(m.get("rotation", 0)),
                        codec=m.get("codec", ""), has_alpha=bool(m.get("has_alpha", False)), hdr=bool(m.get("hdr", False)))
            if meta["frames"] <= 1:
                meta["kind"] = "image"              # 한 프레임짜리 영상(예: 정지 GIF)은 이미지로 다룬다
    except Exception as e:
        problems.append((p.name, f"열 수 없음: {type(e).__name__}: {e}")); continue
    cap_path = p.with_suffix(".txt")
    caption = read_caption(cap_path) if cap_path.exists() else ""
    rows.append(dict(file=p.name, **meta, aspect=round(meta["width"] / max(1, meta["height"]), 3), caption=caption,
                     has_caption=bool(caption), has_trigger=bool(TRIGGER_RE.search(caption))))
if not rows:
    raise RuntimeError(f"{DATA_DIR} 에 이미지·영상이 없습니다")
df = pd.DataFrame(rows)
n_img, n_vid = int((df.kind == "image").sum()), int((df.kind == "video").sum())
print(f"이미지 {n_img}개, 클립 {n_vid}개")
print(df[["file", "kind", "width", "height", "frames", "fps", "duration", "vfr", "rotation", "codec"]].to_string(index=False))
''')

md(r"""
### 1-2. 클립 신호 분석 (검은·멈춘 구간, 장면 전환, 움직임, 흐림, 중복)

- 검은 프레임·멈춤은 ffmpeg 의 blackdetect·freezedetect 기준, 장면 전환은 PySceneDetect 기본 기준을 numpy 로 옮긴 것이다.
- 2D 셀 애니메이션은 같은 그림을 2·3프레임 보여 주는 '홀드'가 정상이라, 멈춤과 따로 `hold_ratio`·`drawing_fps` 로 보여 준다.
- 움직임은 프레임 차이(깜빡임에 속는다) 대신 블록 정합 변위로 잰다: `subject_motion`(피사체), `camera_motion`(화면 전체).
  - 단위는 '초당 화면 긴 변 대비 이동 비율'이라 해상도·fps 가 달라도 비교할 수 있다. 0.05 보다 작으면 거의 정지한 클립이다.
- 중복은 클립마다 같은 위치 8프레임의 pHash 거리로 찾는다 (8비트 이하면 '중복 의심', 자동으로 지우지 않음).
""")
code(r'''
SIGNALS, EVENTS, TIMESTAMPS, SIGNATURES = {}, {}, {}, {}
extra = []
for r in df.itertuples():
    path = DATA_DIR / r.file
    if r.kind == "image":
        arr = load_still(path)
        frames = arr[None]
        s = V.summarize_clip(frames, 1.0)
        SIGNATURES[r.file] = V.clip_signature(frames, k=1)
        extra.append(dict(file=r.file, sha1=hashlib.sha1(arr.tobytes()).hexdigest(), **{k: s.get(k) for k in ("blur", "sat")}))
        continue
    frames, ts = V.read_frames(path, max_side=FIXED["analysis_side"] * 3)
    TIMESTAMPS[r.file] = ts
    fps = r.fps if r.fps == r.fps and r.fps > 0 else A.FPS
    sig = V.frame_signals(frames, fps, analysis_side=FIXED["analysis_side"])
    ev = V.detect_events(sig, fps, timestamps=ts)        # 시각을 주면 VFR 의 긴 정지 화면도 멈춤으로 잡는다
    s = V.summarize_clip(frames, fps, sig, ev)
    SIGNALS[r.file], EVENTS[r.file] = sig, ev
    SIGNATURES[r.file] = V.clip_signature(frames, k=8)
    extra.append(dict(file=r.file, sha1="", decoded=len(frames), **{k: s.get(k) for k in (
        "subject_motion", "camera_motion", "static", "hold_ratio", "drawing_fps", "blur", "n_cuts", "n_black",
        "n_freeze", "sat")}))
df = df.merge(pd.DataFrame(extra), on="file", how="left")
for _c in ("subject_motion", "camera_motion", "static", "hold_ratio", "drawing_fps", "n_cuts", "n_black", "n_freeze",
           "decoded", "blur", "sat"):
    if _c not in df.columns:             # 이미지만 있는 데이터셋: 클립 열을 빈 값으로 둔다
        df[_c] = np.nan

# 중복: 이미지는 화소가 같으면 빼고, 클립은 pHash 거리가 가까우면 사람이 보도록 표시만 한다
dup_img = df.kind.eq("image") & df.duplicated("sha1", keep="first") & df.sha1.ne("")
for f in df.loc[dup_img, "file"]:
    problems.append((f, "같은 이미지가 이미 있어 제외"))
df = df.loc[~dup_img].reset_index(drop=True)
vids = list(df.loc[df.kind == "video", "file"])
DUP_PAIRS = []
for i, a in enumerate(vids):
    for b in vids[i + 1:]:
        d = V.signature_distance(SIGNATURES[a], SIGNATURES[b])
        if d <= 8:
            DUP_PAIRS.append((a, b, d)); problems.append((b, f"{a} 와 거의 같은 클립 (pHash 거리 {d:.1f}, 확인 필요)"))

for r in df.itertuples():
    if r.kind != "video":
        continue
    if r.fps == r.fps and r.fps < A.FPS - 0.5:
        problems.append((r.file, f"원본 {r.fps:.1f}fps < 모델 {A.FPS}fps: 리샘플하면 같은 프레임이 반복됩니다"))
    if r.vfr:
        problems.append((r.file, "가변 프레임률(VFR): 시각 기준으로 리샘플합니다"))
    if r.n_cuts:
        problems.append((r.file, f"장면 전환 {int(r.n_cuts)}곳: 창은 전환을 넘지 않게 고릅니다"))
    if r.n_black or r.n_freeze:
        problems.append((r.file, f"검은 구간 {int(r.n_black)}개, 멈춘 구간 {int(r.n_freeze)}개: 학습 창에서 뺍니다"))
    if r.subject_motion == r.subject_motion and r.subject_motion < V.STATIC_MOTION:
        problems.append((r.file, f"거의 움직이지 않는 클립 (움직임 {r.subject_motion:.3f} < {V.STATIC_MOTION}, 단위: 초당 화면 긴 변 비율)"))
    if r.hold_ratio is not None and r.hold_ratio > 0.3:
        problems.append((r.file, f"홀드 비율 {r.hold_ratio:.2f} (셀 애니메이션이면 정상, fps 를 올려 변환한 흔적일 수도 있음)"))
    if r.hdr:
        problems.append((r.file, "HDR·10비트 영상: 색이 바랠 수 있습니다"))
print(df[["file", "kind", "subject_motion", "camera_motion", "hold_ratio", "n_cuts", "n_black", "n_freeze", "blur"]]
      .round(3).to_string(index=False))
''')

md(r"""
### 1-3. 캡션 검사

- 트리거가 없으면 앞에 붙이고, 캡션이 없으면 트리거만으로 학습한다 (이미지 노트북과 같음).
- 클립 캡션에는 동작과 카메라 표현이 있어야 하고, EDA 측정값과 모순되면 표시한다 (예: 'static camera' 인데 화면 전체가 크게 움직임).
- `CAPTION_FIXED_WORDS` 에 적은 고정 외형 낱말이 캡션에 있으면 표시한다.
""")
code(r'''
cap_issue_col = []
for r in df.itertuples():
    summary = None if r.kind == "image" else dict(subject_motion=r.subject_motion, camera_motion=r.camera_motion)
    iss = V.caption_issues(r.caption, TRIGGER_RE, r.kind == "video", FIXED_WORDS, summary) if r.caption else ["캡션 없음"]
    cap_issue_col.append("; ".join(iss))
    if not r.caption:
        problems.append((r.file, "캡션 파일 없음 → 트리거만으로 학습"))
    elif iss:
        problems.append((r.file, "캡션: " + "; ".join(iss)))
df["caption_issues"] = cap_issue_col
df["train_caption"] = [c if t else (f"{USER['TRIGGER']}, {c}" if c else USER["TRIGGER"])
                       for c, t in zip(df.caption, df.has_trigger)]
df["caption_words"] = [len(c.split()) for c in df.train_caption]
print(df[["file", "kind", "caption_words", "caption_issues"]].to_string(index=False))

# 트리거 대신 넣을 일반 낱말 (그림체 변화 확인·보존 학습). auto 면 캡션에 가장 많이 나온 후보 낱말
CLASS_CANDIDATES = ("girl", "boy", "woman", "man", "child", "kid", "baby", "person", "cat", "dog", "bird", "animal",
                    "creature", "robot", "monster", "character")
KO_CLASS = {"소녀": "girl", "여자아이": "girl", "소년": "boy", "남자아이": "boy", "여자": "woman", "남자": "man", "아이": "child"}
CLASS_WORD = USER["CLASS_WORD"].strip()
if CLASS_WORD in ("", "auto"):
    _PLURAL = dict(girls="girl", boys="boy", women="woman", men="man", children="child", kids="kid", babies="baby",
                   people="person", persons="person", cats="cat", dogs="dog", birds="bird", animals="animal",
                   creatures="creature", robots="robot", monsters="monster", characters="character")
    _words = [_PLURAL.get(w, w) for c in df.caption for w in re.findall(r"[a-z]+", c.lower())]
    _cnt = Counter(w for w in _words if w in CLASS_CANDIDATES)
    _she = sum(w in ("she", "her", "hers", "herself") for w in _words)
    _he = sum(w in ("he", "him", "his", "himself") for w in _words)
    _ko = Counter(v for c in df.caption for k, v in KO_CLASS.items()        # 앞뒤 낱말 경계 (조사 허용, '아이스크림' 은 아님)
                  if re.search(r"(?<![가-힣])" + k + r"(?:가|이|는|은|를|을|와|과|의|도|들)?(?![가-힣])", c))
    if _cnt:
        CLASS_WORD, _why = _cnt.most_common(1)[0][0], "캡션에 가장 많이 나온 낱말"
    elif _she != _he:
        CLASS_WORD, _why = ("girl" if _she > _he else "boy"), "캡션의 대명사(she/he)"
    elif _ko:
        CLASS_WORD, _why = _ko.most_common(1)[0][0], "캡션의 한국어 낱말"
    else:
        CLASS_WORD, _why = "person", "캡션에 알맞은 낱말이 없어 기본값. 직접 정하려면 USER 의 CLASS_WORD 를 girl·boy 등으로 바꾼다"
    print(f"\nCLASS_WORD = {CLASS_WORD} ({_why})")
DRIFT_PROMPTS = [_TRIG.sub(lambda m: CLASS_WORD, p) for p in VALIDATION_PROMPTS[:1]]   # 트리거 없는 같은 장면
''')

md("### 1-4. 문제 목록과 요약")
code(r'''
print(f"이미지 {int((df.kind == 'image').sum())}개, 클립 {int((df.kind == 'video').sum())}개 (중복 이미지 제외 뒤)")
print(df[["width", "height", "frames", "fps", "duration", "caption_words"]].describe().round(2).to_string())
print("\n문제 목록:" if problems else "\n문제 없음")
for f, msg in problems:
    print(f"  - {f}: {msg}")
if A.CONDITIONING == "i2v_concat" and not (df.kind == "video").any():
    raise ValueError(f"{A.FAMILY} 는 클립이 있어야 학습합니다 (1프레임 이미지는 조건 채널에 정답이 들어가 배우는 것이 없음). "
                     "이미지만 있으면 T2V-A14B·TI2V-5B·Wan2.1 모델 경로로 먼저 학습하고, 클립이 생기면 I2V 로 학습합니다.")

if TEST_FAMILY:   # 합성 데이터의 알려진 문제(회전·VFR·검은 구간·멈춤·장면 전환·중복·정지 클립)를 바르게 찾았는지 확인
    V.check_test_expectations(df, EVENTS, DUP_PAIRS, TEST_EXPECT)
    print("시험 데이터 확인 통과")
''')

md(r"""
## 2. 시각화

### 2-1. 분포
""")
code(r'''
vd = df[df.kind == "video"]
fig, axes = plt.subplots(1, 4, figsize=(17, 3.3))
if len(vd):
    axes[0].hist(vd.duration, bins=10, color="#7B3342"); axes[0].set_title("클립 길이 (초)")
    axes[1].hist(vd.fps, bins=10, color="#4A372E"); axes[1].axvline(A.FPS, ls="--", c="k"); axes[1].set_title(f"fps (점선 = 모델 {A.FPS})")
    axes[2].scatter(vd.duration, vd.subject_motion, c=np.where(vd.vfr, "tab:red", "tab:blue"))
    for _, r in vd.iterrows():
        axes[2].annotate(r.file[:12], (r.duration, r.subject_motion), fontsize=6)
    axes[2].axhline(V.STATIC_MOTION, ls=":", c="gray")
    axes[2].set_title("길이 대 움직임 (빨강 = VFR, 점선 = 정지 기준)"); axes[2].set_xlabel("초"); axes[2].set_ylabel("초당 긴 변 비율")
else:
    for ax in axes[:3]:
        ax.axis("off"); ax.text(0.5, 0.5, "클립 없음", ha="center")
axes[3].scatter(df.width, df.height, c=np.where(df.kind == "video", "tab:blue", "tab:orange"))
axes[3].set_title("해상도 (주황 = 이미지)"); axes[3].set_xlabel("W"); axes[3].set_ylabel("H")
plt.tight_layout(); plt.show()
''')

md("### 2-2. 컨택트 시트 (클립마다 같은 간격 8프레임, 이미지는 전체에서 고르게 8장, 비율 유지)")
code(r'''
_rows, _labels = [], []
for r in df.itertuples():
    if r.kind == "video":
        fr, _ = V.read_frames(DATA_DIR / r.file, max_side=FIXED["analysis_side"] * 3)
        _rows.append(fr); _labels.append(f"{r.file} | {r.frames}f {r.fps:.1f}fps")
_all = list(df.loc[df.kind == "image", "file"])
_pick = [_all[j] for j in np.linspace(0, len(_all) - 1, min(8, len(_all))).round().astype(int)] if _all else []
_imgs = [load_still(DATA_DIR / f) for f in _pick]
if _imgs:   # 정사각형으로 늘이지 않고 여백을 둔다 (캐릭터 비율이 보이게)
    _rows.append(np.stack([VM.Dino.letterbox(i, 128) for i in _imgs])); _labels.append(f"이미지 {len(_imgs)}/{len(_all)}장")
if _rows:
    sheet = V.contact_sheet(_rows, labels=_labels, n=8, thumb_h=96)
    plt.figure(figsize=(14, 1.3 * len(_rows) + 1)); plt.imshow(sheet); plt.axis("off"); plt.show()
''')

md("### 2-3. 클립별 신호 곡선 (세로선 = 장면 전환, 회색 = 멈춤, 검정 = 검은 구간)")
code(r'''
_files = list(SIGNALS)[:8]
if _files:
    fig, axes = plt.subplots(len(_files), 1, figsize=(13, 2.0 * len(_files)), squeeze=False)
    for ax, f in zip(axes[:, 0], _files):
        sig, ev = SIGNALS[f], EVENTS[f]
        ax.plot(sig["content"], label="장면 차이 (HSV)", lw=1)
        ax.plot(np.asarray(sig["mad"]) * 255, label="밝기 차이 ×255", lw=1)
        ax.plot(sig["motion_p95"], label="움직임 p95 (px)", lw=1)
        for c in ev["cuts"]:
            ax.axvline(c - 1, c="r", ls="--", lw=1)
        for a, b in ev["freeze"]:
            ax.axvspan(a, b - 1, color="gray", alpha=0.3)
        for a, b in ev["black"]:
            ax.axvspan(a, b - 1, color="k", alpha=0.3)
        ax.set_title(f"{f} | hold {ev['hold_ratio']:.2f}, drawing fps {ev['drawing_fps']:.1f}", fontsize=9)
    axes[0, 0].legend(fontsize=7, ncol=3)
    plt.tight_layout(); plt.show()
else:
    print("클립 없음")
''')

md(r"""
## 3. 전처리

### 3-1. 모델 fps 로 리샘플하고 학습 창 고르기

- 시각 기준으로 모델 fps 격자 시각에 화면에 떠 있는 원본 프레임을 고른다 (ffmpeg fps 필터와 같음, 늘이거나 줄이지 않음).
  - VFR·GIF 에서 오래 떠 있는 프레임은 그만큼 반복되고, 0.25초 이상이면 1-2 의 멈춤 검사가 학습 창에서 뺀다.
- 장면 전환·검은·멈춘 구간을 뺀 '깨끗한 구간' 안에서만 창을 고른다.
- 창 길이는 4n+1 프레임이고, 버킷은 토큰 예산(`token_budget`) 안에서 프레임 수를 우선한다. 업스케일은 하지 않는다.
  - 원본이 기본 버킷보다 작으면(예: TI2V-5B 에 480p 클립, 640x360 클립) 원본 안에 들어가는 더 작은 버킷을 쓴다. 쓸 수 있는 넓이(지원 비율 `aspects` 로 잘라 남는 넓이)가 `max_area` 의 1/8 보다 작은 원본은 뺀다.
  - 미리보기·최종 생성은 가장 흔한 학습 버킷의 비율로, 넓이는 `max_area` 로 만든다 (작은 버킷이 생성 크기를 끌어내리지 않게).
- `min_motion` 보다 움직임이 작은 클립은 뺀다.
- 학습에서 뺀 파일과 이유는 셀 끝에 출력한다.
""")
code(r'''
RUN_DIR = (OUTPUT_PATH / (USER["RUN_NAME"] + (f"_test_{TEST_FAMILY}" if TEST_FAMILY else ""))).resolve()   # 결과 폴더 (절대 경로)
if RUN_DIR.exists() and TEST_FAMILY:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

BUCKETS = V.make_buckets(FIXED["token_budget"], FIXED["frame_options"], FIXED["aspects"], FIXED["max_area"],
                         A.TOKEN_DIV, A.TEMPORAL_FACTOR, A.SPATIAL_MULTIPLE)
IMG_BUCKETS = [b for b in BUCKETS if b[0] == 1]
CLIP_BUCKETS = [b for b in BUCKETS if b[0] >= FIXED["min_frames"]]
# own_buckets·fit_bucket (원본이 기본 버킷보다 작을 때의 버킷)은 0-3 에서 정의했다: 0-3 의 클립 확인과 같은 규칙

rng = np.random.default_rng(FIXED["seed"])
PLAN = []          # 학습 샘플 계획: 디코딩은 3-4 캐시 때 한 번만
_n_prob = len(problems)
_fmin = min((f for f in FIXED["frame_options"] if f >= FIXED["min_frames"]), default=FIXED["min_frames"])   # 가장 짧은 학습 창
for r in df.itertuples():
    if r.kind == "image":
        if A.CONDITIONING == "i2v_concat":
            continue                                    # I2V 학습에는 쓰지 않는다 (검증 시작 이미지로 쓴다)
        b = fit_bucket(IMG_BUCKETS, 1, r.width, r.height, True)
        if b is None:   # 넓이가 1/8 이상인데 빠지면 원인은 지원 비율로 자르기다 (예: 1600x150 → 가장 커야 273x150)
            _a = max(FIXED["aspects"], key=lambda a: min(r.width, r.height * a) * min(r.height, r.width / a))
            problems.append((r.file, f"이미지 {r.width}x{r.height} 가 작아 맞는 버킷이 없음 (업스케일 금지, 쓸 수 있는 넓이가 "
                                     "max_area 의 1/8 미만)" if r.width * r.height < FIXED["max_area"] / 8 else
                             f"이미지 {r.width}x{r.height} 에서 지원 비율(aspects)로 자르면 남는 크기(가장 커야 "
                             f"{min(r.width, r.height * _a):.0f}x{min(r.height, r.width / _a):.0f})가 max_area 의 1/8 "
                             "미만이라 맞는 버킷이 없음 (업스케일 금지)")); continue
        PLAN.append(dict(file=r.file, kind="image", start=0, F=1, H=b[1], W=b[2], tokens=b[3], caption=r.train_caption))
        continue
    if r.subject_motion is not None and r.subject_motion < FIXED["min_motion"]:
        problems.append((r.file, f"움직임 {r.subject_motion:.3f} < min_motion {FIXED['min_motion']}: 학습에서 뺌")); continue
    ts = TIMESTAMPS[r.file]
    idx = V.resample_to_fps(ts, A.FPS)                  # 모델 fps 격자 → 원본 프레임 번호
    seg_of = np.full(len(ts), -1)
    for j, (a, b) in enumerate(V.clean_segments(len(ts), EVENTS[r.file])):
        seg_of[a:b] = j
    lab = seg_of[idx]                                   # 리샘플한 프레임마다 깨끗한 구간 번호
    starts, k = [], 0
    while k < len(lab):
        e = k
        while e < len(lab) and lab[e] == lab[k]:
            e += 1
        if lab[k] >= 0:
            starts.append((k, e))
        k = e
    used = 0
    for a, e in starts:
        b = fit_bucket(CLIP_BUCKETS, e - a, r.width, r.height, False)
        if b is None:
            continue
        for s in V.windows(e - a, b[0], FIXED["window_mode"], FIXED["windows_per_segment"], rng):
            PLAN.append(dict(file=r.file, kind="clip", start=a + s, F=b[0], H=b[1], W=b[2], tokens=b[3],
                             caption=r.train_caption, src_idx=idx[a + s:a + s + b[0]].tolist()))
            used += 1
    if not used:
        _long = max((e - a for a, e in starts), default=0)    # 모델 fps 로 센 가장 긴 깨끗한 구간
        problems.append((r.file, "깨끗한 구간이 없어 쓰지 않음 (장면 전환·검은·멈춘 구간뿐)" if not starts else
                         f"깨끗한 구간이 가장 길어야 {_long}프레임(모델 {A.FPS}fps 기준)이라 가장 짧은 학습 창 {_fmin}프레임보다 "
                         "짧아 쓰지 않음" if _long < _fmin else
                         f"{r.width}x{r.height} 에 맞는 버킷이 없어 쓰지 않음 (업스케일 금지, 쓸 수 있는 넓이가 max_area 의 1/8 미만)"))
_dropped = problems[_n_prob:]
if _dropped:
    print("학습에서 뺀 파일:")
    for _f, _why in _dropped:
        print(f"  - {_f}: {_why}")
plan_df = pd.DataFrame(PLAN)
if plan_df.empty:
    raise RuntimeError("학습 샘플이 하나도 없습니다."
                       + ("".join(f"\n  - {_f}: {_why}" for _f, _why in _dropped) if _dropped else " 1 EDA 의 문제 목록을 확인하세요."))
# 검증 손실용 파일: 학습에서 통째로 뺀다 (같은 클립의 다른 창이 학습에 섞이지 않게 파일 단위)
_kind_of = {}
for p in PLAN:
    _kind_of.setdefault(p["file"], p["kind"])
_quota = Counter(_kind_of.values())
_quota = {k: n // 10 for k, n in _quota.items()}          # 종류(이미지·클립)마다 10개당 1개까지
HOLDOUT = set()
for f in np.random.default_rng(FIXED["seed"] + 3).permutation(sorted(_kind_of)).tolist():
    if len(HOLDOUT) >= FIXED["holdout_files"]:
        break
    if _quota[_kind_of[f]] > 0:
        HOLDOUT.add(f)
        _quota[_kind_of[f]] -= 1
for p in PLAN:
    p["holdout"] = p["file"] in HOLDOUT
print("검증 손실용으로 학습에서 뺀 파일:", sorted(HOLDOUT) or "없음 (종류마다 파일이 10개 미만이라 학습 샘플로 잰다)")
for p in PLAN:
    assert (p["F"] - 1) % A.TEMPORAL_FACTOR == 0 and p["H"] % A.SPATIAL_MULTIPLE == 0 and p["W"] % A.SPATIAL_MULTIPLE == 0, p
print("버킷 후보:", [(b[0], b[1], b[2]) for b in BUCKETS])
print(f"학습 샘플 {len(PLAN)}개 (이미지 {int((plan_df.kind == 'image').sum())}, 클립 창 {int((plan_df.kind == 'clip').sum())})")
''')

md("### 3-2. 버킷 분포")
code(r'''
bc = Counter(f"{p['F']}f {p['W']}x{p['H']}" for p in PLAN)
plt.figure(figsize=(max(6, len(bc) * 1.2), 3)); plt.bar(list(bc.keys()), list(bc.values()), color="#4A372E")
plt.xticks(rotation=30); plt.title("(프레임, 너비x높이) 버킷별 샘플 수"); plt.tight_layout(); plt.show()
print(plan_df.groupby(["kind", "F", "W", "H"]).size().rename("샘플").reset_index().to_string(index=False))
''')

md("### 3-3. 모델이 실제로 볼 창 미리보기 (모델 fps, 자르기 뒤)")
code(r'''
def window_frames(p, cache=None):
    """계획 p 의 프레임을 디코딩해 버킷 크기로 자른다. cache 에 클립별 디코딩 결과를 둔다."""
    if p["kind"] == "image":
        return V.resize_crop_frames(load_still(DATA_DIR / p["file"])[None], p["H"], p["W"])
    if cache is not None and p["file"] in cache:
        fr = cache[p["file"]]
    else:
        fr, _ = V.read_frames(DATA_DIR / p["file"])
        if cache is not None:
            cache.clear(); cache[p["file"]] = fr
    return V.resize_crop_frames(fr[p["src_idx"]], p["H"], p["W"])

_prev = [p for p in PLAN if p["kind"] == "clip"][:3] or PLAN[:1]
_clips = [window_frames(p) for p in _prev]
sheet = V.contact_sheet(_clips, labels=[f"{p['file']} @{p['start']} {p['F']}f" for p in _prev], n=8, thumb_h=96)
plt.figure(figsize=(14, 1.4 * len(_clips) + 1)); plt.imshow(sheet); plt.axis("off"); plt.show()
for j, (p, c) in enumerate(zip(_prev, _clips)):
    if p["kind"] == "clip":
        out = V.save_mp4(RUN_DIR / f"preprocess_preview_{j}.mp4", c, A.FPS)
        if Video is not None and not TEST_FAMILY:
            display(Video(str(out), embed=True))
''')

md(r"""
### 3-4. 잠재값·텍스트 미리 계산 (캐시)

- 텍스트 인코더(UMT5)와 VAE 는 한 번만 쓰고 GPU 에서 내린다. 학습 중에는 학습하는 전문가만 GPU 에 둔다.
- 잠재값은 VAE 사후분포의 평균(mode)을 `(z - latents_mean) / latents_std` 로 정규화한 값이다 (diffusers 파이프라인과 같음).
- I2V 는 창의 첫 프레임으로 조건(마스크 4채널 + [첫 프레임, 0] 의 잠재값)을 만든다.
""")
code(r'''
def to_dev(d):
    return None if d is None else {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in d.items()}

captions = sorted({p["caption"] for p in PLAN})
CAP_IDX = {c: i for i, c in enumerate(captions)}
place(["text_encoder"])
with torch.no_grad():
    TEXT = A.encode_prompts(pipe, captions, DEVICE, WEIGHT_DTYPE)
    VAL_TEXT = A.encode_prompts(pipe, VALIDATION_PROMPTS, DEVICE, WEIGHT_DTYPE)
    DRIFT_TEXT = A.encode_prompts(pipe, DRIFT_PROMPTS, DEVICE, WEIGHT_DTYPE)
    NEG_TEXT = A.encode_prompts(pipe, [NEGATIVE], DEVICE, WEIGHT_DTYPE)[0]
    EMPTY_TEXT = A.encode_prompts(pipe, [""], DEVICE, WEIGHT_DTYPE)[0] if FIXED["caption_dropout"] > 0 else None
    PRES_TEXT = A.encode_prompts(pipe, [TRIGGER_RE.sub(lambda m: CLASS_WORD, c) for c in captions], DEVICE, WEIGHT_DTYPE) \
        if FIXED["preservation"] == "dop" else None
place(["vae"])
SAMPLES, THUMBS, _cache = [], [], {}
with torch.no_grad():
    for p in tqdm(PLAN, desc="VAE"):
        fr = window_frames(p, _cache)
        need_cond = A.CONDITIONING == "i2v_concat" and p["kind"] == "clip"
        enc = A.encode_video(pipe, fr, DEVICE, WEIGHT_DTYPE, condition=need_cond)
        assert torch.isfinite(enc["x0"]).all(), f"{p['file']} 잠재값에 NaN/Inf"
        SAMPLES.append(dict(**{k: p[k] for k in ("file", "kind", "F", "H", "W", "tokens", "holdout")}, cap=CAP_IDX[p["caption"]],
                            x0=enc["x0"], cond=enc.get("cond")))
        if len(THUMBS) < 16:
            THUMBS.append(fr[len(fr) // 2])
    # 잠재값 왕복 확인: 첫 샘플을 다시 그림으로 풀어 원본과 비교한다 (정규화가 틀리면 색이 크게 바뀜)
    _p0 = PLAN[0]; _x = SAMPLES[0]["x0"].to(DEVICE, pipe.vae.dtype)
    _mean = torch.tensor(pipe.vae.config.latents_mean, device=DEVICE, dtype=_x.dtype).view(1, -1, 1, 1, 1)
    _std = torch.tensor(pipe.vae.config.latents_std, device=DEVICE, dtype=_x.dtype).view(1, -1, 1, 1, 1)
    _dec = pipe.vae.decode(_x * _std + _mean, return_dict=False)[0]
    ROUNDTRIP = ((_dec[0, :, 0].float().clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    del _x, _mean, _std, _dec                     # 학습·생성 동안 GPU 에 남기지 않는다 (클립이면 수백 MB)
place([])
_cache.clear()
ORIG0 = window_frames(_p0)[0]
_mse = float(((ROUNDTRIP.astype(np.float32) - ORIG0.astype(np.float32)) ** 2).mean())
ROUNDTRIP_PSNR = None if _mse == 0 else round(10 * math.log10(255 ** 2 / _mse), 2)
print(f"캐시 샘플 {len(SAMPLES)}개 | 캡션 {len(captions)}종 | 잠재값 왕복 PSNR {ROUNDTRIP_PSNR} dB (실제 모델은 보통 25dB 이상)")
''')

md("### 3-5. 전처리 결과 확인")
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].imshow(ORIG0); axes[0].set_title("원본 (자른 뒤)"); axes[0].axis("off")
axes[1].imshow(ROUNDTRIP); axes[1].set_title(f"VAE 왕복 ({ROUNDTRIP_PSNR} dB)"); axes[1].axis("off")
z_mean = [float(s["x0"].mean()) for s in SAMPLES]; z_std = [float(s["x0"].std()) for s in SAMPLES]
axes[2].scatter(z_mean, z_std, c=["tab:orange" if s["kind"] == "image" else "tab:blue" for s in SAMPLES])
axes[2].set_title("잠재값 평균-표준편차 (주황 = 이미지)"); axes[2].set_xlabel("mean"); axes[2].set_ylabel("std")
plt.tight_layout(); plt.show()
print(f"잠재값 전체 평균 {np.mean(z_mean):.3f}, 표준편차 평균 {np.mean(z_std):.3f} (정규화가 맞으면 대략 0, 1 근처)")
''')

md(r"""
## 4. 학습

- 전문가마다 LoRA 를 붙이고, 전문가 차례대로 학습한다 (A14B 기본: 저노이즈 → 고노이즈). 학습하는 전문가만 GPU 에 둔다.
- 노이즈 단계 σ 는 logit-normal 을 shift 로 옮긴 분포에서 뽑되, 그 전문가의 구간 안에 들어올 때까지 다시 뽑는다 (분포 모양 보존).
- 손실: x_t = (1-σ)·x0 + σ·noise, 목표 = noise - x0, 디노이저에는 t = 1000·σ 를 준다.
- 저장(`save_every`)과 검증(`validate_every`)은 전체 단계 기준이다. 검증마다 고정 노이즈 검증 손실과 고정 시드 미리보기 클립을 남긴다.
- 미리보기는 0 단계(=베이스 모델) 클립과 비교해 움직임·깜빡임·색 변화를 잰다.

### 4-1. LoRA 붙이기와 전문가 계획
""")
code(r'''
TARGETS = list(A.LORA_TARGETS[FIXED["target_scope"]])
lora_cfg = LoraConfig(r=FIXED["rank"], lora_alpha=FIXED["alpha"], lora_dropout=FIXED["dropout"],
                      init_lora_weights=FIXED["init_lora_weights"], target_modules=TARGETS)

# 전문가 계획: [(전문가, 단계 수)]
has_clip = any(s["kind"] == "clip" and not s["holdout"] for s in SAMPLES)
if len(A.EXPERTS) == 1:
    PHASES = [(A.EXPERTS[0], FIXED["max_train_steps"])]
else:
    HIGH, LOW = A.EXPERTS
    mode = "both" if USER["EXPERT"] == "auto" else USER["EXPERT"]
    high_ok = has_clip or FIXED["high_uses_images"]
    if mode == "high" and not high_ok:
        raise ValueError("고노이즈 전문가는 기본으로 클립으로만 학습합니다. 클립을 넣거나 high_uses_images=True 로 덮어씁니다.")
    if mode == "both" and high_ok:
        n_low = round(FIXED["max_train_steps"] * FIXED["low_steps_share"])
        PHASES = [(LOW, n_low), (HIGH, FIXED["max_train_steps"] - n_low)]
    elif mode == "both":
        PHASES = [(LOW, FIXED["max_train_steps"])]
        _why = "학습에 쓸 클립 창이 없어 (3-1 의 '학습에서 뺀 파일' 참고)" if (df.kind == "video").any() else "클립이 없어"
        print(f"참고: {_why} 고노이즈 전문가는 학습하지 않습니다. 저노이즈 LoRA 만 {FIXED['max_train_steps']}단계 학습합니다.")
    else:
        PHASES = [(LOW if mode == "low" else HIGH, FIXED["max_train_steps"])]
PHASES = [(e, n) for e, n in PHASES if n > 0]
POOL, VAL_ITEMS = {}, {}
for e, _ in PHASES:
    is_high = len(A.EXPERTS) > 1 and e == A.EXPERTS[0]
    fits = lambda s: s["kind"] == "clip" or not is_high or FIXED["high_uses_images"]
    POOL[e] = [i for i, s in enumerate(SAMPLES) if fits(s) and not s["holdout"]]
    if not POOL[e]:
        raise RuntimeError(f"{A.EXPERT_TAGS[e]} 전문가가 학습할 샘플이 없습니다")
    _hold = [i for i, s in enumerate(SAMPLES) if fits(s) and s["holdout"]]
    VAL_ITEMS[e] = (_hold[:FIXED["val_loss_items"]], True) if _hold else (POOL[e][:FIXED["val_loss_items"]], False)

for e, _ in PHASES:
    den = getattr(pipe, e)
    den.add_adapter(lora_cfg)
    for n, p in den.named_parameters():
        p.requires_grad_("lora_" in n)
        if "lora_" in n:
            p.data = p.data.float()            # 학습 파라미터는 fp32
    n_mod = len({n.rsplit(".lora_", 1)[0] for n, _ in den.named_parameters() if "lora_A" in n})
    if n_mod == 0:
        raise RuntimeError(f"LoRA 를 붙일 층을 찾지 못했습니다: {TARGETS}")
    print(f"{A.EXPERT_TAGS[e]:>5} 전문가 ({e}): LoRA 층 {n_mod}개, 파라미터 "
          f"{sum(p.numel() for n, p in den.named_parameters() if 'lora_' in n) / 1e6:.2f}M, 학습 {dict(PHASES)[e]}단계, "
          f"σ 구간 {tuple(round(x, 3) for x in A.sigma_range(pipe, e))}, 샘플 {len(POOL[e])}개, "
          f"검증 손실 {'따로 뺀 파일' if VAL_ITEMS[e][1] else '학습 샘플 (과적합 판단에 쓰지 않음)'}")
''')
md("### 4-2. 학습 준비: 노이즈 단계·검증 손실·미리보기 (DINO·CCIP 불러오기)")
code(r'''
def expert_shift(e):
    if len(A.EXPERTS) == 1:
        return FIXED["shift"]
    return FIXED["shift_high"] if e == A.EXPERTS[0] else FIXED["shift_low"]

def sample_sigma(gen, e):
    """logit-normal → shift → 전문가 구간 안으로 거절 샘플링. 거의 안 들어오면 구간 안 균등으로 대신한다."""
    lo, hi = A.sigma_range(pipe, e)
    sh = expert_shift(e)
    for _ in range(2000):
        u = torch.sigmoid(torch.normal(FIXED["logit_mean"], FIXED["logit_std"], (1,), generator=gen))
        s = sh * u / (1 + (sh - 1) * u)
        if lo <= float(s) < hi or (hi >= 1.0 and float(s) >= lo):
            return s.clamp(1e-3, 1 - 1e-3)
    return (lo + (hi - lo) * torch.rand((1,), generator=gen)).clamp(1e-3, 1 - 1e-3)

# 검증 손실: 전문가 구간 안의 고정 σ 4개 · 고정 노이즈 · 고정 샘플 → 단계끼리 비교 가능
_vg = torch.Generator("cpu").manual_seed(FIXED["seed"] + 7)
VAL_NOISE = {i: torch.randn(SAMPLES[i]["x0"].shape, generator=_vg) for i in range(len(SAMPLES))}

def sample_mode(s, gen):
    if A.CONDITIONING == "i2v_concat":
        return "i2v"
    if A.CONDITIONING == "ti2v" and s["kind"] == "clip" and float(torch.rand((1,), generator=gen)) < FIXED["ti2v_i2v_share"]:
        return "i2v"
    return "t2v"

@torch.no_grad()
def validation_loss(e):
    den = getattr(pipe, e); was = den.training; den.eval()
    lo, hi = A.sigma_range(pipe, e)
    vals = []
    for i in VAL_ITEMS[e][0]:
        s = SAMPLES[i]
        mode = "i2v" if A.CONDITIONING == "i2v_concat" else "t2v"
        for q in (0.125, 0.375, 0.625, 0.875):
            sig = torch.tensor([lo + (min(hi, 1 - 1e-3) - lo) * q], device=DEVICE)
            vals.append(float(A.forward_loss(pipe, den, s["x0"].to(DEVICE), to_dev({"cond": s["cond"]}) if s["cond"] is not None else None,
                                             to_dev(TEXT[s["cap"]]), VAL_NOISE[i].to(DEVICE), sig, mode)))
    den.train(was)
    return float(np.mean(vals))

def fit_hw(h, w, area, m):
    """(h, w) 의 비율을 지키며 면적 area 안에서 m 의 배수인 가장 큰 (H, W)"""
    s = math.sqrt(area / (h * w))
    return max(m, int(h * s) // m * m), max(m, int(w * s) // m * m)

# 미리보기·최종 생성 크기: 학습 샘플에서 가장 흔한 (H, W) 의 비율, 넓이는 max_area (0-4 생성 추정과 같은 크기).
#   3-1 이 작은 원본에 쓴 작은 버킷이나 긴 클립의 줄인 버킷이 생성 크기를 끌어내리지 않게 한다.
_hw = Counter((s["H"], s["W"]) for s in SAMPLES if s["kind"] == "clip") or Counter((s["H"], s["W"]) for s in SAMPLES)
PREVIEW_HW = fit_hw(*_hw.most_common(1)[0][0], FIXED["max_area"], A.SPATIAL_MULTIPLE)
# I2V 시작 이미지: VALIDATION_FILE images → 데이터셋 이미지 → 클립 첫 프레임.
# 가운데를 잘라 미리보기 크기에 맞추지 않고, 이미지마다 그 비율에 맞는 생성 크기를 쓴다 (세로 전신 그림의 머리가 잘리지 않게).

def start_image(arr):
    """시작 이미지를 미리보기와 같은 면적, 자기 비율의 생성 크기로 줄인다 (배수에 맞추느라 가장자리만 조금 잘림)"""
    return V.resize_crop_frames(arr[None], *fit_hw(*arr.shape[:2], PREVIEW_HW[0] * PREVIEW_HW[1], A.SPATIAL_MULTIPLE))[0]

if VALIDATION_IMAGES:
    START_IMAGES = [start_image(load_still(Path(x))) for x in VALIDATION_IMAGES]
else:
    _imgs = [r.file for r in df.itertuples() if r.kind == "image"][:4]
    START_IMAGES = [start_image(load_still(DATA_DIR / f)) for f in _imgs] or \
                   [start_image(window_frames(p)[0]) for p in [q for q in PLAN if q["kind"] == "clip"][:2]]
NEEDS_IMAGE = A.CONDITIONING == "i2v_concat"
OFFLOAD = bool(FIXED["offload_sampling"]) and DEVICE.type == "cuda" and len(A.EXPERTS) > 1
ACTIVE = []          # 지금 GPU 에 둘 학습 중 전문가

def generate_clip(text, seed, frames, steps=None, image=None, hw=None):
    """캐시한 임베딩으로 고정 시드 클립을 만든다. 끝나면 학습 배치를 되돌린다."""
    flags = {e: getattr(pipe, e).training for e in A.EXPERTS}
    for e in A.EXPERTS:
        getattr(pipe, e).eval()
    if not OFFLOAD:
        place(list(A.EXPERTS) + ["vae"])
    try:
        h, w = hw or (image.shape[:2] if image is not None else PREVIEW_HW)   # 시작 이미지가 있으면 그 크기로
        img = Image.fromarray(image) if image is not None else None
        return A.generate(pipe, to_dev(text), to_dev(NEG_TEXT), height=h, width=w, num_frames=frames,
                          generator=torch.Generator("cpu").manual_seed(seed), image=img, steps=steps,
                          device=DEVICE, offload=OFFLOAD)
    finally:
        for e, t in flags.items():
            getattr(pipe, e).train(t)
        place(ACTIVE)

PREVIEW = [(VAL_TEXT[i], VALIDATION_SEEDS[i], "trigger", i) for i in range(min(FIXED["preview_prompts"], len(VAL_TEXT)))]
if A.CONDITIONING == "ti2v" and any(s["kind"] == "clip" for s in SAMPLES):   # TI2V-5B 가 클립으로 배우는 I2V 모드도 본다
    PREVIEW += [(VAL_TEXT[0], VALIDATION_SEEDS[0], "i2v", 0)]
PREVIEW += [(DRIFT_TEXT[0], VALIDATION_SEEDS[0], "class_word", 0)]          # 마지막은 늘 트리거 없는 클립
DINO = VM.Dino.load("cpu") if FIXED["use_dino"] else None
CCIP = VM.CCIP.load() if FIXED["use_ccip"] else None
REF_FEATS = CCIP.features(THUMBS) if CCIP is not None and THUMBS else None
TRAIN_EMB = DINO.embed(THUMBS) if DINO is not None and THUMBS else None   # 생성 클립과 같은 letterbox 전처리
_pv_hw = sorted({tuple(START_IMAGES[i % len(START_IMAGES)].shape[:2]) if NEEDS_IMAGE or kind == "i2v" else PREVIEW_HW
                 for _, _, kind, i in PREVIEW})   # run_previews 와 같은 식: I2V·'i2v' 미리보기는 시작 이미지 크기
print("미리보기:", f"{len(PREVIEW)}개 × {FIXED['preview_frames']}프레임 {', '.join(f'{w}x{h}' for h, w in _pv_hw)}, "
      f"{FIXED['preview_steps']} step", "| DINO", "켬" if DINO else "끔", "| CCIP", "켬" if CCIP else "끔", "| 생성 오프로드", OFFLOAD)
''')
code(r'''
PREVIEW_HISTORY, STEP_METRICS, CALIB = {}, {}, None

def run_previews(step):
    global CALIB
    clips = []
    for j, (text, seed, kind, i) in enumerate(PREVIEW):
        img = START_IMAGES[i % len(START_IMAGES)] if NEEDS_IMAGE or kind == "i2v" else None
        clip = generate_clip(text, seed, FIXED["preview_frames"], FIXED["preview_steps"], image=img)
        clips.append(clip)
        V.save_mp4(RUN_DIR / f"preview_step{step:05d}_{j}_{kind}.mp4", clip, A.FPS)
    PREVIEW_HISTORY[step] = clips
    base = PREVIEW_HISTORY.get(0)
    if step == 0 and DINO is not None:      # 그림체 변화 기준: 같은 프롬프트 다른 시드의 베이스 차이
        alt = generate_clip(PREVIEW[-1][0], PREVIEW[-1][1] + 7, FIXED["preview_frames"], FIXED["preview_steps"],
                            image=START_IMAGES[0] if NEEDS_IMAGE else None)
        CALIB = VM.style_drift(DINO.embed(alt), DINO.embed(clips[-1]))
    if base is None:
        return
    mets = [VM.clip_metrics(c, base=b, dino=DINO, ccip=CCIP, ref_feats=REF_FEATS) for c, b in zip(clips, base)]
    trig = [m for m, (_, _, kind, _) in zip(mets, PREVIEW) if kind == "trigger"]
    mean = lambda key, ms: (float(np.mean([m[key] for m in ms if m.get(key) is not None]))
                            if any(m.get(key) is not None for m in ms) else None)
    drift = mets[-1].get("drift")
    memo = None
    if TRAIN_EMB is not None and not NEEDS_IMAGE:   # I2V 는 학습 데이터 그림에서 시작하므로 닮음이 암기의 증거가 아니다
        emb = np.concatenate([DINO.embed(c) for c, (_, _, kind, _) in zip(clips, PREVIEW) if kind == "trigger"])
        memo = float((emb @ TRAIN_EMB.T).max())
    # 색 변화·그림체 변화는 트리거 없는 클립으로 본다 (트리거 클립은 캐릭터 색이 바뀌는 것이 정상)
    STEP_METRICS[step] = dict(motion_ratio=mean("motion_ratio", trig), flicker_rel=mean("flicker_rel", trig),
                              color_shift=mets[-1].get("color_shift"), consistency=mean("consistency", trig),
                              drift_rel=(drift / CALIB) if drift is not None and CALIB else None,
                              id_pass=mean("id_pass", trig), id_slope=mean("id_slope", trig), memo=memo)

def save_ckpt(e, step):
    d = RUN_DIR / f"checkpoint-{step:05d}"
    d.mkdir(exist_ok=True)
    sd = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in get_peft_model_state_dict(getattr(pipe, e)).items()}
    meta = dict(family=A.FAMILY, expert=A.EXPERT_TAGS[e], step=str(step), rank=str(FIXED["rank"]), alpha=str(FIXED["alpha"]),
                trigger=USER["TRIGGER"], base_model=USER["MODEL_PATH"], target_scope=FIXED["target_scope"])
    return A.save_lora(sd, str(d / f"{USER['RUN_NAME']}_{A.EXPERT_TAGS[e]}.safetensors"), meta)

# 그래디언트 체크포인트: 켠 것과 끈 것의 손실이 같은지 한 번 확인한 뒤 켠다 (이미지 노트북과 같음)
def enable_gc(e):
    den = getattr(pipe, e)
    if not (FIXED["gradient_checkpointing"] and getattr(den, "_supports_gradient_checkpointing", False)):
        return "끔"
    s = SAMPLES[POOL[e][0]]
    g = torch.Generator("cpu").manual_seed(0)
    nz = torch.randn(s["x0"].shape, generator=g).to(DEVICE)
    sig = torch.tensor([sum(A.sigma_range(pipe, e)) / 2], device=DEVICE)
    mode = "i2v" if A.CONDITIONING == "i2v_concat" else "t2v"
    cond = to_dev({"cond": s["cond"]}) if s["cond"] is not None else None
    with torch.no_grad():
        off = float(A.forward_loss(pipe, den, s["x0"].to(DEVICE), cond, to_dev(TEXT[s["cap"]]), nz, sig, mode))
    den.enable_gradient_checkpointing()
    with torch.enable_grad():
        on = float(A.forward_loss(pipe, den, s["x0"].to(DEVICE), cond, to_dev(TEXT[s["cap"]]), nz, sig, mode).detach())
    den.zero_grad(set_to_none=True)
    tol = 1e-4 if WEIGHT_DTYPE == torch.float32 else 2e-2
    if abs(on - off) > tol * max(1.0, abs(off)):
        den.disable_gradient_checkpointing()
        return f"끔 (켜면 손실이 달라짐 {off:.5f} → {on:.5f})"
    return f"켬 (확인 손실 {off:.5f} / {on:.5f})"
''')
md("### 4-3. 학습 루프")
code(r'''
# 그림체 보존(DOP): LoRA 를 끈 출력(베이스)과 켠 출력의 차이 — v2 이미지 노트북과 같은 방식
_captured = []

def _capture(module, args, kwargs, out):
    out = getattr(out, "sample", out)
    _captured.append(out[0] if isinstance(out, (tuple, list)) else out)

def preservation_loss(den, x0, cond, text, noise, sigma, mode):
    def run():
        _captured.clear()
        h = den.register_forward_hook(_capture, with_kwargs=True)
        try:
            A.forward_loss(pipe, den, x0, cond, text, noise, sigma, mode)
        finally:
            h.remove()
        if len(_captured) != 1:
            raise RuntimeError(f"디노이저가 {len(_captured)}번 불렸습니다")
        return _captured.pop()
    den.disable_adapters()
    try:
        with torch.no_grad():
            prior = run().detach()
    finally:
        den.enable_adapters()
    return torch.mean((run().float() - prior.float()) ** 2)

history, val_history, ckpt_paths = [], [], defaultdict(dict)
gen = torch.Generator("cpu").manual_seed(FIXED["seed"])
order = defaultdict(list)
nan_steps, global_step, t_start = 0, 0, time.time()
step_sec, train_peak, all_peak = 0.0, 0, 0   # 학습 단계만의 시간·최대 GPU 메모리, 미리보기 생성까지 포함한 최대 GPU 메모리
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
run_previews(0)                               # 0 단계 = 베이스 모델 (LoRA B=0)

def pick(e):
    pool = POOL[e]
    kinds = {SAMPLES[i]["kind"] for i in pool}
    kind = ("image" if float(torch.rand((1,), generator=gen)) < FIXED["image_share"] else "clip") if len(kinds) == 2 else kinds.pop()
    if not order[(e, kind)]:
        cand = [i for i in pool if SAMPLES[i]["kind"] == kind]
        order[(e, kind)] = [cand[j] for j in torch.randperm(len(cand), generator=gen).tolist()]
    return order[(e, kind)].pop()

for e, n_steps in PHASES:
    den = getattr(pipe, e)
    ACTIVE[:] = [e]; place(ACTIVE)
    den.train()
    print(f"\n[{A.EXPERT_TAGS[e]} 전문가] 그래디언트 체크포인트:", enable_gc(e))
    params = [p for n, p in den.named_parameters() if "lora_" in n]
    optimizer = torch.optim.AdamW(params, lr=FIXED["learning_rate"], betas=FIXED["betas"],
                                  weight_decay=FIXED["weight_decay"], eps=FIXED["eps"])
    lr_scheduler = get_scheduler(FIXED["lr_scheduler"], optimizer=optimizer, num_warmup_steps=FIXED["warmup_steps"],
                                 num_training_steps=n_steps)
    val_history.append((global_step, A.EXPERT_TAGS[e], validation_loss(e)))
    pbar = tqdm(range(1, n_steps + 1), desc=f"학습 {A.EXPERT_TAGS[e]}")
    for k in pbar:
        global_step += 1
        t_step = time.time()
        if DEVICE.type == "cuda":   # 학습 단계만의 최대 메모리를 미리보기 생성과 따로 잰다
            all_peak = max(all_peak, torch.cuda.max_memory_allocated())
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        i = pick(e); s = SAMPLES[i]
        x0 = s["x0"].to(DEVICE)
        noise = torch.randn(x0.shape, generator=gen).to(DEVICE)
        sigma = sample_sigma(gen, e).to(DEVICE)
        mode = sample_mode(s, gen)
        cond = to_dev({"cond": s["cond"]}) if s["cond"] is not None else None
        drop = EMPTY_TEXT is not None and float(torch.rand((1,), generator=gen)) < FIXED["caption_dropout"]
        text = to_dev(EMPTY_TEXT if drop else TEXT[s["cap"]])
        loss = A.forward_loss(pipe, den, x0, cond, text, noise, sigma, mode)
        pres = float("nan")
        if torch.isfinite(loss):
            loss.backward()
            if PRES_TEXT is not None and global_step % FIXED["preservation_every"] == 0:
                pl = preservation_loss(den, x0, cond, to_dev(PRES_TEXT[s["cap"]]), noise, sigma, mode)
                if torch.isfinite(pl):
                    (FIXED["preservation_weight"] * pl).backward(); pres = float(pl.detach())
                else:
                    loss = pl
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, FIXED["max_grad_norm"])) if torch.isfinite(loss) else float("nan")
        if not math.isfinite(grad_norm):
            nan_steps += 1
            loss = loss.detach(); pl = None          # 건너뛴 단계의 계산 그래프(활성값)를 다음 단계까지 들고 있지 않는다
            optimizer.zero_grad(set_to_none=True)
            if nan_steps > FIXED["max_nan_steps"]:
                raise RuntimeError(f"NaN/Inf 손실·기울기가 {nan_steps}번 나와 멈춥니다. 학습률·데이터를 확인하세요.")
        else:
            optimizer.step(); lr_scheduler.step()
            history.append(dict(step=global_step, expert=A.EXPERT_TAGS[e], loss=float(loss.detach()), pres_loss=pres,
                                lr=lr_scheduler.get_last_lr()[0], grad_norm=grad_norm, sigma=float(sigma),
                                mode=mode, kind=s["kind"], frames=s["F"], tokens=s["tokens"], sec=time.time() - t_start))
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}")
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            train_peak = max(train_peak, torch.cuda.max_memory_allocated())
        step_sec += time.time() - t_step
        if global_step % FIXED["save_every"] == 0 or k == n_steps:
            ckpt_paths[e][global_step] = save_ckpt(e, global_step)
        if global_step % FIXED["validate_every"] == 0 or k == n_steps:
            val_history.append((global_step, A.EXPERT_TAGS[e], validation_loss(e)))
            run_previews(global_step)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
del optimizer, lr_scheduler, params           # AdamW 상태(LoRA 파라미터당 8바이트)를 5절 생성 동안 GPU 에 남기지 않는다
loss = pl = None
ACTIVE[:] = []; place([])
loop_sec = time.time() - t_start
train_sec = step_sec                          # 학습 단계만 (미리보기·검증 손실·저장 제외)
if DEVICE.type == "cuda":
    all_peak = max(all_peak, torch.cuda.max_memory_allocated())
peak_gb = all_peak / 1e9 if DEVICE.type == "cuda" else float("nan")            # 미리보기 생성까지 포함
train_peak_gb = train_peak / 1e9 if DEVICE.type == "cuda" else float("nan")    # 학습 단계만
MAX_TOKENS = max((r["tokens"] for r in history), default=0)
pd.DataFrame(history).to_csv(RUN_DIR / "train_log.csv", index=False)
pd.DataFrame(val_history, columns=["step", "expert", "val_loss"]).to_csv(RUN_DIR / "val_loss.csv", index=False)
print(f"학습 {len(history)}단계: 학습 {train_sec / 60:.1f}분 + 미리보기·검증·저장 {(loop_sec - train_sec) / 60:.1f}분, "
      f"NaN 건너뜀 {nan_steps}번")
if DEVICE.type == "cuda":
    print(f"최대 GPU 메모리: 학습 단계 {train_peak_gb:.1f}GB (가장 큰 샘플 {MAX_TOKENS}토큰) | 미리보기 생성 포함 {peak_gb:.1f}GB")
''')

md("## 5. 검증\n\n### 5-1. 손실 곡선 (전문가별)\n\n"
   "검증 손실은 학습에서 뺀 파일(3-1 절)로 잰다. 이미지·클립 종류마다 10개 미만이면 학습 샘플로 재므로, "
   "암기할수록 계속 내려가 과적합을 보여 주지 못한다.")
code(r'''
h = pd.DataFrame(history)
vh = pd.DataFrame(val_history, columns=["step", "expert", "val_loss"])
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
for tag, g in h.groupby("expert"):
    axes[0].plot(g.step, g.loss, alpha=0.3, label=f"{tag} 단계 손실")
    axes[0].plot(g.step, g.loss.ewm(alpha=0.05).mean(), label=f"{tag} 이동 평균")
axes[0].legend(fontsize=7); axes[0].set_title("학습 손실")
for tag, g in vh.groupby("expert"):
    axes[1].plot(g.step, g.val_loss, marker="o", label=tag)
axes[1].legend(fontsize=7); axes[1].set_title("검증 손실 (고정 노이즈·σ, 전문가 구간별)")
axes[2].plot(h.step, h.lr); axes[2].set_title("학습률")
plt.tight_layout(); plt.show()
print(vh.to_string(index=False))
''')

md(r"""
### 5-2. 단계별 미리보기 (같은 프롬프트·같은 시드)

- 위에서 아래로 단계가 진행된다. 맨 위는 0 단계(베이스 모델)다.
- 마지막 줄의 프롬프트는 트리거를 일반 낱말로 바꾼 것이라 학습해도 베이스와 비슷해야 한다 (그림체 변화 확인).
- 표의 `motion_ratio` 는 베이스 대비 움직임 크기다. 0.6 보다 작으면 움직임이 줄어든 것이다.
""")
code(r'''
steps_ = sorted(PREVIEW_HISTORY)
for j, (_, _, kind, i) in enumerate(PREVIEW):
    sheet = V.contact_sheet([PREVIEW_HISTORY[s][j] for s in steps_], labels=[f"step {s}" for s in steps_], n=6, thumb_h=110)
    plt.figure(figsize=(13, 1.4 * len(steps_) + 0.6)); plt.imshow(sheet); plt.axis("off")
    plt.title(("[트리거 없음] " if kind == "class_word" else "") + (VALIDATION_PROMPTS[i] if kind == "trigger" else DRIFT_PROMPTS[0])[:90], fontsize=9)
    plt.show()
print(pd.DataFrame(STEP_METRICS).T.round(3).to_string())
_last = sorted(RUN_DIR.glob(f"preview_step{steps_[-1]:05d}_0_*.mp4"))
if _last and Video is not None and not TEST_FAMILY:
    display(Video(str(_last[0]), embed=True))
''')

md(r"""
### 5-3. 저장한 LoRA 파일 다시 불러오기 확인

- 마지막 저장 파일(ComfyUI 형식 키)을 diffusers 로 다시 불러와 학습한 값과 같은지 본다.
- 모든 키가 `diffusion_model.blocks.` 로 시작하는지도 확인한다. ComfyUI 에서는 경고 `lora key not loaded` 가 0건이어야 한다 (5-6 절).
""")
code(r'''
RELOAD = {}
for e, _ in PHASES:
    den = getattr(pipe, e)
    final_step = max(ckpt_paths[e])
    path = ckpt_paths[e][final_step]
    from safetensors.torch import load_file
    keys = list(load_file(path).keys())
    bad = [k for k in keys if not k.startswith("diffusion_model.blocks.")]
    assert not bad and all(k.endswith((".lora_A.weight", ".lora_B.weight")) for k in keys), bad[:5]
    trained = {k.split(".lora_")[0] + ".lora_" + k.split(".lora_")[1].split(".")[0]: v.detach().float().cpu()
               for k, v in get_peft_model_state_dict(den).items()}
    den.delete_adapters("default")
    A.load_lora(pipe, path, e, adapter_name=f"reload_{A.EXPERT_TAGS[e]}")
    re_ = {k.split(".lora_")[0] + ".lora_" + k.split(".lora_")[1].split(".")[0]: v.detach().float().cpu()
           for k, v in get_peft_model_state_dict(den, adapter_name=f"reload_{A.EXPERT_TAGS[e]}").items()}
    assert set(re_) == set(trained), (len(re_), len(trained))
    diff = max(float((trained[k] - re_[k]).abs().max()) for k in trained)
    assert diff < 1e-6, f"저장 파일과 학습 값이 다릅니다 ({diff})"   # 어댑터가 파일의 fp32 값을 그대로 넣는다
    RELOAD[A.EXPERT_TAGS[e]] = dict(file=str(path), keys=len(keys), max_diff=diff)
    print(f"{A.EXPERT_TAGS[e]:>5}: {Path(path).name} 키 {len(keys)}개 (모두 ComfyUI 형식), 최대 차이 {diff:.2e}")
check = generate_clip(VAL_TEXT[0], VALIDATION_SEEDS[0], FIXED["preview_frames"], FIXED["preview_steps"],
                      image=START_IMAGES[0] if NEEDS_IMAGE else None)
assert check.ndim == 4 and check.std() > 0, "생성 클립이 한 색입니다"
print("다시 불러온 LoRA 로 생성 성공:", check.shape)
''')

md(r"""
### 5-4. 최종 검증 세트 (LoRA 켬 / 끔 비교)

- 검증 프롬프트 전부를 공식 설정(step 수·shift·guidance)으로 그리고, 같은 시드로 LoRA 를 끈 베이스와 비교한다.
- 지표: 움직임 비율(베이스 대비), 깜빡임 비율, 채도 변화, 시간 일관성(DINO), 정체성 통과율(CCIP, 선택)
- 절댓값보다 베이스와의 비교로 본다. 최종 판정은 5-6 절대로 ComfyUI 에서 사람이 한다.
""")
code(r'''
FINAL_STEPS = FIXED["final_steps"] or A.VAL_STEPS
FINAL, FINAL_METRICS = [], []
for i, (text, seed) in enumerate(zip(VAL_TEXT, VALIDATION_SEEDS)):
    img = START_IMAGES[i % len(START_IMAGES)] if NEEDS_IMAGE else None
    lora = generate_clip(text, seed, FIXED["final_frames"], FINAL_STEPS, image=img)
    for e in A.EXPERTS:
        if getattr(getattr(pipe, e), "_hf_peft_config_loaded", False):
            getattr(pipe, e).disable_adapters()
    try:
        base = generate_clip(text, seed, FIXED["final_frames"], FINAL_STEPS, image=img)
    finally:
        for e in A.EXPERTS:
            if getattr(getattr(pipe, e), "_hf_peft_config_loaded", False):
                getattr(pipe, e).enable_adapters()
    V.save_mp4(RUN_DIR / f"final_{i:02d}_lora.mp4", lora, A.FPS); V.save_mp4(RUN_DIR / f"final_{i:02d}_base.mp4", base, A.FPS)
    m = VM.clip_metrics(lora, base=base, dino=DINO, ccip=CCIP, ref_feats=REF_FEATS)
    FINAL.append((lora, base)); FINAL_METRICS.append(dict(prompt=i, **m))
for i, (lora, base) in enumerate(FINAL):
    sheet = V.contact_sheet([base, lora], labels=["베이스", "LoRA"], n=6, thumb_h=110)
    plt.figure(figsize=(13, 3.2)); plt.imshow(sheet); plt.axis("off"); plt.title(VALIDATION_PROMPTS[i][:90], fontsize=9); plt.show()
FINAL_DF = pd.DataFrame(FINAL_METRICS)
print(FINAL_DF.round(3).to_string(index=False))
''')

md(r"""
### 5-5. 새 프롬프트로 시험 (학습 데이터에 없던 장면)

- `EXTRA_PROMPTS` 를 바꿔 이 셀만 다시 실행해도 된다. 트리거가 없는 문장은 앞에 트리거를 붙인다.
- I2V 계열은 `EXTRA_IMAGE` 에 시작 이미지 경로를 준다 (비우면 검증 시작 이미지를 쓴다).
""")
code(r'''
EXTRA_PROMPTS = [
    "full body, static camera, she spins around once on a swing in a sunset park, clean anime illustration",
    "medium shot, camera slowly pushes in, she laughs and covers her mouth with her sleeve, snowy street, clean anime illustration",
]
EXTRA_IMAGE = ""
EXTRA_SEED = 2001

extra_prompts = [p.strip() if _TRIG.search(p) else f"{USER['TRIGGER']}, {p.strip()}" for p in EXTRA_PROMPTS]
place(["text_encoder"])
with torch.no_grad():
    extra_text = A.encode_prompts(pipe, extra_prompts, DEVICE, WEIGHT_DTYPE)
place([])
_img = start_image(load_still(Path(EXTRA_IMAGE))) if EXTRA_IMAGE else START_IMAGES[0]
extra = [generate_clip(t, EXTRA_SEED + j, FIXED["final_frames"], FINAL_STEPS, image=_img if NEEDS_IMAGE else None)
         for j, t in enumerate(extra_text)]
for j, c in enumerate(extra):
    V.save_mp4(RUN_DIR / f"extra_{j:02d}_seed{EXTRA_SEED + j}.mp4", c, A.FPS)
(RUN_DIR / "extra_prompts.txt").write_text("\n".join(f"{j:02d} seed {EXTRA_SEED + j}: {p}" for j, p in enumerate(extra_prompts)) + "\n",
                                           encoding="utf-8")
sheet = V.contact_sheet(extra, labels=[p[len(USER["TRIGGER"]) + 2:][:40] for p in extra_prompts], n=6, thumb_h=110)
plt.figure(figsize=(13, 1.6 * len(extra) + 0.6)); plt.imshow(sheet); plt.axis("off"); plt.show()
print(f"{len(extra)}개 저장:", RUN_DIR)
''')

md(r"""
### 5-6. ComfyUI 에서 확인할 것 (노트북 밖)

- A14B 는 LoRA 가 전문가마다 하나씩 2개다. 고노이즈 모델 로더 뒤에 `_high` 파일, 저노이즈 모델 로더 뒤에 `_low` 파일을 각각 `LoraLoaderModelOnly` 로 건다 (키 이름이 같아서 바꿔 걸어도 경고가 나지 않는다).
  - 클립 없이 이미지만으로 학습하면 `_low` 파일 하나만 나온다. 고노이즈 모델에는 LoRA 를 걸지 않는다.
  - 두 전문가는 서로 다른 단계 폴더에 저장된다 (저노이즈 먼저, 고노이즈 나중). 6-2 절이 추천한 짝을 쓴다.
- ComfyUI 콘솔에 `lora key not loaded` 경고가 0건인지 본다.
- 랭크와 알파가 같으므로 강도 1.0 이 학습 때 세기다. 움직임이 줄었으면 고노이즈 LoRA 를 0.5 로 낮춰 비교한다.
- T2V-A14B 로 학습한 LoRA 를 I2V-A14B 에 걸어 효과가 이어지는지 따로 확인한다 (블록 모양은 같지만 효과는 미확인).
- 공식 설정으로 비교한다: I2V 40 step, shift 5.0 (diffusers 저장소 기본 3.0 과 다름), CFG 3.5·3.5 / T2V shift 12.0, CFG 4.0(고)·3.0(저).
- 공식 부정 프롬프트에는 '정지 화면'·'그림체' 단어가 들어 있다. 셀화 그림체가 흐려지면 부정 프롬프트를 바꿔 비교한다.
""")

md(r"""
## 6. 보완

### 6-1. 증상 진단

- 미리보기 지표(단계별)와 검증 손실로 증상을 찾는다. 기준값은 `remedy.THR` 에 있다 [제안, 실측으로 다시 맞춘다].
- 지표가 없는 규칙(예: CCIP 를 끈 경우의 정체성)은 건너뛴다.
""")
code(r'''
SYMPTOM_KO = dict(overfit="과적합·암기", underfit="과소학습", identity_weak="정체성 약함", identity_drift="클립 안 정체성 흔들림",
                  motion_loss="움직임 손실", motion_excess="움직임 과다", flicker="깜빡임", color_shift="색 변화",
                  style_drift="그림체 변화")
_main = A.EXPERT_TAGS[PHASES[0][0]]
VAL_HELDOUT = VAL_ITEMS[PHASES[0][0]][1]
# 학습 샘플로 잰 손실은 암기할수록 계속 내려가므로 과적합·추천 단계 판단에 쓰지 않는다
VAL_MAIN = {int(s): v for s, t, v in val_history if t == _main} if VAL_HELDOUT else None
if VAL_MAIN and len(PHASES) > 1:   # 첫 전문가는 다음 전문가를 학습하는 동안 그대로라, 그 뒤 단계의 검증 손실은 마지막 값과 같다
    VAL_MAIN.update({int(s): VAL_MAIN[max(VAL_MAIN)] for s in STEP_METRICS if int(s) > PHASES[0][1]})
if not VAL_HELDOUT:
    print("참고: 검증 손실이 학습 샘플로 잰 값이라 진단에 쓰지 않습니다 (이미지·클립 종류마다 10개 이상이면 따로 뺀 파일로 잽니다).")
# I2V 미리보기는 학습 데이터 그림(시작 이미지)에서 시작해 베이스도 CCIP 를 통과한다: memo 처럼 통과율을 진단·추천에 쓰지 않는다
_SM = {int(k): (dict(v, id_pass=None) if NEEDS_IMAGE else dict(v)) for k, v in STEP_METRICS.items()}
_AFTER = PHASES[0][1] if len(PHASES) > 1 else None   # A14B: 이 단계 뒤 미리보기는 두 전문가의 LoRA 를 함께 쓴 것
if NEEDS_IMAGE and any(m.get("id_pass") is not None for m in STEP_METRICS.values()):
    print("참고: I2V 는 학습 데이터 그림에서 시작해 베이스도 CCIP 를 통과하므로, 정체성 통과율(5-2 표)은 진단·추천에 쓰지 않습니다.")
_MET_KO = dict(motion_ratio="움직임", flicker_rel="깜빡임", color_shift="색 변화", consistency="시간 일관성(DINO)",
               drift_rel="그림체 변화(DINO)", memo="암기(DINO)", id_pass="정체성(CCIP)", id_slope="클립 안 정체성(CCIP)")
_missing = [v for k, v in _MET_KO.items() if STEP_METRICS and all(m.get(k) is None for m in STEP_METRICS.values())
            and not (k == "memo" and NEEDS_IMAGE)]   # I2V 는 암기 지표를 쓰지 않는다
if _missing:
    print("값이 없어 관련 규칙을 건너뛴 지표:", ", ".join(_missing))
    if "움직임" in _missing:
        print("  움직임 비율은 베이스(LoRA 0) 미리보기가 거의 움직이지 않으면 내지 않는다. 움직임이 있는 검증 프롬프트를 쓴다.")
    if any("DINO" in m or "CCIP" in m for m in _missing):
        print("  DINO·CCIP 지표는 끄거나(LORA_FIXED_OVERRIDES 의 use_dino·use_ccip) 4-2 셀에서 불러오지 못하면 빠진다 "
              "(이유는 4-2 셀 출력).")
SYMPTOMS = RM.diagnose(_SM, VAL_MAIN, after=_AFTER)
if SYMPTOMS:
    print(pd.DataFrame([dict(증상=SYMPTOM_KO.get(s["symptom"], s["symptom"]), 근거=s["evidence"], 정도=s["severity"])
                        for s in SYMPTOMS]).to_string(index=False))
else:
    print("뚜렷한 증상 없음 (지표 기준). 컨택트 시트와 ComfyUI 결과로 최종 판단한다.")
''')

md("### 6-2. 체크포인트 추천")
code(r'''
BEST_STEP = RM.pick_best_step(_SM, VAL_MAIN, after=_AFTER)
# 전문가는 차례로 학습하므로, 어느 단계의 미리보기는 '그때까지 학습한 전문가들의 그 시점 LoRA' 로 그린 것이다.
# 실제로 함께 잰 짝만 추천한다: 고른 단계 뒤에 학습한 전문가는 LoRA 없이(None), 앞서 끝난 전문가는 마지막 파일.
RECOMMEND, _start = {}, 0
for e, n in PHASES:
    steps_e = sorted(ckpt_paths[e])
    if BEST_STEP is None or BEST_STEP >= _start + n:
        RECOMMEND[A.EXPERT_TAGS[e]] = ckpt_paths[e][steps_e[-1]]
    elif BEST_STEP <= _start:
        RECOMMEND[A.EXPERT_TAGS[e]] = None
    else:
        cand = [s for s in steps_e if s <= BEST_STEP]
        RECOMMEND[A.EXPERT_TAGS[e]] = ckpt_paths[e][cand[-1]] if cand else None
    _start += n
print("추천 단계:", BEST_STEP)
for tag, pth in RECOMMEND.items():
    print(f"  {tag:>5}: {pth if pth else '없음 (고른 단계는 이 전문가를 학습하기 전이라 LoRA 없이 쓴다)'}")
_late = [A.EXPERT_TAGS[e] for e, _ in PHASES if RECOMMEND[A.EXPERT_TAGS[e]] is None and ckpt_paths[e]]
if _late:
    print(f"참고: {', '.join(_late)} LoRA 까지 함께 쓰려면 마지막 단계({global_step})의 짝(전문가마다 마지막 파일)을 씁니다. "
          "그 단계의 지표는 5-2 표에서 봅니다.")
print("ComfyUI 에서는 이 파일들을 model-inbox 절차로 등록해 5-6 절대로 확인한다.")
''')

md(r"""
### 6-3. 다음 실행 설정 만들기

- 증상의 우선순위대로 한 번에 2개까지만 값을 바꾼다 (무엇이 효과였는지 알 수 있게).
- 결과는 `run_many.py` 목록 형식의 `next_run.json` 이다. 실행 이름은 판이 올라간다 (예: `_v1` → `_v2`).
- 바로 돌리지 말고, 컨택트 시트와 ComfyUI 결과를 보고 사람이 승인한 뒤 실행한다.
""")
code(r'''
# 이 계열·이 데이터에서 효과가 없는 값은 제안 후보에서 뺀다 (바꿀 수 있는 2개를 헛되이 쓰지 않게)
_drop = {"low_steps_share", "shift_low", "shift_high", "high_uses_images"} if len(A.EXPERTS) == 1 else {"shift"}
_trained = {e for e, _ in PHASES}
if len(A.EXPERTS) > 1 and A.EXPERTS[0] not in _trained:   # 고노이즈를 학습하지 않음 (클립 없음 또는 EXPERT=low)
    _drop |= {"shift_high", "low_steps_share", "high_uses_images"}
if len(A.EXPERTS) > 1 and A.EXPERTS[1] not in _trained:   # 저노이즈를 학습하지 않음 (EXPERT=high)
    _drop |= {"shift_low", "low_steps_share"}
_pool_kinds = [{SAMPLES[i]["kind"] for i in POOL[e]} for e in _trained]   # 학습한 전문가가 실제로 뽑은 샘플 종류
_kinds = set().union(*_pool_kinds)
if A.CONDITIONING == "i2v_concat" or not any(len(k) > 1 for k in _pool_kinds):   # image_share 는 두 종류가 섞인 풀에서만 쓴다
    _drop |= {"image_share"}
if "image" not in _kinds:
    _drop |= {"high_uses_images"}
if "clip" not in _kinds:                                   # 클립에만 쓰는 값
    _drop |= {"min_motion", "min_frames", "window_mode", "windows_per_segment", "ti2v_i2v_share"}
NEXT_OVERRIDES, CHANGES = RM.propose({k: v for k, v in FIXED.items() if k not in _drop}, SYMPTOMS)
# 이번 실행에 이미 덮어쓴 값(OVERRIDES)을 이어받아야 이전 보완이 되돌아가지 않는다
ALL_OVERRIDES = {**OVERRIDES, **NEXT_OVERRIDES}
if "rank" in ALL_OVERRIDES:
    ALL_OVERRIDES["alpha"] = ALL_OVERRIDES["rank"]
# 다음 실행에는 이번에 정한 계열·낱말·범위를 그대로 적는다 (auto 를 풀어 쓴 값)
#   계열은 사용자가 정한 그대로(보통 auto): next_run.json 에서 model_path 만 바꿔도 새 모델에 맞춰 다시 정한다
_user_next = dict(USER, MODEL_FAMILY=USER["MODEL_FAMILY"].strip() or "auto", CLASS_WORD=CLASS_WORD, TARGET_SCOPE=SCOPE)
NEXT_PLAN = RM.next_run_plan(_user_next, NEXT_OVERRIDES, CHANGES, BEST_STEP, base_overrides=OVERRIDES)
(RUN_DIR / "next_run.json").write_text(json.dumps(NEXT_PLAN, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
if CHANGES:
    print(pd.DataFrame([dict(값=c.get("key"), 지금=c.get("old"), 다음=c.get("new"),
                             이유=SYMPTOM_KO.get(c.get("symptom"), c.get("symptom"))) for c in CHANGES]).to_string(index=False))
    print("\n다음 실행:", RUN_DIR / "next_run.json")
    print(f"  python run_many.py {RUN_DIR / 'next_run.json'} --notebook lora_training_video.ipynb")
    print("  또는 LORA_FIXED_OVERRIDES='" + json.dumps(ALL_OVERRIDES, ensure_ascii=False) + "' 와 새 LORA_RUN_NAME 으로 이 노트북 실행")
else:
    if SYMPTOMS:   # 증상은 있는데 이 계열·데이터에서 바꿀 수 있는 값이 없음
        print("증상(" + ", ".join(SYMPTOM_KO.get(s["symptom"], s["symptom"]) for s in SYMPTOMS) + ")은 있지만, "
              "이 모델·데이터에서 바꿀 수 있는 학습 값이 없습니다.")
        print("  예: 이미지만 있는 데이터의 움직임 손실은 클립을 넣어야 나아집니다.")
    else:
        print("바꿀 값 없음 (지표상 증상 없음).")
    print("  데이터·캡션을 보강하거나 더 이른 체크포인트를 비교합니다. next_run.json 은 판만 올린 같은 설정입니다:",
          RUN_DIR / "next_run.json")
''')

md("### 6-4. 결과 요약 저장")
code(r'''
summary = dict(
    notebook="video", family=A.FAMILY, model_path=USER["MODEL_PATH"] if not TEST_FAMILY else "tiny-test", run_dir=str(RUN_DIR),
    trigger=USER["TRIGGER"], expert_plan=[(A.EXPERT_TAGS[e], n) for e, n in PHASES],
    family_auto=not TEST_FAMILY and USER["MODEL_FAMILY"].strip() in ("", "auto"), target_scope=FIXED["target_scope"],
    class_word=CLASS_WORD,
    data=dict(images=int((df.kind == "image").sum()), clips=int((df.kind == "video").sum()), samples=len(SAMPLES),
              buckets=dict(Counter(f"{s['F']}x{s['H']}x{s['W']}" for s in SAMPLES)), problems=len(problems)),
    fixed=dict(FIXED), overrides=OVERRIDES, steps_done=len(history), nan_steps=nan_steps,
    # train_minutes 는 학습 단계만, preview_minutes 는 학습 중 미리보기·검증 손실·저장. peak_gpu_gb 는 미리보기 생성까지 포함한
    # 최대, train_peak_gpu_gb 는 학습 단계만 (가장 큰 샘플이 max_train_tokens 토큰)
    train_minutes=round(train_sec / 60, 2), preview_minutes=round((loop_sec - train_sec) / 60, 2),
    peak_gpu_gb=None if math.isnan(peak_gb) else round(peak_gb, 2),
    train_peak_gpu_gb=None if math.isnan(train_peak_gb) else round(train_peak_gb, 2), max_train_tokens=MAX_TOKENS,
    val_loss=[dict(step=int(s), expert=t, val_loss=round(v, 5)) for s, t, v in val_history],
    checkpoints={A.EXPERT_TAGS[e]: {int(s): str(p) for s, p in d.items()} for e, d in ckpt_paths.items()},
    reload=RELOAD, holdout_files=sorted(HOLDOUT), val_heldout=VAL_HELDOUT,
    step_metrics={int(k): v for k, v in STEP_METRICS.items()},
    final_metrics=FINAL_METRICS, symptoms=SYMPTOMS, best_step=BEST_STEP, recommend=RECOMMEND,
    next_overrides=ALL_OVERRIDES, roundtrip_psnr=ROUNDTRIP_PSNR,
    # run_many.py 기록용: 따로 뺀 파일로 잰 검증 손실이 가장 낮은 저장 단계(첫 전문가), 추천 LoRA 파일 크기 합(MB),
    # 마지막 20단계 평균 손실
    best_val_step=min(((s, v) for s, t, v in val_history if t == _main and s in ckpt_paths[PHASES[0][0]]),
                      key=lambda x: x[1], default=(None, None))[0] if VAL_HELDOUT else None,
    lora_file_mb=round(sum(Path(p).stat().st_size for p in RECOMMEND.values() if p) / 1e6, 1),
    final_train_loss=round(float(np.mean([r["loss"] for r in history[-20:]])), 5) if history else None,
    versions={k: _md.version(k) for k in list(PINNED) + ["av", "ftfy"]},
)
(RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("family", "expert_plan", "data", "train_minutes", "preview_minutes", "peak_gpu_gb",
                                           "train_peak_gpu_gb", "max_train_tokens", "best_step", "recommend")},
                 ensure_ascii=False, indent=1, default=str))
if TEST_FAMILY:
    print(f"\nNOTEBOOK TEST OK {TEST_FAMILY}")
''')

for i, c in enumerate(cells):   # 셀 id 를 고정해, 다시 만들 때 diff 에 내용 변경만 남게 한다
    c["id"] = f"cell-{i:02d}"
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.validate(nb)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(nbf.writes(nb), encoding="utf-8", newline="\n")
print("wrote", OUT, len(cells), "cells")
