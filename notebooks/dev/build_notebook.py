"""Build lora_training.ipynb from cell sources + adapter files.

  python build_notebook.py <adapters_dir> <out.ipynb>
"""
import sys
from pathlib import Path

import nbformat as nbf

ADIR = Path(sys.argv[1])
OUT = Path(sys.argv[2])
FAMILIES = ["qwen_image", "qwen_image21", "flux2_klein", "z_image", "krea2", "flux1", "chroma", "longcat_image",
            "sdxl", "sd3"]

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# 캐릭터 LoRA 학습 노트북 (diffusers + peft)

이 노트북은 이미지 생성 모델에 캐릭터 LoRA 를 학습한다. 베이스 모델은 얼리고 LoRA 어댑터만 학습한다.

| 단계 | 하는 일 |
|---|---|
| 0. 준비 | 라이브러리·버전 확인, 설정(고정값), 모델 계열별 어댑터, 모델 불러오기 |
| 1. EDA | 데이터셋 이미지·캡션을 읽고 크기·비율·중복·흐림·트리거를 검사 |
| 2. 전처리 | 비율별 묶음(bucket)으로 자르고, 이미지 잠재값과 텍스트 임베딩을 미리 계산 |
| 3. 시각화 | 전처리 결과, 비율·캡션 분포, 잠재값 통계를 그림으로 확인 |
| 4. 학습 | LoRA 를 붙여 학습하고, 정해진 단계마다 저장·검증 |
| 5. 검증 | 손실 곡선, 단계별 샘플 비교, 저장 파일 다시 불러오기 확인, 결과 요약 |

- 지원 모델 계열: `qwen_image`(Qwen-Image, 2512), `qwen_image21`, `flux2_klein`, `z_image`, `krea2`, `flux1`, `chroma`, `longcat_image`, `sdxl`, `sd3`.
- 사용자가 바꾸는 값은 `0-2` 의 `USER` 뿐이다. 학습 조건(랭크, 학습률, optimizer, 단계 수 등)은 `FIXED` 에 고정되어 있고, 실행 중에 바꿀 수 없다.
- 모델은 diffusers 형식 폴더(`model_index.json` 이 있는 폴더) 또는 Hugging Face 저장소 이름(예: `Qwen/Qwen-Image-2512`)으로 불러온다. 폴더면 `model_index.json` 의 파이프라인 종류가 `MODEL_FAMILY` 와 맞는지 먼저 확인한다.
- 메인 서버 ComfyUI 의 단일 `.safetensors` 파일(예: `qwen_image_2512_fp8_e4m3fn.safetensors`)은 그대로 쓸 수 없다. 같은 모델의 diffusers 형식 폴더를 따로 받아 둔다.
- 노트북은 실행한 폴더에 `lora_adapters/` 를 만든다.

### 데이터셋 폴더

```
dataset/nkgreen/
  nkgreen_01.png   nkgreen_01.txt   # 같은 이름의 .txt 에 영문 캡션 한 줄
  nkgreen_02.png   nkgreen_02.txt
  ...
```

- 캡션은 `nkgreen, full body, front view, standing, plain light gray background, clean anime illustration` 처럼 트리거 + 바뀌는 것만 적는다 (`prompts/nkgreen_qwen2512_prompts_*.md` 참고).
- 캡션 파일이 없으면 트리거만으로 학습하고 경고를 남긴다.

### 시험 실행 (오류 검사)

환경변수 `LORA_NB_TEST=<모델 계열>` 을 주고 실행하면, 같은 구조의 아주 작은 무작위 모델과 가짜 데이터셋으로 모든 셀을 끝까지 돌린다 (계열당 CPU 30초 안팎). 실제 학습 전에 환경이 맞는지 확인할 때 쓴다. 작은 토크나이저를 Hugging Face 에서 내려받으므로 인터넷이 필요하다. `LORA_NB_TEST_DEVICE=cuda` 를 더하면 GPU·bf16 으로 시험한다.

```
LORA_NB_TEST=qwen_image jupyter nbconvert --to notebook --execute lora_training.ipynb --output test_run.ipynb
```
""")

md("## 0. 준비\n\n### 0-1. 라이브러리 버전 확인과 불러오기")
code(r'''
# 고정 버전: 이 버전 조합에서 모든 모델 계열을 작은 무작위 모델로 시험했다.
#   pip install torch==2.15.0 diffusers==0.42.1 peft==0.21.2 transformers==5.22.1 accelerate==1.15.0 \
#               safetensors sentencepiece protobuf pillow numpy pandas matplotlib tqdm
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
    print("\n경고: 고정 버전과 다릅니다. 시험하지 않은 조합이라 계열에 따라 오류가 날 수 있습니다.")

import contextlib, copy, gc, hashlib, importlib, json, math, os, random, re, shutil, sys, tempfile, time, types
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch
import diffusers
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from tqdm.auto import tqdm

if not hasattr(sys, "ps1") and "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
''')

md(r"""
### 0-2. 설정

- `USER` 만 바꾼다. 환경변수로도 줄 수 있다 (예: `LORA_MODEL_FAMILY`, `LORA_MODEL_PATH`).
- `FIXED` 는 계획서(`lora-training-plan.md` 3-3 절)와 diffusers 공식 학습 스크립트 기본값을 따른 고정값이다. 읽기 전용이라 실행 중에 바꾸면 오류가 난다.
""")
code(r'''
TEST_FAMILY = os.environ.get("LORA_NB_TEST", "").strip()   # 시험 실행이면 모델 계열 이름

USER = MappingProxyType(dict(
    MODEL_FAMILY=os.environ.get("LORA_MODEL_FAMILY", TEST_FAMILY or "qwen_image"),
    MODEL_PATH=os.environ.get("LORA_MODEL_PATH", "/srv/models/diffusers/Qwen-Image-2512"),
    DATASET_DIR=os.environ.get("LORA_DATASET_DIR", "dataset/nkgreen"),
    OUTPUT_DIR=os.environ.get("LORA_OUTPUT_DIR", "outputs"),
    TRIGGER=os.environ.get("LORA_TRIGGER", "nkgreen"),
    RUN_NAME=os.environ.get("LORA_RUN_NAME", "nkgreen_v1"),
))

FIXED = MappingProxyType(dict(
    seed=42,
    # LoRA
    rank=16, alpha=16, dropout=0.0, init_lora_weights="gaussian",
    # optimizer (torch.optim.AdamW)
    optimizer="AdamW", learning_rate=1e-4, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8,
    lr_scheduler="constant_with_warmup", warmup_steps=100,
    # 학습 길이
    max_train_steps=1500, batch_size=1, grad_accum_steps=1, max_grad_norm=1.0,
    # 해상도: 한 변 1024 의 넓이를 기준으로 비율별 묶음
    resolution=1024, bucket_aspects=(0.5, 0.5625, 0.625, 0.6667, 0.75, 0.8, 1.0, 1.25, 1.3333, 1.5, 1.6, 1.7778, 2.0),
    # 노이즈 단계 뽑기: flow 계열은 logit-normal(0,1), eps/v 계열은 균등
    timestep_sampling="logit_normal", logit_mean=0.0, logit_std=1.0,
    # 정밀도·메모리
    mixed_precision="bf16", gradient_checkpointing=True,
    # 저장·검증
    save_every=250, validate_every=250, val_loss_images=4, val_sigmas=(0.1, 0.3, 0.5, 0.7, 0.9),
    max_nan_steps=10, min_image_side=512,
))

TEST_DEVICE = os.environ.get("LORA_NB_TEST_DEVICE", "cpu")   # 시험 실행 장치: cpu(fp32) 또는 cuda(bf16)
if TEST_FAMILY:   # 시험 실행: 같은 코드 경로를 아주 작은 값으로
    FIXED = MappingProxyType({**FIXED, "max_train_steps": 4, "warmup_steps": 1, "save_every": 2, "validate_every": 2,
                              "val_loss_images": 2, "min_image_side": 8,
                              "mixed_precision": "bf16" if TEST_DEVICE == "cuda" else "fp32"})

VALIDATION_PROMPTS = [
    f"{USER['TRIGGER']}, full body, front view, standing, plain light gray background, clean anime illustration",
    f"{USER['TRIGGER']}, upper body, three-quarter view from the left, shy small smile, summer meadow, clean anime illustration",
    f"{USER['TRIGGER']}, close-up, side view, calm look into the distance, plain light gray background, clean anime illustration",
    f"{USER['TRIGGER']}, full body, back view, standing, forest path, clean anime illustration",
]
VALIDATION_NEGATIVE = "photorealistic, 3d render, text, watermark, blurry, deformed, extra limbs, hands, fingers, multiple people"
VALIDATION_SEEDS = [1001, 1002, 1003, 1004]
print(json.dumps(dict(USER), ensure_ascii=False, indent=1))
''')

md(r"""
### 0-3. 모델 계열별 어댑터 (수정하지 않는다)

모델마다 텍스트 인코딩, 잠재값 정규화, 노이즈 공식, 잠재값 묶기(packing), LoRA 대상 층, 저장 형식이 다르다.
아래 셀들은 계열마다 같은 함수 이름(`encode_prompts`, `encode_images`, `forward_loss`, `save_lora`, `generate`)을 가진 파일을 `lora_adapters/` 에 쓰고, 선택한 계열 하나만 불러온다.
각 어댑터는 diffusers v0.42.1 의 공식 학습 스크립트와 대조했다 (diffusers 에 스크립트가 없는 longcat_image 는 Meituan 공식 LoRA 스크립트). 모든 계열은 아주 작은 무작위 모델로 CPU(fp32)와 GPU(bf16)에서 처음부터 끝까지 실행해 확인했고, 실제 가중치로는 아직 학습해 보지 않았다.
""")
for fam in FAMILIES:
    src = (ADIR / f"{fam}.py").read_text(encoding="utf-8").rstrip() + "\n"
    code(f"%%writefile lora_adapters/{fam}.py\n" + src)

code(r'''
ADAPTER_DIR = Path("lora_adapters").resolve()
if str(ADAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(ADAPTER_DIR))
SUPPORTED = sorted(p.stem for p in ADAPTER_DIR.glob("*.py"))
if USER["MODEL_FAMILY"] not in SUPPORTED:
    raise ValueError(f"MODEL_FAMILY='{USER['MODEL_FAMILY']}' 는 지원하지 않습니다. 고를 수 있는 값: {SUPPORTED}")
A = importlib.import_module(USER["MODEL_FAMILY"])
for _name in ["PIPELINE_CLASS", "DENOISER_ATTR", "PREDICTION", "PIXEL_MULTIPLE", "DEFAULT_STEPS", "DEFAULT_GUIDANCE",
              "LORA_TARGET_MODULES", "load_pipeline", "tiny_pipeline", "encode_prompts", "encode_images",
              "forward_loss", "save_lora", "generate"]:
    if not hasattr(A, _name):
        raise AttributeError(f"어댑터 {USER['MODEL_FAMILY']} 에 {_name} 가 없습니다")
print("어댑터:", A.FAMILY, "| 파이프라인:", A.PIPELINE_CLASS.__name__, "| 예측:", A.PREDICTION,
      "| 픽셀 배수:", A.PIXEL_MULTIPLE, "| LoRA 대상:", A.LORA_TARGET_MODULES)
''')
# replace %%writefile target dir creation: writefile needs the dir to exist -> create it before the adapter cells
cells.insert(len(cells) - len(FAMILIES) - 1, nbf.v4.new_code_cell('os.makedirs("lora_adapters", exist_ok=True)'))

md("### 0-4. 장치·정밀도 결정, 시드 고정, 모델 불러오기")
code(r'''
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(FIXED["seed"])
if TEST_FAMILY:
    DEVICE = torch.device("cuda" if TEST_DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if FIXED["mixed_precision"] == "bf16" and DEVICE.type == "cuda" and torch.cuda.is_bf16_supported():
    WEIGHT_DTYPE = torch.bfloat16
elif FIXED["mixed_precision"] in ("bf16", "fp16") and DEVICE.type == "cuda":
    WEIGHT_DTYPE = torch.float16     # bf16 을 못 쓰는 GPU
else:
    WEIGHT_DTYPE = torch.float32     # CPU 는 fp32
print("장치:", DEVICE, "| 가중치 dtype:", WEIGHT_DTYPE)

def check_model_path(mp):
    """로컬 폴더면 diffusers 형식과 모델 계열을 확인하고, 아니면 Hugging Face 저장소 이름인지 확인한다."""
    p = Path(mp).expanduser()
    if p.exists():
        if p.is_file():
            raise ValueError(f"MODEL_PATH 가 파일입니다: {mp}. ComfyUI 용 단일 .safetensors 는 쓸 수 없고, "
                             "model_index.json 이 있는 diffusers 형식 폴더가 필요합니다.")
        mi = p / "model_index.json"
        if not mi.exists():
            raise FileNotFoundError(f"{mp} 에 model_index.json 이 없습니다. diffusers 형식 폴더가 필요합니다.")
        cls_name = json.loads(mi.read_text(encoding="utf-8")).get("_class_name", "")
        ok = {c.__name__ for c in A.PIPELINE_CLASS.__mro__ if issubclass(c, diffusers.DiffusionPipeline)}
        if cls_name and cls_name not in ok:
            raise ValueError(f"이 폴더는 {cls_name} 모델입니다. MODEL_FAMILY='{A.FAMILY}' 는 "
                             f"{A.PIPELINE_CLASS.__name__} 용입니다. MODEL_FAMILY 를 확인하세요.")
        return str(p)
    if p.is_absolute() or mp.startswith((".", "~")) or not re.fullmatch(r"[\w.-]+/[\w.-]+", mp):
        raise FileNotFoundError(f"MODEL_PATH 가 없습니다: {mp}")
    # Hugging Face 저장소 이름 (예: Qwen/Qwen-Image-2512): 큰 파일을 받기 전에 model_index.json 만 받아 계열 확인
    try:
        from huggingface_hub import hf_hub_download
        cls_name = json.loads(Path(hf_hub_download(mp, "model_index.json")).read_text(encoding="utf-8")).get("_class_name", "")
    except Exception as e:
        print(f"참고: {mp} 의 model_index.json 을 받지 못해 계열 확인을 건너뜁니다 ({type(e).__name__})")
        return mp
    ok = {c.__name__ for c in A.PIPELINE_CLASS.__mro__ if issubclass(c, diffusers.DiffusionPipeline)}
    if cls_name and cls_name not in ok:
        raise ValueError(f"{mp} 는 {cls_name} 모델입니다. MODEL_FAMILY='{A.FAMILY}' 는 {A.PIPELINE_CLASS.__name__} 용입니다.")
    return mp

t0 = time.time()
if TEST_FAMILY:
    pipe = A.tiny_pipeline(WEIGHT_DTYPE)
else:
    pipe = A.load_pipeline(check_model_path(USER["MODEL_PATH"]), WEIGHT_DTYPE)
if not isinstance(pipe, A.PIPELINE_CLASS):
    raise TypeError(f"불러온 파이프라인 {type(pipe).__name__} 이 {A.PIPELINE_CLASS.__name__} 가 아닙니다. MODEL_FAMILY 를 확인하세요.")
denoiser = getattr(pipe, A.DENOISER_ATTR)
for _c in pipe.components.values():
    if isinstance(_c, torch.nn.Module):
        _c.requires_grad_(False)
        _c.eval()
_param_rows = [(k, type(v).__name__, sum(p.numel() for p in v.parameters()) / 1e6)
               for k, v in pipe.components.items() if isinstance(v, torch.nn.Module)]
display_df = pd.DataFrame(_param_rows, columns=["구성요소", "클래스", "파라미터(백만)"])
print(display_df.to_string(index=False))
print(f"불러오기 {time.time() - t0:.1f}초")
''')

md("## 1. EDA (데이터 살펴보기)")
code(r'''
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def make_test_dataset(root, n=6):
    """시험 실행용 가짜 데이터셋: 서로 다른 비율의 무작위 색 이미지 + 캡션"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sizes = [(64, 96), (96, 64), (80, 80), (64, 112), (112, 64), (72, 72)]
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        arr = (rng.random((h, w, 3)) * 255).astype("uint8")
        Image.fromarray(arr).save(root / f"img_{i:02d}.png")
        if i != n - 1:   # 마지막 하나는 캡션 없음 → 경고 경로 시험
            (root / f"img_{i:02d}.txt").write_text(f"{USER['TRIGGER']}, test image {i}", encoding="utf-8")
    Image.fromarray(arr).save(root / "dup_00.png")   # 중복 → 제외 경로 시험
    # 문제 사례: EXIF 회전(저장은 가로, 보이는 것은 세로), 16비트 흑백, 여러 캡션 인코딩
    exif = Image.Exif(); exif[0x0112] = 6
    Image.fromarray((rng.random((64, 96, 3)) * 255).astype("uint8")).save(root / "rot_exif.jpg", exif=exif.tobytes())
    (root / "rot_exif.txt").write_bytes(f"{USER['TRIGGER']}, rotated".encode("utf-8-sig"))          # BOM
    Image.fromarray((rng.random((80, 64)) * 65535).astype("uint16")).save(root / "gray16.png")
    (root / "gray16.txt").write_bytes(f"({USER['TRIGGER']}) 회색 이미지".encode("cp949"))              # CP949
    Image.fromarray((rng.random((72, 88, 3)) * 255).astype("uint8")).save(root / "utf16.png")
    (root / "utf16.txt").write_bytes(f"{USER['TRIGGER']}, utf16 caption".encode("utf-16"))           # UTF-16
    Image.fromarray(rng.random((70, 90)).astype("float32"), "F").save(root / "float01.tif")           # 0~1 실수
    (root / "float01.txt").write_bytes(f"{USER['TRIGGER']}가 서 있음".encode("utf-16-le"))             # BOM 없는 UTF-16
    rgba = (rng.random((66, 84, 4)) * 255).astype("uint8"); rgba[..., 3] = 0
    Image.fromarray(rgba, "RGBA").save(root / "clear.png")                                           # 완전 투명
    (root / "clear.txt").write_text(f"{USER['TRIGGER']}_x, transparent", encoding="utf-8")           # 트리거 아님
    return root

DATA_DIR = make_test_dataset(Path(tempfile.mkdtemp()) / "data") if TEST_FAMILY else Path(USER["DATASET_DIR"])
if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"DATASET_DIR 가 없습니다: {DATA_DIR}")

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

def read_caption(path):
    """UTF-8(BOM 포함)·UTF-16·CP949 캡션을 읽는다. 모르는 인코딩은 글자를 바꾸고 문제 목록에 남긴다."""
    b = path.read_bytes()
    if b.startswith((b"\xff\xfe", b"\xfe\xff")):
        return b.decode("utf-16").strip()
    if len(b) >= 4 and b"\x00" in b:          # BOM 없는 UTF-16: 한쪽 바이트가 대부분 0
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
        problems.append((path.name, "캡션을 CP949 로 읽음 (내용 확인 권장, UTF-8 로 저장 권장)"))
        return text
    except UnicodeDecodeError:
        pass
    problems.append((path.name, "캡션 인코딩을 알 수 없어 일부 글자를 바꿈 (UTF-8 로 저장 권장)"))
    return b.decode("utf-8", errors="replace").strip()

# 왼쪽은 낱말 경계, 오른쪽은 영문·숫자·밑줄만 막는다 (한글 트리거 뒤에 조사가 붙은 '니키가' 도 트리거로 인정)
TRIGGER_RE = re.compile(r"(?<!\w)" + re.escape(USER["TRIGGER"]) + r"(?![A-Za-z0-9_])")

def blur_score(gray):
    """라플라시안 분산: 낮을수록 흐림"""
    g = gray.astype(np.float32)
    lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())

rows, problems = [], []
for p in sorted(DATA_DIR.iterdir()):
    if p.suffix.lower() not in IMG_EXT:
        continue
    try:
        with Image.open(p) as im:
            im.load()
            mode, info = im.mode, dict(im.info)
            im_t = to_8bit(ImageOps.exif_transpose(im))      # EXIF 회전 뒤 크기로 묶음을 고른다
            w, h = im_t.size
            rgb = np.asarray(flatten_alpha(im_t))
    except Exception as e:
        problems.append((p.name, f"열 수 없음: {e}")); continue
    if mode.startswith("I;16") or mode in ("I", "F"):
        problems.append((p.name, f"{mode} 이미지를 8비트로 줄여 사용"))
    cap_path = p.with_suffix(".txt")
    caption = read_caption(cap_path) if cap_path.exists() else ""
    rows.append(dict(file=p.name, width=w, height=h, aspect=round(w / h, 3), mode=mode,
                     has_alpha=mode in ("RGBA", "LA", "PA") or "transparency" in info,
                     caption=caption, caption_words=len(caption.split()), has_caption=bool(caption),
                     has_trigger=bool(TRIGGER_RE.search(caption)),
                     sha1=hashlib.sha1(rgb.tobytes()).hexdigest(), blur=blur_score(rgb.mean(axis=2)),
                     brightness=float(rgb.mean())))
if not rows:
    raise RuntimeError(f"{DATA_DIR} 에 이미지가 없습니다 ({sorted(IMG_EXT)})")
df = pd.DataFrame(rows)

# 문제 처리: 중복은 빼고, 작은 이미지는 경고, 캡션·트리거가 없으면 채운다
dup = df.duplicated("sha1", keep="first")
for f in df.loc[dup, "file"]:
    problems.append((f, "같은 이미지가 이미 있어 제외"))
df = df.loc[~dup].reset_index(drop=True)
for _, r in df.iterrows():
    if min(r.width, r.height) < FIXED["min_image_side"]:
        problems.append((r.file, f"짧은 변 {min(r.width, r.height)}px < {FIXED['min_image_side']}px (업스케일 권장)"))
    if not r.has_caption:
        problems.append((r.file, "캡션 파일 없음 → 트리거만으로 학습"))
    elif not r.has_trigger:
        problems.append((r.file, "캡션에 트리거 없음 → 앞에 붙임"))
df["train_caption"] = [c if t else (f"{USER['TRIGGER']}, {c}" if c else USER["TRIGGER"])
                       for c, t in zip(df.caption, df.has_trigger)]
low_blur = df.blur < df.blur.median() * 0.25
for f in df.loc[low_blur, "file"]:
    problems.append((f, "다른 이미지보다 많이 흐림 (확인 권장)"))

if TEST_FAMILY:   # 문제 사례가 바르게 처리됐는지 확인
    _r = df.set_index("file")
    assert _r.loc["rot_exif.jpg", "height"] > _r.loc["rot_exif.jpg", "width"], "EXIF 회전 미반영"
    assert all(_r.loc[f, "has_trigger"] for f in ["rot_exif.jpg", "gray16.png", "utf16.png"]), "트리거 판정 오류"
    assert not _r.loc["rot_exif.jpg", "train_caption"].startswith(USER["TRIGGER"] + ", " + USER["TRIGGER"])
    assert "회색" in _r.loc["gray16.png", "caption"], "CP949 캡션 해석 실패"
    assert 60 < _r.loc["gray16.png", "brightness"] < 200, "16비트 이미지가 하얗게 잘림"
    assert int(dup.sum()) == 1
    assert _r.loc["float01.tif", "has_trigger"] and "\x00" not in _r.loc["float01.tif", "caption"], "BOM 없는 UTF-16 오류"
    assert 60 < _r.loc["float01.tif", "brightness"] < 200, "0~1 실수 이미지가 검게 나옴"
    assert _r.loc["clear.png", "brightness"] > 250, "투명 이미지가 흰 배경에 합성되지 않음"
    assert not _r.loc["clear.png", "has_trigger"], "nkgreen_x 를 트리거로 잘못 인정"
print(f"이미지 {len(df)}장 (제외 {int(dup.sum())}장)")
print(df[["width", "height", "aspect", "caption_words", "blur", "brightness"]].describe().round(2).to_string())
print("\n이미지 모드:", dict(Counter(df["mode"])), "| 투명도 있음:", int(df.has_alpha.sum()))
print("\n문제 목록:" if problems else "\n문제 없음")
for f, msg in problems:
    print(f"  - {f}: {msg}")
if len(df) < 5 and not TEST_FAMILY:
    print("경고: 이미지가 5장보다 적습니다. 계획서 기준은 20~30장입니다.")
''')

md("## 2. 전처리\n\n### 2-1. 비율별 묶음(bucket) 정하기와 자르기")
code(r'''
M = A.PIXEL_MULTIPLE
TARGET = (2 * M) if TEST_FAMILY else FIXED["resolution"]   # 시험 실행은 아주 작게

def make_buckets(target, aspects, multiple):
    """넓이가 target^2 에 가깝고 multiple 의 배수인 (w,h) 목록"""
    out = set()
    for a in aspects:
        w = max(multiple, int(round(target * math.sqrt(a) / multiple)) * multiple)
        h = max(multiple, int(round(target / math.sqrt(a) / multiple)) * multiple)
        out.add((w, h))
    return sorted(out, key=lambda wh: wh[0] / wh[1])

BUCKETS = make_buckets(TARGET, FIXED["bucket_aspects"], M)

def pick_bucket(w, h):
    a = w / h
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - math.log(a)))

def load_rgb(path):
    with Image.open(path) as im:
        return flatten_alpha(to_8bit(ImageOps.exif_transpose(im)))

def resize_crop(im, bw, bh):
    """비율을 지키며 덮도록 키운 뒤 가운데를 자름"""
    scale = max(bw / im.width, bh / im.height)
    nw, nh = max(bw, round(im.width * scale)), max(bh, round(im.height * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    return im.crop((left, top, left + bw, top + bh))

def to_tensor(im):
    arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()

df["bucket"] = [pick_bucket(w, h) for w, h in zip(df.width, df.height)]
processed = [resize_crop(load_rgb(DATA_DIR / f), *b) for f, b in zip(df.file, df.bucket)]
for im, b in zip(processed, df.bucket):
    assert im.size == b and b[0] % M == 0 and b[1] % M == 0, (im.size, b)
print("묶음 후보:", BUCKETS)
print("사용된 묶음:", dict(Counter(df.bucket)))
''')

md("### 2-2. 잠재값과 텍스트 임베딩 미리 계산 (캐시)\n\n텍스트 인코더와 VAE 는 한 번만 쓰고 GPU 에서 내려, 학습 중에는 디노이저만 GPU 에 둔다.")
code(r'''
def offload(names, device):
    for n in names:
        m = getattr(pipe, n, None)
        if isinstance(m, torch.nn.Module):
            m.to(device)

DENOISER_NAME = A.DENOISER_ATTR
OTHER_MODULES = [k for k, v in pipe.components.items() if isinstance(v, torch.nn.Module) and k != DENOISER_NAME]
offload(OTHER_MODULES, DEVICE)

with torch.no_grad():
    caption_conds = A.encode_prompts(pipe, list(df.train_caption), DEVICE, WEIGHT_DTYPE)
    val_conds = A.encode_prompts(pipe, VALIDATION_PROMPTS, DEVICE, WEIGHT_DTYPE)
    neg_cond = A.encode_prompts(pipe, [VALIDATION_NEGATIVE], DEVICE, WEIGHT_DTYPE)[0]
    latents = [A.encode_images(pipe, to_tensor(im), DEVICE, WEIGHT_DTYPE) for im in tqdm(processed, desc="VAE")]

offload(OTHER_MODULES, "cpu")
if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

assert len(caption_conds) == len(df) == len(latents)
for i, (c, z) in enumerate(zip(caption_conds, latents)):
    assert torch.isfinite(z).all(), f"{df.file[i]} 잠재값에 NaN/Inf"
    for k, v in c.items():
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"{df.file[i]} 텍스트 임베딩 {k} 에 NaN/Inf"
print("잠재값 모양 예:", tuple(latents[0].shape), "| 텍스트 키:", sorted(caption_conds[0].keys()))
''')

md("## 3. 시각화")
code(r'''
n = len(processed)
cols = min(6, n)
rows_ = math.ceil(n / cols)
fig, axes = plt.subplots(rows_, cols, figsize=(3 * cols, 3.6 * rows_), squeeze=False)
for ax in axes.flat:
    ax.axis("off")
for ax, im, cap, b in zip(axes.flat, processed, df.train_caption, df.bucket):
    ax.imshow(im)
    ax.set_title(f"{b[0]}x{b[1]}\n" + (cap[:38] + "…" if len(cap) > 38 else cap), fontsize=7)
plt.suptitle("전처리된 학습 이미지와 캡션"); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
axes[0].hist(df.aspect, bins=12, color="#7B3342"); axes[0].set_title("원본 가로/세로 비율")
bc = Counter(f"{w}x{h}" for w, h in df.bucket)
axes[1].bar(list(bc.keys()), list(bc.values()), color="#4A372E"); axes[1].set_title("묶음별 장수")
axes[1].tick_params(axis="x", rotation=45)
axes[2].hist(df.caption_words, bins=10, color="#4A4A52"); axes[2].set_title("캡션 낱말 수")
z_mean = [float(z.mean()) for z in latents]; z_std = [float(z.std()) for z in latents]
axes[3].scatter(z_mean, z_std, color="#6B4A35"); axes[3].set_title("잠재값 평균-표준편차")
axes[3].set_xlabel("mean"); axes[3].set_ylabel("std")
plt.tight_layout(); plt.show()
print(f"잠재값 전체 평균 {np.mean(z_mean):.3f}, 표준편차 평균 {np.mean(z_std):.3f} (정규화가 맞으면 대략 0, 1 근처)")
''')

md(r"""
## 4. 학습

- LoRA 를 디노이저의 어텐션 층(어댑터의 `LORA_TARGET_MODULES`)에 붙이고, 그 층만 학습한다.
- 노이즈 단계: flow 계열은 logit-normal(0, 1) 로 시그마를 뽑고, eps/v 계열은 균등하게 뽑는다.
- `save_every` 마다 LoRA 파일을, `validate_every` 마다 검증 손실과 고정 시드 샘플을 남긴다.
- 손실이 NaN/Inf 인 단계는 건너뛰고, `max_nan_steps` 를 넘으면 멈춘다.
""")
code(r'''
RUN_DIR = Path(USER["OUTPUT_DIR"]) / (USER["RUN_NAME"] + (f"_test_{TEST_FAMILY}" if TEST_FAMILY else ""))
if RUN_DIR.exists() and TEST_FAMILY:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

lora_cfg = LoraConfig(r=FIXED["rank"], lora_alpha=FIXED["alpha"], lora_dropout=FIXED["dropout"],
                      init_lora_weights=FIXED["init_lora_weights"], target_modules=list(A.LORA_TARGET_MODULES))
denoiser.to(DEVICE)
denoiser.add_adapter(lora_cfg)
lora_params = [p for n, p in denoiser.named_parameters() if "lora_" in n]
if not lora_params:
    raise RuntimeError(f"LoRA 를 붙일 층을 찾지 못했습니다: {A.LORA_TARGET_MODULES}")
for n, p in denoiser.named_parameters():
    p.requires_grad_("lora_" in n)
for p in lora_params:            # 학습 파라미터는 fp32 (공식 스크립트의 cast_training_params 와 같음)
    p.data = p.data.float()
n_targets = len({n.rsplit(".lora_", 1)[0] for n, _ in denoiser.named_parameters() if "lora_A" in n})
print(f"LoRA 층 {n_targets}개, 학습 파라미터 {sum(p.numel() for p in lora_params) / 1e6:.2f}M")

def to_dev(cond):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in cond.items()}

# diffusers 0.42.1 의 Chroma 는 체크포인트 경로에서 single block 에 attention_mask 를 넘기지 않는다
# (학습 때만 패딩 토큰을 보게 됨). 그래서 Chroma 는 끄고, 나머지는 켠 뒤 같은 입력의 손실이 같은지 확인한다.
NO_GC_FAMILIES = {"chroma"}
denoiser.train()
if FIXED["gradient_checkpointing"] and A.FAMILY not in NO_GC_FAMILIES \
        and getattr(denoiser, "_supports_gradient_checkpointing", False):
    _g = torch.Generator("cpu").manual_seed(0)
    _x0 = latents[0].to(DEVICE); _nz = torch.randn(_x0.shape, generator=_g).to(DEVICE)
    _s = torch.tensor([0.5], device=DEVICE)
    with torch.no_grad():      # 기준값: 체크포인트 없는 경로, 활성값을 저장하지 않아 메모리를 쓰지 않음
        _l_off = float(A.forward_loss(pipe, denoiser, _x0, to_dev(caption_conds[0]), _nz, _s))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    denoiser.enable_gradient_checkpointing()
    with torch.enable_grad():  # 체크포인트 경로는 기울기를 계산할 때만 쓰인다
        _l_on = float(A.forward_loss(pipe, denoiser, _x0, to_dev(caption_conds[0]), _nz, _s))
    denoiser.zero_grad(set_to_none=True)
    _tol = 1e-4 if WEIGHT_DTYPE == torch.float32 else 2e-2
    if abs(_l_on - _l_off) > _tol * max(1.0, abs(_l_off)):
        denoiser.disable_gradient_checkpointing()
        print(f"경고: 그래디언트 체크포인트를 켜면 손실이 달라져({_l_off:.5f} → {_l_on:.5f}) 끕니다. 메모리를 더 씁니다.")
    else:
        print(f"그래디언트 체크포인트 켬 (확인 손실 {_l_off:.5f} / {_l_on:.5f})")
else:
    print("그래디언트 체크포인트 끔")

optimizer = torch.optim.AdamW(lora_params, lr=FIXED["learning_rate"], betas=FIXED["betas"],
                              weight_decay=FIXED["weight_decay"], eps=FIXED["eps"])
lr_scheduler = get_scheduler(FIXED["lr_scheduler"], optimizer=optimizer, num_warmup_steps=FIXED["warmup_steps"],
                             num_training_steps=FIXED["max_train_steps"])

def train_sigma_shift():
    """공식 학습 레시피의 시그마 이동(shift). SD3·Z-Image 는 스케줄러의 고정 shift,
    LongCat 은 학습 해상도의 mu 로 exp(mu), 나머지 flow 계열(FLUX·Qwen·Krea2·Chroma)은 이동 없음."""
    cfg = pipe.scheduler.config
    if A.FAMILY in ("sd3", "z_image") and not cfg.get("use_dynamic_shifting", False):
        return float(cfg.get("shift", 1.0))
    if A.FAMILY == "longcat_image" and cfg.get("use_dynamic_shifting", False):
        seq = (TARGET // 16) ** 2                       # VAE 8배 축소 × 2x2 묶기
        m = (cfg["max_shift"] - cfg["base_shift"]) / (cfg["max_image_seq_len"] - cfg["base_image_seq_len"])
        return float(math.exp(seq * m + cfg["base_shift"] - m * cfg["base_image_seq_len"]))
    return 1.0

SIGMA_SHIFT = train_sigma_shift() if A.PREDICTION == "flow" else 1.0
print(f"시그마 이동(shift): {SIGMA_SHIFT:.3f}")

def sample_sigma(gen):
    if A.PREDICTION == "flow" and FIXED["timestep_sampling"] == "logit_normal":
        u = torch.normal(FIXED["logit_mean"], FIXED["logit_std"], (FIXED["batch_size"],), generator=gen)
        s = torch.sigmoid(u)
        if SIGMA_SHIFT != 1.0:
            s = SIGMA_SHIFT * s / (1 + (SIGMA_SHIFT - 1) * s)
        return s.clamp(1e-3, 1 - 1e-3)
    return torch.rand((FIXED["batch_size"],), generator=gen).clamp(1e-3, 1 - 1e-3)

# 검증 손실: 고정 이미지 · 고정 노이즈 · 고정 시그마 → 단계끼리 비교 가능
_vg = torch.Generator("cpu").manual_seed(FIXED["seed"] + 7)
VAL_IDX = list(range(min(FIXED["val_loss_images"], len(latents))))
VAL_NOISE = {i: torch.randn(latents[i].shape, generator=_vg) for i in VAL_IDX}

@torch.no_grad()
def validation_loss():
    denoiser.eval()
    vals = []
    for i in VAL_IDX:
        x0 = latents[i].to(DEVICE)
        for s in FIXED["val_sigmas"]:
            vals.append(float(A.forward_loss(pipe, denoiser, x0, to_dev(caption_conds[i]),
                                             VAL_NOISE[i].to(DEVICE), torch.tensor([s], device=DEVICE))))
    denoiser.train()
    return float(np.mean(vals))

SAMPLE_SIZE = (2 * M, 2 * M) if TEST_FAMILY else pick_bucket(2, 3)   # 세로 전신 비율
SAMPLE_STEPS = 2 if TEST_FAMILY else A.DEFAULT_STEPS

GEN_MODULES = [m for m in OTHER_MODULES if "text_encoder" not in m]   # 생성에 필요한 VAE 등 (텍스트 인코더는 CPU 에 둠)

@torch.no_grad()
def generate_images(conds, seeds):
    """캐시한 임베딩으로 고정 시드 이미지 생성. VAE 를 잠깐 GPU 에 올렸다가 내린다."""
    was_training = denoiser.training
    denoiser.eval()
    offload(GEN_MODULES, DEVICE)
    try:
        out = [A.generate(pipe, to_dev(c), to_dev(neg_cond), SAMPLE_SIZE[1], SAMPLE_SIZE[0], SAMPLE_STEPS,
                          A.DEFAULT_GUIDANCE, torch.Generator("cpu").manual_seed(s)) for c, s in zip(conds, seeds)]
    finally:
        offload(GEN_MODULES, "cpu")
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        denoiser.train(was_training)
    return out

def make_samples(step):
    out = generate_images(val_conds, VALIDATION_SEEDS)
    for img, seed in zip(out, VALIDATION_SEEDS):
        img.save(RUN_DIR / f"sample_step{step:05d}_seed{seed}.png")
    return out

def save_checkpoint(step):
    d = RUN_DIR / f"checkpoint-{step:05d}"
    d.mkdir(exist_ok=True)
    sd = {k: v.detach().to("cpu", torch.float32).contiguous() for k, v in get_peft_model_state_dict(denoiser).items()}
    return A.save_lora(pipe, sd, str(d))

history, val_history, sample_history, ckpt_paths = [], [], {}, {}
val_history.append((0, validation_loss()))
sample_history[0] = make_samples(0)             # LoRA B=0 으로 시작하므로 0 단계 = 베이스 모델
print(f"0 단계 검증 손실 {val_history[-1][1]:.4f}")

gen = torch.Generator("cpu").manual_seed(FIXED["seed"])
order, nan_steps, t_start = [], 0, time.time()
if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
pbar = tqdm(range(1, FIXED["max_train_steps"] + 1), desc="학습")
for step in pbar:
    optimizer.zero_grad(set_to_none=True)
    step_loss = 0.0
    for _ in range(FIXED["grad_accum_steps"]):
        if not order:                       # 한 에폭마다 섞기 (시드 고정)
            order = torch.randperm(len(latents), generator=gen).tolist()
        i = order.pop()
        x0 = latents[i].to(DEVICE)
        noise = torch.randn(x0.shape, generator=gen).to(DEVICE)
        sigma = sample_sigma(gen).to(DEVICE)
        loss = A.forward_loss(pipe, denoiser, x0, to_dev(caption_conds[i]), noise, sigma)
        if not torch.isfinite(loss):
            break
        (loss / FIXED["grad_accum_steps"]).backward()
        step_loss += float(loss) / FIXED["grad_accum_steps"]
    grad_norm = float("nan")
    if math.isfinite(step_loss) and torch.isfinite(loss):
        grad_norm = float(torch.nn.utils.clip_grad_norm_(lora_params, FIXED["max_grad_norm"]))
    if not math.isfinite(grad_norm):     # 손실 또는 기울기가 NaN/Inf → 이 단계는 가중치를 바꾸지 않음
        nan_steps += 1
        optimizer.zero_grad(set_to_none=True)
        if nan_steps > FIXED["max_nan_steps"]:
            raise RuntimeError(f"NaN/Inf 손실·기울기가 {nan_steps}번 나와 멈춥니다. 학습률·데이터를 확인하세요.")
    else:
        optimizer.step(); lr_scheduler.step()
        history.append(dict(step=step, loss=step_loss, lr=lr_scheduler.get_last_lr()[0], grad_norm=grad_norm,
                            sigma=float(sigma.mean()), sec=time.time() - t_start))
        pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{history[-1]['lr']:.1e}")
    # 건너뛴 단계여도 저장·검증은 한다 (가중치는 직전 단계와 같음)
    if step % FIXED["save_every"] == 0 or step == FIXED["max_train_steps"]:
        ckpt_paths[step] = save_checkpoint(step)
    if step % FIXED["validate_every"] == 0 or step == FIXED["max_train_steps"]:
        val_history.append((step, validation_loss()))
        sample_history[step] = make_samples(step)

train_sec = time.time() - t_start
peak_gb = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else float("nan")
pd.DataFrame(history).to_csv(RUN_DIR / "train_log.csv", index=False)
print(f"학습 {len(history)}단계, {train_sec / 60:.1f}분, NaN 건너뜀 {nan_steps}번, 최대 GPU 메모리 {peak_gb:.1f}GB")
''')

md("## 5. 검증\n\n### 5-1. 손실 곡선")
code(r'''
h = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4))
axes[0].plot(h.step, h.loss, alpha=0.35, label="단계 손실")
axes[0].plot(h.step, h.loss.ewm(alpha=0.05).mean(), label="이동 평균"); axes[0].legend(); axes[0].set_title("학습 손실")
vs, vl = zip(*val_history)
axes[1].plot(vs, vl, marker="o", color="#7B3342"); axes[1].set_title("검증 손실 (고정 노이즈)")
axes[2].plot(h.step, h.lr); axes[2].set_title("학습률")
plt.tight_layout(); plt.show()
_bad = [s for s, v in val_history if not math.isfinite(v)]
if _bad:
    print("경고: 검증 손실이 NaN/Inf 인 단계:", _bad)
best_step = min((s for s, v in val_history if s in ckpt_paths and math.isfinite(v)),
                key=lambda s: dict(val_history)[s], default=max(ckpt_paths))
print("단계별 검증 손실:", {s: round(v, 4) for s, v in val_history})
print("검증 손실이 가장 낮은 저장 단계:", best_step, "(최종 선택은 샘플을 사람이 보고 정한다)")
''')

md("### 5-2. 단계별 샘플 비교 (같은 프롬프트·같은 시드)")
code(r'''
steps_ = sorted(sample_history)
fig, axes = plt.subplots(len(steps_), len(VALIDATION_PROMPTS), figsize=(3 * len(VALIDATION_PROMPTS), 3.3 * len(steps_)), squeeze=False)
for r, s in enumerate(steps_):
    for c, img in enumerate(sample_history[s]):
        ax = axes[r][c]; ax.imshow(img); ax.axis("off")
        if r == 0:
            ax.set_title(VALIDATION_PROMPTS[c][len(USER["TRIGGER"]) + 2:][:34], fontsize=7)
        if c == 0:
            ax.text(-0.05, 0.5, f"step {s}", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9)
plt.suptitle("0 단계 = 베이스 모델, 아래로 갈수록 학습이 진행된 LoRA"); plt.tight_layout(); plt.show()
''')

md("### 5-3. 저장한 LoRA 파일 다시 불러오기 확인\n\n학습한 어댑터를 지우고 마지막 저장 파일을 `load_lora_weights` 로 다시 불러와, 값이 같은지와 이미지가 나오는지 확인한다.")
code(r'''
final_step = max(ckpt_paths)
final_path = ckpt_paths[final_step]
trained = {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(denoiser).items()}
denoiser.delete_adapters("default") if hasattr(denoiser, "delete_adapters") else None
pipe.load_lora_weights(final_path if Path(final_path).is_file() else str(Path(final_path).parent), adapter_name="reload")
reloaded = {k: v.detach().float().cpu() for k, v in get_peft_model_state_dict(denoiser, adapter_name="reload").items()}
assert len(reloaded) == len(trained), (len(reloaded), len(trained))
max_diff = max(float((a - b).abs().max()) for a, b in zip(sorted(trained.items()), sorted(reloaded.items()))
               for a, b in [(a[1], b[1])])
print(f"다시 불러온 LoRA 텐서 {len(reloaded)}개, 최대 차이 {max_diff:.2e}")
assert max_diff < (1e-2 if WEIGHT_DTYPE != torch.float32 else 1e-5), "저장 파일과 학습 값이 다릅니다"
check_img = generate_images(val_conds[:1], VALIDATION_SEEDS[:1])[0]
assert np.asarray(check_img).std() > 0, "생성 이미지가 한 색입니다"
check_img.save(RUN_DIR / "reload_check.png")
print("다시 불러온 LoRA 로 생성 성공:", check_img.size)
''')

md("### 5-4. 결과 요약 저장")
code(r'''
dataset_hash = hashlib.sha1("".join(sorted(df.sha1 + df.train_caption)).encode()).hexdigest()
summary = dict(
    family=A.FAMILY, model_path=USER["MODEL_PATH"] if not TEST_FAMILY else "tiny-test", run_dir=str(RUN_DIR),
    trigger=USER["TRIGGER"], images=len(df), dataset_sha1=dataset_hash, fixed=dict(FIXED),
    steps_done=len(history), nan_steps=nan_steps, train_minutes=round(train_sec / 60, 2),
    peak_gpu_gb=None if math.isnan(peak_gb) else round(peak_gb, 2),
    final_train_loss=float(pd.DataFrame(history).loss.tail(20).mean()) if history else None,
    val_loss={int(s): round(v, 5) for s, v in val_history}, best_val_step=int(best_step),
    checkpoints={int(s): str(p) for s, p in ckpt_paths.items()},
    lora_file_mb=round(Path(final_path).stat().st_size / 1e6, 2) if Path(final_path).is_file() else None,
    versions={k: _md.version(k) for k in PINNED},
)
(RUN_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
if TEST_FAMILY:
    print(f"\nNOTEBOOK TEST OK {TEST_FAMILY}")
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.validate(nb)
OUT.write_text(nbf.writes(nb), encoding="utf-8", newline="\n")
print("wrote", OUT, len(cells), "cells")
