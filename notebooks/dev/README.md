# 노트북 개발 도구

`../lora_training.ipynb`(v1), `../lora_training_v2.ipynb`(v2), `../lora_training_video.ipynb`(영상)는 이 폴더의 빌더로 만든다. 노트북을 직접 고치지 말고 빌더나 어댑터를 고친 뒤 다시 만든다.

| 파일 | 하는 일 |
|---|---|
| `build_notebook.py` | 셀 원본과 `adapters/*.py` 를 합쳐 노트북을 만든다 |
| `v2_parts.py` | v2 에만 들어가는 부분. 빌더 셀 원본의 `# [v2:이름]`(한 줄 추가), `# [v1:이름] ... # [/v1]`(v1 부분 바꾸기) 자리에 들어간다 |
| `adapters/<계열>.py` | 모델 계열별 어댑터 (규격은 `INTERFACE.md`) |
| `harness.py` | 어댑터 하나를 아주 작은 무작위 모델로 시험한다 (인코딩, 손실, 기울기, LoRA 저장·다시 불러오기, 생성) |
| `run_nb.py` | 노트북 전체를 시험 모드로 실행한다 |
| `test_v2.py` | 10개 계열마다 v1 과 v2(설정 4가지)를 시험 모드로 돌려, v2 가 환경변수만 바꿔 도는지와 v1 과 손실이 같은지 확인한다 |

```
python build_notebook.py adapters ../lora_training_v2.ipynb v2
python build_notebook.py adapters ../test_runs/v1_check.ipynb      # v1 은 확인용 경로에 만든다 (아래 참고)
NB=../lora_training_v2.ipynb python run_nb.py qwen_image cpu x   # v2 시험
KN=<커널> python test_v2.py <짧은 작업 폴더> [--device cuda]      # v1·v2 전체 시험 (계열당 5번)
python harness.py qwen_image                 # HARNESS OK qwen_image
python run_nb.py qwen_image cpu x            # qwen_image cpu OK
python run_nb.py qwen_image cuda x           # GPU·bf16 시험
```

- 시험 버전: torch 2.15.0 (CPU 판, CUDA 13.2 판), diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0, Python 3.14.
- 작은 토크나이저를 Hugging Face 에서 내려받으므로 시험에는 인터넷이 필요하다.
- `../lora_training.ipynb`(v1)에는 실행한 출력이 들어 있다. 빌더로 그 자리에 다시 만들면 출력이 지워지므로, 확인할 때는 `../test_runs/` 에 만들어 셀 원본만 비교한다.
  - 예: `python -c "import nbformat as n; f=lambda p:[(c.id,c.cell_type,c.source) for c in n.read(p,4).cells]; print(f('../test_runs/v1_check.ipynb')==f('../lora_training.ipynb'))"`
- v2 를 고칠 때도 위처럼 v1 셀 원본이 그대로인지 확인한다 (표시 줄은 v1 에서 지워진다).
- Windows 에서는 시험 폴더 경로가 길면 LoRA 저장이 실패한다. 사용자 폴더 바로 아래처럼 짧은 경로에서 돌린다.

## 영상 노트북 (`../lora_training_video.ipynb`)

| 파일 | 하는 일 |
|---|---|
| `build_video_notebook.py` | 셀 원본과 아래 모듈을 `%%writefile` 로 합쳐 영상 노트북을 만든다 |
| `VIDEO_INTERFACE.md` | 영상 어댑터와 도구 모듈의 규격 (영문) |
| `video_adapters/<계열>.py`, `video_adapters/wan_common.py` | Wan 계열 어댑터 4개와 공통 코드 (인코딩, 손실, ComfyUI 키 저장·다시 불러오기, 생성) |
| `video_tools/video_io.py` | 영상 읽기(PyAV)·EDA 신호·리샘플·창 고르기·버킷·시험 데이터셋 |
| `video_tools/video_metrics.py`, `video_tools/remedy.py` | 검증 지표(움직임·깜빡임·색·정체성)와 보완 규칙(증상 → 다음 실행 설정) |
| `video_tools/test_*.py` | 위 세 모듈의 시험 |
| `video_harness.py` | 어댑터 하나를 작은 무작위 모델로 시험한다 (CPU fp32·bf16, 잘못된 모델 폴더 거부, ComfyUI 키) |
| `test_video.py` | 노트북 전체를 시험 모드(작은 모델 + 문제를 넣은 합성 데이터)로 4개 계열 실행 |
| `test_video_user.py` | 사용자 방식 시험: 작은 모델을 diffusers 폴더로 저장해 설정 5개(TRIGGER, MODEL_PATH, RUN_NAME, DATASET_DIR, TARGET_SCOPE)만 주고 실행. 계열 자동 인식, 이미지만 있는 데이터, 이미지 20장 + 클립 1개, 잘못된 값 7가지도 확인 |

```
python build_video_notebook.py ../lora_training_video.ipynb
KN=<커널> python test_video.py <짧은 작업 폴더> [계열 ...] [--device cuda]       # ALL OK
KN=<커널> python test_video_user.py <짧은 작업 폴더> [계열 ...] [--device cuda]  # USER MODE ALL OK
python video_harness.py wan22_t2v                                           # VIDEO HARNESS OK
python video_tools/test_video_io.py <폴더>                                   # VIDEO IO OK
```

- 시험에는 `hf-internal-testing/tiny-random-t5` 가 필요하다 (`HF_HOME` 에 받아 두면 `HF_HUB_OFFLINE=1` 로 돌릴 수 있다).
- `LORA_NB_SMOKE=1` 은 `test_video_user.py` 가 쓰는 개발용 값이다. 실제 경로는 그대로 두고 학습 크기만 시험 모드처럼 줄인다. GPU 가 없어도 CPU 로 돈다 (SMOKE·시험 모드가 아닌 실제 실행은 GPU 가 보이지 않으면 0-4 셀에서 멈춘다).
- 시험 데이터셋을 만들려면 PyAV 에 libx264·libvpx-vp9 인코더가 있어야 한다 (pip 의 av 휠에는 들어 있다).
