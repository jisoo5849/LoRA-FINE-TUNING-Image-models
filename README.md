# LoRA 미세조정 (이미지·영상 모델)

메인 GPU 서버(사내 72GB GPU 1장)에서 스토리 등장인물의 외형만 담은 LoRA 를 학습하는 계획과 결과를 모으는 저장소입니다.

- 이미지 모델: `notebooks/lora_training.ipynb`(v1), `notebooks/lora_training_v2.ipynb`(v2)
- 영상 모델(Wan 2.1·2.2): `notebooks/lora_training_video.ipynb` (2026-11-05, 아래 '영상 모델 학습 노트북' 절)

- 계획서: [lora-training-plan.md](lora-training-plan.md) (2026-10-29 초안, 2026-11-02 개정)
- 첫 대상은 "니키와 이상한 나침반" 의 니키 한 명이고, 베이스 모델은 FLUX.2 [klein] Base 4B 를 제안합니다.
- 학습 데이터셋은 새로 만들어진 것을 받아 쓰고, 이 저장소에서 만들지 않습니다 (2026-11-02).
- 모델별 학습할 모델·실제로 쓸 모델·라이선스와 LoRA 외 방법 비교는 계획서 1-1 절과 8 절에 있습니다.
- 메인 GPU 서버 코드: [main-GPU-server-Documents](https://github.com/jisoo5849/main-GPU-server-Documents)
- 컨셉아트: [all-about-Niki](https://github.com/jisoo5849/all-about-Niki)

## 폴더

| 경로 | 내용 |
|---|---|
| `prompts/nkgreen_qwen2512_prompts_en.md` | 니키 학습 이미지 24컷을 Qwen-Image-2512 로 만드는 영문 프롬프트 (긍정·부정·학습 캡션) |
| `prompts/nkgreen_qwen2512_prompts_ko.md` | 같은 내용의 한글판과 프롬프트 작성 규칙 |
| `prompts/nkgreen_shots.json`, `prompts/build_nkgreen_prompts.py` | 컷별 데이터와 두 문서를 만드는 스크립트 |
| `notebooks/lora_training.ipynb` | LoRA 학습 노트북 (EDA → 전처리 → 시각화 → 학습 → 검증) |
| `notebooks/lora_training_v2.ipynb` | v2: 학습 범위 넓히기 + 그림체 보존 학습. v1 과 따로 둔 별도 노트북 |
| `notebooks/lora_training_video.ipynb` | 영상 모델(Wan 2.1·2.2) LoRA 학습 노트북 (EDA → 시각화 → 전처리 → 학습 → 검증 → 보완) |
| `notebooks/run_many.py`, `notebooks/run_many_example.json` | 여러 모델·캐릭터 학습을 목록대로 차례로 실행하는 스크립트와 목록 예 |
| `notebooks/dev/` | 노트북을 만드는 빌더, 모델 계열별 어댑터 원본, 시험 도구 |

## 프롬프트 (니키 `nkgreen`)

- 기준 이미지는 Qwen-Image-2512 로 만든 니키 전신 3장(S06-Q1~Q3)이고, 그 외형을 고정값으로 매 컷 같은 문장으로 넣습니다.
  - 예: 새까만 턱선 단발, 와인빛 케이프 코트, 손을 덮는 긴 크림색 소매, 모카 반바지, 회색 타이츠, 갈색 끈 부츠
- 컷마다 바꾸는 것은 화면 크기(전신·상반신·얼굴), 각도, 표정·포즈, 배경뿐입니다 (24컷).
- 학습 캡션은 트리거 `nkgreen` 과 바뀌는 것만 적습니다.
- 이 외형은 2026-10-29 의 '상상도 우선'(짙은 갈색 머리, 파란 눈, 주근깨) 결정과 다릅니다 [TBD] 니키의 정본을 무엇으로 할지.
- 2026-11-02 부터 학습 데이터셋은 새로 받아 쓰므로, 이 프롬프트로 데이터셋을 만들지 않습니다 (참고용).

## 학습 노트북 (이미지 모델)

- 지원 모델 계열: `qwen_image`(Qwen-Image·2512), `qwen_image21`, `flux2_klein`, `z_image`, `krea2`, `flux1`, `chroma`, `longcat_image`, `sdxl`, `sd3`.
- 사용자가 정하는 값은 모델 계열, 모델 경로, 데이터셋 폴더, 출력 폴더, 트리거, 실행 이름, 검증 프롬프트 파일(선택)뿐입니다.
- 학습 조건은 공통값에 계열별 시작값을 덮어써 고정합니다 (출처는 계획서 3-3 절).
  - 예: 공통값은 LoRA 랭크 16·알파 16, AdamW(학습률 1e-4, betas 0.9/0.999, weight decay 1e-4), 워밍업 100 단계, 1,500 단계, 배치 1, 기울기 자르기 1.0, 1024px 비율별 묶음, bf16, 시드 42
  - 예: FLUX.2 klein·Qwen-Image 는 2,000 단계, Krea 2 는 랭크 32·학습률 3e-4·1,000 단계, Z-Image 는 랭크 32
- 랭크와 알파는 늘 같게 둡니다. ComfyUI 는 alpha 텐서가 없는 LoRA 를 강도 1(알파 = 랭크)로 적용하기 때문입니다.
- 학습할 모델 경로가 증류판이거나 비상업·매출 조건부 모델이면 주의 문구를 보여 줍니다 (증류판: v1 은 klein 증류판과 Z-Image·Krea 2 의 Turbo, v2 는 경로에 `turbo` 가 들어간 모든 모델).
- 검증 샘플은 학습한 모델로 그린 진행 확인용이고, 최종 판정은 ComfyUI 에서 실제로 쓸 모델로 합니다 (노트북 5-6 절).
- 학습 데이터에 없던 장면으로 시험하는 셀(5-4)이 있습니다. `EXTRA_PROMPTS` 를 바꿔 그 셀만 다시 실행하면 되고, 그림은 `<실행 폴더>/extra_*.png` 에 남습니다.
- 모델마다 다른 부분(텍스트 인코딩, 잠재값 정규화, 노이즈 공식, 잠재값 묶기, LoRA 대상 층, 저장 형식)은 계열별 어댑터가 맡고, 각 어댑터는 diffusers v0.42.1 공식 학습 스크립트와 대조했습니다.
- 버전 고정: torch 2.15.0, diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0
- 모델은 diffusers 형식 폴더나 Hugging Face 저장소 이름으로 불러옵니다. ComfyUI 용 단일 `.safetensors` 파일은 쓸 수 없습니다.

### 모델이나 캐릭터를 바꿀 때 바꾸는 값

v1·v2 모두 코드는 고치지 않고 환경변수(`LORA_*`)나 0-2 절의 `USER` 값만 바꿉니다.

| 언제 | 바꾸는 값 | 예 |
|---|---|---|
| 모델을 바꿀 때 | `LORA_MODEL_FAMILY`, `LORA_MODEL_PATH`, `LORA_RUN_NAME` | `flux2_klein`, `/home/mgs/models/diffusers/FLUX.2-klein-base-4B`, `nkq7_klein4b_v1` |
| 캐릭터를 바꿀 때 | `LORA_DATASET_DIR`, `LORA_TRIGGER`, `LORA_RUN_NAME`, (선택) `LORA_VALIDATION_FILE`, (v2) `LORA_CLASS_WORD` | `dataset/quillins`, `qlq3`, `qlq3_qwen2512_v1`, `man` |
| v2 학습 방식을 바꿀 때 | `LORA_TARGET_SCOPE`, `LORA_PRESERVATION`, `LORA_RUN_NAME` | `attn_mlp`, `off`, `nkq7_qwen2512_v2_mlp` |

- `LORA_RUN_NAME` 은 매번 새 이름으로 합니다. v1 은 같은 이름이면 앞선 결과를 덮어씁니다. v2 는 같은 이름의 결과 폴더가 있으면 시작하지 않습니다.
- 예시 경로는 메인 서버에서 Qwen-Image-2512 를 학습할 때 실제로 쓴 위치(`/home/mgs/models/diffusers`) 기준입니다. 계획서와 `run_many_example.json` 의 `/srv/models/diffusers` 와 다르므로, 모델을 받아 둔 실제 폴더로 바꿔 씁니다.
- 계열을 고르면 아래는 저절로 바뀝니다: 랭크·학습률·단계 수(계열별 시작값), LoRA 대상 층, 노이즈 공식과 시그마 이동, 잠재값 묶기, 검증 샘플의 step 수와 guidance, LoRA 저장 형식.
- 모델 폴더는 미리 받아 둡니다. FLUX.2 klein 9B, Krea 2, FLUX.1, SD3.5 는 Hugging Face 로그인(사용 동의)이 필요합니다.
- 학습할 모델과 실제로 쓸 모델이 다른 계열(klein, Krea 2, Z-Image)은 노트북 첫 셀의 표를 따릅니다.

### v2 노트북: 학습 범위 넓히기 + 그림체 보존 (`notebooks/lora_training_v2.ipynb`, 2026-11-04)

- v1 과 같은 빌더에서 만든 별도 노트북이다. 바뀐 부분만 `notebooks/dev/v2_parts.py` 에 있고, v1 노트북은 바뀌지 않는다.
- v1 처럼 환경변수나 `USER` 값만 바꿔 10개 계열 모두 학습한다. v2 에서 더 정할 수 있는 값은 아래 3개이고, 빼면 기본값을 쓴다.
- `LORA_TARGET_SCOPE` 로 LoRA 를 붙일 층을 고른다 (기본 `all`).
  - `attn`: v1 과 같은 층
  - `attn_all`: + 나머지 어텐션 (Qwen-Image·FLUX.2 의 글 쪽 어텐션 등)
  - `attn_mlp`: + MLP
  - `all`: + 블록 안 조절(modulation) 층
  - 예: Qwen-Image 랭크 16 의 LoRA 는 `attn` 파라미터 약 2,360만(파일 약 94MB), `all` 약 1억 4,750만(약 590MB). LoRA 는 fp32 로 저장된다.
  - 계열별 파라미터 수는 노트북 첫 셀의 표에 있다 (실제 모델 설정으로 셈). `all` 은 `attn` 의 1.2배(Krea 2)~6.3배(Qwen-Image).
- 더하는 층은 모두 블록 안 선형층이고, 계열마다 층 이름을 정규식으로 정해 두었다 (노트북 0-3-1 절).
- v1 기본 층은 v1 학습률 그대로, 새로 더한 층은 낮춘다 (글 쪽 어텐션·MLP 0.5배, 조절 층 0.25배) [제안].
- `LORA_PRESERVATION=dop`(기본)이면 ostris ai-toolkit 의 Differential Output Preservation 과 같은 보존 학습을 함께 한다.
  - 트리거를 `LORA_CLASS_WORD`(기본 `girl`)로 바꾼 캡션의 출력이, LoRA 를 끈 베이스 모델의 출력과 같도록 학습한다.
  - 한 단계에 디노이저를 세 번 계산하므로 학습 시간이 v1 의 약 2~2.5배로 추정된다 (실측 전).
- 트리거를 `CLASS_WORD` 로 바꾼 검증 프롬프트를 단계마다 그려 0 단계와 비교한다 (노트북 5-2-1 절, `drift_*.png`).
- `RUN_NAME` 기본값은 `nkgreen_v2` 이고, 같은 이름의 결과 폴더가 이미 있으면 0-2 절과 학습 셀에서 멈춘다 (덮어쓰기 방지).
- `MODEL_PATH` 기본값은 `/home/mgs/models/diffusers/Qwen-Image-2512` 다 (v1 은 `/srv/models/diffusers/Qwen-Image-2512`).
- 경로에 `turbo` 가 들어간 모델은 계열과 상관없이 증류판 주의 문구를 보여 준다.
- `run_many.py --notebook lora_training_v2.ipynb` 로 돌리면 목록에 `target_scope`, `preservation`, `class_word` 를 줄 수 있다.
- 시험 (2026-11-04, 작은 무작위 모델, `notebooks/dev/test_v2.py`)
  - 10개 계열마다 v1 1번과 v2 4번(`attn`+`off`, `attn_all`+`dop`, `attn_mlp`+`off`, `all`+`dop`), 모두 50번을 CPU(fp32)와 GPU(RTX 5060, bf16)에서 끝까지 실행했다.
  - 10개 계열 모두 v2 `attn` + `off` 의 단계별 손실이 v1 과 같았다.
  - 범위마다 정한 묶음이 LoRA 에 들어가 학습됐다. 단, 마지막 블록의 글 쪽 출력 층(Qwen-Image 의 `txt_mlp`·`to_add_out`·`add_q_proj`, SD3 의 `add_q_proj`)은 모델이 결과를 버리는 층이라 학습되지 않는다. v1 에서도 같다.
  - SD3 의 `attn_extra`(`attn2`, SD3.5 Medium 에만 있음)는 1층 시험 모델에 없어서, 2층 시험 모델로 학습·저장·다시 불러오기를 따로 한 번 확인했다.
  - 실제 모델 설정으로 만든 빈 모델에서 정한 묶음이 모두 블록 안 층과만 맞는 것과, 10개 계열의 공식 모델 폴더(`model_index.json`)를 노트북이 받아들이는 것을 확인했다.
  - 보존 손실은 작은 Qwen-Image 에서 LoRA 를 끈 출력이 원래 모델과 같고, 그래디언트 체크포인트를 켜고 꺼도 기울기가 같았다.
- 실제 가중치로는 학습해 보지 않았다 [TBD] 메인 서버에서 Qwen-Image-2512 로 v1(attn) 과 같은 트리거·같은 시드로 비교.
- ComfyUI 가 더한 층의 키를 읽는지는 코드로만 확인했다 [TBD] 실제로 불러 `lora key not loaded` 확인.

### 여러 모델·캐릭터를 차례로 학습 (`notebooks/run_many.py`)

- 실행 목록(JSON)의 항목마다 노트북을 새 커널로 한 번씩 차례로 돌립니다.
  - 예: [run_many_example.json](notebooks/run_many_example.json) 은 니키를 klein Base 4B 와 Qwen-Image-2512 에 차례로 학습
- 항목의 값은 노트북 `USER` 값(`LORA_*` 환경변수)이 되고, `defaults` 는 항목에 없는 값을 채웁니다. 상대 경로는 목록 파일 폴더 기준입니다.
- 시작 전에 목록 전체를 검사해, 문제가 하나라도 있으면 아무것도 실행하지 않습니다.
  - 예: 지원하지 않는 계열, 빠진 값, 같은 실행 이름, 없는 데이터셋·모델·검증 파일
- `summary.json` 이 있는 실행은 건너뛰고, 중단·실패로 끝나지 않은 폴더는 `<실행 이름>.incomplete-<시각>` 으로 옮긴 뒤 처음부터 다시 합니다.
- 한 실행이 실패해도 다음으로 넘어가고(`--stop-on-error` 면 멈춤), 끝에 결과 표를 보여 줍니다.
- 실행한 노트북은 `<실행 폴더>/executed.ipynb` 에, 실행마다 한 줄 기록(시간, 최대 메모리, 파일 크기, 손실)은 `<output_dir>/run_many_log.csv` 에 남습니다.
- 학습용 가상환경의 python 으로 실행하고, 메인 서버에서는 관리자 '대기열 멈춤' 상태에서만 돌립니다 (계획서 P2).
- 영상 노트북은 `--notebook lora_training_video.ipynb` 로 돌립니다.
  - 목록에서 `family` 는 빼도 됩니다 (노트북이 `model_path` 로 정함). 단, `--test` 에는 계열 이름이 필요합니다.
  - `target_scope` 는 `all`·`attn`·`kvo` 를 쓰고, `output_dir` 를 빼면 `<목록 파일 폴더>/outputs_video` 에 저장합니다.

```
cd notebooks
python run_many.py run_many_example.json --dry-run   # 검사와 실행 계획만
python run_many.py run_many_example.json --test      # 시험 모드(작은 무작위 모델)로 계열별 확인
python run_many.py run_many_example.json             # 실제 학습
```

- 이 스크립트는 환경변수를 읽어 `summary.json` 만 쓰는 가짜 노트북으로 8가지 경우를 시험했습니다 (2026-11-02).
  - 예: 목록 검사 오류, 미리 보기, 실행 간 환경변수 초기화, 건너뛰기, 끝나지 않은 폴더 옮기기, 실패 뒤 계속, `--stop-on-error`, 시험 모드
- 실제 학습으로는 아직 돌려 보지 않았습니다 [TBD].

### 오류 검사 결과 (2026-11-02)

- 10개 계열 모두, 같은 구조의 아주 작은 무작위 모델로 노트북 전체를 CPU(fp32)와 GPU(RTX 5060, bf16)에서 끝까지 실행해 통과했습니다.
  - 예: `LORA_NB_TEST=qwen_image jupyter nbconvert --to notebook --execute notebooks/lora_training.ipynb`
- 시험 데이터에는 EXIF 회전 사진, 16비트 흑백 이미지, BOM·CP949·UTF-16 캡션, 중복 이미지, 캡션 없는 이미지가 들어 있습니다.
- 독립 검토에서 확인되어 고친 것은 아래와 같습니다.
  - Chroma 는 diffusers 0.42.1 의 그래디언트 체크포인트 경로가 텍스트 마스크를 빠뜨려 체크포인트를 끕니다. 다른 계열은 켜기 전에 켠 것과 끈 것의 손실이 같은지 확인합니다.
  - SD3·Z-Image·LongCat 은 공식 레시피처럼 노이즈 단계를 높은 쪽으로 옮깁니다(shift).
  - Z-Image 는 노이즈 단계 값을 bf16 으로 바꾸지 않고 fp32 로 넘깁니다.
  - 손실이나 기울기가 NaN 인 단계는 가중치를 바꾸지 않고 건너뛰며, 그래도 저장·검증은 합니다.
  - Qwen-Image 2.1 은 torchvision 없이도 불러지게 했습니다 (torch 2.15 와 짝인 torchvision 이 아직 나오지 않음).
- 실제 가중치로는 아직 학습해 보지 않았습니다 [TBD] 메인 서버 파일럿(계획서 P2)에서 시간·메모리·파일 크기를 재서 기록.
- 같은 날 개정(계열별 시작값, 검증 프롬프트 파일, 모델 주의 문구, 셀 id 고정) 뒤에는 설정 셀만 23가지 경우로 따로 시험했습니다.
- 새 프롬프트 시험 셀(5-4)은 가짜 부품으로 3가지 경우(트리거 붙이기, 프롬프트·시드 바꿔 다시 실행, 프롬프트 1개)와 셀 제목·순서를 시험했습니다.
  - 노트북 전체를 작은 무작위 모델로 다시 돌리는 시험은 하지 않았습니다 [TBD] 시험 환경을 다시 만들어 확인.
  - 5-4 셀은 실제 모델에서 실행해 보지 않았습니다 [TBD] Qwen-Image-2512 파일럿에서 확인.

## 영상 모델 학습 노트북 (`notebooks/lora_training_video.ipynb`, 2026-11-05)

- 영상 생성 모델에 캐릭터 LoRA 를 학습하는 별도 노트북입니다. 순서는 EDA → 시각화 → 전처리 → 학습 → 검증 → 보완입니다.
- 이미지 v2 처럼 코드는 고치지 않고 0-2 절의 `USER` 값(또는 `LORA_*` 환경변수)만 바꿉니다. 모델을 바꿀 때 고치는 값은 5개입니다.

| 값 | 환경변수 | 예 |
|---|---|---|
| `TRIGGER` | `LORA_TRIGGER` | `nkq7` |
| `MODEL_PATH` | `LORA_MODEL_PATH` | `/home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers` |
| `RUN_NAME` | `LORA_RUN_NAME` | `nkq7_wan22t2v_v1` |
| `DATASET_DIR` | `LORA_DATASET_DIR` | `~/lora_datasets/nkq7_final` |
| `TARGET_SCOPE` | `LORA_TARGET_SCOPE` | `all`(어텐션 + FFN, 기본), `attn`(어텐션만), `kvo`(어텐션의 k·v·o 만) |

- 모델 계열은 `MODEL_PATH` 폴더의 `model_index.json`, transformer·vae 설정으로 자동으로 정합니다. 계열별 값(fps, 버킷, 검증 step·guidance, 전문가 수)도 따라 바뀝니다.
  - Hugging Face 공식 Diffusers 저장소 8개의 설정 파일로 확인했습니다. 지원하는 5개는 맞는 계열을 고르고, Wan2.1 I2V·FLF2V·VACE 는 이유를 보여 주고 멈춥니다.
- `TARGET_SCOPE` 는 이미지 v2 이름도 받습니다 (`attn_mlp` = `all`, `attn_all` = `attn`). Wan 블록의 선형층은 어텐션과 FFN 이 전부이기 때문입니다.
- 경로에는 `~` 를 쓸 수 있고, 상대 경로는 노트북 폴더 기준입니다. `MODEL_PATH` 에는 Hugging Face 저장소 이름(`Wan-AI/Wan2.2-T2V-A14B-Diffusers`)이나 주소도 됩니다.
- 없는 경로, 지원하지 않는 모델, 클립이 없는 I2V 데이터는 모델(수십 GB)을 불러오기 전에 알려 줍니다.
- `RUN_NAME` 폴더에 저장된 LoRA 나 `summary.json` 이 있으면 0-2 절에서 멈춥니다. 첫 저장 전에 멈춘 실행의 폴더는 이어서 씁니다.
- 결과 위치: `<노트북 폴더>/outputs_video/<RUN_NAME>/checkpoint-<5자리 단계>/<RUN_NAME>_<high|low|main>.safetensors`
  - 예: `outputs_video/nkq7_wan22t2v_v1/checkpoint-00250/nkq7_wan22t2v_v1_low.safetensors` (250단계마다, 마지막 단계도)
  - ComfyUI 가 읽는 Wan 원래 키 이름(`diffusion_model.blocks.N.self_attn.q.lora_A.weight` 등)으로 저장합니다.
  - A14B 는 저노이즈를 먼저(기본 1,500단계), 고노이즈를 나중에(500단계) 학습해서 `_low`·`_high` 가 서로 다른 단계 폴더에 생깁니다. ComfyUI 에서 각 전문가 모델에 따로 걸고, 짝은 6-2 절이 추천한 파일을 씁니다.
  - 클립 없이 이미지만으로 학습하면 `_low` 하나만 나옵니다 (저노이즈가 2,000단계 전부 학습, 고노이즈 모델에는 LoRA 를 걸지 않음).
- 이 밖의 값(`EXPERT`, `CLASS_WORD`, `VALIDATION_FILE`, `FIXED_OVERRIDES` 등)은 보통 그대로 둡니다.
  - `CLASS_WORD` 는 캡션의 girl·boy 등, 없으면 she/he 대명사로 정하고, 둘 다 없으면 `person` 입니다. 니키 캡션은 `person` 이 됩니다. 바꾸려면 `LORA_CLASS_WORD=girl`.

### 지원 모델 (2026-11-05 조사)

| 모델 | 학습 데이터 | LoRA 파일 | 쓰임 |
|---|---|---|---|
| Wan2.2-T2V-A14B | 이미지·클립 | 고노이즈·저노이즈 2개 | 지금 가진 이미지로 학습하는 1단계. 이미지만 있으면 저노이즈 LoRA 만 만든다 |
| Wan2.2-I2V-A14B | 클립만 | 2개 | 회사 기본 영상 모델. 클립이 생기면 학습 |
| Wan2.2-TI2V-5B | 이미지·클립 | 1개 | 가벼운 파일럿 (공식 해상도 1280x704 면적까지). 5B LoRA 는 A14B 에 쓸 수 없다 |
| Wan2.1-T2V-1.3B·14B | 이미지·클립 | 1개 | 코드 점검(1.3B)·대안(14B) |

- 모두 Apache-2.0 입니다. 뺀 후보와 이유 (라이선스 원문 확인)
  - HunyuanVideo·HunyuanVideo-1.5·MiniMax H3: 라이선스가 대한민국을 제외
  - LTX-2.x: 연매출 US$10M 이상이면 유료 라이선스, 출력물로 다른 모델을 상업적으로 학습할 수 없음
  - Kandinsky 5·6: 모델은 MIT 지만 Tencent HunyuanVideo VAE 를 포함 (법률 검토 필요)
- 애니 특화 Index-AniSora V3.2 는 Wan2.2-I2V-A14B 구조라, diffusers 형식으로 바꾸면 I2V 로 학습할 수 있습니다 [미검증].

### 메인 서버에서 처음 실행할 때

1. 학습 가상환경에 영상 라이브러리를 더합니다: `pip install av ftfy` (선택: 애니 캐릭터 정체성 지표용 `pip install onnxruntime`)
2. 모델과 검증 지표용 작은 모델을 받습니다.
   - 예: `hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir /home/mgs/models/diffusers/Wan2.2-T2V-A14B-Diffusers`
   - 지표: `hf download facebook/dinov2-small config.json model.safetensors`
   - 선택(CCIP): `hf download deepghs/ccip_onnx ccip-caformer-24-randaug-pruned/model_feat.onnx ccip-caformer-24-randaug-pruned/model_metrics.onnx ccip-caformer-24-randaug-pruned/metrics.json`
   - 파일 이름을 적어 받습니다. `--include` 는 값을 하나만 받습니다: 뒤에 이어 쓴 값은 파일 이름이 되어 `--include` 가 무시되고 첫 값이 빠집니다 (글롭이면 오류). 패턴을 여러 개 쓰려면 값마다 `--include` 를 붙입니다.
   - 실행 공간에서 인터넷이 막혀 있으면 미리 받아 두고 `HF_HUB_OFFLINE=1` 로 실행합니다. 아니면 4-2 셀이 내려받기를 몇 분 기다린 뒤 그 지표를 빼고 넘어갑니다. 지표가 필요 없으면 `LORA_FIXED_OVERRIDES='{"use_dino": false, "use_ccip": false}'` 로 끕니다.
3. 시험 모드로 한 번 돌려, 리눅스에서도 영상 읽기(회전·VFR·프레임 수)가 맞는지 봅니다. 작은 토크나이저를 받으므로 Hugging Face 접속이 필요합니다.
   - `LORA_NB_TEST=wan22_t2v jupyter nbconvert --to notebook --execute lora_training_video.ipynb --output test_run.ipynb`
4. ComfyUI 대기열을 멈추고, 5개 값을 정해 실행합니다.

- A14B 는 전문가 2개(bf16 각 약 28.6GB)를 번갈아 GPU 에 올리고, 쉬는 쪽은 CPU 메모리에 둡니다. 미리보기·최종 생성 때는 모든 구성요소(약 69GB)가 CPU 메모리에 있습니다.
- 0-4 셀이 모델을 불러오기 전에 GPU 가 보이는지와 메모리를 확인하고, 모자라면 필요한 양을 보여 주고 멈춥니다. 메모리 확인을 건너뛰려면 `LORA_SKIP_RAM_CHECK=1` 을 줍니다.
  - GPU 가 보이지 않으면(`torch.cuda.is_available()` 이 False) 멈춥니다. 컨테이너를 `--gpus all` 로 띄웠는지 확인합니다.
  - 시스템 메모리: 가중치 + 8GB 가 비어 있어야 합니다. 컨테이너 메모리 한도나 서버 RAM 이 그보다 작으면 다른 프로그램을 멈춰도 안 되므로, 원인(한도·RAM)과 들어가는 더 작은 모델을 알립니다.
  - GPU 메모리: 가장 큰 구성요소 + 4GB, 학습 단계 추정(LoRA·AdamW 포함), 생성(미리보기·최종) 추정 가운데 가장 큰 값이 비어 있어야 합니다. 생성 추정은 전문가를 번갈아 올리는 A14B 면 전문가 하나 + 생성 토큰, 아니면 디노이저 모두 + VAE 디코드입니다.
  - 가중치 파일이나 목록 파일(`*.safetensors.index.json`)이 깨졌으면(git-lfs 포인터·0바이트·데이터가 잘린 파일) 그 파일 경로를 알려 주고 멈춥니다.
- 시간과 GPU 메모리 [추정: 실제 폭의 블록을 노트북 GPU 에서 재고 연산량·층 수로 늘려 잡음. 첫 실행으로 확인]
  - A14B·이미지만·기본값이면 연산의 약 27% 가 학습, 30% 가 미리보기(9번 × 3클립), 43% 가 최종 생성(10클립)입니다. 서버 GPU 의 bf16 실효 성능을 200 TFLOPS 로 잡으면 약 2시간(학습만 약 33분)입니다.
  - 미리보기를 줄이려면 `LORA_FIXED_OVERRIDES='{"validate_every": 500}'` 를 줍니다 (9번 → 5번).
  - A14B 학습 단계의 GPU 메모리는 약 30GB + 토큰당 약 1MB 입니다. 클립이 있으면 기본 예산 14,040토큰에서 약 44GB 입니다.
  - A14B 미리보기·최종 생성은 전문가 하나 + 생성 토큰(기본 33프레임 480x832 = 14,040토큰, 토큰당 약 0.25MB)으로 잡아, 이미지만 학습해도 0-4 셀은 약 35GB 가 비어 있기를 요구합니다.
  - 토큰 예산(`token_budget`)은 주로 클립에 걸립니다. 이미지는 `max_area`(480x832)에 묶여 한 장이 최대 1,560토큰(약 32GB)이라, 이미지만 학습하면 `token_budget` 을 늘려도 학습이 같습니다. 1,560 아래로 내리면 이미지도 작아집니다 (예: 1,000 이면 512x768 → 400x608). 이미지 크기는 `max_area` 로 바꿉니다 (예: `{"max_area": 921600}` = 720p 면적, 한 장 최대 3,600토큰).
- 첫 실행 뒤 `summary.json` 의 `train_minutes`(학습 단계만), `preview_minutes`(학습 중 미리보기·검증·저장), `train_peak_gpu_gb`(학습 단계 최대), `max_train_tokens` 를 봅니다.
  - 클립이 있으면 0-4 셀이 보여 주는 '지금 들어가는 클립 token_budget' 안에서 `LORA_FIXED_OVERRIDES='{"token_budget": 20280}'` 처럼 늘립니다 (`FIXED` 에 있는 키만 받고, 보완 규칙에 범위가 있는 값은 그 안에서만 받음).
  - `FIXED` 의 정수 값은 1 이상만 받습니다. 0 이 뜻 있는 `seed`·`warmup_steps`·`holdout_files`·`final_steps`(0 = 공식 step 수)·`max_nan_steps` 만 0 을 받습니다.
  - 실수·목록·문자열 값은 0-3 셀이 학습 셀과 같은 객체(버킷·AdamW·학습률 스케줄러)를 만들어 확인하고, 오류 없이 학습을 망치는 값도 거절합니다 (예: `max_grad_norm` 0 은 '자르지 않음' 이 아니라 기울기를 0 으로 만듭니다. `allow_upscale` 은 1 이상, `aspects` 는 비우지 않습니다).
- 6 보완 단계가 다음 실행 설정(`next_run.json`, run_many 형식)을 만듭니다. 결과를 사람이 확인한 뒤 `python run_many.py <RUN_DIR>/next_run.json --notebook lora_training_video.ipynb` 로 돌립니다.

### 시험 결과 (2026-11-06)

- 같은 구조의 아주 작은 무작위 모델로 노트북 전체를 CPU(fp32)와 GPU(RTX 5060, bf16)에서 끝까지 실행했습니다.
  - 시험 모드(`notebooks/dev/test_video.py`): 4개 계열 모두 통과. 문제를 일부러 넣은 합성 데이터(회전·VFR·검은 구간·장면 전환·멈춤·중복·움직이는 WebP 등 18개 파일)의 EDA 결과를 정답과 비교합니다.
  - 사용자 방식(`notebooks/dev/test_video_user.py`): 작은 모델을 diffusers 폴더로 저장하고 설정 5개만 줘서 4개 계열 모두 통과. 학습 크기만 개발용 값 `LORA_NB_SMOKE=1` 로 줄였습니다.
    - 계열 자동 인식, `TARGET_SCOPE` 별 LoRA 층(`all`·`attn`·`kvo`·`attn_mlp`), ComfyUI 키, `next_run.json` 의 절대 경로를 확인했습니다.
    - 이미지만 있는 데이터(니키 캡션 형식, 1프레임 mp4 포함)로 T2V-A14B 를 돌리면 저노이즈 LoRA 만 만들고, `CLASS_WORD` 는 `person` 이 됩니다.
    - 이미지 20장 + 클립 1개면 검증용으로 이미지 2장만 빼고, 클립은 고노이즈 학습에 남습니다 (`CLASS_WORD` 는 캡션의 her 로 `girl`).
    - 기본 크기(2,000단계·480p)로도 처음부터 끝까지 돌렸습니다 (작은 모델, `LORA_` 값 5개만): T2V-A14B 이미지만(첫 실제 실행 모양)·이미지+클립, I2V 640x360 클립, TI2V-5B 480p 클립만.
    - 잘못된 값 7가지는 이유를 보여 주고 멈춥니다: 같은 RUN_NAME, 없는 MODEL_PATH·DATASET_DIR, 틀린 TARGET_SCOPE, 규칙에 맞지 않는 RUN_NAME, 계열이 다른 MODEL_FAMILY, 이미지만 준 I2V (같은 RUN_NAME 외에는 모델을 불러오기 전).
- 이 시험과 검토로 찾아 고친 것
  - 데이터·학습
    - 이미지만 있는 데이터셋에서 EDA 1-2 셀이 클립 전용 열을 찾다 멈추던 문제
    - 1프레임 mp4(이미지를 영상으로 저장한 파일)를 시작 이미지로 읽지 못하던 문제
    - 검증용으로 뺀 파일이 하나뿐인 클립을 가져가, 고노이즈 전문가가 학습할 샘플이 없어 멈추던 문제 (이제 종류마다 10개당 1개까지만 뺌)
    - A14B 체크포인트 추천이 함께 재지 않은 짝(저노이즈 학습 중인 단계의 고노이즈 파일)을 고르던 문제
    - I2V 시작 이미지를 가운데만 잘라 세로 전신 그림의 머리가 잘리던 문제 (이제 이미지 비율대로 생성)
    - 시험 모드에서 고노이즈 전문가가 학습률 0 인 1단계만 돌던 문제
    - NaN 으로 건너뛴 단계의 계산 그래프가 다음 단계까지 GPU 에 남던 문제
    - 기본 버킷보다 작은 클립·이미지(TI2V-5B 의 480p 2~5초 클립, 640x360 클립 등)를 이유를 알리지 않고 빼던 문제 (이제 원본 안에 들어가는 더 작은 버킷을 쓰고, 뺀 파일과 이유를 3-1 에 출력)
    - A14B 를 클립으로 학습하면 6-2 가 고노이즈 LoRA 를 추천하지 않던 문제 (검증용 파일이 있거나, CCIP 로 잰 정체성을 저노이즈가 먼저 통과하면 늘 그랬음), I2V 에서 CCIP 를 쓰면 첫 검증 단계(250)를 추천하던 문제 (I2V 미리보기는 학습 데이터 그림에서 시작해 베이스도 통과하므로 이제 정체성은 진단·추천에 쓰지 않음)
    - 작은 클립이 섞이면 미리보기·최종 생성까지 그 작은 크기로 만들던 문제 (이제 가장 흔한 학습 버킷의 비율로 `max_area` 넓이에서 생성)
    - 0-3 의 클립 확인이 3-1 의 규칙(가장 짧은 학습 창, 원본 크기)과 달라, 쓰지 못할 클립만 있는 I2V 가 모델을 다 불러온 뒤 3-1 에서 멈추던 문제
    - I2V 미리보기 크기 출력이 실제 생성 크기와 다르던 문제, 2-2 컨택트 시트가 이미지를 정사각형으로 늘이던 문제, 학습 셀의 UserWarning
  - 메모리
    - 메모리 확인이 모델을 불러온 뒤에 돌아, 모자라면 커널이 이유를 남기지 않고 죽던 문제 (이제 불러오기 전에 확인하고 멈춤)
    - GPU 확인이 가장 큰 구성요소만 봐서, 클립 학습(토큰 예산)이나 미리보기 VAE 디코드에 모자란 메모리를 통과시키던 문제
    - 깨진 가중치 파일에서 빈 MemoryError 가 나던 문제, 컨테이너 한도가 원인인데 다른 프로그램을 멈추라고 안내하던 문제
    - 셀 오류·중지 뒤 다시 실행하면 앞 실행의 모델이 메모리에 남던 문제 (학습 셀이 멈춘 뒤 전역 `loss` 가 쥔 계산 그래프, 체크포인트 재계산·DOP 중 오류의 traceback, 오류 셀 바로 다음에 0-4 를 실행할 때 IPython 이 쥐는 앞 셀의 실행 결과 포함)
    - GPU 추정이 LoRA·AdamW 를 rank 와 상관없는 상수로 세고, A14B 의 생성(미리보기·최종)과 오프로드를 끈 A14B 의 두 전문가를 세지 않던 문제
    - 학습에 쓰지 못할 짧은 움직이는 파일(모델 fps 로 17프레임 미만)만 있어도 클립 예산으로 세어 멈추던 문제, 토큰 예산 제안이 생성 몫을 보지 않고 다른 덮어쓴 값을 지우던 문제
    - 데이터 부분이 잘린 가중치 파일과 깨진 목록 파일(index.json)을 불러오기 전에 잡지 못하던 문제
    - 메모리 한도가 있는 컨테이너에서 회수할 수 있는 파일 캐시를 쓰는 중으로 세던 문제, `--cgroupns=host` 컨테이너의 한도를 보지 못하던 문제, 한도가 없는 서버에서도 docker 한도를 늘리라고 안내하던 문제
    - GPU 가 보이지 않으면 CPU·fp32 로 세어 메모리 부족으로 안내하던 문제 (이제 GPU 가 없다고 알리고 멈춤)
    - 학습 뒤 AdamW 상태와 3-4 왕복 확인 디코드가 GPU 에 남던 문제
  - 기록·설정·문서
    - `train_minutes` 에 미리보기 생성 시간이 섞여 학습 시간이 약 2배로 기록되던 문제
    - `summary.json` 의 체크포인트 경로가 상대 경로로 남던 문제
    - `FIXED_OVERRIDES` 가 `preview_steps` 0, `weight_decay` -1, `max_grad_norm` 0 같은 값을 받아 모델을 불러온 뒤 학습 셀에서 멈추거나 LoRA 를 0 으로 남기던 문제 (이제 모델을 불러오기 전 0-2·0-3 에서 거절), 쓰지 않는 `batch_size` 를 받던 문제
    - 0-4 의 안내 줄과 오류 문구가 같은 값을 다른 숫자로 보이던 문제, 3-1 이 비율이 극단적인 큰 그림(예: 1600x150)도 '작아' 뺀다고 안내하던 문제, 과적합 근거가 재지 않은 단계를 적던 문제
    - 문서의 DINO 내려받기 명령이 지금 hf CLI 에서 아무것도 받지 않던 문제, `--include`·`token_budget` 설명이 실제 동작과 다르던 문제
- 구성요소 검증 (어댑터·영상 읽기·지표·보완 규칙, 각 모듈 시험 통과)
  - 어댑터가 디노이저에 넣는 값이 diffusers 파이프라인과 같습니다 (차이 0~1.2e-7).
  - 저장한 LoRA 를 ComfyUI 가 실제 Wan 모델 헤더에서 모두 읽습니다 (빠진 키 0).
  - bf16 모델에 다시 불러온 LoRA 가 학습한 fp32 값과 비트 단위로 같습니다.
  - 리샘플이 ffmpeg fps 필터 규칙(vf_fps.c)을 옮긴 기준식과 15가지 경우에서 같습니다 (`test_video_io.py`).
- 실제 가중치로는 아직 학습하지 않았습니다 [TBD] 메인 서버에서 Wan2.2-T2V-A14B 와 니키 이미지 30장으로 첫 실행, 시간·메모리 기록.
- 움직임 지표(`block_motion_p95`)는 합성 벤치 2,168 사례에서 1~16 px/프레임을 참값의 0.9~1.4 배로 읽고, 속도 순서가 늘 맞습니다. 어둡거나 흐린 클립(화면에 밝은 정지 요소가 없을 때, 무늬가 빽빽한 그림 포함), 망점 무늬, 검은 프레임·페이드, 프레임마다 번갈아 오는 강한 깜빡임·검은 프레임도 같은 속도면 비슷한 값으로 읽고, 시간 잡음이 낀 어두운 정지 클립을 움직임으로 읽지 않습니다. 남은 한계가 있습니다. 1 px/프레임 근처는 1.2~1.4 배 크게 읽고, 화면의 5% 보다 작은 피사체와 시간 잡음이 낀 어둡거나 흐린 클립의 움직임은 작게 읽습니다(대비가 아주 낮으면 빠른 움직임도 0.5~0.6 배). 832x480 에서 약 32 px/프레임보다 빠르면 값이 더 커지지 않습니다. 화면 전체를 덮는 규칙 무늬(줄무늬·망점)는 한 주기 어긋나게 읽을 수 있고, 255 로 잘리는 번쩍임 프레임과 흰 바탕 선화의 ±25% 넘는 노출 깜빡임은 움직임으로 읽습니다(깜빡임 지표로는 걸림). [TBD] 실제 Wan 미리보기 클립으로 기준(motion_low 0.6, motion_high 1.6) 다시 맞추기.
