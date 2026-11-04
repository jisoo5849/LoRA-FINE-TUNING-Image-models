# LoRA 미세조정 (이미지 모델)

메인 GPU 서버(사내 72GB GPU 1장)에서 스토리 등장인물의 외형만 담은 LoRA 를 학습하는 계획과 결과를 모으는 저장소입니다.

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
| `notebooks/run_many.py`, `notebooks/run_many_example.json` | 여러 모델·캐릭터 학습을 목록대로 차례로 실행하는 스크립트와 목록 예 |
| `notebooks/dev/` | 노트북을 만드는 빌더, 모델 계열별 어댑터 원본, 시험 도구 |

## 프롬프트 (니키 `nkgreen`)

- 기준 이미지는 Qwen-Image-2512 로 만든 니키 전신 3장(S06-Q1~Q3)이고, 그 외형을 고정값으로 매 컷 같은 문장으로 넣습니다.
  - 예: 새까만 턱선 단발, 와인빛 케이프 코트, 손을 덮는 긴 크림색 소매, 모카 반바지, 회색 타이츠, 갈색 끈 부츠
- 컷마다 바꾸는 것은 화면 크기(전신·상반신·얼굴), 각도, 표정·포즈, 배경뿐입니다 (24컷).
- 학습 캡션은 트리거 `nkgreen` 과 바뀌는 것만 적습니다.
- 이 외형은 2026-10-29 의 '상상도 우선'(짙은 갈색 머리, 파란 눈, 주근깨) 결정과 다릅니다 [TBD] 니키의 정본을 무엇으로 할지.
- 2026-11-02 부터 학습 데이터셋은 새로 받아 쓰므로, 이 프롬프트로 데이터셋을 만들지 않습니다 (참고용).

## 학습 노트북

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
