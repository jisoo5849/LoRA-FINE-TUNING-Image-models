# LoRA 미세조정 (이미지 모델)

메인 GPU 서버(사내 72GB GPU 1장)에서 스토리 등장인물의 외형만 담은 LoRA 를 학습하는 계획과 결과를 모으는 저장소입니다.

- 계획서: [lora-training-plan.md](lora-training-plan.md) (2026-10-29 초안)
- 첫 대상은 "니키와 이상한 나침반" 의 니키 한 명이고, 베이스 모델은 FLUX.2 [klein] Base 4B 를 제안합니다.
- 메인 GPU 서버 코드: [main-GPU-server-Documents](https://github.com/jisoo5849/main-GPU-server-Documents)
- 컨셉아트(학습 데이터 출발점): [all-about-Niki](https://github.com/jisoo5849/all-about-Niki)

## 폴더

| 경로 | 내용 |
|---|---|
| `prompts/nkgreen_qwen2512_prompts_en.md` | 니키 학습 이미지 24컷을 Qwen-Image-2512 로 만드는 영문 프롬프트 (긍정·부정·학습 캡션) |
| `prompts/nkgreen_qwen2512_prompts_ko.md` | 같은 내용의 한글판과 프롬프트 작성 규칙 |
| `prompts/nkgreen_shots.json`, `prompts/build_nkgreen_prompts.py` | 컷별 데이터와 두 문서를 만드는 스크립트 |
| `notebooks/lora_training.ipynb` | LoRA 학습 노트북 (EDA → 전처리 → 시각화 → 학습 → 검증) |
| `notebooks/dev/` | 노트북을 만드는 빌더, 모델 계열별 어댑터 원본, 시험 도구 |

## 프롬프트 (니키 `nkgreen`)

- 기준 이미지는 Qwen-Image-2512 로 만든 니키 전신 3장(S06-Q1~Q3)이고, 그 외형을 고정값으로 매 컷 같은 문장으로 넣습니다.
  - 예: 새까만 턱선 단발, 와인빛 케이프 코트, 손을 덮는 긴 크림색 소매, 모카 반바지, 회색 타이츠, 갈색 끈 부츠
- 컷마다 바꾸는 것은 화면 크기(전신·상반신·얼굴), 각도, 표정·포즈, 배경뿐입니다 (24컷).
- 학습 캡션은 트리거 `nkgreen` 과 바뀌는 것만 적습니다.
- 이 외형은 2026-10-29 의 '상상도 우선'(짙은 갈색 머리, 파란 눈, 주근깨) 결정과 다릅니다 [TBD] 니키의 정본을 무엇으로 할지.

## 학습 노트북

- 지원 모델 계열: `qwen_image`(Qwen-Image·2512), `qwen_image21`, `flux2_klein`, `z_image`, `krea2`, `flux1`, `chroma`, `longcat_image`, `sdxl`, `sd3`.
- 사용자가 정하는 값은 모델 계열, 모델 경로, 데이터셋 폴더, 출력 폴더, 트리거, 실행 이름뿐입니다.
- 학습 조건은 고정되어 있습니다.
  - 예: LoRA 랭크 16·알파 16, AdamW(학습률 1e-4, betas 0.9/0.999, weight decay 1e-4), 워밍업 100 단계, 1,500 단계, 배치 1, 기울기 자르기 1.0, 1024px 비율별 묶음, bf16, 시드 42
- 모델마다 다른 부분(텍스트 인코딩, 잠재값 정규화, 노이즈 공식, 잠재값 묶기, LoRA 대상 층, 저장 형식)은 계열별 어댑터가 맡고, 각 어댑터는 diffusers v0.42.1 공식 학습 스크립트와 대조했습니다.
- 버전 고정: torch 2.15.0, diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0
- 모델은 diffusers 형식 폴더나 Hugging Face 저장소 이름으로 불러옵니다. ComfyUI 용 단일 `.safetensors` 파일은 쓸 수 없습니다.

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
