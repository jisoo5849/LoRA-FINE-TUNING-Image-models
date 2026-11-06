# 캐릭터 LoRA 학습 계획서 (2026-10-29 초안, 2026-11-02 개정)

메인 GPU 서버에서 스토리 등장인물의 외형만 담은 LoRA 를 학습해, 베이스 모델은 그대로 두고 캐릭터 가중치 파일(.safetensors)만 따로 만드는 계획이다.
첫 대상은 "니키와 이상한 나침반" 의 니키 한 명이고, 결과를 보고 나머지 8명과 허브 기능으로 넓힌다.

2026-11-02 개정에서 바뀐 것은 아래와 같다.

- 학습 데이터셋은 새로 만들어진 것을 받아 쓰고, 우리가 만들지 않는다 (사용자 결정).
- 학습 도구를 자체 노트북(`notebooks/lora_training.ipynb`)으로 정하고, 첫 학습 때 diffusers 공식 스크립트와 비교해 구현을 검증한다.
- 모델별로 학습할 모델, 실제로 쓸 모델, 라이선스, 시작값을 정리했다 (1-1 절).
- LoRA 외 미세조정 방법을 조사해 비교했다 (8 절).

## 0. 목표와 범위

- 같은 캐릭터를 어느 컷, 어느 각도, 어느 장면에서 그려도 같은 얼굴·체형·머리·옷으로 나오게 한다.
- 캐릭터 고정 외형은 트리거 낱말 하나로 부르고, 포즈·표정·장면은 프롬프트로 자유롭게 바꾼다.
- 지금 편집으로 고치는 특징(니키의 소매 속 손)을 LoRA 가 배워 편집 단계를 줄이는지 확인한다.
- 범위는 받은 데이터셋으로 LoRA 어댑터를 학습·검증하는 데까지이며, 데이터셋 생성과 베이스 모델 전체 미세조정은 하지 않는다.

## 1. 정해야 할 것

| 항목 | 제안 | 이유 | 상태 |
|---|---|---|---|
| 파일럿 베이스 모델 | FLUX.2 [klein] Base 4B | 메인에 이미 있고(`flux-2-klein-base-4b`), Apache 2.0 으로 상업 이용 가능, 4B 라 학습이 빠름 | [TBD] 승인 |
| 실행(추론) 모델 | FLUX.2 [klein] 4B (증류판) | BFL 블로그: Base 로 학습한 LoRA 가 증류판에 그대로 불러지고 대개 더 좋음 | 문서 확인, 파일럿에서 실측 [TBD] |
| 2순위 베이스 | Qwen-Image-2512 | 2차 컨셉아트 그림체와 같은 모델, Apache 2.0, 다만 20B 라 학습이 느림 | 파일럿 뒤 결정 |
| 학습 도구 | 자체 노트북 (diffusers 0.42.1 + peft 0.21.2) | 10개 계열 어댑터가 이미 있고, Apache 2.0 라이브러리만 써서 도구 라이선스 문제가 없음 | 자체 구현으로 정함 (2026-11-02) |
| 구현 검증용 도구 | diffusers 공식 `train_dreambooth_lora_flux2_klein.py` | 노트북 어댑터가 이 스크립트를 옮긴 것이라 같은 조건으로 비교할 수 있음 | P2 에서 1회 |
| 영상 학습 도구 | 자체 영상 노트북 (`notebooks/lora_training_video.ipynb`, Wan 2.1·2.2) | 이미지 노트북과 같은 라이브러리(Apache 2.0)로 만들고, ComfyUI 가 읽는 Wan 원래 키로 저장함. ai-toolkit·Musubi Tuner 는 비교용 | 2026-11-05 만듦, 실제 가중치 학습 [TBD] |
| 트리거 낱말 | `nkgreen` | 실제 영어 낱말과 겹치지 않는 짧은 말 | [TBD] 캐릭터별 규칙 |
| 데이터셋 | 새로 만들어진 것을 받음 | 2026-11-02 사용자 결정 | 받는 시점 [TBD] |
| 캡션 | 데이터셋과 함께 오는지 확인 | 없으면 3-2 절 방식으로 만듦 | [TBD] |
| 니키 외형 정본 | 상상도(짙은 갈색 단발, 파란 눈, 주근깨) 또는 컨셉 시트판(검은 단발) | LoRA 는 데이터셋의 외형을 그대로 배우므로, 받는 데이터셋이 정본을 따라야 함 | [TBD] |
| 데이터셋 승인자 | 작가(아트 디렉터) | 학습 데이터의 정본을 사람이 정해야 함 | [TBD] 담당 |

- FLUX.2 [klein] 9B 는 비상업 라이선스라 쓰지 않는다.
- 4B 로 학습한 LoRA 는 9B 에서 쓸 수 없고, 반대도 같다.
- Illustrious XL v2.0 은 애니 특화지만 메인에 없고, 원본 모델 카드의 라이선스를 직접 확인해야 한다 [TBD].

### 1-1. 모델별 학습할 모델과 실제로 쓸 모델 (2026-11-02 조사)

대부분 증류하지 않은 Base 모델로 학습하고, 실제 생성은 빠른 증류판에서 한다.

| 계열 | 학습할 모델 | 실제로 쓸 모델 | 라이선스 | 알려진 시작값 (출처) | 판단 |
|---|---|---|---|---|---|
| FLUX.2 klein 4B | Base 4B | 증류판 4B (4 step) | Apache-2.0 | 캐릭터 1,500~3,000 단계, 학습률 8e-5~1e-4, 1024px (BFL 문서) | 1순위, 파일럿 |
| Qwen-Image 2512 | 2512 | 2512 | Apache-2.0 | 랭크 16, 학습률 1e-4, 2,000 단계 (ai-toolkit 예시) | 2순위 (20B 라 느림) |
| Krea 2 | Raw | Turbo (8 step) | 계열사 포함 연매출 US$1M 미만만 상업 이용 | 랭크·알파 32, 학습률 3e-4, 1,000 단계 (diffusers 예시) | 회사 매출 조건 확인 뒤 |
| Z-Image | base | Turbo | Apache-2.0 | dim 32, 학습률 1e-4 (musubi-tuner 예시) | 후순위 |
| Wan 2.2 A14B (영상) | high·low 두 전문가 | 같은 모델 | Apache-2.0 | 이미지만, 랭크 32, 학습률 1e-4, 2,000 단계 (ai-toolkit 예시). 영상 노트북 기본은 랭크 16, 2,000 단계 | 영상 1순위 |
| LTX-2.3 (영상) | 2.3 | 같은 모델 | 연매출 US$10M 미만 | 공식 `ltx-trainer` | 제외 (출력물로 다른 모델 상업 학습 금지, 2026-11-05) |

- klein 은 BFL 블로그에 Base 로 학습한 LoRA 가 증류판에서 "대개 더 좋다" 고 나와 있다.
- Z-Image 는 base 로 학습한 LoRA 가 Turbo 에서 듣는지 공식 근거가 없고, 커뮤니티 보고도 엇갈린다.
  - 예: 강도 2.5 이상이면 된다는 보고와 효과가 미미하다는 보고
- Krea 2 라이선스에는 콘텐츠 필터를 둘 의무가 있고, 파생 모델을 배포하면 이름 앞에 "Krea" 를 붙여야 한다.
- LTX 는 Lightricks 권장 버전이 2.5(2026-07)로 바뀌었고, 2.3 에서 학습한 LoRA 는 2.5 에서 쓸 수 없다.
- Qwen-Image 2.1 은 Qwen RESEARCH License(비상업)라 작품용에서 뺀다.
- LoRA 는 학습한 계열에서만 동작하므로, 모델을 하나 늘릴 때마다 9명분 학습·검증·재학습이 함께 늘어난다.
- 영상은 캐릭터 LoRA 로 만든 첫 화면을 Wan 2.2 I2V 에 넣는 방식을 기본으로 하고, 뒤돌거나 가려졌다 나오는 장면에서 얼굴이 흔들릴 때 Wan 2.2 LoRA 를 더한다 [판단].
- Wan 2.2 A14B LoRA 는 클립이 있으면 high-noise·low-noise 두 파일, 이미지만이면 low-noise 파일 하나로 나오고 (고노이즈 모델에는 LoRA 를 걸지 않음), 영상 노트북(2026-11-05)으로 학습한다.
  - 영상 노트북은 Wan2.2-T2V-A14B·I2V-A14B·TI2V-5B, Wan2.1-T2V 를 지원하고, 계열은 모델 폴더에서 자동으로 정한다.
  - HunyuanVideo·HunyuanVideo-1.5·MiniMax H3 는 라이선스가 대한민국을 제외해서, Kandinsky 5·6 은 HunyuanVideo VAE 를 포함해서 뺐다.

## 2. 전체 흐름

| 단계 | 하는 일 | 결과물 | 사용 도구 |
|---|---|---|---|
| 1. 데이터셋 받기 | 새로 만들어진 데이터셋을 받아 검사한다 | `dataset/<트리거>/*.png` | 노트북 1단계(EDA) + 사람 검토 |
| 2. 캡션 | 고정 외형은 트리거로, 바뀌는 것만 글로 적는다 (함께 오면 규칙만 확인) | `dataset/<트리거>/*.txt` | `qwen3-vl:8b`(메인 Ollama) + 사람 검토 |
| 3. 학습 | 베이스는 얼린 채 LoRA 어댑터만 학습한다 | `<트리거>_v1.safetensors` + 단계별 샘플 | 자체 노트북, 메인 GPU |
| 4. 검증·패키징 | 실제로 쓸 모델로 고정 프롬프트를 그려 일관성을 재고, 통과하면 묶어 보관한다 | 캐릭터 패키지 | ComfyUI 고정 워크플로우, 결과 자동 검사 |

## 3. 단계별 계획

### 3-1. 받은 데이터셋 검사

- 학습 데이터셋은 새로 만들어진 것을 받아 쓰고, 우리가 만들지 않는다 (2026-11-02 사용자 결정).
- 받은 한 벌을 모든 베이스 모델에 똑같이 쓴다.
  - 예: 모델마다 다른 데이터로 학습하면 모델 수만큼 서로 다른 니키가 생김
- 노트북 1단계가 자동으로 보는 것은 아래와 같다.
  - 예: 짧은 변 512px 미만, 같은 이미지 중복, 흐림, 캡션 인코딩, 캡션의 트리거
- 사람이 보는 것은 아래와 같다.
  - 예: 한 장에 한 인물만 있는지, 장수(20~30장)와 각도·화면 크기·배경이 고르게 섞였는지
  - 예: 니키는 손이 소매 밖으로 나온 이미지가 없는지 (핵심 특징)
  - 예: 니키 외형 정본 [TBD] 을 따르는지
  - 예: 팔·손이 하나 더 있는 것, 다른 인물과 붙은 것, 글자가 들어간 것은 없는지
- 성장 시트의 '맨손' 모습은 트리거와 섞이지 않게 이번 데이터셋에서 뺀다 [TBD] 별도 태그로 넣을지.
- 데이터셋을 만든 모델의 라이선스를 확인한다.
  - 예: FLUX.2 klein 4B·Qwen-Image 2512·Z-Image·Wan 2.2 출력은 Apache-2.0 이라 학습 데이터로 문제없음
  - 예: LTX-2.3 출력은 다른 모델 학습에 상업적으로 쓸 수 없음 (라이선스 부록 A 18번)
  - 예: klein 9B 출력은 FLUX 와 경쟁하는 모델 학습에 쓸 수 없고, Krea 2 출력의 상업 이용에는 매출 조건이 붙음
- 문제가 있는 이미지는 빼거나 다시 요청하고, 그 기록을 `dataset/<트리거>/selection.md` 에 남긴다.

### 3-2. 캡션과 트리거

| 구분 | 넣는 곳 | 예 |
|---|---|---|
| 고정 외형 | 트리거 `nkgreen` 하나에 담음 (캡션에 쓰지 않음) | 정본 머리 모양·색, 눈, 와인빛 케이프, 크림색 긴 소매, 작은 체구 |
| 바뀌는 것 | 캡션에 글로 적음 | 정면/측면/후면, 표정, 포즈, 배경, 구도, 조명 |

- 캡션이 데이터셋과 함께 오면 아래 규칙에 맞는지 확인하고, 없으면 만든다 [TBD].
- 캡션 모양은 `nkgreen, <각도>, <표정>, <포즈>, <배경>` 처럼 짧은 영문 한 줄로 맞춘다.
  - 예: `nkgreen, side view, shy downcast eyes, standing with arms at her sides, plain light gray background`
- 고정 외형을 캡션에 적으면 그 특징이 트리거가 아닌 글에 붙어 버리므로, 자동 캡션 지시에서 막는다.
- 의상이 여러 벌인 캐릭터는 의상을 캡션에 적고, 트리거에는 얼굴·머리·체형만 담는다 [제안].
  - 예: 니키는 1장 집 안 옷(크림 셔츠·모카 반바지)과 2장부터 외출복(와인빛 케이프)이 있어, 케이프를 트리거에 담으면 집 안 장면을 그리기 어려움
- 자동 캡션은 메인 Ollama 의 `qwen3-vl:8b` 로 하며, 허브의 다른 LLM 호출처럼 DB 대기열을 거친다.
- 사람이 전부 한 번 읽고 고친 뒤 학습에 넣는다.

### 3-3. 학습

- 베이스 모델 가중치는 얼리고(freeze), LoRA 어댑터만 학습한다.
- 노트북은 공통값에 계열별 시작값을 덮어쓴 고정값으로 학습하고, 파일럿 결과를 보고 고친다 [TBD] 측정 후 확정.

| 설정 | 공통 | FLUX.2 klein | Qwen-Image | Krea 2 | Z-Image |
|---|---|---|---|---|---|
| LoRA 랭크 = 알파 | 16 | 16 | 16 | 32 | 32 |
| 학습률 (AdamW) | 1e-4 | 1e-4 | 1e-4 | 3e-4 | 1e-4 |
| 학습 단계 | 1,500 | 2,000 | 2,000 | 1,000 | 1,500 |
| 근거 | 계획서 파일럿 시작값 (10-29) | BFL 캐릭터 범위 1,500~3,000 | ai-toolkit 예시 | diffusers README_krea2 예시 | musubi-tuner 예시 |

- 해상도(1024px 크기별 묶음), 배치 1, bf16, 워밍업 100 단계는 모든 계열이 같다.
- 250 단계마다 LoRA 파일과 고정 프롬프트 샘플을 남겨, 더 이른 단계도 고를 수 있게 한다.
- 랭크와 알파는 늘 같게 둔다.
  - 예: diffusers 로 저장한 파일에는 alpha 텐서가 없고, ComfyUI 는 이때 강도를 1(알파 = 랭크)로 적용하므로 둘이 다르면 세기가 달라짐 (노트북은 다르면 멈춤)
- 노트북의 검증 샘플은 학습한 모델(klein 은 Base 50 step, cfg 4)로 그린 진행 확인용이고, 최종 판정은 3-4 절대로 ComfyUI 에서 한다.
- 검증 프롬프트는 캐릭터별 JSON 파일(`LORA_VALIDATION_FILE`)로 바꿀 수 있고, 기본 부정어에는 손·손가락을 넣지 않는다.
  - 예: 손을 숨기는 특징을 LoRA 가 배웠는지 봐야 하고, 실제로 쓸 증류판(cfg 1)은 부정어를 거의 듣지 않음
- 학습할 모델 경로가 증류판(klein 증류판, Turbo)이거나 비상업·매출 조건부 모델이면 노트북이 주의 문구를 보여 준다.
- 산출물은 단계별 LoRA 파일, 샘플 이미지, 학습 기록(손실값, 시간, 최대 메모리)이다.
- 파일 크기·학습 시간·최대 메모리는 파일럿에서 잰 값으로 이 문서를 고친다 [TBD].
  - 예: BFL 안내 기준 4B LoRA 는 24GB 안에 들어가고 RTX 4090 에서 약 1시간, Qwen-Image 는 musubi-tuner 기준 메모리 절약 옵션 없이 42GB

### 3-4. 검증과 패키징

- 고정 프롬프트 세트를 정하고 시드도 고정해 매번 같은 조건으로 그린다.
  - 예: 정면 전신, 측면 전신, 후면, 앉은 자세, 다른 인물과 둘이 있는 장면, 처음 보는 배경
- 그림은 1-1 절 표의 '실제로 쓸 모델' 로 ComfyUI 고정 워크플로우에서 그린다.
  - 예: klein 은 증류판 4B 4 step cfg 1
- 처음 불러올 때 ComfyUI 콘솔의 `lora key not loaded` 경고가 0건인지 확인한다.
  - ComfyUI 는 모르는 키를 경고만 남기고 버려서, 경고가 있으면 LoRA 일부가 빠진 채 그려진다.
  - 예: Krea 2 어댑터는 LoRA 대상 층이 15종(img_in, txt_in, time_embed 등)이라 특히 확인
- 각 그림은 허브의 결과 자동 검사(시각 모델)로 해부학 오류와 '꼭 보여야 할 특징' 을 확인한다.
  - 예: `hands hidden inside long cream sleeves`, `wine-red cape coat`, 정본 머리 모양·색
- 캐릭터가 같은지는 CCIP(애니 캐릭터 동일성 판별 모델)로 재고, 사람이 라벨링한 50장으로 기준값을 맞춘다 [제안].
  - 예: DINOv2·CLIP 유사도는 사람 판단과 50.7%·61.0% 만 일치해 단독 기준으로 쓰지 않음
- 사람이 어느 LoRA 단계인지 모르게 보고 고르는 블라인드 검수를 한다.
- 캐릭터 LoRA 두 개를 함께 건 두 인물 장면을 그려 특징이 섞이는지 본다.
- 시중 그림체 LoRA 와 함께 썼을 때 캐릭터가 무너지는지도 본다 (화풍 충돌 확인).
- 통과하면 아래를 한 폴더로 묶어 '캐릭터 패키지' 로 보관한다.
  - 예: LoRA 파일, 캐릭터 시트, 캐릭터 설명(바이블), 트리거와 권장 강도, 고정 워크플로우, 검증 결과, 학습 설정·데이터셋 해시

## 4. 메인 서버 반영 (허브 설계)

학습도 GPU 를 수십 분~수 시간 쓰므로, 기존 설계 원칙(메인 GPU 는 한 번에 한 작업, 모든 GPU 사용은 DB 를 거침)을 그대로 따른다.

| 항목 | 지금 | 바꿀 것 |
|---|---|---|
| 작업 종류 | `jobs.kind` 가 `comfy`, `llm`, `embed` 만 허용 | `train` 추가, 같은 `jobs_one_gpu_holder` 규칙 적용 |
| GPU 를 붙이는 곳 | `dockerctl.run_worker()` 만 | 학습 전용 `run_trainer()` 추가, 같은 방식으로 GPU 를 붙임 |
| 실행 공간 | ComfyUI 런타임 이미지 | 자체 노트북과 고정 버전(torch 2.15.0, diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, accelerate 1.15.0)이 든 `images/lora-trainer/` 이미지 추가 |
| 학습용 모델 | ComfyUI 용 단일 `.safetensors` 만 있음 | 노트북이 읽는 diffusers 형식 폴더를 `/srv/models/diffusers/<모델>/` 에 따로 받음 [TBD] 디스크 여유 확인 |
| 입력 고정 | 실행 접수 때 btrfs 스냅샷 | 데이터셋 폴더·캡션·설정 파일을 접수 때 읽기 전용 스냅샷으로 고정 |
| 결과 | `results/<사용자>/...` | `results/<사용자>/training/<실행 이름>/` 에 단계별 LoRA·샘플·기록 |
| 모델 등록 | `model-inbox` → 포털 '모델 추가' | 고른 LoRA 를 inbox 로 보내 같은 절차로 `/srv/models/loras` 에 등록 |
| 기록 | `jobs`, `llm_calls` | `train_runs` 표 추가 (베이스, 데이터셋 해시, 설정, 단계, 손실, 시간, 최대 메모리) |
| 메모리 정리 | 작업 뒤 NVML 확인 | 학습 뒤에도 같은 확인 |

- 지금 원칙은 "GPU 는 `run_worker()` 에서만 요청한다" 이므로, `run_trainer()` 를 더하면 메인 GPU 서버 저장소 `CLAUDE.md` 의 이 문장도 "`run_worker()`·`run_trainer()` 두 곳에서만" 으로 함께 고친다.
  - 예: 두 함수 모두 스케줄러가 DB 에서 GPU 차례를 받은 뒤에만 부르므로 '한 번에 한 작업' 은 그대로 지켜짐
- 학습이 끝날 때까지 다른 사람의 이미지 작업이 막히지 않게, 아래 둘 중 하나를 고른다 [TBD].
  - 예: 중간 저장마다 대기열을 보고, 기다리는 작업이 있으면 멈췄다가 비면 이어서 학습 (양보 모드)
  - 예: 밤 시간대나 대기열이 빈 때만 학습 작업을 꺼냄 (시간대 모드)
- 처음 구현은 시간대 모드가 단순하고, 양보 모드는 중간 저장에서 이어 학습이 되는지 확인한 뒤 붙인다.
- 패널에는 '학습' 카드를 따로 두어, 진행 단계와 최근 샘플 이미지를 보여 준다 [TBD] 화면 설계.

## 5. 진행 순서

| 단계 | 내용 | 끝났다고 보는 기준 |
|---|---|---|
| P0 결정 | 1절 표의 [TBD] 를 정함 | 베이스 모델, 트리거, 니키 정본, 승인자 확정 |
| P1 데이터셋 받기 | 새 데이터셋을 받아 3-1 절대로 검사하고, 캡션을 확인하거나 만듦 | 작가가 데이터셋 목록을 확인 |
| P2 파일럿 학습 | 관리자가 대기열을 멈춘 상태에서 니키 × klein Base 4B 를 노트북으로 1회 학습하고, 같은 데이터·설정으로 diffusers 공식 스크립트도 1회 학습 | 두 결과의 손실 흐름과 샘플이 비슷하고, 시간·메모리·파일 크기를 이 문서에 기록 |
| P3 검증 | 3-4 절 방식으로 단계별 LoRA 를 ComfyUI 증류판에서 비교 | `lora key not loaded` 0건, 일관성이 1·2차 컨셉아트(편집 포함)보다 낫거나 같음 |
| P4 허브 반영 | 4절의 `train` 작업, 학습 실행 공간, 기록, 패널 카드 | 일반 사용자가 대기열로 학습을 넣고 결과를 받음 |
| P5 확대 | 나머지 8명 → Qwen-Image-2512 → Wan 2.2 (영상) | 9명 캐릭터 패키지 보관 |

- P2 에서 공식 스크립트와 비교하는 것은 노트북을 작은 무작위 모델로만 시험했기 때문이다.
  - 예: 잠재값 정규화 같은 실수는 실제 가중치에서만 드러남
  - 예: 공식 README 예시는 모델 이름이 증류판 `FLUX.2-klein-4B` 라서 Base(`FLUX.2-klein-base-4B`)로 바꿔 돌려야 함
- P2 는 설계 원칙의 예외이므로, 반드시 대기열을 멈추고(관리자 '대기열 멈춤') 하며 기록을 남긴다.
- 여러 모델·캐릭터를 학습할 때는 `notebooks/run_many.py` 에 실행 목록(JSON)을 주어 차례로 돌린다 (허브 `train` 작업이 생기기 전까지).
  - 예: 끝난 실행은 건너뛰고, 실행마다 시간·최대 메모리·파일 크기를 `run_many_log.csv` 에 남겨 7 절의 측정에 씀
- 메인에서 컨테이너를 띄우는 `sudo` 명령은 사용자가 직접 실행하고, Claude 는 명령만 안내한다.

## 6. 위험과 대응

| 위험 | 대응 |
|---|---|
| 받은 데이터셋(AI 생성 이미지)의 작은 오류까지 배움 | 사람이 한 장씩 확인하고, 문제 이미지는 빼거나 다시 요청 |
| 과적합 (같은 포즈·배경만 나옴) | 각도·포즈·배경이 고르게 섞였는지 받을 때 확인하고, 단계별 샘플로 이른 단계를 고를 수 있게 함 |
| 트리거에 배경·그림체까지 붙음 | 배경과 그림체는 캡션에 적고, 배경이 여러 가지인지 확인 |
| ComfyUI 가 LoRA 키 일부를 못 읽음 | 첫 로드 때 `lora key not loaded` 경고 0건 확인 |
| ComfyUI 에서 LoRA 세기가 학습 때와 다름 | 랭크와 알파를 같게 고정 (노트북이 검사) |
| 캐릭터 LoRA 를 여러 개 걸면 특징이 섞임 | 파일럿 때 두 인물 장면을 시험하고, 여러 인물 장면은 생성 뒤 편집으로 합성하는 방식을 주력으로 (8 절) |
| 다른 그림체 LoRA 와 충돌 | 검증에 그림체 LoRA 결합 시험을 넣고 권장 강도를 패키지에 적음 |
| 학습이 GPU 를 오래 잡음 | 시간대 모드부터 시작, 양보 모드는 이어 학습 확인 뒤 |
| 라이선스 | 베이스 모델은 Apache 2.0 만 우선 사용하고, 데이터셋을 만든 모델과 결합 LoRA 의 라이선스도 패키지에 적음 |

## 7. 파일럿에서 잴 것 [TBD]

- 학습 1회 시간 (단계 수별)
- 최대 GPU 메모리
- LoRA 파일 크기 (랭크별)
- 노트북과 diffusers 공식 스크립트 결과의 차이 (손실 흐름, 같은 시드 샘플)
- ComfyUI `lora key not loaded` 경고 수
- 고정 프롬프트 세트에서 자동 검사 통과율
- 손을 숨긴 모습이 편집 없이 나오는 비율

## 8. LoRA 외 미세조정 방법 (2026-11-02 조사)

LoRA 보다 확실히 나은 미세조정 방법의 근거는 찾지 못했고, 아래 방법은 LoRA 를 대신하기보다 옆에 붙여 쓰는 보완책으로 본다.

| 방법 | 용도 | 비용 | ComfyUI | 판단 |
|---|---|---|---|---|
| LoRA 랭크 올리기 | 정체성 강화 | 낮음 | 지원 | 16 으로 시작, 특징이 약하면 32 |
| LoKr (LyCORIS) | 작은 파일 | 학습 약 2.8배 느림 | 읽는 경로는 있음 (실행 미검증) | 비교용으로만 |
| DoRA·LoHa·OFT·BOFT | LoRA 변형 | 2~3배 느리거나 메모리 큼 | PEFT 형식 DoRA·OFT 는 못 읽음 | 비권장 |
| 전체 미세조정 뒤 LoRA 추출 | 9명을 한 모델에 | klein 4B 약 40GB, 20B 는 어려움 | 추출한 LoRA 로 | 지금은 안 함 |
| Textual Inversion | 트리거 낱말만 학습 | 낮음 | — | 새 모델용 학습 도구를 찾지 못함 |
| 편집 모델 다중 참조 | 단발성 컷, 변주 | 학습 없음 | 지원 | 적극 사용 |
| 편집 LoRA (전·후 쌍 학습) | 반복 편집 자동화 | 중간 | 지원 | 유력 후보 |
| 생성 뒤 편집으로 합성 | 여러 인물 장면 | 중간 | 지원 | 주력 |
| 영역 LoRA (Create Hook LoRA + 마스크) | 여러 인물 장면 | 인물 수만큼 느림 | 지원 | 보조 |
| 영상 참조 (Wan VACE, Phantom, LTX IC-LoRA Ingredients) | 영상 일관성 | 학습 없음 | 지원 | 소규모 시험 |
| LTX ID-LoRA, Stand-In | 실사 얼굴·목소리 | — | — | 우리 용도에 맞지 않음 |

- LoRA 변형을 직접 비교한 근거는 PEFT 공식 비교 하나뿐이다 (klein Base 4B, 고양이 한 종, DINOv2 점수, 시드 1개).
  - 예: LoRA r64 0.743 > OFT 0.708 > LoRA r32·LoKr 0.697 > DoRA 0.668 > LoHa 0.662 > BOFT 0.645
  - 같은 LoRA 학습 안에서도 저장 단계마다 0.67~0.72 로 흔들려, 이 차이는 잡음 수준이고 애니 데이터도 아니다.
- 같은 비교에서 전체 미세조정(300 단계, 설정 조정 없음)은 0.577 로 LoRA 보다 낮았고, klein 4B 전체 미세조정은 512px 에서 최대 39.9GB 를 썼다.
- Textual Inversion 은 diffusers 예제가 FLUX.1 의 CLIP·T5 용뿐이고, Qwen3·Qwen2.5-VL·Mistral 텍스트 인코더용 학습 도구는 찾지 못했다.
- 편집 LoRA 는 지금 매번 손으로 하는 소매 편집을 한 번에 하게 만드는 용도로 맞다 [미검증].
  - 예: 손을 숨긴 그림을 정답, 손이 보이게 편집한 그림을 입력으로 쌍을 만들어 Qwen-Image-Edit-2511 에 학습
  - 예: 지원 도구는 musubi-tuner(`--model_version edit-2511`), ai-toolkit, diffusers klein img2img 스크립트
- 다중 참조는 Qwen-Image-Edit 가 1~3장(2509 모델 카드), klein 이 최대 4장까지 받고, 컷마다 세부가 흔들리기 쉬워 반복 생산에는 LoRA 가 더 안정적이다 [판단].
- 캐릭터 LoRA 여러 개를 한 그림에 걸면 특징이 섞인다는 보고가 있고(Mix-of-Show), 영역 LoRA 는 hook 묶음마다 따로 계산해 인물 수만큼 느려진다.
- LTX 의 'ID-LoRA' 는 참조 이미지와 약 5초 음성으로 실사 인물의 얼굴·목소리를 맞추는 연구 모델이고, 'IC-LoRA Union Control' 은 구조 제어다.
- 캐릭터용 LTX 기능은 레퍼런스 시트를 넣는 'IC-LoRA Ingredients' 인데, 예시가 3D 애니풍이라 2D 셀화에서는 검증되지 않았다.

## 참고

- [Black Forest Labs: FLUX.2 klein training](https://docs.bfl.ai/flux_2/flux2_klein_training)
- [Fine-tune FLUX.2 [klein] with a LoRA under 60 minutes](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
- diffusers v0.42.1 학습 안내: [FLUX.2](https://github.com/huggingface/diffusers/blob/v0.42.1/examples/dreambooth/README_flux2.md), [Krea 2](https://github.com/huggingface/diffusers/blob/v0.42.1/examples/dreambooth/README_krea2.md)
- [ai-toolkit (ostris)](https://github.com/ostris/ai-toolkit), [Wan 2.2 14B LoRA 예시 설정](https://github.com/ostris/ai-toolkit/blob/main/config/examples/train_lora_wan22_14b_24gb.yaml)
- [Musubi Tuner (kohya-ss)](https://github.com/kohya-ss/musubi-tuner)
- [ComfyUI LoRA 키 매핑 (v0.42.0 `comfy/lora.py`)](https://github.com/Comfy-Org/ComfyUI/blob/v0.42.0/comfy/lora.py)
- 라이선스: [Krea 2](https://www.krea.ai/krea-2-licensing), [Qwen-Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE), [LTX-2](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2)
- [Z-Image base LoRA 를 Turbo 에 쓰는 논의](https://huggingface.co/Tongyi-MAI/Z-Image/discussions/18)
- [PEFT 이미지 생성 방법 비교](https://github.com/huggingface/peft/tree/main/method_comparison/image-gen)
- [LTX-2.3 IC-LoRA Ingredients](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients), [ID-LoRA 논문](https://arxiv.org/abs/2603.10256)
- [ComfyUI 영역·일정 LoRA (Hook)](https://blog.comfy.org/p/masking-and-scheduling-lora-and-model-weights)
- [CCIP (deepghs)](https://huggingface.co/deepghs/ccip), [DINO·CLIP 유사도와 사람 판단 비교](https://arxiv.org/abs/2406.16855)
- [Illustrious XL v2.0 (OnomaAI)](https://huggingface.co/OnomaAIResearch/Illustrious-XL-v2.0)
- 메인 GPU 서버 저장소: [docs/prompt-helper.md](https://github.com/jisoo5849/main-GPU-server-Documents/blob/main/docs/prompt-helper.md) (확장 기능과 결과 자동 검사), [hub/app/schema.sql](https://github.com/jisoo5849/main-GPU-server-Documents/blob/main/hub/app/schema.sql) (작업 표), [hub/app/dockerctl.py](https://github.com/jisoo5849/main-GPU-server-Documents/blob/main/hub/app/dockerctl.py) (GPU 실행 공간)
- 컨셉아트 결과: [all-about-Niki](https://github.com/jisoo5849/all-about-Niki) (1차 루트, 2차 `v2_qwen_anime/`)
