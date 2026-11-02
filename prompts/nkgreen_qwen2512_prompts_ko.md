# 니키(`nkgreen`) LoRA 학습 이미지 · Qwen-Image-2512 프롬프트 (한글)

기준 이미지는 Qwen-Image-2512 로 만든 니키 전신 3장(S06-Q1~Q3)이고, 그 모습을 기본값으로 고정한다.
실제 생성에는 영문판(`nkgreen_qwen2512_prompts_en.md`)을 넣는다. 이 문서는 같은 내용을 한글로 옮긴 것이다.

## 기준 이미지와 캐릭터 바이블의 차이

- 이 프롬프트는 기준 이미지(컨셉 시트 버전)를 따른다: 새까만 단발, 하늘빛이 살짝 도는 어두운 눈, 주근깨 없음, 현대 애니 일러스트 그림체.
- 2026-10-29 의 '상상도 우선' 결정(짙은 갈색 단발, 맑은 파란 눈, 주근깨, 손그림 극장 애니메이션풍)과는 다르다 [TBD] 둘 중 무엇을 니키의 정본으로 할지.

## 프롬프트 구조

`<화면 크기>, <각도>, <표정·포즈>. <고정 캐릭터 문장>. <배경> <그림체>`

- 고정 캐릭터 문장은 모든 컷에 글자 하나 바꾸지 않고 넣고, 화면 크기·각도·표정·포즈·배경만 바꾼다.
- 상반신·얼굴 컷에는 반바지·타이츠·부츠를 쓰지 않는다.
  - 예: 화면 밖에 있을 옷을 적으면 전신을 그렸다.
- 긍정 프롬프트에는 손을 쓰지 않고 긴 소매만 적으며, 손은 부정 프롬프트로 막는다.
  - 예: 손 단어가 있으면 손을 그렸다. 그래도 손이 나오면 Qwen-Image-Edit-2511 로 소매를 고친다.
- 뒷모습에 '돌아봄' 을 쓰지 않는다.
  - 예: '뒷모습 + 어깨 너머로 돌아봄' 이 비스듬한 앞모습이 됐다.

## 고정 캐릭터 문장

| 부분 | 한글 | 영문 |
|---|---|---|
| 머리·얼굴 (모든 컷) | 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입 | Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth |
| 상의 (모든 컷) | 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매 | a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs |
| 하의 (전신 컷만) | 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠 | high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots |
| 그림체 (모든 컷) | 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인. | Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design. |
| 무지 배경 | 글자 없는 밝은 회색 무지 배경. | Plain light gray background without any lettering. |

- 색은 컨셉 시트 색 코드를 따른다.
  - 예: 케이프 #7B3342, 크림 상의 #F3E8D6, 모카 반바지 #4A372E, 회색 타이츠 #4A4A52, 갈색 부츠 #6B4A35

## 부정 프롬프트

Qwen-Image-2512 는 cfg 4 라 부정 프롬프트가 듣는다.

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

| 더하는 경우 | 추가 부정어 |
|---|---|
| 상반신 | `full body, legs, shorts, boots` |
| 얼굴 | `full body, legs, shorts, boots, waist` |
| 뒷모습 | `face, looking back, looking at viewer, front view` |

- 부정어 뜻: 사진풍·3D·글자·워터마크·흐림·저화질·변형·팔다리 추가·손·손가락·여러 사람·머리 잘림·갈색/금발/긴 머리·포니테일·주근깨·모자·반팔·맨다리·치마·운동화·하이힐·성인 얼굴·화장을 막는다.

## ComfyUI 설정

| 설정 | 값 |
|---|---|
| 모델 / Model | `qwen_image_2512_fp8_e4m3fn.safetensors` (UNETLoader, weight_dtype default) |
| 텍스트 인코더 / Text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (CLIPLoader type `qwen_image`) |
| VAE | `qwen_image_vae.safetensors` |
| ModelSamplingAuraFlow shift | 3.1 |
| KSampler | steps 20 (공식 템플릿 50, 시간 때문에 20 / official template 50), cfg 4, euler, simple, denoise 1.0 |
| 크기 / Size | 전신 1024×1536, 상반신 1024×1280, 얼굴 1024×1024 |
| 시드 / Seed | 31000+번호 (예: nkgreen_01 = 31001) |
| LoRA | 쓰지 않음 (Lightning 4-step LoRA 는 반사실풍으로 바뀌어 학습 데이터에 쓰지 않음) |

## 컷 구성 (24장)

| 번호 | 화면 | 각도 | 배경 | 크기 | 시드 |
|---|---|---|---|---|---|
| nkgreen_01 | full | front | plain | 1024×1536 | 31001 |
| nkgreen_02 | full | 3ql | plain | 1024×1536 | 31002 |
| nkgreen_03 | full | 3qr | plain | 1024×1536 | 31003 |
| nkgreen_04 | full | side | plain | 1024×1536 | 31004 |
| nkgreen_05 | full | back | plain | 1024×1536 | 31005 |
| nkgreen_06 | full | front | plain | 1024×1536 | 31006 |
| nkgreen_07 | full | 3qr | forest | 1024×1536 | 31007 |
| nkgreen_08 | full | front | square | 1024×1536 | 31008 |
| nkgreen_09 | full | 3ql | meadow | 1024×1536 | 31009 |
| nkgreen_10 | full | side | bridge | 1024×1536 | 31010 |
| nkgreen_11 | waist | front | plain | 1024×1280 | 31011 |
| nkgreen_12 | waist | 3ql | plain | 1024×1280 | 31012 |
| nkgreen_13 | waist | 3qr | library | 1024×1280 | 31013 |
| nkgreen_14 | waist | side | plain | 1024×1280 | 31014 |
| nkgreen_15 | waist | front | rain | 1024×1280 | 31015 |
| nkgreen_16 | waist | 3qr | forest | 1024×1280 | 31016 |
| nkgreen_17 | waist | back | plain | 1024×1280 | 31017 |
| nkgreen_18 | face | front | plain | 1024×1024 | 31018 |
| nkgreen_19 | face | 3ql | plain | 1024×1024 | 31019 |
| nkgreen_20 | face | 3qr | plain | 1024×1024 | 31020 |
| nkgreen_21 | face | side | plain | 1024×1024 | 31021 |
| nkgreen_22 | face | front | night | 1024×1024 | 31022 |
| nkgreen_23 | face | 3ql | meadow | 1024×1024 | 31023 |
| nkgreen_24 | face | front | room | 1024×1024 | 31024 |

- 화면 크기: 전신 1024×1536, 상반신 1024×1280, 얼굴 1024×1024.
- 전신 10장(무지 6, 장면 4), 상반신 7장(무지 4, 장면 3), 얼굴 7장(무지 4, 장면 3)으로 각도·표정·배경을 고르게 섞었다.

## 학습 캡션

고정 외형은 트리거 `nkgreen` 하나에 담고, 캡션에는 컷마다 바뀌는 것만 영문으로 적는다 (계획서 3-2절).

`nkgreen, <화면 크기>, <각도>, <표정·포즈>, <배경>, clean anime illustration`

- 고정 외형(검은 단발, 와인 케이프, 긴 크림 소매 등)을 캡션에 적으면 그 특징이 트리거가 아닌 글에 붙으므로 적지 않는다.
- 그림체와 배경은 캡션에 적어, 트리거에 그림체·배경까지 붙지 않게 한다.

## 생성 뒤 검수

- 손이 소매 밖으로 나온 컷, 머리색·케이프색이 바뀐 컷, 상반신 컷이 전신으로 나온 컷, 뒷모습이 앞모습으로 나온 컷은 다시 뽑는다.
- 같은 컷을 시드만 바꿔 2장씩 뽑고 나은 쪽을 고르면 검수 시간이 줄어든다.

### nkgreen_01 · full · front · 1024×1536 · 시드 31001

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 정면, 팔을 내리고 가만히 선 자세, 잔잔하고 수줍은 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, front view, standing, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_02 · full · 3ql · 1024×1536 · 시드 31002

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 왼쪽으로 살짝 돈 3/4 각도, 가만히 선 자세, 잔잔하고 수줍은 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, three-quarter view from the left, standing, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_03 · full · 3qr · 1024×1536 · 시드 31003

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 오른쪽으로 살짝 돈 3/4 각도, 가만히 선 자세, 옅은 미소. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, three-quarter view from the right, standing, faint smile, plain light gray background, clean anime illustration
```

### nkgreen_04 · full · side · 1024×1536 · 시드 31004

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 오른쪽을 향한 옆모습, 눈은 하나만 보임, 가만히 서서 앞을 봄. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, side view, standing, looking ahead, plain light gray background, clean anime illustration
```

### nkgreen_05 · full · back · 1024×1536 · 시드 31005

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 뒷모습, 화면 안쪽을 향해 서서 등과 뒤통수·머리카락만 보임, 가만히 선 자세. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, face, looking back, looking at viewer, front view
```

캡션

```
nkgreen, full body, back view, standing, plain light gray background, clean anime illustration
```

### nkgreen_06 · full · front · 1024×1536 · 시드 31006

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 정면, 작은 걸음으로 앞으로 걸어옴. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, front view, walking, plain light gray background, clean anime illustration
```

### nkgreen_07 · full · 3qr · 1024×1536 · 시드 31007

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 오른쪽으로 살짝 돈 3/4 각도, 길을 따라 천천히 걸음. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 배경: 아침 햇살이 드는 조용한 숲길. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, three-quarter view from the right, walking, forest path, clean anime illustration
```

### nkgreen_08 · full · front · 1024×1536 · 시드 31008

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 정면, 나무 벤치에 앉아 긴 소매를 무릎 위에 올려 둠. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 배경: 작은 돌길 마을 광장. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, front view, sitting on a bench, town square, clean anime illustration
```

### nkgreen_09 · full · 3ql · 1024×1536 · 시드 31009

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 왼쪽으로 살짝 돈 3/4 각도, 쪼그려 앉아 작은 흰 꽃을 들여다봄. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 배경: 맑은 하늘 아래 초록 여름 들판. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, three-quarter view from the left, crouching, looking at a flower, summer meadow, clean anime illustration
```

### nkgreen_10 · full · side · 1024×1536 · 시드 31010

긍정 (한글)

```
머리부터 발끝까지 한 인물 전신 일러스트, 오른쪽을 향한 옆모습, 눈은 하나만 보임, 다리 위에 서서 강을 내려다봄. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매, 허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠. 배경: 잔잔한 강 위의 오래된 돌다리. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

캡션

```
nkgreen, full body, side view, standing, looking down, stone bridge over a river, clean anime illustration
```

### nkgreen_11 · waist · front · 1024×1280 · 시드 31011

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 정면, 수줍은 작은 미소. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, front view, shy small smile, plain light gray background, clean anime illustration
```

### nkgreen_12 · waist · 3ql · 1024×1280 · 시드 31012

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 왼쪽으로 살짝 돈 3/4 각도, 입을 살짝 벌린 놀란 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, three-quarter view from the left, surprised, plain light gray background, clean anime illustration
```

### nkgreen_13 · waist · 3qr · 1024×1280 · 시드 31013

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 오른쪽으로 살짝 돈 3/4 각도, 눈썹을 모은 걱정스러운 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 키 큰 나무 책장이 있는 아늑한 도서관. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, three-quarter view from the right, worried, library, clean anime illustration
```

### nkgreen_14 · waist · side · 1024×1280 · 시드 31014

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 오른쪽을 향한 옆모습, 눈은 하나만 보임, 위쪽의 무언가를 올려다봄. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, side view, looking up, plain light gray background, clean anime illustration
```

### nkgreen_15 · waist · front · 1024×1280 · 시드 31015

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 정면, 다정한 미소. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 가게 차양 아래에서 본 비 오는 거리. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, front view, gentle smile, rainy street, clean anime illustration
```

### nkgreen_16 · waist · 3qr · 1024×1280 · 시드 31016

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 오른쪽으로 살짝 돈 3/4 각도, 결심한 듯한 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 아침 햇살이 드는 조용한 숲길. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

캡션

```
nkgreen, upper body, three-quarter view from the right, determined, forest path, clean anime illustration
```

### nkgreen_17 · waist · back · 1024×1280 · 시드 31017

긍정 (한글)

```
허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖, 뒷모습, 화면 안쪽을 향해 서서 등과 뒤통수·머리카락만 보임, 가만히 선 자세. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, face, looking back, looking at viewer, front view
```

캡션

```
nkgreen, upper body, back view, standing, plain light gray background, clean anime illustration
```

### nkgreen_18 · face · front · 1024×1024 · 시드 31018

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 정면, 보는 사람을 바라보는 잔잔하고 수줍은 눈. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, front view, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_19 · face · 3ql · 1024×1024 · 시드 31019

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 왼쪽으로 살짝 돈 3/4 각도, 옅은 미소. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, three-quarter view from the left, faint smile, plain light gray background, clean anime illustration
```

### nkgreen_20 · face · 3qr · 1024×1024 · 시드 31020

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 오른쪽으로 살짝 돈 3/4 각도, 눈물이 고인 슬픈 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, three-quarter view from the right, teary sad look, plain light gray background, clean anime illustration
```

### nkgreen_21 · face · side · 1024×1024 · 시드 31021

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 오른쪽을 향한 옆모습, 눈은 하나만 보임, 먼 곳을 바라보는 잔잔한 표정. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 글자 없는 밝은 회색 무지 배경. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, side view, calm look into the distance, plain light gray background, clean anime illustration
```

### nkgreen_22 · face · front · 1024×1024 · 시드 31022

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 정면, 크게 뜬 놀란 눈. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 언덕 위 별이 가득한 밤하늘. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, front view, wide surprised eyes, starry night sky, clean anime illustration
```

### nkgreen_23 · face · 3ql · 1024×1024 · 시드 31023

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 왼쪽으로 살짝 돈 3/4 각도, 눈을 감고 웃음. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 맑은 하늘 아래 초록 여름 들판. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, three-quarter view from the left, laughing, eyes closed, summer meadow, clean anime illustration
```

### nkgreen_24 · face · front · 1024×1024 · 시드 31024

긍정 (한글)

```
한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름, 정면, 수줍게 아래를 보는 눈. 니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, 하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입, 엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, 케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매. 배경: 따뜻한 스탠드 불빛의 작은 방. 깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인.
```

부정

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

캡션

```
nkgreen, close-up, front view, eyes looking down shyly, warm bedroom, clean anime illustration
```
