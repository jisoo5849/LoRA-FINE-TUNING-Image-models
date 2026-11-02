# Niki (`nkgreen`) LoRA training images — Qwen-Image-2512 prompts

Reference: three full-body Niki images made with Qwen-Image-2512 (S06-Q1~Q3). Their look is fixed as the default.

## How the prompt is built

`<framing>, <view>, <expression / pose>. <fixed character block>. <background> <style>`

- The fixed character block is repeated word for word in every shot; only framing, view, expression, pose and background change.
- Waist-up and close-up shots leave out the shorts, tights and boots (writing off-frame clothes made the model draw full bodies).
- Hands are never mentioned in the positive prompt; the long sleeves are described instead, and hands are blocked in the negative prompt.
- Rear views never say "looking back" (it turns into an angled front view).

## Fixed character block

| Part | Prompt |
|---|---|
| Head (all shots) | Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth |
| Upper body (all shots) | a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs |
| Lower body (full-body shots only) | high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots |
| Style (all shots) | Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design. |
| Plain background | Plain light gray background without any lettering. |

## Negative prompt

Qwen-Image-2512 runs at cfg 4, so the negative prompt takes effect.

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

| Added for | Extra negative |
|---|---|
| Waist-up | `full body, legs, shorts, boots` |
| Close-up | `full body, legs, shorts, boots, waist` |
| Rear view | `face, looking back, looking at viewer, front view` |

## ComfyUI settings

| Setting | Value |
|---|---|
| 모델 / Model | `qwen_image_2512_fp8_e4m3fn.safetensors` (UNETLoader, weight_dtype default) |
| 텍스트 인코더 / Text encoder | `qwen_2.5_vl_7b_fp8_scaled.safetensors` (CLIPLoader type `qwen_image`) |
| VAE | `qwen_image_vae.safetensors` |
| ModelSamplingAuraFlow shift | 3.1 |
| KSampler | steps 20 (공식 템플릿 50, 시간 때문에 20 / official template 50), cfg 4, euler, simple, denoise 1.0 |
| 크기 / Size | 전신 1024×1536, 상반신 1024×1280, 얼굴 1024×1024 |
| 시드 / Seed | 31000+번호 (예: nkgreen_01 = 31001) |
| LoRA | 쓰지 않음 (Lightning 4-step LoRA 는 반사실풍으로 바뀌어 학습 데이터에 쓰지 않음) |

## Shot list (24 shots)

| # | Framing | View | Background | Size | Seed |
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

## Training captions

The fixed look goes into the trigger word `nkgreen`, so captions only list what changes.

`nkgreen, <framing>, <view>, <expression / pose>, <background>, clean anime illustration`

### nkgreen_01 · full · front · 1024×1536 · seed 31001

Positive

```
Full-body illustration of a single character from head to toe, front view facing the viewer, standing still with her arms at her sides, a calm shy look. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, front view, standing, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_02 · full · 3ql · 1024×1536 · seed 31002

Positive

```
Full-body illustration of a single character from head to toe, three-quarter view turned slightly to the left, standing still, a calm shy look. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, three-quarter view from the left, standing, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_03 · full · 3qr · 1024×1536 · seed 31003

Positive

```
Full-body illustration of a single character from head to toe, three-quarter view turned slightly to the right, standing still, a faint small smile. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, three-quarter view from the right, standing, faint smile, plain light gray background, clean anime illustration
```

### nkgreen_04 · full · side · 1024×1536 · seed 31004

Positive

```
Full-body illustration of a single character from head to toe, side profile view facing right, in profile with only one eye visible, standing still, looking ahead. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, side view, standing, looking ahead, plain light gray background, clean anime illustration
```

### nkgreen_05 · full · back · 1024×1536 · seed 31005

Positive

```
Full-body illustration of a single character from head to toe, rear view, the character stands facing away from the viewer toward the background, so we see only the back, the back of the head and the hair, standing still. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, face, looking back, looking at viewer, front view
```

Caption

```
nkgreen, full body, back view, standing, plain light gray background, clean anime illustration
```

### nkgreen_06 · full · front · 1024×1536 · seed 31006

Positive

```
Full-body illustration of a single character from head to toe, front view facing the viewer, walking forward with small steps. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, front view, walking, plain light gray background, clean anime illustration
```

### nkgreen_07 · full · 3qr · 1024×1536 · seed 31007

Positive

```
Full-body illustration of a single character from head to toe, three-quarter view turned slightly to the right, walking slowly along the path. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Background: a quiet forest path with soft morning light. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, three-quarter view from the right, walking, forest path, clean anime illustration
```

### nkgreen_08 · full · front · 1024×1536 · seed 31008

Positive

```
Full-body illustration of a single character from head to toe, front view facing the viewer, sitting on a wooden bench with the long sleeves resting on her lap. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Background: a small cobblestone town square. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, front view, sitting on a bench, town square, clean anime illustration
```

### nkgreen_09 · full · 3ql · 1024×1536 · seed 31009

Positive

```
Full-body illustration of a single character from head to toe, three-quarter view turned slightly to the left, crouching down to look at a small white flower. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Background: a green summer meadow under a clear sky. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, three-quarter view from the left, crouching, looking at a flower, summer meadow, clean anime illustration
```

### nkgreen_10 · full · side · 1024×1536 · seed 31010

Positive

```
Full-body illustration of a single character from head to toe, side profile view facing right, in profile with only one eye visible, standing on the bridge and looking down at the river. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs, high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up ankle boots. Background: an old stone bridge over a calm river. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup
```

Caption

```
nkgreen, full body, side view, standing, looking down, stone bridge over a river, clean anime illustration
```

### nkgreen_11 · waist · front · 1024×1280 · seed 31011

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, front view facing the viewer, a shy small smile. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, front view, shy small smile, plain light gray background, clean anime illustration
```

### nkgreen_12 · waist · 3ql · 1024×1280 · seed 31012

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, three-quarter view turned slightly to the left, a surprised look with slightly open mouth. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, three-quarter view from the left, surprised, plain light gray background, clean anime illustration
```

### nkgreen_13 · waist · 3qr · 1024×1280 · seed 31013

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, three-quarter view turned slightly to the right, a worried look, eyebrows drawn together. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a cozy library with tall wooden bookshelves. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, three-quarter view from the right, worried, library, clean anime illustration
```

### nkgreen_14 · waist · side · 1024×1280 · seed 31014

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, side profile view facing right, in profile with only one eye visible, looking up at something above. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, side view, looking up, plain light gray background, clean anime illustration
```

### nkgreen_15 · waist · front · 1024×1280 · seed 31015

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, front view facing the viewer, a gentle smile. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a rainy street seen from under a shop awning. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, front view, gentle smile, rainy street, clean anime illustration
```

### nkgreen_16 · waist · 3qr · 1024×1280 · seed 31016

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, three-quarter view turned slightly to the right, a determined look. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a quiet forest path with soft morning light. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots
```

Caption

```
nkgreen, upper body, three-quarter view from the right, determined, forest path, clean anime illustration
```

### nkgreen_17 · waist · back · 1024×1280 · seed 31017

Positive

```
Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs are out of frame, rear view, the character stands facing away from the viewer toward the background, so we see only the back, the back of the head and the hair, standing still. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, face, looking back, looking at viewer, front view
```

Caption

```
nkgreen, upper body, back view, standing, plain light gray background, clean anime illustration
```

### nkgreen_18 · face · front · 1024×1024 · seed 31018

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, front view facing the viewer, calm shy eyes looking at the viewer. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, front view, calm shy look, plain light gray background, clean anime illustration
```

### nkgreen_19 · face · 3ql · 1024×1024 · seed 31019

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, three-quarter view turned slightly to the left, a faint small smile. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, three-quarter view from the left, faint smile, plain light gray background, clean anime illustration
```

### nkgreen_20 · face · 3qr · 1024×1024 · seed 31020

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, three-quarter view turned slightly to the right, teary eyes, a sad look. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, three-quarter view from the right, teary sad look, plain light gray background, clean anime illustration
```

### nkgreen_21 · face · side · 1024×1024 · seed 31021

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, side profile view facing right, in profile with only one eye visible, a calm look into the distance. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Plain light gray background without any lettering. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, side view, calm look into the distance, plain light gray background, clean anime illustration
```

### nkgreen_22 · face · front · 1024×1024 · seed 31022

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, front view facing the viewer, wide surprised eyes. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a night sky full of stars over a hill. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, front view, wide surprised eyes, starry night sky, clean anime illustration
```

### nkgreen_23 · face · 3ql · 1024×1024 · seed 31023

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, three-quarter view turned slightly to the left, laughing with her eyes closed. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a green summer meadow under a clear sky. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, three-quarter view from the left, laughing, eyes closed, summer meadow, clean anime illustration
```

### nkgreen_24 · face · front · 1024×1024 · seed 31024

Positive

```
Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just below the shoulders, front view facing the viewer, eyes looking down shyly. Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a soft blush, a small mouth, a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the cape and ending in soft rounded cuffs. Background: a small bedroom lit by a warm desk lamp. Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth gentle gradients, natural muted colors, soft even lighting, original character design.
```

Negative

```
photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, sneakers, high heels, adult woman, mature face, makeup, full body, legs, shorts, boots, waist
```

Caption

```
nkgreen, close-up, front view, eyes looking down shyly, warm bedroom, clean anime illustration
```
