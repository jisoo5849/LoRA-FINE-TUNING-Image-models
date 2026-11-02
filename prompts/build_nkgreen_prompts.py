"""니키(nkgreen) LoRA 학습 이미지용 Qwen-Image-2512 프롬프트를 한글·영문 MD 로 만든다 (2026-10-31).

기준 이미지: Qwen-Image-2512 로 만든 니키 전신 3장 (S06-Q1~Q3, 사용자가 '기본값으로 고정' 하라고 지정).
고정 외형은 매 컷 같은 문장으로 반복하고, 컷마다 바뀌는 것(각도·화면 크기·표정·포즈·배경)만 바꾼다.

  python prompts/build_nkgreen_prompts.py  ->  prompts/nkgreen_qwen2512_prompts_en.md, _ko.md, nkgreen_shots.json
"""
import json
from pathlib import Path

OUT = Path(__file__).parent
TRIGGER = "nkgreen"
SEED0 = 31000

# ---------------------------------------------------------------- 고정 외형 (영문 / 한글)
HEAD_EN = ("Niki, a petite twelve-year-old girl with a slim build, a glossy jet-black chin-length bob with short "
           "straight bangs just above her eyes, large dark eyes with a faint sky-blue tint, pale cream skin with a "
           "soft blush, a small mouth")
HEAD_KO = ("니키, 마른 체형의 작은 12살 여자아이, 윤기 나는 새까만 턱선 길이 단발에 눈썹 바로 위까지 오는 짧은 일자 앞머리, "
           "하늘빛이 살짝 도는 크고 어두운 눈, 볼이 발그레한 창백한 크림빛 피부, 작은 입")
# 손 단어를 쓰면 손을 그린다 (character-bible 1절). 소매가 길게 늘어진 모습만 적는다.
UPPER_EN = ("a deep wine-red wool cape coat reaching her hips, with a turn-down collar and a row of dark buttons down "
            "the front, an ivory cream top with a round neckline, its very long cream sleeves hanging well below the "
            "cape and ending in soft rounded cuffs")
UPPER_KO = ("엉덩이까지 오는 짙은 와인빛 울 케이프 코트(접힌 깃, 앞섶에 어두운 단추 한 줄), 둥근 목선의 아이보리 크림색 상의, "
            "케이프 아래로 길게 늘어져 부드럽게 둥근 소맷부리로 끝나는 아주 긴 크림색 소매")
LOWER_EN = ("high-waisted dark mocha brown wool shorts, opaque charcoal-gray tights, low-heeled brown leather lace-up "
            "ankle boots")
LOWER_KO = "허리선이 높은 다크 모카 울 반바지, 비치지 않는 짙은 회색 타이츠, 굽 낮은 갈색 가죽 끈 앵클부츠"

STYLE_EN = ("Clean modern anime character illustration: delicate thin dark outlines, soft cel shading with smooth "
            "gentle gradients, natural muted colors, soft even lighting, original character design.")
STYLE_KO = "깔끔한 현대 애니메이션 캐릭터 일러스트: 가늘고 섬세한 어두운 윤곽선, 부드러운 그라데이션의 셀 채색, 차분한 자연색, 고른 조명, 오리지널 캐릭터 디자인."
PLAIN_EN = "Plain light gray background without any lettering."
PLAIN_KO = "글자 없는 밝은 회색 무지 배경."

NEG = ("photorealistic, 3d render, cgi, text, letters, watermark, logo, signature, blurry, lowres, jpeg artifacts, "
       "deformed, bad anatomy, extra limbs, extra arms, hands, fingers, visible hands, multiple people, duplicate, "
       "cropped head, brown hair, blonde hair, long hair, ponytail, freckles, hat, short sleeves, bare legs, skirt, "
       "sneakers, high heels, adult woman, mature face, makeup")
NEG_EXTRA = {
    "waist": ", full body, legs, shorts, boots",
    "face": ", full body, legs, shorts, boots, waist",
    "back": ", face, looking back, looking at viewer, front view",
}

# ---------------------------------------------------------------- 화면 크기 · 각도
FRAME = {
    "full": ("Full-body illustration of a single character from head to toe", "머리부터 발끝까지 한 인물 전신 일러스트", 1024, 1536),
    "waist": ("Medium shot from the waist up of a single character, the picture is cropped at the waist so the legs "
              "are out of frame", "허리 위 한 인물 상반신, 허리에서 잘라 다리는 화면 밖", 1024, 1280),
    "face": ("Close-up portrait of a single character, the face and shoulders fill the whole picture, cropped just "
             "below the shoulders", "한 인물 얼굴 클로즈업, 얼굴과 어깨가 화면을 채우고 어깨 바로 아래에서 자름", 1024, 1024),
}
VIEW = {
    "front": ("front view facing the viewer", "정면"),
    "3ql": ("three-quarter view turned slightly to the left", "왼쪽으로 살짝 돈 3/4 각도"),
    "3qr": ("three-quarter view turned slightly to the right", "오른쪽으로 살짝 돈 3/4 각도"),
    "side": ("side profile view facing right, in profile with only one eye visible", "오른쪽을 향한 옆모습, 눈은 하나만 보임"),
    # '돌아봄' 을 넣으면 비스듬한 앞모습이 된다 (character-bible 1절)
    "back": ("rear view, the character stands facing away from the viewer toward the background, so we see only the "
             "back, the back of the head and the hair", "뒷모습, 화면 안쪽을 향해 서서 등과 뒤통수·머리카락만 보임"),
}
SCENE = {
    None: (PLAIN_EN, PLAIN_KO, "plain light gray background"),
    "forest": ("Background: a quiet forest path with soft morning light.", "배경: 아침 햇살이 드는 조용한 숲길.", "forest path"),
    "square": ("Background: a small cobblestone town square.", "배경: 작은 돌길 마을 광장.", "town square"),
    "meadow": ("Background: a green summer meadow under a clear sky.", "배경: 맑은 하늘 아래 초록 여름 들판.", "summer meadow"),
    "bridge": ("Background: an old stone bridge over a calm river.", "배경: 잔잔한 강 위의 오래된 돌다리.", "stone bridge over a river"),
    "library": ("Background: a cozy library with tall wooden bookshelves.", "배경: 키 큰 나무 책장이 있는 아늑한 도서관.", "library"),
    "rain": ("Background: a rainy street seen from under a shop awning.", "배경: 가게 차양 아래에서 본 비 오는 거리.", "rainy street"),
    "night": ("Background: a night sky full of stars over a hill.", "배경: 언덕 위 별이 가득한 밤하늘.", "starry night sky"),
    "room": ("Background: a small bedroom lit by a warm desk lamp.", "배경: 따뜻한 스탠드 불빛의 작은 방.", "warm bedroom"),
}

# (화면, 각도, 배경, 내용 영문, 내용 한글, 캡션용 표정·포즈)
SHOTS = [
    ("full", "front", None, "standing still with her arms at her sides, a calm shy look", "팔을 내리고 가만히 선 자세, 잔잔하고 수줍은 표정", "standing, calm shy look"),
    ("full", "3ql", None, "standing still, a calm shy look", "가만히 선 자세, 잔잔하고 수줍은 표정", "standing, calm shy look"),
    ("full", "3qr", None, "standing still, a faint small smile", "가만히 선 자세, 옅은 미소", "standing, faint smile"),
    ("full", "side", None, "standing still, looking ahead", "가만히 서서 앞을 봄", "standing, looking ahead"),
    ("full", "back", None, "standing still", "가만히 선 자세", "standing"),
    ("full", "front", None, "walking forward with small steps", "작은 걸음으로 앞으로 걸어옴", "walking"),
    ("full", "3qr", "forest", "walking slowly along the path", "길을 따라 천천히 걸음", "walking"),
    ("full", "front", "square", "sitting on a wooden bench with the long sleeves resting on her lap", "나무 벤치에 앉아 긴 소매를 무릎 위에 올려 둠", "sitting on a bench"),
    ("full", "3ql", "meadow", "crouching down to look at a small white flower", "쪼그려 앉아 작은 흰 꽃을 들여다봄", "crouching, looking at a flower"),
    ("full", "side", "bridge", "standing on the bridge and looking down at the river", "다리 위에 서서 강을 내려다봄", "standing, looking down"),
    ("waist", "front", None, "a shy small smile", "수줍은 작은 미소", "shy small smile"),
    ("waist", "3ql", None, "a surprised look with slightly open mouth", "입을 살짝 벌린 놀란 표정", "surprised"),
    ("waist", "3qr", "library", "a worried look, eyebrows drawn together", "눈썹을 모은 걱정스러운 표정", "worried"),
    ("waist", "side", None, "looking up at something above", "위쪽의 무언가를 올려다봄", "looking up"),
    ("waist", "front", "rain", "a gentle smile", "다정한 미소", "gentle smile"),
    ("waist", "3qr", "forest", "a determined look", "결심한 듯한 표정", "determined"),
    ("waist", "back", None, "standing still", "가만히 선 자세", "standing"),
    ("face", "front", None, "calm shy eyes looking at the viewer", "보는 사람을 바라보는 잔잔하고 수줍은 눈", "calm shy look"),
    ("face", "3ql", None, "a faint small smile", "옅은 미소", "faint smile"),
    ("face", "3qr", None, "teary eyes, a sad look", "눈물이 고인 슬픈 표정", "teary sad look"),
    ("face", "side", None, "a calm look into the distance", "먼 곳을 바라보는 잔잔한 표정", "calm look into the distance"),
    ("face", "front", "night", "wide surprised eyes", "크게 뜬 놀란 눈", "wide surprised eyes"),
    ("face", "3ql", "meadow", "laughing with her eyes closed", "눈을 감고 웃음", "laughing, eyes closed"),
    ("face", "front", "room", "eyes looking down shyly", "수줍게 아래를 보는 눈", "eyes looking down shyly"),
]
CAP_FRAME = {"full": "full body", "waist": "upper body", "face": "close-up"}
CAP_VIEW = {"front": "front view", "3ql": "three-quarter view from the left", "3qr": "three-quarter view from the right",
            "side": "side view", "back": "back view"}


def describe(frame, lang):
    head, upper, lower = (HEAD_EN, UPPER_EN, LOWER_EN) if lang == "en" else (HEAD_KO, UPPER_KO, LOWER_KO)
    # 화면 밖에 있을 옷은 쓰지 않는다: 상반신·얼굴 컷에 반바지·부츠를 적으면 전신을 그렸다
    return ", ".join([head, upper] + ([lower] if frame == "full" else []))


def shots():
    out = []
    for i, (frame, view, scene, c_en, c_ko, cap) in enumerate(SHOTS, 1):
        f_en, f_ko, w, h = FRAME[frame]
        bg_en, bg_ko, bg_cap = SCENE[scene]
        en = f"{f_en}, {VIEW[view][0]}, {c_en}. {describe(frame, 'en')}. {bg_en} {STYLE_EN}"
        ko = f"{f_ko}, {VIEW[view][1]}, {c_ko}. {describe(frame, 'ko')}. {bg_ko} {STYLE_KO}"
        neg = NEG + NEG_EXTRA.get(frame, "") + (NEG_EXTRA["back"] if view == "back" else "")
        caption = f"{TRIGGER}, {CAP_FRAME[frame]}, {CAP_VIEW[view]}, {cap}, {bg_cap}, clean anime illustration"
        out.append(dict(id=f"nkgreen_{i:02d}", frame=frame, view=view, scene=scene or "plain", w=w, h=h,
                        seed=SEED0 + i, positive=en, positive_ko=ko, negative=neg, caption=caption))
    return out


SETTINGS = [
    ("모델 / Model", "`qwen_image_2512_fp8_e4m3fn.safetensors` (UNETLoader, weight_dtype default)"),
    ("텍스트 인코더 / Text encoder", "`qwen_2.5_vl_7b_fp8_scaled.safetensors` (CLIPLoader type `qwen_image`)"),
    ("VAE", "`qwen_image_vae.safetensors`"),
    ("ModelSamplingAuraFlow shift", "3.1"),
    ("KSampler", "steps 20 (공식 템플릿 50, 시간 때문에 20 / official template 50), cfg 4, euler, simple, denoise 1.0"),
    ("크기 / Size", "전신 1024×1536, 상반신 1024×1280, 얼굴 1024×1024"),
    ("시드 / Seed", f"{SEED0}+번호 (예: nkgreen_01 = {SEED0 + 1})"),
    ("LoRA", "쓰지 않음 (Lightning 4-step LoRA 는 반사실풍으로 바뀌어 학습 데이터에 쓰지 않음)"),
]


def main():
    S = shots()
    (OUT / "nkgreen_shots.json").write_text(json.dumps(S, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    sizes = {k: f"{v[2]}×{v[3]}" for k, v in FRAME.items()}

    en = ["# Niki (`nkgreen`) LoRA training images — Qwen-Image-2512 prompts", "",
          "Reference: three full-body Niki images made with Qwen-Image-2512 (S06-Q1~Q3). Their look is fixed as the default.",
          "", "## How the prompt is built", "",
          "`<framing>, <view>, <expression / pose>. <fixed character block>. <background> <style>`", "",
          "- The fixed character block is repeated word for word in every shot; only framing, view, expression, pose and background change.",
          "- Waist-up and close-up shots leave out the shorts, tights and boots (writing off-frame clothes made the model draw full bodies).",
          "- Hands are never mentioned in the positive prompt; the long sleeves are described instead, and hands are blocked in the negative prompt.",
          "- Rear views never say \"looking back\" (it turns into an angled front view).", "",
          "## Fixed character block", "",
          "| Part | Prompt |", "|---|---|",
          f"| Head (all shots) | {HEAD_EN} |", f"| Upper body (all shots) | {UPPER_EN} |",
          f"| Lower body (full-body shots only) | {LOWER_EN} |", f"| Style (all shots) | {STYLE_EN} |",
          f"| Plain background | {PLAIN_EN} |", "",
          "## Negative prompt", "", "Qwen-Image-2512 runs at cfg 4, so the negative prompt takes effect.", "",
          "```", NEG, "```", "",
          "| Added for | Extra negative |", "|---|---|",
          f"| Waist-up | `{NEG_EXTRA['waist'].lstrip(', ')}` |", f"| Close-up | `{NEG_EXTRA['face'].lstrip(', ')}` |",
          f"| Rear view | `{NEG_EXTRA['back'].lstrip(', ')}` |", "",
          "## ComfyUI settings", "", "| Setting | Value |", "|---|---|"] + [f"| {a} | {b} |" for a, b in SETTINGS] + [
          "", "## Shot list (24 shots)", "",
          "| # | Framing | View | Background | Size | Seed |", "|---|---|---|---|---|---|"] + [
          f"| {s['id']} | {s['frame']} | {s['view']} | {s['scene']} | {s['w']}×{s['h']} | {s['seed']} |" for s in S] + [
          "", "## Training captions", "",
          f"The fixed look goes into the trigger word `{TRIGGER}`, so captions only list what changes.", "",
          f"`{TRIGGER}, <framing>, <view>, <expression / pose>, <background>, clean anime illustration`", ""]
    for s in S:
        en += [f"### {s['id']} · {s['frame']} · {s['view']} · {s['w']}×{s['h']} · seed {s['seed']}", "",
               "Positive", "", "```", s["positive"], "```", "", "Negative", "", "```", s["negative"], "```", "",
               "Caption", "", "```", s["caption"], "```", ""]
    (OUT / "nkgreen_qwen2512_prompts_en.md").write_text("\n".join(en), encoding="utf-8", newline="\n")

    ko = ["# 니키(`nkgreen`) LoRA 학습 이미지 · Qwen-Image-2512 프롬프트 (한글)", "",
          "기준 이미지는 Qwen-Image-2512 로 만든 니키 전신 3장(S06-Q1~Q3)이고, 그 모습을 기본값으로 고정한다.",
          "실제 생성에는 영문판(`nkgreen_qwen2512_prompts_en.md`)을 넣는다. 이 문서는 같은 내용을 한글로 옮긴 것이다.", "",
          "## 기준 이미지와 캐릭터 바이블의 차이", "",
          "- 이 프롬프트는 기준 이미지(컨셉 시트 버전)를 따른다: 새까만 단발, 하늘빛이 살짝 도는 어두운 눈, 주근깨 없음, 현대 애니 일러스트 그림체.",
          "- 2026-10-29 의 '상상도 우선' 결정(짙은 갈색 단발, 맑은 파란 눈, 주근깨, 손그림 극장 애니메이션풍)과는 다르다 [TBD] 둘 중 무엇을 니키의 정본으로 할지.",
          "", "## 프롬프트 구조", "",
          "`<화면 크기>, <각도>, <표정·포즈>. <고정 캐릭터 문장>. <배경> <그림체>`", "",
          "- 고정 캐릭터 문장은 모든 컷에 글자 하나 바꾸지 않고 넣고, 화면 크기·각도·표정·포즈·배경만 바꾼다.",
          "- 상반신·얼굴 컷에는 반바지·타이츠·부츠를 쓰지 않는다.",
          "  - 예: 화면 밖에 있을 옷을 적으면 전신을 그렸다.",
          "- 긍정 프롬프트에는 손을 쓰지 않고 긴 소매만 적으며, 손은 부정 프롬프트로 막는다.",
          "  - 예: 손 단어가 있으면 손을 그렸다. 그래도 손이 나오면 Qwen-Image-Edit-2511 로 소매를 고친다.",
          "- 뒷모습에 '돌아봄' 을 쓰지 않는다.",
          "  - 예: '뒷모습 + 어깨 너머로 돌아봄' 이 비스듬한 앞모습이 됐다.", "",
          "## 고정 캐릭터 문장", "", "| 부분 | 한글 | 영문 |", "|---|---|---|",
          f"| 머리·얼굴 (모든 컷) | {HEAD_KO} | {HEAD_EN} |", f"| 상의 (모든 컷) | {UPPER_KO} | {UPPER_EN} |",
          f"| 하의 (전신 컷만) | {LOWER_KO} | {LOWER_EN} |", f"| 그림체 (모든 컷) | {STYLE_KO} | {STYLE_EN} |",
          f"| 무지 배경 | {PLAIN_KO} | {PLAIN_EN} |", "",
          "- 색은 컨셉 시트 색 코드를 따른다.",
          "  - 예: 케이프 #7B3342, 크림 상의 #F3E8D6, 모카 반바지 #4A372E, 회색 타이츠 #4A4A52, 갈색 부츠 #6B4A35", "",
          "## 부정 프롬프트", "", "Qwen-Image-2512 는 cfg 4 라 부정 프롬프트가 듣는다.", "", "```", NEG, "```", "",
          "| 더하는 경우 | 추가 부정어 |", "|---|---|",
          f"| 상반신 | `{NEG_EXTRA['waist'].lstrip(', ')}` |", f"| 얼굴 | `{NEG_EXTRA['face'].lstrip(', ')}` |",
          f"| 뒷모습 | `{NEG_EXTRA['back'].lstrip(', ')}` |", "",
          "- 부정어 뜻: 사진풍·3D·글자·워터마크·흐림·저화질·변형·팔다리 추가·손·손가락·여러 사람·머리 잘림·갈색/금발/긴 머리·포니테일·주근깨·모자·반팔·맨다리·치마·운동화·하이힐·성인 얼굴·화장을 막는다.", "",
          "## ComfyUI 설정", "", "| 설정 | 값 |", "|---|---|"] + [f"| {a} | {b} |" for a, b in SETTINGS] + [
          "", "## 컷 구성 (24장)", "",
          "| 번호 | 화면 | 각도 | 배경 | 크기 | 시드 |", "|---|---|---|---|---|---|"] + [
          f"| {s['id']} | {s['frame']} | {s['view']} | {s['scene']} | {s['w']}×{s['h']} | {s['seed']} |" for s in S] + [
          "", f"- 화면 크기: 전신 {sizes['full']}, 상반신 {sizes['waist']}, 얼굴 {sizes['face']}.",
          "- 전신 10장(무지 6, 장면 4), 상반신 7장(무지 4, 장면 3), 얼굴 7장(무지 4, 장면 3)으로 각도·표정·배경을 고르게 섞었다.", "",
          "## 학습 캡션", "",
          f"고정 외형은 트리거 `{TRIGGER}` 하나에 담고, 캡션에는 컷마다 바뀌는 것만 영문으로 적는다 (계획서 3-2절).", "",
          f"`{TRIGGER}, <화면 크기>, <각도>, <표정·포즈>, <배경>, clean anime illustration`", "",
          "- 고정 외형(검은 단발, 와인 케이프, 긴 크림 소매 등)을 캡션에 적으면 그 특징이 트리거가 아닌 글에 붙으므로 적지 않는다.",
          "- 그림체와 배경은 캡션에 적어, 트리거에 그림체·배경까지 붙지 않게 한다.", "",
          "## 생성 뒤 검수", "",
          "- 손이 소매 밖으로 나온 컷, 머리색·케이프색이 바뀐 컷, 상반신 컷이 전신으로 나온 컷, 뒷모습이 앞모습으로 나온 컷은 다시 뽑는다.",
          "- 같은 컷을 시드만 바꿔 2장씩 뽑고 나은 쪽을 고르면 검수 시간이 줄어든다.", ""]
    for s in S:
        ko += [f"### {s['id']} · {s['frame']} · {s['view']} · {s['w']}×{s['h']} · 시드 {s['seed']}", "",
               "긍정 (한글)", "", "```", s["positive_ko"], "```", "", "부정", "", "```", s["negative"], "```", "",
               "캡션", "", "```", s["caption"], "```", ""]
    (OUT / "nkgreen_qwen2512_prompts_ko.md").write_text("\n".join(ko), encoding="utf-8", newline="\n")
    print(len(S), "shots")


if __name__ == "__main__":
    main()
