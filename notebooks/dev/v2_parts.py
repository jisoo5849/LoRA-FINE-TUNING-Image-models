"""lora_training_v2.ipynb 에만 들어가는 부분. build_notebook.py 의 # [v2:이름] / # [v1:이름] 표시 자리에 들어간다.

v2 = v1 + 학습 범위 선택(TARGET_SCOPE) + 층 묶음별 학습률 + 그림체 보존 학습(DOP) + 그림체 변화 확인.
"""

V2 = {}

V2["title"] = r"""
# 캐릭터 LoRA 학습 노트북 v2 (학습 범위 넓히기 + 그림체 보존)
"""

V2["intro"] = r"""
### v1 과 다른 점 (v2)

v2 는 `lora_training.ipynb`(v1)에서 아래만 바꾼 별도 노트북이다. 나머지 셀은 v1 과 같다.

| 항목 | v1 | v2 |
|---|---|---|
| LoRA 를 붙이는 층 | 공식 학습 스크립트 기본값 (계열 대부분은 어텐션만) | `TARGET_SCOPE` 로 고른다. 기본 `all` = 0-3-1 절에 정한 묶음 전부 (대부분 계열에서 블록 안 선형층 전부) |
| 층별 학습률 | 모두 같음 | v1 기본 층은 v1 그대로, 새로 더한 층은 낮춤 (`lr_mult`) |
| 그림체 보존 | 없음 | `PRESERVATION="dop"`: 트리거를 일반 낱말(`CLASS_WORD`)로 바꾼 문장은 베이스 모델과 같은 출력을 내도록 함께 학습 |
| 그림체 변화 확인 | 없음 | 트리거 없는 프롬프트 샘플을 0 단계(베이스 모델)와 비교 (5-2-1) |

- `TARGET_SCOPE` 값 (뒤로 갈수록 넓음)
  - `attn`: v1 과 같은 층
  - `attn_all`: + 나머지 어텐션 (Qwen-Image·FLUX.2 의 글 쪽 어텐션 `add_*_proj`, `to_add_out` 등)
  - `attn_mlp`: + MLP (어텐션 다음에 토큰마다 따로 거치는 2층 신경망)
  - `all`: + 블록 안의 조절(modulation) 층 (노이즈 단계에 따라 각 블록의 세기를 바꾸는 층)
- 계열마다 v1 기본 층이 달라서 범위를 넓힐 때 늘어나는 양도 다르다. 아래는 실제 모델 설정으로 센 LoRA 파라미터 수다 (랭크는 계열별 시작값).

| 계열 (랭크) | `attn` | `attn_all` | `attn_mlp` | `all` | `all` / `attn` |
|---|---|---|---|---|---|
| Qwen-Image 2512 (16) | 2,360만 | 4,720만 | 1억 620만 | 1억 4,750만 | 6.3배 |
| Qwen-Image 2.1 (16) | 1,680만 | 1,680만 | 4,190만 | 4,190만 | 2.5배 |
| FLUX.2 klein 4B (16) | 1,670만 | 1,870만 | 2,410만 | 2,410만 | 1.4배 |
| Z-Image (32) | 3,340만 | 3,340만 | 7,940만 | 9,540만 | 2.9배 |
| Krea 2 (32) | 9,590만 | 9,590만 | 1억 1,730만 | 1억 1,730만 | 1.2배 |
| FLUX.1 dev (16) | 4,480만 | 4,480만 | 6,540만 | 8,590만 | 1.9배 |
| Chroma (16) | 4,480만 | 4,480만 | 6,540만 | 6,540만 | 1.5배 |
| LongCat-Image (16) | 2,360만 | 2,360만 | 3,440만 | 4,520만 | 1.9배 |
| SDXL (16) | 2,320만 | 2,320만 | 4,190만 | 4,190만 | 1.8배 |
| SD3.5 Medium (16) | 940만 | 1,190만 | 2,350만 | 3,260만 | 3.5배 |
| SD3.5 Large (16) | 2,360만 | 2,360만 | 5,280만 | 7,330만 | 3.1배 |

  - LoRA 파일은 fp32 로 저장되므로 크기는 대략 파라미터 수 × 4바이트다 (예: Qwen-Image `attn` 약 94MB, `all` 약 590MB). 저장 단계마다 파일이 하나씩 생긴다.
  - FLUX.1·Chroma·LongCat 의 v1 은 두 갈래 블록(`transformer_blocks`)의 MLP 까지만 학습한다. 범위를 넓히면 한 갈래 블록(`single_transformer_blocks`)의 MLP 가 더해지고, FLUX.1·LongCat 은 두 갈래·한 갈래 블록의 조절 층도 더해진다 (Chroma 는 조절 층이 없는 구조).
- v2 가 더하는 층은 모두 블록 안에 있다. 블록 밖의 층(입력 임베딩, 출력층, FLUX.2 의 공유 조절 층 등)은 더하지 않는다. ComfyUI 가 읽지 못하는 층이 있기 때문이다.
  - 단, v1 기본 층에 이미 블록 밖 층이 있는 계열은 v1 처럼 그대로 학습한다 (예: Krea 2 의 `img_in`, `txt_in`, `time_embed`, `final_layer`).
  - SDXL 의 `all` 은 어텐션·MLP 까지다. 조절 층(`time_emb_proj`)을 넣으면 diffusers 의 LoRA 다시 불러오기가 일부만 불러와서 뺐고, 어텐션 앞뒤의 `proj_in`·`proj_out` 도 넣지 않는다.
- `TARGET_SCOPE="attn"`, `PRESERVATION="off"` 로 두면 v1 과 같은 학습이 된다. 10개 계열 모두 시험 실행에서 단계별 손실이 v1 과 같았다.
- `RUN_NAME` 기본값은 `nkgreen_v2` 다. 같은 이름의 결과 폴더가 이미 있으면 0-2 절과 학습 셀에서 멈춘다 (앞선 결과를 덮어쓰지 않게). 모델·설정을 바꿀 때마다 `RUN_NAME` 도 바꾼다.
- `MODEL_PATH` 기본값은 메인 서버에서 Qwen 학습 때 쓴 `/home/mgs/models/diffusers/Qwen-Image-2512` 다.
- 보존 학습(DOP)은 한 단계에 디노이저를 세 번 계산한다. 학습 시간은 v1 의 약 2~2.5배로 추정한다 (실측 전).
- 층 묶음별 학습률 배율과 보존 손실 가중치는 근거가 약한 시작값이다 [제안]. 출처는 0-2 절 주석에 있다.
"""

V2["setup_md"] = r"""
- v2 설정: `TARGET_SCOPE`(학습 범위), `PRESERVATION`(`dop` 또는 `off`), `CLASS_WORD`(보존 학습에서 트리거 대신 넣을 일반 낱말, 기본 `girl`).
  - 예: `LORA_TARGET_SCOPE=attn_mlp LORA_PRESERVATION=dop LORA_CLASS_WORD=girl`
  - `CLASS_WORD` 는 캐릭터가 속한 넓은 부류다 (예: 여자아이면 `girl`, 남자 어른이면 `man`). `PRESERVATION="dop"` 이면 이 낱말로 그린 그림이 베이스 모델에 가깝게 남도록 함께 학습한다. `off` 여도 5-2-1 의 그림체 확인에는 쓴다.
"""

V2["user"] = r"""
TARGET_SCOPE=os.environ.get("LORA_TARGET_SCOPE", "all"),   # v2: attn / attn_all / attn_mlp / all
PRESERVATION=os.environ.get("LORA_PRESERVATION", "dop"),   # v2: dop(그림체 보존 학습) / off
CLASS_WORD=os.environ.get("LORA_CLASS_WORD", "girl"),      # v2: 보존 학습에서 트리거 대신 넣는 일반 낱말
"""

V2["fixed"] = r"""
# v2 시작값 [제안]. 근거가 약하므로 결과를 보고 고친다.
#   - lr_mult: v1 기본 층(base)은 v1 학습률 그대로 두고, 새로 더한 층만 낮춘다. 층마다 세기를 alpha 로 바꾸면
#     ComfyUI(alpha 텐서 없음 → 강도 1)와 세기가 달라지므로 학습률로 조절한다.
#     참고: musubi-tuner 는 Qwen-Image 선형층 전부를 학습률 5e-5 로, 조절 층은 빼고 학습한다.
#   - preservation_weight: ostris ai-toolkit 의 Differential Output Preservation 기본 배율 1.0
COMMON_FIXED.update(
    lr_mult=dict(base=1.0, attn_extra=0.5, mlp=0.5, mod=0.25),
    preservation_weight=1.0,
    drift_prompts=2,          # 그림체 변화 확인에 쓸 검증 프롬프트 수 (트리거를 CLASS_WORD 로 바꿔 그림)
)
"""

V2["user_check"] = r"""
SCOPES = {"attn": [], "attn_all": ["attn_extra"], "attn_mlp": ["attn_extra", "mlp"], "all": ["attn_extra", "mlp", "mod"]}
if USER["TARGET_SCOPE"] not in SCOPES:
    raise ValueError(f"TARGET_SCOPE='{USER['TARGET_SCOPE']}' 는 쓸 수 없습니다. 고를 수 있는 값: {list(SCOPES)}")
if USER["PRESERVATION"] not in ("dop", "off"):
    raise ValueError(f"PRESERVATION='{USER['PRESERVATION']}' 는 쓸 수 없습니다. dop 또는 off")
if USER["PRESERVATION"] == "dop" and (not USER["CLASS_WORD"].strip() or USER["CLASS_WORD"].strip() == USER["TRIGGER"]):
    raise ValueError("CLASS_WORD 는 트리거와 다른 일반 낱말이어야 합니다 (예: girl)")
_run_dir = Path(USER["OUTPUT_DIR"]) / USER["RUN_NAME"]
if not TEST_FAMILY and _run_dir.is_dir() and any(_run_dir.iterdir()):
    raise FileExistsError(f"{_run_dir} 에 이전 실행 결과가 있습니다. 덮어쓰지 않도록 RUN_NAME(LORA_RUN_NAME)을 바꾸거나, "
                          "지워도 되는 폴더면 지운 뒤 다시 실행합니다. 모델·설정을 바꿀 때마다 RUN_NAME 도 바꿉니다.")
print("v2 | 학습 범위:", USER["TARGET_SCOPE"], "| 그림체 보존:", USER["PRESERVATION"],
      f"(일반 낱말 '{USER['CLASS_WORD']}')" if USER["PRESERVATION"] == "dop" else "")
"""

V2["groups_md"] = r"""
### 0-3-1. 학습 범위 (v2)

계열마다 v1 기본 층(어댑터의 `LORA_TARGET_MODULES`) 밖에 더할 수 있는 블록 안 선형층을 묶음으로 정해 둔다.

| 묶음 | 뜻 | 들어가는 범위 |
|---|---|---|
| `base` | v1 기본 층 | 모든 범위 |
| `attn_extra` | v1 에 없던 나머지 어텐션 (주로 글 쪽) | `attn_all` 부터 |
| `mlp` | v1 에 없던 MLP | `attn_mlp` 부터 |
| `mod` | 블록 안 조절(modulation) 층 | `all` |

- 층 이름은 diffusers v0.42.1 의 작은 시험 모델과, 실제 모델 설정(config)으로 만든 빈 모델에서 확인했다 (2026-11-04). 정한 묶음은 모두 블록 안 층과만 맞는다.
- 이 표에 없는 계열(새로 더한 어댑터)은 v1 기본 층만 학습한다.
- ComfyUI(2026-11-03 master) 코드를 읽어 본 결과, 이 묶음들의 LoRA 키는 모두 읽힌다. 실제로 불러 보지는 않았으므로 5-6 절대로 `lora key not loaded` 를 확인한다.
  - SDXL 은 v1 기본 층(어텐션)의 저장 형식이 ComfyUI 와 맞지 않을 수 있다는 코드 분석이 있다 [TBD] 실제로 불러 확인.
- 정해 둔 층이 없는 묶음은 그 계열에 해당 층이 없거나, 이미 v1 기본 층에 들어 있는 경우다. SDXL 의 `time_emb_proj`·`proj_in`·`proj_out` 은 일부러 넣지 않았다.
"""

V2["groups_code"] = r'''
# 층 이름 전체와 맞추는 정규식 (peft 는 target_modules 가 문자열이면 re.fullmatch 로 고른다)
LAYER_GROUPS = {
    "qwen_image": dict(
        attn_extra=r"transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)",
        mlp=r"transformer_blocks\.\d+\.(img_mlp|txt_mlp)\.net\.(0\.proj|2)",
        mod=r"transformer_blocks\.\d+\.(img_mod|txt_mod)\.1"),
    "qwen_image21": dict(
        mlp=r"transformer_blocks\.\d+\.img_mlp\.(proj|gate_layer|out)"),            # 블록 안 조절 층 없음 (공유 modulation 은 블록 밖)
    "flux2_klein": dict(
        attn_extra=r"transformer_blocks\.\d+\.attn\.(add_q_proj|add_k_proj|add_v_proj|to_add_out)",
        mlp=r"transformer_blocks\.\d+\.(ff|ff_context)\.(linear_in|linear_out)"),   # 공유 조절 층은 블록 밖이고 ComfyUI 가 읽지 않음
    "z_image": dict(
        mlp=r"(layers|noise_refiner|context_refiner)\.\d+\.feed_forward\.(w1|w2|w3)",
        mod=r"(layers|noise_refiner)\.\d+\.adaLN_modulation\.0"),
    "krea2": dict(
        mlp=r"(transformer_blocks|text_fusion\.layerwise_blocks|text_fusion\.refiner_blocks)\.\d+\.ff\.gate"),  # 나머지는 v1 에 있음
    "flux1": dict(
        mlp=r"single_transformer_blocks\.\d+\.(proj_mlp|proj_out)",
        mod=r"transformer_blocks\.\d+\.(norm1|norm1_context)\.linear|single_transformer_blocks\.\d+\.norm\.linear"),
    "chroma": dict(
        mlp=r"single_transformer_blocks\.\d+\.(proj_mlp|proj_out)"),               # 조절 층이 없는 구조 (pruned norm)
    "longcat_image": dict(
        mlp=r"single_transformer_blocks\.\d+\.(proj_mlp|proj_out)",
        mod=r"transformer_blocks\.\d+\.(norm1|norm1_context)\.linear|single_transformer_blocks\.\d+\.norm\.linear"),
    "sdxl": dict(
        mlp=r"(down_blocks|up_blocks|mid_block)\..*\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"),
        # time_emb_proj(블록마다 노이즈 단계를 넣는 층)를 넣으면 diffusers 0.42.1 의 load_lora_weights 가 일부만 불러와서 뺐다
    "sd3": dict(
        attn_extra=r"transformer_blocks\.\d+\.attn2\.(to_q|to_k|to_v|to_out\.0)",     # SD3.5 Medium 의 두 번째 어텐션 (SD3·SD3.5 Large 에는 없음)
        mlp=r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.(0\.proj|2)",
        mod=r"transformer_blocks\.\d+\.(norm1|norm1_context)\.linear"),
}
_BASE_RE = re.compile("|".join(r"(?:.*\.)?" + re.escape(t) for t in A.LORA_TARGET_MODULES))   # peft 의 이름 끝 맞추기와 같음
_GROUPS = LAYER_GROUPS.get(A.FAMILY, {})
if A.FAMILY not in LAYER_GROUPS:
    print(f"참고: {A.FAMILY} 는 LAYER_GROUPS 에 없어 v1 기본 층만 학습합니다")


def layer_group(module_name):
    """LoRA 를 붙인 층이 어느 묶음인지. v1 기본 층이면 base."""
    if _BASE_RE.fullmatch(module_name):
        return "base"
    for g, pat in _GROUPS.items():
        if re.fullmatch(pat, module_name):
            return g
    raise ValueError(f"묶음을 모르는 층: {module_name}")


def lora_target(scope):
    """peft LoraConfig 의 target_modules. attn 이면 v1 과 같은 목록을 그대로 쓴다."""
    extra = [_GROUPS[g] for g in SCOPES[scope] if g in _GROUPS]
    if not extra:
        return list(A.LORA_TARGET_MODULES)
    return "(?:" + "|".join([_BASE_RE.pattern] + extra) + ")"


_more = [g for g in SCOPES[USER["TARGET_SCOPE"]] if g in _GROUPS]
print(f"{A.FAMILY} | 범위 {USER['TARGET_SCOPE']} | v1 기본 층에 더하는 묶음: {_more or '없음 (v1 과 같은 층)'}")
'''

V2["encode"] = r"""
# v2: 보존 학습용 캡션(트리거 → 일반 낱말)과 그림체 변화 확인용 프롬프트
PRES_CAPTIONS = [TRIGGER_RE.sub(USER["CLASS_WORD"], c) for c in df.train_caption]
assert not any(TRIGGER_RE.search(c) for c in PRES_CAPTIONS), "보존 캡션에 트리거가 남음"
DRIFT_PROMPTS = [_TRIG.sub(USER["CLASS_WORD"], p) for p in VALIDATION_PROMPTS[:FIXED["drift_prompts"]]]
DRIFT_SEEDS = VALIDATION_SEEDS[:len(DRIFT_PROMPTS)]
pres_conds = A.encode_prompts(pipe, PRES_CAPTIONS, DEVICE, WEIGHT_DTYPE) if USER["PRESERVATION"] == "dop" else None
drift_conds = A.encode_prompts(pipe, DRIFT_PROMPTS, DEVICE, WEIGHT_DTYPE)
"""

V2["train_md"] = r"""
- LoRA 를 `TARGET_SCOPE` 의 층에 붙이고, 그 층만 학습한다 (0-3-1 절). 층 묶음마다 학습률에 `lr_mult` 배율을 곱한다.
- `PRESERVATION="dop"` 이면 한 단계에 아래 두 손실을 함께 줄인다 (ostris ai-toolkit 의 Differential Output Preservation 과 같은 방식).
  - 일반 손실: 트리거 캡션으로 학습 이미지를 맞춘다 (v1 과 같음).
  - 보존 손실: 같은 노이즈·같은 시그마에서, 트리거를 `CLASS_WORD` 로 바꾼 캡션의 출력이 LoRA 를 끈 베이스 모델의 출력과 같도록 한다.
  - 그래서 LoRA 의 변화가 트리거 낱말에 모이고, 트리거 없는 문장의 그림(그림체 포함)은 베이스 모델에 가깝게 남는다.
  - 트리거가 들어간 그림의 그림체 변화를 직접 막지는 않는다. 그래서 5-2 와 5-2-1 을 함께 본다.
"""

V2["lora_cfg"] = r"""
LORA_TARGET = lora_target(USER["TARGET_SCOPE"])
lora_cfg = LoraConfig(r=FIXED["rank"], lora_alpha=FIXED["alpha"], lora_dropout=FIXED["dropout"],
                      init_lora_weights=FIXED["init_lora_weights"], target_modules=LORA_TARGET)
"""

V2["groups_report"] = r"""
GROUP_OF = {m: layer_group(m) for m in {n.rsplit(".lora_", 1)[0] for n, _ in denoiser.named_parameters() if "lora_A" in n}}
_gc = Counter(GROUP_OF.values())
for _g in SCOPES[USER["TARGET_SCOPE"]]:
    if _g in _GROUPS and not _gc.get(_g):   # 예: SD3 의 attn2 는 SD3.5 Medium 에만 있다
        print(f"참고: '{_g}' 묶음에 맞는 층이 이 모델에는 없습니다 ({_GROUPS[_g]})")
_gp = Counter()
for n, p in denoiser.named_parameters():
    if "lora_" in n:
        _gp[GROUP_OF[n.rsplit(".lora_", 1)[0]]] += p.numel()
print(pd.DataFrame([(g, _gc[g], round(_gp[g] / 1e6, 2), FIXED["learning_rate"] * FIXED["lr_mult"][g]) for g in FIXED["lr_mult"] if g in _gc],
                   columns=["묶음", "층 수", "파라미터(백만)", "학습률"]).to_string(index=False))
LORA_PARAMS_M = round(sum(p.numel() for p in lora_params) / 1e6, 2)
"""

V2["optimizer"] = r"""
_param_groups = {}
for n, p in denoiser.named_parameters():
    if "lora_" in n:
        _param_groups.setdefault(GROUP_OF[n.rsplit(".lora_", 1)[0]], []).append(p)
optimizer = torch.optim.AdamW([dict(params=_param_groups[g], lr=FIXED["learning_rate"] * FIXED["lr_mult"][g], group=g)
                               for g in FIXED["lr_mult"] if g in _param_groups],   # base 가 첫 묶음 (진행 막대·기록의 학습률)
                              lr=FIXED["learning_rate"], betas=FIXED["betas"], weight_decay=FIXED["weight_decay"], eps=FIXED["eps"])
"""

V2["pres_fn"] = r'''
_captured = []


def _capture(module, args, kwargs, out):
    out = getattr(out, "sample", out)
    if isinstance(out, tuple):            # return_dict=False → (출력,)
        out = out[0]
    if isinstance(out, list):             # Z-Image 는 이미지마다 텐서 하나씩인 목록을 낸다
        out = torch.cat([t.reshape(-1) for t in out])
    elif A.FAMILY == "qwen_image21":      # 글+그림 토큰을 함께 내므로, 어댑터처럼 넣은 그림 토큰 수만큼 뒤쪽만 쓴다
        out = out[:, -kwargs["hidden_states"].shape[1]:]
    _captured.append(out)


def denoiser_output(x0, cond, noise, sigma):
    """어댑터의 forward_loss 를 그대로 돌리고, 디노이저가 낸 예측을 가로챈다 (손실 값은 쓰지 않음).
    두 번 모두 같은 입력이라 묶기(packing)·부호가 같아, 디노이저 출력끼리 바로 비교할 수 있다."""
    _captured.clear()
    _h = denoiser.register_forward_hook(_capture, with_kwargs=True)
    try:
        A.forward_loss(pipe, denoiser, x0, cond, noise, sigma)
    finally:
        _h.remove()
    if len(_captured) != 1:
        raise RuntimeError(f"디노이저가 {len(_captured)}번 불렸습니다 (1번이어야 함)")
    out = _captured.pop()
    if out.numel() != x0.numel():         # 예측은 잠재값과 크기가 같아야 한다 (글 토큰이 섞이거나 일부만 남으면 다름)
        raise RuntimeError(f"디노이저 출력 {tuple(out.shape)} 의 크기가 잠재값 {tuple(x0.shape)} 와 다릅니다")
    return out


def preservation_loss(x0, cond, noise, sigma):
    """보존 손실: LoRA 를 켠 예측과 끈 예측(베이스 모델)의 차이"""
    denoiser.disable_adapters()          # peft 가 LoRA 파라미터의 requires_grad 를 끈다
    try:
        with torch.no_grad():
            prior = denoiser_output(x0, cond, noise, sigma).detach()
    finally:
        denoiser.enable_adapters()       # 다시 켜면 requires_grad 도 돌아온다
    pred = denoiser_output(x0, cond, noise, sigma)
    return torch.mean((pred.float() - prior.float()) ** 2)
'''

V2["drift_init"] = r"""
drift_history = {}
"""

V2["drift"] = r"""
drift_history[step] = generate_images(drift_conds, DRIFT_SEEDS)   # v2: 트리거 없는 프롬프트
for img, seed in zip(drift_history[step], DRIFT_SEEDS):
    img.save(RUN_DIR / f"drift_step{step:05d}_seed{seed}.png")
"""

V2["step_init"] = r"""
step_pres = 0.0
"""

V2["train_step"] = r"""
loss = A.forward_loss(pipe, denoiser, x0, to_dev(caption_conds[i]), noise, sigma)
if not torch.isfinite(loss):
    break
(loss / FIXED["grad_accum_steps"]).backward()   # 보존 계산 전에 역전파해 활성값 메모리를 비운다
step_loss += float(loss) / FIXED["grad_accum_steps"]
if pres_conds is not None:
    pres_loss = preservation_loss(x0, to_dev(pres_conds[i]), noise, sigma)
    if not torch.isfinite(pres_loss):
        loss = pres_loss                          # 아래에서 NaN 단계로 처리
        break
    (FIXED["preservation_weight"] * pres_loss / FIXED["grad_accum_steps"]).backward()
    step_pres += float(pres_loss) / FIXED["grad_accum_steps"]
"""

V2["history_pres"] = r"""
history[-1]["pres_loss"] = step_pres if pres_conds is not None else float("nan")
"""

V2["drift_md"] = r"""
### 5-2-1. 그림체 변화 확인 (v2, 트리거 없는 프롬프트)

검증 프롬프트의 트리거를 `CLASS_WORD` 로 바꿔 같은 시드로 그린 그림이다. 학습이 그림체를 바꾸지 않았다면 모든 단계가 0 단계(베이스 모델)와 거의 같아야 한다.

- 숫자는 0 단계 그림과의 PSNR(dB)이다. 높을수록 0 단계와 비슷하고, 픽셀이 완전히 같으면 '0 단계와 같음'(기록은 null)이다.
- 구도만 조금 바뀌어도 숫자가 크게 떨어지므로, 숫자보다 그림을 보고 판단한다. 숫자는 단계끼리 비교하는 데 쓴다.
- 보존 손실(DOP)을 켰다면 아래 그래프에 보존 손실도 함께 보인다. 보존 손실이 계속 커지면 LoRA 가 트리거 없는 문장까지 바꾸고 있다는 뜻이다.
"""

V2["drift_code"] = r'''
def psnr(a, b):
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return None if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 2)


d_steps = sorted(drift_history)
DRIFT_PSNR = {s: [psnr(im, im0) for im, im0 in zip(drift_history[s], drift_history[0])] for s in d_steps if s}
fig, axes = plt.subplots(len(d_steps), len(DRIFT_PROMPTS), figsize=(3 * len(DRIFT_PROMPTS), 3.3 * len(d_steps)), squeeze=False)
for r, s in enumerate(d_steps):
    for c, img in enumerate(drift_history[s]):
        ax = axes[r][c]; ax.imshow(img); ax.axis("off")
        ax.set_title((DRIFT_PROMPTS[c][:30] + "\n" if r == 0 else "") + (("0 단계와 같음" if DRIFT_PSNR[s][c] is None else f"{DRIFT_PSNR[s][c]} dB") if s else "step 0"), fontsize=7)
        if c == 0:
            ax.text(-0.05, 0.5, f"step {s}", transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9)
plt.suptitle(f"트리거 없는 프롬프트 ('{USER['CLASS_WORD']}'): 0 단계와 같을수록 그림체가 유지된 것"); plt.tight_layout(); plt.show()

h = pd.DataFrame(history)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
_ms = [float(np.mean([v for v in DRIFT_PSNR[s] if v is not None] or [float("nan")])) for s in d_steps if s]
axes[0].plot([s for s in d_steps if s], _ms, marker="o", color="#7B3342"); axes[0].set_title("0 단계와의 PSNR 평균 (dB)")
if "pres_loss" in h and h.pres_loss.notna().any():
    axes[1].plot(h.step, h.pres_loss, alpha=0.35); axes[1].plot(h.step, h.pres_loss.ewm(alpha=0.05).mean())
    axes[1].set_title("보존 손실 (LoRA 켠 출력 - 베이스 출력)")
else:
    axes[1].axis("off"); axes[1].text(0.5, 0.5, "보존 학습 끔", ha="center")
plt.tight_layout(); plt.show()
print("단계별 PSNR(dB):", DRIFT_PSNR)
'''

V2["summary"] = r"""
notebook="v2", target_scope=USER["TARGET_SCOPE"], preservation=USER["PRESERVATION"], class_word=USER["CLASS_WORD"],
lora_groups=dict(Counter(GROUP_OF.values())), lora_params_m=LORA_PARAMS_M,
final_pres_loss=float(pd.DataFrame(history).pres_loss.tail(20).mean()) if history and USER["PRESERVATION"] == "dop" else None,
drift_psnr={int(s): v for s, v in DRIFT_PSNR.items()},
"""

V2["comfy"] = r"""
- v2 는 v1 보다 많은 층을 저장하므로 `lora key not loaded` 경고를 꼭 확인한다. v2 가 더한 층은 모두 블록 안이고 ComfyUI 코드상으로는 읽히지만, 실제로 불러 보지는 않았다.
- 범위를 넓힌 LoRA 는 강도 1.0 에서 v1 보다 세게 들 수 있다. 0.6~1.0 사이도 함께 그려 본다.
"""

V2["test_cmd"] = r"""
LORA_NB_TEST=qwen_image jupyter nbconvert --to notebook --execute lora_training_v2.ipynb --output test_run.ipynb
LORA_NB_TEST=qwen_image LORA_TARGET_SCOPE=attn_mlp LORA_PRESERVATION=off jupyter nbconvert --to notebook --execute lora_training_v2.ipynb --output test_run.ipynb
"""

V2["run_name"] = r"""
RUN_NAME=os.environ.get("LORA_RUN_NAME", "nkgreen_v2"),
"""

V2["turbo_note"] = r"""
if "turbo" in p:   # v2: 계열과 상관없이 경로에 turbo 가 있으면 (예: Qwen-Image-2.1-Turbo, sdxl-turbo)
    notes.append("Turbo(증류) 모델을 학습하면 적은 step 생성이 깨진다. 증류하지 않은 모델로 학습하고 "
                 "(예: Z-Image 는 base, Krea 2 는 Raw, Qwen-Image 2.1 은 2.1), 만든 LoRA 는 Turbo 에서 쓴다.")
"""

V2["model_path"] = r"""
MODEL_PATH=os.environ.get("LORA_MODEL_PATH", "/home/mgs/models/diffusers/Qwen-Image-2512"),   # 메인 서버에서 Qwen 학습 때 쓴 위치
"""

V2["run_dir_guard"] = r"""
if not TEST_FAMILY and RUN_DIR.is_dir() and any(RUN_DIR.iterdir()):   # v2: 같은 커널에서 학습 셀만 다시 돌려도 덮어쓰지 않게
    raise FileExistsError(f"{RUN_DIR} 에 이전 실행 결과가 있습니다. RUN_NAME 을 바꾸거나, 지워도 되는 폴더면 지운 뒤 0-2 절부터 다시 실행합니다.")
"""
