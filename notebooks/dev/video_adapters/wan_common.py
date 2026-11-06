"""Shared Wan 2.1 / 2.2 code for the video LoRA adapters (diffusers 0.42.1, peft 0.21.2).

diffusers v0.42.1 ships no Wan training script, so this mirrors the inference code the LoRA must match:
pipelines/wan/pipeline_wan.py (WanPipeline), pipeline_wan_i2v.py (WanImageToVideoPipeline),
models/transformers/transformer_wan.py, and the original Wan-Video/Wan2.1 and Wan2.2 wan/configs.
  * text: UMT5 through pipe.encode_prompt(max_sequence_length=512) (encode_prompt defaults to 226; __call__ and the
    original text_len use 512); rows after the real token count are zeros.
  * pixels: px = 2 * (uint8 / 255) - 1 as VaeImageProcessor, shape (1, 3, F, H, W), F = 4k + 1.
  * latents: x0 = (vae.encode(px).latent_dist.mode() - latents_mean) * (1 / latents_std), i.e. (z - mean) / std written
    the way prepare_latents computes it; inverse as the pipeline decode: z = x0 / (1 / latents_std) + latents_mean.
  * flow matching: x_t = (1 - s) * x0 + s * noise, target = noise - x0 (UniPC flow_prediction: x0 = x_t - s * target);
    transformer timestep t = 1000 * s as float (the pipeline passes UniPC's int64(1000 * s)).
  * "i2v_concat" (Wan2.2 I2V-A14B): hidden_states = cat([x_t, mask4, normalized latent of cat([frame0, zeros(F-1)])]),
    the condition built exactly as WanImageToVideoPipeline.prepare_latents.
  * "ti2v" (Wan2.2 TI2V-5B, expand_timesteps): per-token timestep (mask[0, 0][:, ::2, ::2] * t).flatten(); in I2V mode
    the first latent frame is the clean x0 (timestep 0 on its tokens) and is left out of the loss.
  * A14B: `transformer` (high noise) runs while t >= boundary_ratio * 1000, `transformer_2` (low noise) below.
  * ComfyUI export: original Wan names, the inverse of diffusers _convert_non_diffusers_wan_lora_to_diffusers for
    block linears, no alpha keys (train with lora_alpha == rank -> scale 1 in ComfyUI and in diffusers).
  * reload: load_lora writes the file's fp32 values back into the loaded adapter (peft casts it to the bf16 base
    dtype), so a reload is bit-exact against the fp32 LoRA trained on a bf16 base.
  * ftfy: encode_prompts requires it (diffusers' Wan prompt cleaning; WanImageToVideoPipeline fails without it).
"""
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLWan,
    DiffusionPipeline,
    UniPCMultistepScheduler,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
)
from diffusers.utils import is_ftfy_available
from safetensors.torch import save_file

TEXT_LEN = 512  # wan/configs/shared_config.py text_len
NUM_TRAIN_TIMESTEPS = 1000
TINY_T5 = "hf-internal-testing/tiny-random-t5"
# Wan2.2 wan/configs/shared_config.py sample_neg_prompt, verbatim (the Wan2.1 shared_config.py has the same string).
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
    "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

_QKVO = (("to_q", "q"), ("to_k", "k"), ("to_v", "v"), ("to_out.0", "o"))
_ATTN = [f"{a}.{d}" for a in ("attn1", "attn2") for d, _ in _QKVO]
# peft target_modules suffixes, block linears only (musubi-tuner / DiffSynth q,k,v,o,ffn.0,ffn.2 set).
LORA_TARGETS = {
    "all": _ATTN + ["ffn.net.0.proj", "ffn.net.2"],
    "attn": _ATTN,
    "kvo": [f"{a}.{d}" for a in ("attn1", "attn2") for d in ("to_k", "to_v", "to_out.0")],  # queries untouched
}
# diffusers module inside blocks.N -> original Wan / ComfyUI module
COMFY_NAMES = {f"{a}.{d}": f"{n}.{o}" for a, n in (("attn1", "self_attn"), ("attn2", "cross_attn")) for d, o in _QKVO}
COMFY_NAMES.update({"ffn.net.0.proj": "ffn.0", "ffn.net.2": "ffn.2"})
_LORA_KEY = re.compile(r"(?:transformer\.)?blocks\.(\d+)\.(.+)\.(lora_A|lora_B)\.weight")


# ----------------------------------------------------------------------------------------------- pipelines
def tiny_pipeline(pipeline_class, dtype, *, in_channels=16, out_channels=16, z_dim=16, two_experts=False,
                  boundary_ratio=None, expand_timesteps=False):
    """Random tiny Wan pipeline. Configs from diffusers v0.42.1 tests/pipelines/wan/test_wan_22.py and
    test_wan_22_image_to_video.py (get_dummy_components); UniPC flow_prediction / flow_shift 3.0 as in those tests
    (the Wan2.1 / A14B -Diffusers repos ship 3.0, TI2V-5B ships 5.0; generate() always sets the shift).
    transformer_2 is seeded differently so the two experts differ, as the real ones do."""
    from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

    torch.manual_seed(0)
    vae_kw = dict(base_dim=3, z_dim=z_dim, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                  temperal_downsample=[False, True, True])
    if z_dim == 48:  # TI2V-5B VAE: 2x2 patchify, 16x spatial
        vae_kw.update(in_channels=12, out_channels=12, is_residual=True, patch_size=2, latents_mean=[0.0] * 48,
                      latents_std=[1.0] * 48, scale_factor_spatial=16, scale_factor_temporal=4)
    vae = AutoencoderKLWan(**vae_kw)
    torch.manual_seed(0)
    scheduler = UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)
    text_encoder = T5EncoderModel(AutoConfig.from_pretrained(TINY_T5)).eval()  # eval(): T5 dropout off
    tokenizer = AutoTokenizer.from_pretrained(TINY_T5)

    def tx(seed):
        torch.manual_seed(seed)
        return WanTransformer3DModel(
            patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=12, in_channels=in_channels,
            out_channels=out_channels, text_dim=32, freq_dim=256, ffn_dim=32, num_layers=2, cross_attn_norm=True,
            qk_norm="rms_norm_across_heads", rope_max_seq_len=32,
        )

    pipe = pipeline_class(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, scheduler=scheduler,
                          transformer=tx(0), transformer_2=tx(1) if two_experts else None,
                          boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)
    if dtype != torch.float32:  # as load_pipeline: VAE fp32, transformer _keep_in_fp32_modules fp32, rest in dtype
        text_encoder.to(dtype=dtype)
        for m in [m for m in (pipe.transformer, pipe.transformer_2) if m is not None]:
            for name, p in m.named_parameters():
                keep = any(k in name.split(".") for k in m._keep_in_fp32_modules or [])
                p.data = p.data.to(torch.float32 if keep else dtype)
    return pipe


def check_variant(family, where, index, transformers, vae, *, two_experts, in_channels, z_dim, expand_timesteps):
    """Raise ValueError when config dicts (model_index / transformer(_2) / vae) are not the variant `family` trains."""
    bad = []
    if index.get("_class_name", "WanPipeline") not in ("WanPipeline", "WanImageToVideoPipeline"):
        bad.append(f"pipeline is {index.get('_class_name')}, not a Wan pipeline")
    br = index.get("boundary_ratio")
    has1, has2 = transformers.get("transformer") is not None, transformers.get("transformer_2") is not None
    if two_experts and (br is None or not (has1 and has2)):
        bad.append(f"needs two experts (transformer + transformer_2) and boundary_ratio; found transformer_2={has2}, "
                   f"boundary_ratio={br} (a single-expert Wan2.1 / TI2V-5B folder?)")
    if not two_experts and (br is not None or has2 or not has1):
        bad.append(f"needs one transformer and no boundary_ratio; found transformer_2={has2}, boundary_ratio={br} "
                   "(a two-expert Wan2.2 A14B folder?)")
    enc = (index.get("image_encoder") or [None, None])[-1]  # model_index entry / pipe.config: (library, class)
    if enc is not None:
        bad.append(f"image_encoder={enc} (Wan2.1 I2V with CLIP image tokens, Wan2.2 has none)")
    for name, c in transformers.items():
        if c is None:
            continue
        if c.get("_class_name", "WanTransformer3DModel") != "WanTransformer3DModel":
            bad.append(f"{name} is {c.get('_class_name')}")
        if c.get("in_channels") != in_channels:
            bad.append(f"{name}.in_channels={c.get('in_channels')}, this family needs {in_channels} "
                       "(16 T2V / Wan2.1, 36 I2V, 48 TI2V-5B)")
        if c.get("image_dim") is not None:
            bad.append(f"{name}.image_dim={c.get('image_dim')} (Wan2.1 I2V with CLIP image tokens)")
    if vae.get("z_dim") != z_dim:
        bad.append(f"vae.z_dim={vae.get('z_dim')}, this family needs {z_dim} (16 Wan2.1 VAE, 48 Wan2.2 TI2V-5B VAE)")
    if bool(index.get("expand_timesteps") or False) != expand_timesteps:
        bad.append(f"expand_timesteps={index.get('expand_timesteps')}, this family needs {expand_timesteps}")
    if bad:
        raise ValueError(f"{family}: {where} is the wrong Wan variant: " + "; ".join(bad))


def check_pipeline(pipe, family, **spec):
    tx = {n: (dict(getattr(pipe, n).config) if getattr(pipe, n, None) is not None else None)
          for n in ("transformer", "transformer_2")}
    check_variant(family, type(pipe).__name__, dict(pipe.config), tx, dict(pipe.vae.config), **spec)


def read_configs(model_path):
    """model_index.json, transformer(_2)/config.json, vae/config.json only (cheap even for a hub repo id)."""
    index = dict(DiffusionPipeline.load_config(model_path))
    present = {n for n in ("transformer", "transformer_2", "vae") if (index.get(n) or [None])[0]}
    tx = {n: (dict(WanTransformer3DModel.load_config(model_path, subfolder=n)) if n in present else None)
          for n in ("transformer", "transformer_2")}
    vae = dict(AutoencoderKLWan.load_config(model_path, subfolder="vae")) if "vae" in present else {}
    return index, tx, vae


def load_pipeline(model_path, dtype, pipeline_class, family, **spec):
    """Variant check on the JSON configs first (before 2 x 57 GB of A14B weights), then from_pretrained with the VAE in
    fp32 (as the diffusers Wan examples load it) and everything else in `dtype`, then the same check on the objects."""
    check_variant(family, model_path, *read_configs(model_path), **spec)
    pipe = pipeline_class.from_pretrained(model_path, dtype={"vae": torch.float32, "default": dtype})
    check_pipeline(pipe, family, **spec)
    return pipe


def sigma_range(pipe, expert, experts):
    if expert not in experts:
        raise ValueError(f"unknown expert {expert!r}, expected one of {experts}")
    if len(experts) == 1:
        return (0.0, 1.0)
    b = pipe.config.boundary_ratio
    if b is None:
        raise ValueError("two-expert family but pipe.config.boundary_ratio is None")
    return (float(b), 1.0) if expert == experts[0] else (0.0, float(b))


# ----------------------------------------------------------------------------------------------- encoding
def require_ftfy():
    """Raise ImportError unless diffusers found ftfy. diffusers 0.42.1 cleans Wan prompts with ftfy.fix_text:
    WanImageToVideoPipeline calls it unconditionally (NameError without ftfy), WanPipeline silently skips it (cached
    embeddings would then differ from a run with ftfy). diffusers checks at import, so install, then restart."""
    if not is_ftfy_available():
        raise ImportError("ftfy is required for diffusers' Wan prompt cleaning (WanImageToVideoPipeline raises "
                          "NameError without it, WanPipeline silently skips ftfy.fix_text): pip install ftfy, then "
                          "restart the Python kernel")


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    require_ftfy()
    out = []
    for p in prompts:  # one at a time: (1, 512, D), zeros after the real tokens
        emb, _ = pipe.encode_prompt(prompt=[p], do_classifier_free_guidance=False, max_sequence_length=TEXT_LEN,
                                    device=device, dtype=dtype)
        out.append({"prompt_embeds": emb.cpu()})
    return out


def to_pixels(frames):
    """np.uint8 (F, H, W, 3) -> float32 (1, 3, F, H, W) in [-1, 1], bit-identical to VaeImageProcessor."""
    a = np.asarray(frames)
    if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"frames must be np.uint8 (F, H, W, 3), got {a.dtype} {a.shape}")
    px = 2.0 * torch.from_numpy(a.astype(np.float32) / 255.0) - 1.0
    return px.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def latent_stats(vae, device):
    """(latents_mean, 1 / latents_std) as (1, z, 1, 1, 1) float32, built exactly like prepare_latents does."""
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, torch.float32)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, torch.float32)
    return mean, inv_std


def normalize(vae, z):
    mean, inv_std = latent_stats(vae, z.device)
    return (z.to(torch.float32) - mean) * inv_std


def denormalize(vae, x0):
    mean, inv_std = latent_stats(vae, x0.device)
    return x0.to(torch.float32) / inv_std + mean


@torch.no_grad()
def vae_mode(vae, px, device):
    """Posterior mode (retrieve_latents(..., sample_mode="argmax")) as float32."""
    return vae.encode(px.to(device=device, dtype=vae.dtype)).latent_dist.mode().to(torch.float32)


@torch.no_grad()
def i2v_condition(pipe, first_px, num_frames, device):
    """(1, 4 + z, f, h, w) float32 = cat([mask4, normalized latent of cat([frame0, zeros(F - 1)])]), exactly
    WanImageToVideoPipeline.prepare_latents for batch 1 without last_image. first_px: (1, 3, 1, H, W) in [-1, 1]."""
    vae, tf, sf = pipe.vae, pipe.vae_scale_factor_temporal, pipe.vae_scale_factor_spatial
    hh, ww = first_px.shape[-2:]
    h, w = hh // sf, ww // sf
    video = torch.cat([first_px, first_px.new_zeros(1, 3, num_frames - 1, hh, ww)], dim=2)
    lat = normalize(vae, vae_mode(vae, video, device))
    mask = torch.ones(1, 1, num_frames, h, w)
    mask[:, :, 1:] = 0
    mask = torch.cat([torch.repeat_interleave(mask[:, :, 0:1], dim=2, repeats=tf), mask[:, :, 1:]], dim=2)
    mask = mask.view(1, -1, tf, h, w).transpose(1, 2).to(lat.device)
    return torch.cat([mask, lat], dim=1)


@torch.no_grad()
def encode_video(pipe, frames, device, *, conditioning, multiple, condition=False):
    """{"x0": (1, z, f, h, w)} float32 CPU; plus {"cond"} for "i2v_concat" with condition=True ("ti2v" conditions on
    x0[:, :, :1] itself, "t2v" has no condition). The VAE runs in its own dtype on `device`."""
    px = to_pixels(frames)
    n, hh, ww = px.shape[2:]
    tf = pipe.vae_scale_factor_temporal
    if n % tf != 1 or hh % multiple or ww % multiple:
        raise ValueError(f"frames {n}x{hh}x{ww}: need F = {tf}k+1 and H, W multiples of {multiple}")
    out = {"x0": normalize(pipe.vae, vae_mode(pipe.vae, px, device)).cpu()}
    if condition and conditioning == "i2v_concat":
        out["cond"] = i2v_condition(pipe, px[:, :, :1], n, device).cpu()
    return out


# ----------------------------------------------------------------------------------------------- loss
def flow_loss(denoiser, x0, noise, sigma, text, *, conditioning, mode, cond=None):
    """One denoiser call; fp32 MSE against noise - x0 (see module docstring for the three conditioning types)."""
    ok = {"t2v": ("t2v",), "i2v_concat": ("i2v",), "ti2v": ("t2v", "i2v")}
    if mode not in ok.get(conditioning, ()):
        raise ValueError(f"mode {mode!r} is not available for conditioning {conditioning!r}")
    dev = x0.device
    wdtype = getattr(denoiser, "dtype", None) or next(denoiser.parameters()).dtype
    x0 = x0.to(torch.float32)
    noise = noise.to(device=dev, dtype=torch.float32)
    b, _, f, h, w = x0.shape
    if mode == "i2v" and f < 2:
        raise ValueError("mode 'i2v' needs a clip with at least 2 latent frames (F >= 5)")
    sig = sigma.to(device=dev, dtype=torch.float32).reshape(-1)
    s = sig.view(-1, 1, 1, 1, 1)
    x_t = (1.0 - s) * x0 + s * noise
    target = noise - x0
    t = (sig * NUM_TRAIN_TIMESTEPS).expand(b)  # float, same unit as the pipeline's int64 UniPC timesteps
    first = 0
    if conditioning == "i2v_concat":
        if cond is None or "cond" not in cond:
            raise ValueError("i2v_concat needs cond from encode_video(..., condition=True)")
        c = cond["cond"].to(device=dev, dtype=torch.float32)
        if c.shape[2:] != x0.shape[2:]:
            raise ValueError(f"cond {tuple(c.shape)} does not match x0 {tuple(x0.shape)}")
        inp, timestep = torch.cat([x_t, c.expand(b, -1, -1, -1, -1)], dim=1), t
    elif conditioning == "ti2v":
        pt, ph, pw = getattr(getattr(denoiser, "config", None), "patch_size", (1, 2, 2))
        m = torch.ones(1, 1, f, h, w, device=dev)
        if mode == "i2v":
            m[:, :, 0] = 0  # clean first latent frame
            first = 1
        inp = (1 - m) * x0 + m * x_t
        timestep = m[0, 0][::pt, ::ph, ::pw].flatten()[None] * t[:, None]  # (b, tokens), 0 on the clean frame
    else:
        inp, timestep = x_t, t
    emb = text["prompt_embeds"].to(device=dev, dtype=wdtype)
    if emb.shape[0] != b:
        emb = emb.expand(b, -1, -1)
    pred = denoiser(hidden_states=inp.to(wdtype), timestep=timestep, encoder_hidden_states=emb, return_dict=False)[0]
    return F.mse_loss(pred[:, :, first:].float(), target[:, :, first:])


# ----------------------------------------------------------------------------------------------- LoRA files
def lora_to_comfy(peft_state_dict):
    """peft names ("blocks.0.attn1.to_q.lora_A.weight", optional "transformer.") -> original Wan / ComfyUI names
    ("diffusion_model.blocks.0.self_attn.q.lora_A.weight"). Raises on anything else; writes no alpha keys."""
    out, parts = {}, {}
    for k, v in peft_state_dict.items():
        m = _LORA_KEY.fullmatch(k)
        if not m or m.group(2) not in COMFY_NAMES:
            raise ValueError(f"lora_to_comfy: cannot map {k!r}; only lora_A/lora_B weights of the block linears "
                             f"{sorted(COMFY_NAMES)}")
        mod = f"diffusion_model.blocks.{m.group(1)}.{COMFY_NAMES[m.group(2)]}"
        key = f"{mod}.{m.group(3)}.weight"
        if key in out:
            raise ValueError(f"lora_to_comfy: {k!r} maps to {key!r} twice")
        out[key] = v
        parts.setdefault(mod, set()).add(m.group(3))
    unpaired = [mod for mod, ab in parts.items() if ab != {"lora_A", "lora_B"}]
    if not out or unpaired:
        raise ValueError(f"lora_to_comfy: empty state dict or lora_A/lora_B missing for {unpaired[:3]}")
    return out


def save_lora(peft_state_dict, path, metadata):
    """lora_to_comfy(...) as fp32 contiguous safetensors (one file per expert); metadata values become strings."""
    sd = {k: v.detach().to(device="cpu", dtype=torch.float32).contiguous()
          for k, v in lora_to_comfy(peft_state_dict).items()}
    meta = {"format": "pt", **{str(k): str(v) for k, v in (metadata or {}).items()}}
    if "lora_adapter_metadata" in meta:  # diffusers would read it as the LoraConfig and override alpha == rank
        raise ValueError("metadata key 'lora_adapter_metadata' is reserved by diffusers")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    save_file(sd, path, metadata=meta)
    return path


def load_lora(pipe, path, expert, adapter_name):
    """pipe.load_lora_weights(path, adapter_name=..., load_into_transformer_2=(expert == "transformer_2")), then the
    file's exact fp32 values written back into the new adapter. peft casts a loaded adapter to its base layer's dtype
    (bf16 on the GPU pipeline) while training keeps the LoRA fp32 on a bf16 base; without the write-back the reload is
    a bf16-rounded copy (outputs off by a few % of the LoRA effect), with it the reload is bit-exact."""
    if expert not in ("transformer", "transformer_2") or getattr(pipe, expert, None) is None:
        raise ValueError(f"the pipeline has no expert {expert!r}")
    den = getattr(pipe, expert)
    before = set(getattr(den, "peft_config", None) or {})
    pipe.load_lora_weights(path, adapter_name=adapter_name, load_into_transformer_2=(expert == "transformer_2"))
    name = adapter_name
    if name is None:  # diffusers named it (default_<n>)
        (name,) = set(den.peft_config) - before
    # The values load_lora_weights loaded: lora_state_dict applies the same conversion (ComfyUI -> diffusers names),
    # and load_lora_adapter keeps only the "transformer." keys, prefix removed, for either expert
    # (load_into_transformer_2 loads the same "transformer." keys). peft names the parameter
    # <module>.lora_A.<adapter>.weight; matching the adapter right before ".weight" keeps adapter names that look like
    # module names ("attn1", "0") from matching elsewhere.
    sd = pipe.lora_state_dict(path)
    sd = sd[0] if isinstance(sd, tuple) else sd
    vals = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    pat = re.compile(rf"(.+)\.(lora_A|lora_B)\.{re.escape(name)}\.(weight|bias)")
    done, miss = 0, []
    for n, p in den.named_parameters():
        m = pat.fullmatch(n)
        if m is None:
            continue
        v = vals.get(f"{m.group(1)}.{m.group(2)}.{m.group(3)}")
        if v is None or v.shape != p.shape:
            miss.append(n)
            continue
        p.data = v.to(device=p.device, dtype=torch.float32, copy=True)  # own memory, never the file's buffer
        done += 1
    if miss or not done:
        raise ValueError(f"load_lora: {path} has no matching value for {miss[:3] or 'any parameter'} of adapter "
                         f"{name!r} in {expert}")


# ----------------------------------------------------------------------------------------------- sampling
def as_i2v(pipe):
    """The same modules as a WanImageToVideoPipeline (TI2V-5B ships as WanPipeline). Built directly because
    from_pipe() casts every shared module to its default dtype (float32)."""
    if isinstance(pipe, WanImageToVideoPipeline):
        return pipe
    return WanImageToVideoPipeline(**pipe.components, boundary_ratio=pipe.config.boundary_ratio,
                                   expand_timesteps=pipe.config.expand_timesteps)


def with_flow_shift(scheduler, shift):
    """Copy of a flow-sigma scheduler (UniPC as in the -Diffusers repos) with flow_shift = shift."""
    c = scheduler.config
    if not (c.get("use_flow_sigmas") and c.get("prediction_type") == "flow_prediction" and "flow_shift" in c):
        raise ValueError(f"{type(scheduler).__name__} is not a flow_prediction scheduler with use_flow_sigmas")
    return type(scheduler).from_config(c, flow_shift=float(shift))


@torch.no_grad()
def generate(pipe, text, negative, *, height, width, num_frames, generator, image, steps, shift, guidance, device,
             offload, conditioning):
    """uint8 (T, H, W, 3) through the pipeline with cached embeddings, the scheduler temporarily set to `shift`.
    guidance = (high-noise expert, low-noise expert) -> guidance_scale, guidance_scale_2 (single expert: [0] only).
    offload=True: diffusers model CPU offload on CUDA for this call, hooks removed afterwards; elsewhere a plain call.
    device=None: CUDA when offload=True and CUDA is available (the device enable_model_cpu_offload() picks by itself),
    otherwise the device the modules are on (pipe._execution_device). Without that default, offload=True with the
    modules parked on CPU would silently denoise on CPU."""
    if conditioning == "t2v" and image is not None:
        raise ValueError("text-to-video family: image must be None")
    if conditioning == "i2v_concat" and image is None:
        raise ValueError("image-to-video family: pass the first frame as a PIL image")
    run = as_i2v(pipe) if image is not None else pipe
    g_hi, g_lo = (float(g) for g in guidance)
    two = run.config.boundary_ratio is not None
    if two and g_hi <= 1.0 < g_lo:
        raise ValueError("diffusers WanPipeline runs CFG only when guidance_scale (high-noise expert) > 1")
    if g_hi > 1.0 and negative is None:
        raise ValueError("guidance > 1 needs cached negative embeddings (encode_prompts(pipe, [NEGATIVE_PROMPT]))")
    if device is None:
        device = "cuda" if offload and torch.cuda.is_available() else run._execution_device
    device = torch.device(device)
    den = run.transformer if run.transformer is not None else run.transformer_2
    kw = dict(prompt_embeds=text["prompt_embeds"].to(device, den.dtype), height=height, width=width,
              num_frames=num_frames, num_inference_steps=steps, guidance_scale=g_hi, generator=generator,
              output_type="np")
    if g_hi > 1.0:
        kw["negative_prompt_embeds"] = negative["prompt_embeds"].to(device, den.dtype)
    if two:
        kw["guidance_scale_2"] = g_lo
    if image is not None:
        kw["image"] = image
    sched, hooked = run.scheduler, offload and device.type == "cuda"
    run.scheduler = with_flow_shift(sched, shift)
    try:
        if hooked:
            run.enable_model_cpu_offload(device=device)
        frames = run(**kw).frames[0]
    finally:
        if hooked:
            run.remove_all_hooks()  # modules stay on CPU; the caller re-places them
        run.scheduler = sched
    return (np.asarray(frames) * 255).round().clip(0, 255).astype(np.uint8)
