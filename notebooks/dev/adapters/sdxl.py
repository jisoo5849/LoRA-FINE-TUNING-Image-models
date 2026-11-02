"""SDXL adapter (StableDiffusionXLPipeline + UNet2DConditionModel).

Mirrors diffusers v0.42.1 examples/text_to_image/train_text_to_image_lora_sdxl.py:
  - latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor (VAE in fp32, no shift)
  - noise scheduler = DDPMScheduler built from the model's scheduler config (official uses
    DDPMScheduler.from_pretrained(..., subfolder="scheduler")), x_t = add_noise(x0, noise, t)
  - integer timesteps in [0, num_train_timesteps) fed to the UNet
  - target = noise (epsilon) or get_velocity(x0, noise, t) (v_prediction), per scheduler prediction_type
  - added_cond_kwargs = {"text_embeds": pooled (text_encoder_2), "time_ids": orig_size + crop + target_size}
  - prompt_embeds = concat(penultimate hidden states of both CLIP encoders), no attention mask
  - LoRA on UNet attention: to_k, to_q, to_v, to_out.0
"""
import os

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.utils import convert_state_dict_to_diffusers

FAMILY = "sdxl"
PIPELINE_CLASS = StableDiffusionXLPipeline
DENOISER_ATTR = "unet"
PREDICTION = "eps"  # default; actual target follows pipe.scheduler.config.prediction_type
PIXEL_MULTIPLE = 64
DEFAULT_STEPS = 50  # StableDiffusionXLPipeline.__call__ default (official script validation uses 25)
DEFAULT_GUIDANCE = 5.0  # StableDiffusionXLPipeline.__call__ default
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def load_pipeline(model_path: str, dtype: torch.dtype):
    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype)
    return pipe


def tiny_pipeline(dtype: torch.dtype):
    # copied from tests/pipelines/stable_diffusion_xl/test_stable_diffusion_xl.py::get_dummy_components
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(2, 4),
        layers_per_block=2,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,  # 6 * 8 + 32
        cross_attention_dim=64,
        norm_num_groups=1,
    )
    scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        steps_offset=1,
        beta_schedule="scaled_linear",
        timestep_spacing="leading",
    )
    torch.manual_seed(0)
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
        sample_size=128,
    )
    torch.manual_seed(0)
    te_cfg = CLIPTextConfig(
        bos_token_id=0,
        eos_token_id=2,
        hidden_size=32,
        intermediate_size=37,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=5,
        pad_token_id=1,
        vocab_size=1000,
        hidden_act="gelu",
        projection_dim=32,
    )
    text_encoder = CLIPTextModel(te_cfg)
    tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    text_encoder_2 = CLIPTextModelWithProjection(te_cfg)
    tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        image_encoder=None,
        feature_extractor=None,
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=p,
            prompt_2=p,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        out.append({
            "prompt_embeds": pe.to("cpu", dtype),  # (1, 77, 2048) penultimate layers concat
            "pooled_prompt_embeds": pooled.to("cpu", dtype),  # (1, 1280) text_encoder_2 projection
        })
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    orig_dtype = vae.dtype
    # official script keeps the SDXL VAE in fp32 (fp16/bf16 NaN issue) unless a fixed VAE is supplied
    if orig_dtype != torch.float32:
        vae.to(torch.float32)
    try:
        px = pixel_values.to(device=device, dtype=torch.float32)
        lat = vae.encode(px).latent_dist.sample()
        lat = lat * vae.config.scaling_factor
    finally:
        if orig_dtype != torch.float32:
            vae.to(orig_dtype)
    return lat.float().cpu()


def _train_scheduler(pipe):
    sch = getattr(pipe, "_lvw_ddpm", None)
    if sch is None or sch.config.prediction_type != pipe.scheduler.config.prediction_type:
        sch = DDPMScheduler.from_config(pipe.scheduler.config)
        pipe._lvw_ddpm = sch
    return sch


def _time_ids(pipe, x0, device, dtype, cond=None):
    # original_size + crops_coords_top_left + target_size, in pixels.
    # Official script (train_dreambooth_lora_sdxl.py) uses each image's real original size, its random-crop
    # top-left and target_size=(resolution, resolution). The notebook may pass these per sample via optional
    # cond keys "original_size" (h, w) and "crop_top_left" (top, left); otherwise default to the bucketed
    # training size with no crop, which matches how inference sets height/width for aspect-ratio buckets.
    vsf = pipe.vae_scale_factor
    h, w = x0.shape[-2] * vsf, x0.shape[-1] * vsf
    cond = cond or {}
    orig = cond.get("original_size", (h, w))
    crop = cond.get("crop_top_left", (0, 0))
    def _pair(v):
        v = torch.as_tensor(v, dtype=torch.float32).reshape(-1, 2)
        return v.expand(x0.shape[0], 2) if v.shape[0] == 1 else v
    tgt = torch.tensor([[h, w]], dtype=torch.float32).expand(x0.shape[0], 2)
    ids = torch.cat([_pair(orig), _pair(crop), tgt], dim=1)
    return ids.to(device=device, dtype=dtype)


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    sch = _train_scheduler(pipe)
    device = x0.device
    wdtype = next(p for p in denoiser.parameters()).dtype
    T = sch.config.num_train_timesteps
    t = (sigma.float() * (T - 1)).round().long().clamp(0, T - 1).to(device)
    if t.shape[0] != x0.shape[0]:
        t = t.expand(x0.shape[0])
    x0f, nf = x0.float(), noise.float()
    x_t = sch.add_noise(x0f, nf, t)
    pe = cond["prompt_embeds"].to(device, wdtype)
    pooled = cond["pooled_prompt_embeds"].to(device, wdtype)
    if pe.shape[0] != x0.shape[0]:
        pe = pe.expand(x0.shape[0], -1, -1)
        pooled = pooled.expand(x0.shape[0], -1)
    added = {"text_embeds": pooled, "time_ids": _time_ids(pipe, x0, device, wdtype, cond)}
    pred = denoiser(x_t.to(wdtype), t, pe, added_cond_kwargs=added, return_dict=False)[0]
    ptype = sch.config.prediction_type
    if ptype == "epsilon":
        target = nf
    elif ptype == "v_prediction":
        target = sch.get_velocity(x0f, nf, t)
    else:
        raise ValueError(f"Unknown prediction type {ptype}")
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=out_dir,
        unet_lora_layers=convert_state_dict_to_diffusers(lora_state_dict),
        safe_serialization=True,
    )
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.unet.dtype
    pe = prompt_cond["prompt_embeds"].to(dev, dt)
    pooled = prompt_cond["pooled_prompt_embeds"].to(dev, dt)
    if negative_cond is not None:
        npe = negative_cond["prompt_embeds"].to(dev, dt)
        npooled = negative_cond["pooled_prompt_embeds"].to(dev, dt)
    else:
        # force_zeros_for_empty_prompt behaviour of SDXL (no text encoder call needed)
        npe, npooled = torch.zeros_like(pe), torch.zeros_like(pooled)
    if npe.shape[1] != pe.shape[1]:
        L = max(npe.shape[1], pe.shape[1])
        pe = F.pad(pe, (0, 0, 0, L - pe.shape[1]))
        npe = F.pad(npe, (0, 0, 0, L - npe.shape[1]))
    return pipe(
        prompt_embeds=pe,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=npe,
        negative_pooled_prompt_embeds=npooled,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
    ).images[0]
