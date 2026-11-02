"""Stable Diffusion 3 / 3.5 adapter (StableDiffusion3Pipeline + SD3Transformer2DModel, rectified flow).

Mirrors diffusers v0.42.1 examples/dreambooth/train_dreambooth_lora_sd3.py:
  * latents  = (vae.encode(x).latent_dist.sample() - shift_factor) * scaling_factor
  * x_t      = (1 - sigma) * x0 + sigma * noise
  * timestep = sigma * num_train_timesteps (scheduler.timesteps units, 0..1000), passed as-is
  * no packing, no ids, no attention mask, no guidance embedding (SD3 transformer patchifies internally)
  * text: CLIP-L + CLIP-G penultimate hidden states (concat, zero-padded to T5 width) ++ T5 (max_sequence_length=77
    in the training script), pooled = concat of both CLIP pooled projections
  * loss: official default --precondition_outputs=1 -> pred_x0 = model_pred * (-sigma) + x_t, target = x0.
    Set PRECONDITION_OUTPUTS = False to use target = noise - x0 directly (the script's --precondition_outputs=0).
"""
import os
import warnings

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

FAMILY = "sd3"
PIPELINE_CLASS = StableDiffusion3Pipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor 8 * transformer patch_size 2
DEFAULT_STEPS = 28  # StableDiffusion3Pipeline.__call__ default
DEFAULT_GUIDANCE = 7.0  # StableDiffusion3Pipeline.__call__ default
LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]
MAX_SEQUENCE_LENGTH = 77  # official training script default (pipeline inference default is 256)
PRECONDITION_OUTPUTS = True  # official training script default (--precondition_outputs=1)


def load_pipeline(model_path: str, dtype: torch.dtype):
    # works for a local diffusers-format directory (model_index.json) or a HF repo id
    return StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    # same as diffusers tests/pipelines/stable_diffusion_3 get_dummy_components()
    from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

    torch.manual_seed(0)
    transformer = SD3Transformer2DModel(
        sample_size=32, patch_size=1, in_channels=4, num_layers=1, attention_head_dim=8, num_attention_heads=4,
        caption_projection_dim=32, joint_attention_dim=32, pooled_projection_dim=64, out_channels=4,
    )
    clip_cfg = CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1, vocab_size=1000, hidden_act="gelu",
        projection_dim=32,
    )
    torch.manual_seed(0)
    text_encoder = CLIPTextModelWithProjection(clip_cfg)
    torch.manual_seed(0)
    text_encoder_2 = CLIPTextModelWithProjection(clip_cfg)
    torch.manual_seed(0)
    text_encoder_3 = T5EncoderModel(AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5"))
    tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    torch.manual_seed(0)
    vae = AutoencoderKL(
        sample_size=32, in_channels=3, out_channels=3, block_out_channels=(4,), layers_per_block=1,
        latent_channels=4, norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False,
        shift_factor=0.0609, scaling_factor=1.5035,
    )
    pipe = StableDiffusion3Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, text_encoder_3=text_encoder_3,
        tokenizer_3=tokenizer_3, transformer=transformer, image_encoder=None, feature_extractor=None,
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:
        # pipe.encode_prompt == the training script's encode_prompt (clip penultimate layer, zero-pad, concat T5);
        # if text_encoder_3 is None the pipeline substitutes zeros for T5, like the official pipeline.
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=p, prompt_2=None, prompt_3=None, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False, max_sequence_length=MAX_SEQUENCE_LENGTH,
        )
        out.append({
            "prompt_embeds": pe.to("cpu", dtype),
            "pooled_prompt_embeds": pooled.to("cpu", dtype),
        })
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    # official script keeps the VAE in fp32 for latent encoding (train_dreambooth_lora_sd3.py:1237)
    vae = pipe.vae
    orig_dtype = vae.dtype
    vae.to(dtype=torch.float32)
    try:
        lat = vae.encode(pixel_values.to(device, torch.float32)).latent_dist.sample()
        lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
    finally:
        vae.to(dtype=orig_dtype)
    return lat.float().cpu()


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    wdt = next(denoiser.parameters()).dtype
    x0 = x0.to(wdt)
    noise = noise.to(wdt)
    s = sigma.to(x0.device, wdt).view(-1, *([1] * (x0.ndim - 1)))
    x_t = (1.0 - s) * x0 + s * noise
    timesteps = sigma.to(x0.device, torch.float32) * pipe.scheduler.config.num_train_timesteps
    pe = cond["prompt_embeds"].to(x0.device, wdt)
    pooled = cond["pooled_prompt_embeds"].to(x0.device, wdt)
    if pe.shape[0] != x0.shape[0]:
        pe = pe.expand(x0.shape[0], -1, -1)
        pooled = pooled.expand(x0.shape[0], -1)
    model_pred = denoiser(
        hidden_states=x_t, timestep=timesteps, encoder_hidden_states=pe, pooled_projections=pooled,
        return_dict=False,
    )[0]
    if PRECONDITION_OUTPUTS:
        model_pred = model_pred * (-s) + x_t
        target = x0
    else:
        target = noise - x0
    return torch.mean((model_pred.float() - target.float()) ** 2)


def save_lora(pipe, lora_state_dict, out_dir):
    # official script passes get_peft_model_state_dict() output unchanged (peft lora_A/lora_B keys)
    os.makedirs(out_dir, exist_ok=True)
    StableDiffusion3Pipeline.save_lora_weights(
        out_dir, transformer_lora_layers=lora_state_dict
    )
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.transformer.dtype
    kw = dict(
        prompt_embeds=prompt_cond["prompt_embeds"].to(dev, dt),
        pooled_prompt_embeds=prompt_cond["pooled_prompt_embeds"].to(dev, dt),
    )
    if negative_cond is not None:
        kw["negative_prompt_embeds"] = negative_cond["prompt_embeds"].to(dev, dt)
        kw["negative_pooled_prompt_embeds"] = negative_cond["pooled_prompt_embeds"].to(dev, dt)
    elif guidance > 1.0:
        # Official validation runs CFG with the pipeline's empty negative prompt. Encode "" once with the text
        # encoders on whatever device they currently sit on (CPU during training), cache it on the pipe, move to dev.
        neg = getattr(pipe, "_lvw_empty_negative_cond", None)
        if neg is None and pipe.text_encoder is not None:
            te_dev = next(pipe.text_encoder.parameters()).device
            neg = encode_prompts(pipe, [""], te_dev, dt)[0]
            pipe._lvw_empty_negative_cond = neg
        if neg is not None:
            kw["negative_prompt_embeds"] = neg["prompt_embeds"].to(dev, dt)
            kw["negative_pooled_prompt_embeds"] = neg["pooled_prompt_embeds"].to(dev, dt)
        else:
            warnings.warn("sd3.generate: no negative_cond and no text encoder; CFG disabled (guidance=1.0)")
            guidance = 1.0
    return pipe(
        height=height, width=width, num_inference_steps=steps, guidance_scale=guidance, generator=generator,
        max_sequence_length=MAX_SEQUENCE_LENGTH, **kw,
    ).images[0]
