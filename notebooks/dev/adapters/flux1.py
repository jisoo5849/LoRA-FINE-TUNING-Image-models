"""FLUX.1 (dev / schnell) adapter, mirroring diffusers v0.42.1
examples/dreambooth/train_dreambooth_lora_flux.py."""
import os

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)

FAMILY = "flux1"
PIPELINE_CLASS = FluxPipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16          # vae_scale_factor (8) * 2 (2x2 latent packing)
DEFAULT_STEPS = 28           # FluxPipeline.__call__ default (FLUX.1-dev). schnell: 4
DEFAULT_GUIDANCE = 3.5       # distilled guidance for FLUX.1-dev. schnell: 0.0 (guidance_embeds=False -> ignored)
TRAIN_GUIDANCE = 3.5         # official --guidance_scale default, used only when transformer.config.guidance_embeds
MAX_SEQUENCE_LENGTH = 512    # official --max_sequence_length default (schnell supports at most 256)
LORA_TARGET_MODULES = [
    "attn.to_k",
    "attn.to_q",
    "attn.to_v",
    "attn.to_out.0",
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "ff.net.0.proj",
    "ff.net.2",
    "ff_context.net.0.proj",
    "ff_context.net.2",
]


def _max_seq_len(pipe):
    # schnell has no guidance embedding and was trained with 256 T5 tokens
    if not pipe.transformer.config.guidance_embeds:
        return 256
    return MAX_SEQUENCE_LENGTH


def load_pipeline(model_path: str, dtype: torch.dtype):
    # works for a local diffusers-format dir (model_index.json) or a HF repo id
    return FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    """Same component classes as the real checkpoint, shrunk (copied from tests/pipelines/flux get_dummy_components).
    guidance_embeds=True so the FLUX.1-dev guidance path is exercised."""
    from transformers import AutoConfig, AutoTokenizer, CLIPTextConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel

    torch.manual_seed(0)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        axes_dims_rope=[4, 4, 8],
        guidance_embeds=True,
    )
    clip_cfg = CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-05,
        num_attention_heads=4, num_hidden_layers=5, pad_token_id=1, vocab_size=1000, hidden_act="gelu",
        projection_dim=32,
    )
    torch.manual_seed(0)
    text_encoder = CLIPTextModel(clip_cfg)
    torch.manual_seed(0)
    t5_cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
    text_encoder_2 = T5EncoderModel(t5_cfg)
    tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    torch.manual_seed(0)
    vae = AutoencoderKL(
        sample_size=32, in_channels=3, out_channels=3, block_out_channels=(4,), layers_per_block=1,
        latent_channels=1, norm_num_groups=1, use_quant_conv=False, use_post_quant_conv=False,
        shift_factor=0.0609, scaling_factor=1.5035,
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = FluxPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, transformer=transformer,
        image_encoder=None, feature_extractor=None,
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    """One dict per prompt: prompt_embeds (1,L,D) T5, pooled_prompt_embeds (1,D') CLIP pooled, text_ids (1,L,3)."""
    for m in (pipe.text_encoder, pipe.text_encoder_2):
        m.to(device)
    out = []
    for p in prompts:
        pe, pooled, text_ids = pipe.encode_prompt(
            prompt=[p], prompt_2=None, device=device, num_images_per_prompt=1,
            max_sequence_length=_max_seq_len(pipe),
        )
        out.append({
            "prompt_embeds": pe.to(dtype).cpu(),
            "pooled_prompt_embeds": pooled.to(dtype).cpu(),
            "text_ids": text_ids.to(dtype).unsqueeze(0).cpu(),  # stored with batch dim 1; squeezed for the model
        })
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    vae.to(device)
    lat = vae.encode(pixel_values.to(device=device, dtype=vae.dtype)).latent_dist.sample()
    lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
    return lat.float().cpu()


def _txt_ids(cond, device, dtype):
    t = cond["text_ids"]
    if t.ndim == 3:
        t = t[0]
    return t.to(device=device, dtype=dtype)


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    dtype = next(denoiser.parameters()).dtype
    device = x0.device
    B, C, h, w = x0.shape
    vae_scale_factor = pipe.vae_scale_factor

    s = sigma.to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    noisy = (1.0 - s) * x0.float() + s * noise.float()
    noisy = noisy.to(dtype)

    packed = FluxPipeline._pack_latents(noisy, batch_size=B, num_channels_latents=C, height=h, width=w)
    img_ids = FluxPipeline._prepare_latent_image_ids(B, h // 2, w // 2, device, dtype)

    if denoiser.config.guidance_embeds:
        guidance = torch.tensor([TRAIN_GUIDANCE], device=device, dtype=torch.float32).expand(B)
    else:
        guidance = None

    pe = cond["prompt_embeds"].to(device=device, dtype=dtype)
    pooled = cond["pooled_prompt_embeds"].to(device=device, dtype=dtype)
    if pe.shape[0] != B:
        pe = pe.expand(B, -1, -1)
        pooled = pooled.expand(B, -1)

    # official: timestep = timesteps / 1000 == sigma (transformer multiplies by 1000 internally)
    timestep = sigma.to(device=device, dtype=dtype).view(-1).expand(B)
    model_pred = denoiser(
        hidden_states=packed,
        timestep=timestep,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=pe,
        txt_ids=_txt_ids(cond, device, dtype),
        img_ids=img_ids,
        return_dict=False,
    )[0]
    model_pred = FluxPipeline._unpack_latents(
        model_pred, height=h * vae_scale_factor, width=w * vae_scale_factor, vae_scale_factor=vae_scale_factor
    )
    target = noise.float() - x0.float()
    return torch.mean((model_pred.float() - target) ** 2)


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    FluxPipeline.save_lora_weights(out_dir, transformer_lora_layers=lora_state_dict, safe_serialization=True)
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    """guidance -> FLUX.1 distilled guidance_scale (ignored by schnell). negative_cond is passed through but only
    used if true_cfg_scale > 1, which the official validation does not do (true_cfg_scale=1.0)."""
    dev = pipe._execution_device
    dtype = pipe.transformer.dtype
    kw = dict(
        prompt_embeds=prompt_cond["prompt_embeds"].to(dev, dtype),
        pooled_prompt_embeds=prompt_cond["pooled_prompt_embeds"].to(dev, dtype),
    )
    if negative_cond is not None:
        kw["negative_prompt_embeds"] = negative_cond["prompt_embeds"].to(dev, dtype)
        kw["negative_pooled_prompt_embeds"] = negative_cond["pooled_prompt_embeds"].to(dev, dtype)
    out = pipe(
        height=height, width=width, num_inference_steps=steps, guidance_scale=guidance,
        true_cfg_scale=1.0, generator=generator, max_sequence_length=_max_seq_len(pipe), **kw,
    )
    return out.images[0]
