"""Krea 2 adapter (diffusers 0.42.1).

Mirrors examples/dreambooth/train_dreambooth_lora_krea2.py at tag v0.42.1:
  * latents: vae.encode(pixels[:, :, None]).latent_dist.sample(), then (z - latents_mean) / latents_std
    (the script multiplies by 1/latents_std), kept 5D (B, C, 1, h, w)
  * flow matching: x_t = (1 - s) * x0 + s * noise, target = noise - x0
  * timestep passed to the transformer = timesteps / 1000 = sigma in [0, 1] (the transformer scales by 1e3 itself)
  * 2x2 patch packing -> (B, (h/2)*(w/2), C*4); position_ids from Krea2Pipeline.prepare_position_ids
  * text: stacked Qwen3-VL hidden states (B, 512, num_text_layers, D) + bool mask, fixed length 512
  * LoRA target modules: the script's default list (Krea 2 authors' recommended config)
"""
import torch

from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    Krea2Pipeline,
    Krea2Transformer2DModel,
)

FAMILY = "krea2"
PIPELINE_CLASS = Krea2Pipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor (8) * patch_size (2) for the real model
DEFAULT_STEPS = 28  # pipeline default for the base checkpoint (turbo/distilled: 8 steps, guidance 0.0)
DEFAULT_GUIDANCE = 4.5  # Krea 2 convention: cond + g * (cond - uncond); 0 disables CFG
MAX_SEQUENCE_LENGTH = 512  # official --max_sequence_length default

LORA_TARGET_MODULES = [
    "img_in",
    "final_layer.linear",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "to_gate",
    "ff.up",
    "ff.down",
    "text_fusion.projector",
    "txt_in.linear_1",
    "txt_in.linear_2",
    "time_embed.linear_1",
    "time_embed.linear_2",
    "time_mod_proj",
]


def load_pipeline(model_path: str, dtype: torch.dtype):
    # Works for a local diffusers-format dir (model_index.json) or a HF repo id.
    return Krea2Pipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    # Copied from tests/pipelines/krea2/test_krea2.py::get_dummy_components (v0.42.1).
    from transformers import Qwen2Tokenizer, Qwen3VLConfig, Qwen3VLModel

    torch.manual_seed(0)
    transformer = Krea2Transformer2DModel(
        in_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        timestep_embed_dim=8,
        text_hidden_dim=16,
        num_text_layers=3,
        text_num_attention_heads=2,
        text_num_key_value_heads=1,
        text_intermediate_size=16,
        num_layerwise_text_blocks=1,
        num_refiner_text_blocks=1,
        axes_dims_rope=(4, 2, 2),
        rope_theta=1000.0,
    )
    torch.manual_seed(0)
    z_dim = 4
    vae = AutoencoderKLQwenImage(
        base_dim=z_dim * 6,
        z_dim=z_dim,
        dim_mult=[1, 2, 4],
        num_res_blocks=1,
        temperal_downsample=[False, True],
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=6400,
    )
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": 16,
        },
        vocab_size=152064,
    )
    text_encoder = Qwen3VLModel(config).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    pipe = Krea2Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        text_encoder_select_layers=(0, 1, 2),
    )
    return pipe if dtype == torch.float32 else pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    te = pipe.text_encoder
    orig = next(te.parameters()).device
    te.to(device)
    out = []
    for p in prompts:
        emb, mask = pipe.encode_prompt(prompt=[p], device=device, max_sequence_length=MAX_SEQUENCE_LENGTH)
        out.append({
            "prompt_embeds": emb.to(dtype).cpu(),
            "prompt_embeds_mask": mask.bool().cpu(),
        })
    te.to(orig)
    return out


def _latent_stats(vae, device, dtype):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
    return mean, inv_std


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    if px.ndim == 4:
        px = px.unsqueeze(2)  # (B, 3, 1, H, W) as in the official script's collate_fn
    lat = vae.encode(px).latent_dist.sample()
    mean, inv_std = _latent_stats(vae, lat.device, torch.float32)
    lat = (lat.float() - mean) * inv_std
    return lat.float().cpu()  # (B, C, 1, H/8, W/8)


def _pack(x, p=2):
    # (B, C, 1, H, W) -> (B, (H/p)*(W/p), C*p*p), identical to the script's inlined Krea2Pipeline._pack_latents
    b, c, _, h, w = x.shape
    x = x.permute(0, 2, 1, 3, 4).reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // p) * (w // p), c * p * p)


def _unpack(x, h, w, p=2):
    b, _, ch = x.shape
    x = x.view(b, h // p, w // p, ch // (p * p), p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, ch // (p * p), 1, h, w)


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    p = getattr(pipe, "patch_size", 2)
    wdtype = denoiser.img_in.weight.dtype  # norms stay fp32 via _keep_in_fp32_modules
    if x0.ndim == 4:
        x0, noise = x0.unsqueeze(2), noise.unsqueeze(2)
    b = x0.shape[0]
    s = sigma.to(x0.device, torch.float32).view(b, 1, 1, 1, 1)
    x_t = ((1.0 - s) * x0.float() + s * noise.float()).to(wdtype)
    h, w = x0.shape[-2], x0.shape[-1]

    emb = cond["prompt_embeds"].to(x0.device, wdtype)
    mask = cond["prompt_embeds_mask"].to(x0.device).bool()
    if emb.shape[0] != b:
        emb = emb.expand(b, *emb.shape[1:])
        mask = mask.expand(b, *mask.shape[1:])
    position_ids = Krea2Pipeline.prepare_position_ids(emb.shape[1], h // p, w // p, x0.device)

    pred = denoiser(
        hidden_states=_pack(x_t, p),
        encoder_hidden_states=emb,
        timestep=sigma.to(x0.device, torch.float32).view(b),  # official: fp32 timesteps / 1000 == sigma
        position_ids=position_ids,
        encoder_attention_mask=mask,
        return_dict=False,
    )[0]
    pred = _unpack(pred, h, w, p)
    target = noise.float() - x0.float()
    return torch.mean((pred.float() - target) ** 2)


def save_lora(pipe, lora_state_dict, out_dir):
    import os

    Krea2Pipeline.save_lora_weights(out_dir, transformer_lora_layers=lora_state_dict)
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.transformer.dtype
    kw = dict(
        prompt_embeds=prompt_cond["prompt_embeds"].to(dev, dt),
        prompt_embeds_mask=prompt_cond["prompt_embeds_mask"].to(dev),
    )
    if guidance > 0:
        if negative_cond is not None:
            kw["negative_prompt_embeds"] = negative_cond["prompt_embeds"].to(dev, dt)
            kw["negative_prompt_embeds_mask"] = negative_cond["prompt_embeds_mask"].to(dev)
        elif getattr(pipe, "text_encoder", None) is None:
            raise ValueError("Krea 2 CFG needs negative embeddings (encode '' with encode_prompts) when no text encoder.")
        # else: pipeline encodes "" as the negative prompt itself
    return pipe(
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        max_sequence_length=kw["prompt_embeds"].shape[1],
        output_type="pil",
        **kw,
    ).images[0]
