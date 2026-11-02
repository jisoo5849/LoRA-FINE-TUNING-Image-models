"""Qwen-Image adapter (QwenImagePipeline / QwenImageTransformer2DModel).

Mirrors diffusers v0.42.1 examples/dreambooth/train_dreambooth_lora_qwen_image.py:
  * latents: vae.encode(pixels[:, :, None]).latent_dist.sample(); x0 = (z - latents_mean) / latents_std  (5D B,C,1,h,w)
  * flow matching: x_t = (1 - s) * x0 + s * noise ; target = noise - x0
  * transformer timestep = timesteps / 1000 = sigma (0..1)
  * latents permuted to (B,1,C,h,w) then QwenImagePipeline._pack_latents (2x2 patches), img_shapes=[(1,h/2,w/2)]
  * encoder_hidden_states_mask passed (None when all ones, as encode_prompt does)
  * LoRA targets: ["to_k", "to_q", "to_v", "to_out.0"]
"""
import os

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

FAMILY = "qwen_image"
PIPELINE_CLASS = QwenImagePipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor (8) * 2x2 packing
DEFAULT_STEPS = 50  # QwenImagePipeline.__call__ default num_inference_steps
DEFAULT_GUIDANCE = 4.0  # true_cfg_scale default (Qwen-Image uses true CFG, no guidance embedding)
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
MAX_SEQUENCE_LENGTH = 512  # official script --max_sequence_length default


def load_pipeline(model_path: str, dtype: torch.dtype):
    # Works for a local diffusers-format dir (model_index.json) or a HF repo id.
    return QwenImagePipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    # Same as diffusers tests/pipelines/qwenimage/test_qwenimage.py get_dummy_components()
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    torch.manual_seed(0)
    transformer = QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=3,
        joint_attention_dim=16,
        guidance_embeds=False,
        axes_dims_rope=(8, 4, 4),
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
    torch.manual_seed(0)
    scheduler = FlowMatchEulerDiscreteScheduler()
    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
            "rope_theta": 1000000.0,
        },
        vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 16},
        hidden_size=16,
        vocab_size=152064,
        vision_end_token_id=151653,
        vision_start_token_id=151652,
        vision_token_id=151654,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration(config).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    pipe = QwenImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:  # one at a time -> no cross-prompt padding
        emb, mask = pipe.encode_prompt(prompt=[p], device=device, max_sequence_length=MAX_SEQUENCE_LENGTH)
        if mask is None:
            mask = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        out.append({
            "prompt_embeds": emb.to(dtype=dtype).cpu(),
            "prompt_embeds_mask": mask.to(torch.long).cpu(),
        })
    return out


def _latent_stats(vae, device, dtype):
    z = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean).view(1, z, 1, 1, 1).to(device, dtype)
    std = torch.tensor(vae.config.latents_std).view(1, z, 1, 1, 1).to(device, dtype)
    return mean, std


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    if px.ndim == 4:
        px = px.unsqueeze(2)  # (B,3,1,H,W): Qwen VAE is a video VAE
    lat = vae.encode(px).latent_dist.sample()  # official script samples
    mean, std = _latent_stats(vae, lat.device, lat.dtype)
    lat = (lat - mean) / std
    return lat.float().cpu()  # (B, C, 1, h, w)


def _mask_or_none(mask):
    if mask is None or bool(mask.all()):
        return None
    return mask


def collate_conds(conds):
    """Batch a list of per-prompt cond dicts (each batch 1) into one cond for forward_loss.

    Mirrors concat_prompt_embedding_batches in the official train_dreambooth_lora_qwen_image.py:
    embeds are zero-padded to the longest sequence and padded positions get mask 0.
    """
    if not conds:
        raise ValueError("collate_conds needs at least one cond")
    target = max(c["prompt_embeds"].shape[1] for c in conds)
    embs, masks = [], []
    for c in conds:
        e = c["prompt_embeds"]
        m = c.get("prompt_embeds_mask")
        if m is None:
            m = torch.ones(e.shape[:2], dtype=torch.long, device=e.device)
        pad = target - e.shape[1]
        if pad > 0:
            e = torch.cat([e, e.new_zeros(e.shape[0], pad, e.shape[2])], dim=1)
            m = torch.cat([m, m.new_zeros(m.shape[0], pad)], dim=1)
        embs.append(e)
        masks.append(m)
    return {"prompt_embeds": torch.cat(embs, dim=0), "prompt_embeds_mask": torch.cat(masks, dim=0)}


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    wdtype = next(p for p in denoiser.parameters()).dtype
    device = x0.device
    x0 = x0.to(wdtype)
    noise = noise.to(wdtype)
    bsz, C, _, h, w = x0.shape
    s = sigma.to(device=device, dtype=wdtype).view(-1, 1, 1, 1, 1)
    noisy = (1.0 - s) * x0 + s * noise

    img_shapes = [(1, h // 2, w // 2)] * bsz
    packed = QwenImagePipeline._pack_latents(noisy.permute(0, 2, 1, 3, 4), bsz, C, h, w)

    # cond: batch 1 (broadcast to bsz) or batch == bsz. For per-image captions, build the
    # batched cond with collate_conds() (zero-pad + mask 0, as the official script does).
    emb = cond["prompt_embeds"].to(device=device, dtype=wdtype)
    mask = cond.get("prompt_embeds_mask")
    mask = _mask_or_none(mask.to(device) if mask is not None else None)
    if emb.shape[0] != bsz:
        if emb.shape[0] != 1:
            raise ValueError(
                f"cond batch {emb.shape[0]} != latent batch {bsz}; use collate_conds() to batch per-image conds"
            )
        emb = emb.expand(bsz, -1, -1)
        mask = mask.expand(bsz, -1) if mask is not None else None

    if denoiser.config.guidance_embeds:
        # diffusers' QwenImageTransformer2DModel guidance path is broken (time_text_embed arity);
        # Qwen-Image / Qwen-Image-2512 use guidance_embeds=False.
        raise NotImplementedError("guidance_embeds=True is not supported for qwen_image")
    kwargs = {}

    pred = denoiser(
        hidden_states=packed,
        encoder_hidden_states=emb,
        encoder_hidden_states_mask=mask,
        timestep=sigma.to(device=device, dtype=wdtype),  # == timesteps / 1000
        img_shapes=img_shapes,
        return_dict=False,
        **kwargs,
    )[0]
    vsf = pipe.vae_scale_factor
    pred = QwenImagePipeline._unpack_latents(pred, h * vsf, w * vsf, vsf)
    target = noise - x0
    return torch.mean((pred.float() - target.float()) ** 2)


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    QwenImagePipeline.save_lora_weights(out_dir, transformer_lora_layers=lora_state_dict)
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    device = pipe._execution_device
    dtype = pipe.transformer.dtype

    def prep(c):
        e = c["prompt_embeds"].to(device=device, dtype=dtype)
        m = c.get("prompt_embeds_mask")
        m = m.to(device) if m is not None else None
        return e, m  # pipeline.encode_prompt drops an all-ones mask itself

    pe, pm = prep(prompt_cond)
    kwargs = dict(prompt_embeds=pe, prompt_embeds_mask=pm)
    if negative_cond is not None and guidance is not None and guidance > 1.0:
        ne, nm = prep(negative_cond)
        kwargs.update(negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm, true_cfg_scale=float(guidance))
    else:
        kwargs["true_cfg_scale"] = 1.0
    if pipe.transformer.config.guidance_embeds:
        raise NotImplementedError("guidance_embeds=True is not supported for qwen_image")
    return pipe(
        height=height, width=width, num_inference_steps=steps, generator=generator,
        max_sequence_length=MAX_SEQUENCE_LENGTH, output_type="pil", **kwargs,
    ).images[0]
