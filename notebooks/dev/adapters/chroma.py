"""Chroma (lodestones/Chroma1-HD) adapter.

diffusers 0.42.1 ships NO official Chroma training script; this follows
examples/dreambooth/train_dreambooth_lora_flux.py (Chroma is a pruned Flux-schnell derivative)
combined with ChromaPipeline.__call__ for the Chroma-specific bits:
  * T5 only (no CLIP pooled embeds), T5 run WITH the tokenizer attention mask,
  * text mask keeps the real tokens + the first padding token, extended with ones for image tokens,
  * no guidance embedding (Chroma's distilled_guidance_layer is fed timestep only), real CFG at inference.
"""
import os

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ChromaPipeline,
    ChromaTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

FAMILY = "chroma"
PIPELINE_CLASS = ChromaPipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16          # vae 8x * 2x2 packing
DEFAULT_STEPS = 35           # ChromaPipeline.__call__ default
DEFAULT_GUIDANCE = 5.0       # ChromaPipeline.__call__ default (true CFG with negative prompt)
MAX_SEQ_LEN = 512
# Default target modules of train_dreambooth_lora_flux.py (same module names exist in ChromaTransformer2DModel)
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


def load_pipeline(model_path: str, dtype: torch.dtype):
    # works for a local diffusers-format dir (model_index.json) or a hub repo id
    return ChromaPipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    from transformers import AutoConfig, AutoTokenizer, T5EncoderModel

    torch.manual_seed(0)
    transformer = ChromaTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        axes_dims_rope=[4, 4, 8],
        approximator_hidden_dim=32,
        approximator_layers=1,
        approximator_num_channels=16,
    )
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-t5")
    text_encoder = T5EncoderModel(config).eval()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")
    torch.manual_seed(0)
    vae = AutoencoderKL(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
        shift_factor=0.0609,
        scaling_factor=1.5035,
    )
    pipe = ChromaPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        image_encoder=None,
        feature_extractor=None,
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    device = torch.device(device)
    if next(pipe.text_encoder.parameters()).device != device:
        pipe.text_encoder.to(device)
    for p in prompts:
        # pipeline helper: padding="max_length", T5 run with tokenizer mask, chroma text mask (real + 1st pad)
        embeds, mask = pipe._get_t5_prompt_embeds(
            prompt=[p], num_images_per_prompt=1, max_sequence_length=MAX_SEQ_LEN, device=device, dtype=dtype
        )
        out.append({
            "prompt_embeds": embeds.detach().to("cpu"),
            "prompt_attention_mask": mask.detach().to("cpu"),
        })
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    lat = vae.encode(px).latent_dist.sample()
    lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor  # as in train_dreambooth_lora_flux.py
    return lat.to("cpu", torch.float32)


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    wdtype = denoiser.dtype
    B, C, h, w = x0.shape
    device = x0.device
    vsf = pipe.vae_scale_factor
    s = sigma.to(device=device, dtype=x0.dtype).view(-1, 1, 1, 1)
    noisy = (1.0 - s) * x0 + s * noise

    packed = ChromaPipeline._pack_latents(noisy, B, C, h, w).to(wdtype)
    img_ids = ChromaPipeline._prepare_latent_image_ids(B, h // 2, w // 2, device, wdtype)

    prompt_embeds = cond["prompt_embeds"].to(device=device, dtype=wdtype)
    pmask = cond["prompt_attention_mask"].to(device=device)
    if prompt_embeds.shape[0] != B:
        prompt_embeds = prompt_embeds.expand(B, -1, -1)
        pmask = pmask.expand(B, -1)
    txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=wdtype)
    # extend the text mask with ones for the image tokens (ChromaPipeline._prepare_attention_mask)
    attn_mask = torch.cat([pmask.to(wdtype), torch.ones(B, packed.shape[1], device=device, dtype=wdtype)], dim=1)

    timesteps = sigma.to(device) * 1000.0  # scheduler timesteps; denoiser receives t/1000 (= sigma)
    pred = denoiser(
        hidden_states=packed,
        timestep=(timesteps / 1000).to(wdtype),
        encoder_hidden_states=prompt_embeds,
        txt_ids=txt_ids,
        img_ids=img_ids,
        attention_mask=attn_mask,
        return_dict=False,
    )[0]
    pred = ChromaPipeline._unpack_latents(pred, height=h * vsf, width=w * vsf, vae_scale_factor=vsf)
    target = noise - x0
    return F.mse_loss(pred.float(), target.float(), reduction="mean")


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ChromaPipeline.save_lora_weights(
        out_dir, transformer_lora_layers=lora_state_dict, safe_serialization=True
    )
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.transformer.dtype
    kw = dict(
        prompt_embeds=prompt_cond["prompt_embeds"].to(dev, dt),
        prompt_attention_mask=prompt_cond["prompt_attention_mask"].to(dev, dt),
    )
    if guidance > 1:
        if negative_cond is None:
            raise ValueError("Chroma uses true CFG: pass negative_cond (e.g. encode_prompts(pipe, [''])) when guidance > 1")
        kw["negative_prompt_embeds"] = negative_cond["prompt_embeds"].to(dev, dt)
        kw["negative_prompt_attention_mask"] = negative_cond["prompt_attention_mask"].to(dev, dt)
    out = pipe(
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
        **kw,
    )
    return out.images[0]
