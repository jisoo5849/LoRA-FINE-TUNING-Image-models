"""Z-Image (Tongyi-MAI/Z-Image, Z-Image-Turbo) adapter.

Mirrors diffusers v0.42.1 examples/dreambooth/train_dreambooth_lora_z_image.py:
  latents   = (vae.encode(x).latent_dist.mode() - shift_factor) * scaling_factor
  x_t       = (1 - sigma) * x0 + sigma * noise
  timestep  = (1000 - sigma*1000) / 1000 = 1 - sigma   (Z-Image's "t" runs 0=noise -> 1=clean)
  denoiser  = transformer(list[(C,1,H,W)], t, list[(L_i, D)])  -> list[(C,1,H,W)]
  model_pred = -stack(out).squeeze(2)          (Z-Image negates the prediction)
  target    = noise - x0
LoRA targets: ["to_k", "to_q", "to_v", "to_out.0"] (script default).
Text conditioning: Qwen3 hidden_states[-2] after chat template, with padding tokens REMOVED
(variable-length list, no attention mask needed by the transformer).
"""
import os

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)

FAMILY = "z_image"
PIPELINE_CLASS = ZImagePipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor (8) * 2 (patch size)
# ZImagePipeline.__call__ defaults (official validation calls the pipeline with defaults; base Z-Image).
# For Z-Image-Turbo use steps=8..9, guidance=0.0.
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 5.0
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]
MAX_SEQUENCE_LENGTH = 512  # train script default --max_sequence_length


def load_pipeline(model_path: str, dtype: torch.dtype):
    return ZImagePipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    # Copied from diffusers v0.42.1 tests/pipelines/z_image/test_z_image.py get_dummy_components()
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    transformer = ZImageTransformer2DModel(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=32,
        n_layers=2,
        n_refiner_layers=1,
        n_heads=2,
        n_kv_heads=2,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=16,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 4, 4],
        axes_lens=[256, 32, 32],
    )
    with torch.no_grad():
        transformer.x_pad_token.copy_(torch.ones_like(transformer.x_pad_token.data))
        transformer.cap_pad_token.copy_(torch.ones_like(transformer.cap_pad_token.data))

    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[32, 64],
        layers_per_block=1,
        latent_channels=16,
        norm_num_groups=32,
        sample_size=32,
        scaling_factor=0.3611,
        shift_factor=0.1159,
    )
    torch.manual_seed(0)
    scheduler = FlowMatchEulerDiscreteScheduler()
    torch.manual_seed(0)
    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=512,
    )
    text_encoder = Qwen3Model(config)
    tokenizer = Qwen2Tokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    pipe = ZImagePipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:
        # same call as compute_text_embeddings() in the official script (one prompt at a time -> no padding)
        embeds = pipe._encode_prompt(prompt=[p], device=device, max_sequence_length=MAX_SEQUENCE_LENGTH)
        e = embeds[0].to(dtype=dtype).detach().cpu()  # (L, D), padding already stripped
        out.append({"prompt_embeds": e.unsqueeze(0)})  # (1, L, D)
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    lat = vae.encode(px).latent_dist.mode()
    lat = (lat.float() - vae.config.shift_factor) * vae.config.scaling_factor
    return lat.float().cpu()


def _embeds_list(cond, device, dtype, bsz):
    pe = cond["prompt_embeds"].to(device=device, dtype=dtype)  # (1, L, D)
    lst = [pe[0]]
    if bsz > 1:
        lst = lst * bsz
    return lst


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    device = x0.device
    wdtype = next(p for p in denoiser.parameters()).dtype
    x0 = x0.float()
    noise = noise.float()
    s = sigma.to(device=device, dtype=torch.float32).view(-1, *([1] * (x0.ndim - 1)))
    noisy = (1.0 - s) * x0 + s * noise
    # official: timestep_normalized = (1000 - timesteps) / 1000 with timesteps = sigma * 1000
    t = (1000.0 - sigma.to(device=device, dtype=torch.float32) * 1000.0) / 1000.0
    bsz = x0.shape[0]
    t = t.expand(bsz) if t.numel() == 1 else t
    inp = list(noisy.to(wdtype).unsqueeze(2).unbind(dim=0))  # list of (C,1,H,W)
    # timestep stays fp32 like the official script / pipeline (bf16 would bin t*1000 coarsely)
    out = denoiser(inp, t.float(), _embeds_list(cond, device, wdtype, bsz), return_dict=False)[0]
    pred = -torch.stack(out, dim=0).squeeze(2).float()  # Z-Image negates the prediction
    target = noise - x0
    return F.mse_loss(pred, target, reduction="mean")


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    kwargs = {}
    den = getattr(pipe, DENOISER_ATTR)
    cfg = getattr(den, "peft_config", None)
    if cfg and "default" in cfg:
        kwargs["transformer_lora_adapter_metadata"] = cfg["default"].to_dict()
    ZImagePipeline.save_lora_weights(
        out_dir, transformer_lora_layers=lora_state_dict, safe_serialization=True, **kwargs
    )
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    pe = [prompt_cond["prompt_embeds"].to(device=device, dtype=dtype)[0]]
    kwargs = {}
    if guidance and guidance > 0:
        if negative_cond is not None:
            kwargs["negative_prompt_embeds"] = [negative_cond["prompt_embeds"].to(device=device, dtype=dtype)[0]]
        elif getattr(pipe, "text_encoder", None) is not None:
            kwargs["negative_prompt_embeds"] = [
                e.to(device=device, dtype=dtype)
                for e in pipe._encode_prompt(prompt=[""], device=device, max_sequence_length=MAX_SEQUENCE_LENGTH)
            ]
        else:
            guidance = 0.0  # no negative embeds and no text encoder -> CFG impossible
    return pipe(
        prompt_embeds=pe,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        output_type="pil",
        **kwargs,
    ).images[0]
