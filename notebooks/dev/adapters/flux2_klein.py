"""FLUX.2 [klein] adapter (4B Base by default; works for 9B Base too).

Mirrors diffusers v0.42.1 examples/dreambooth/train_dreambooth_lora_flux2_klein.py:
  - latents = vae.encode(px).latent_dist.mode() -> _patchify_latents (2x2 -> 4C channels)
    -> (x - bn.running_mean) / sqrt(bn.running_var + batch_norm_eps)
  - x_t = (1 - s) * x0 + s * noise, target = noise - x0
  - img_ids = Flux2KleinPipeline._prepare_latent_ids(x0), hidden_states = _pack_latents(x_t)
  - transformer(timestep = timesteps/1000 == sigma, guidance = None unless config.guidance_embeds,
                encoder_hidden_states = prompt_embeds, txt_ids = text_ids, img_ids = img_ids)
  - prediction unpacked with _unpack_latents_with_ids; no attention mask (official script/pipeline pass none;
    prompts are padded to max_sequence_length=512).
"""
import os

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)

FAMILY = "flux2_klein"
PIPELINE_CLASS = Flux2KleinPipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor (8) * 2x2 patchify
DEFAULT_STEPS = 50  # Flux2KleinPipeline.__call__ default (Base model; distilled Klein uses 4 steps / guidance 1.0)
DEFAULT_GUIDANCE = 4.0  # Flux2KleinPipeline.__call__ default (true CFG for the non-distilled Base model)
MAX_SEQUENCE_LENGTH = 512
# Pipeline default hidden-state layers of Qwen3 used for prompt embeds (what the model was trained with and what
# Flux2KleinPipeline uses at inference). NOTE: the official training script's CLI default is [10, 20, 30].
TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)

# Official script default (no --lora_layers): double-block attention + single-block fused qkv/mlp proj + to_out.
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"] + [
    f"single_transformer_blocks.{i}.attn.to_out" for i in range(24)
]


def _out_layers(pipe):
    return getattr(pipe, "_lvw_text_encoder_out_layers", TEXT_ENCODER_OUT_LAYERS)


def load_pipeline(model_path: str, dtype: torch.dtype):
    return Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=dtype)


def tiny_pipeline(dtype: torch.dtype):
    # Mirrors tests/pipelines/flux2/test_pipeline_flux2_klein.py get_dummy_components (3 stacked layers here,
    # so joint_attention_dim = 3 * hidden_size like the real model).
    from transformers import Qwen2TokenizerFast, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=48,
        timestep_guidance_channels=256,
        axes_dims_rope=[4, 4, 4, 4],
        guidance_embeds=False,
    )
    config = Qwen3Config(
        intermediate_size=16,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=151936,
        max_position_embeddings=1024,
    )
    torch.manual_seed(0)
    text_encoder = Qwen3ForCausalLM(config)
    tokenizer = Qwen2TokenizerFast.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=1,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    scheduler = FlowMatchEulerDiscreteScheduler()
    pipe = Flux2KleinPipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
    )
    pipe.to(dtype=dtype)
    pipe._lvw_text_encoder_out_layers = (0, 1, 2)
    return pipe


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=p,
            device=device,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
            text_encoder_out_layers=_out_layers(pipe),
        )
        out.append({
            "prompt_embeds": prompt_embeds.to(dtype=dtype).cpu(),
            "text_ids": text_ids.cpu(),
        })
    return out


def _bn_stats(vae, device, dtype):
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device=device, dtype=dtype)
    return mean, std


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    lat = vae.encode(px).latent_dist.mode()
    lat = Flux2KleinPipeline._patchify_latents(lat).float()
    mean, std = _bn_stats(vae, lat.device, torch.float32)
    lat = (lat - mean) / std
    return lat.float().cpu()


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    wdtype = next(denoiser.parameters()).dtype
    device = x0.device
    B = x0.shape[0]
    s = sigma.to(device=device, dtype=torch.float32).view(-1, 1, 1, 1)
    x0f, noisef = x0.float(), noise.float()
    noisy = (1.0 - s) * x0f + s * noisef

    img_ids = Flux2KleinPipeline._prepare_latent_ids(x0).to(device=device)
    packed = Flux2KleinPipeline._pack_latents(noisy).to(wdtype)

    prompt_embeds = cond["prompt_embeds"].to(device=device, dtype=wdtype)
    text_ids = cond["text_ids"].to(device=device)
    if prompt_embeds.shape[0] != B:
        prompt_embeds = prompt_embeds.expand(B, -1, -1)
        text_ids = text_ids.expand(B, -1, -1)

    if denoiser.config.guidance_embeds:
        guidance = torch.full((B,), 3.5, device=device, dtype=wdtype)  # official --guidance_scale default
    else:
        guidance = None

    # official: timestep = timesteps / 1000, with timesteps = sigma * num_train_timesteps (1000)
    timestep = sigma.to(device=device, dtype=wdtype).view(-1).expand(B)
    model_pred = denoiser(
        hidden_states=packed,
        timestep=timestep,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=img_ids,
        return_dict=False,
    )[0]
    model_pred = model_pred[:, : packed.size(1)]
    model_pred = Flux2KleinPipeline._unpack_latents_with_ids(model_pred, img_ids)

    target = noisef - x0f
    return F.mse_loss(model_pred.float(), target, reduction="mean")


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    Flux2KleinPipeline.save_lora_weights(
        save_directory=out_dir, transformer_lora_layers=lora_state_dict, safe_serialization=True
    )
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.transformer.dtype
    kwargs = dict(
        prompt_embeds=prompt_cond["prompt_embeds"].to(device=dev, dtype=dt),
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        text_encoder_out_layers=_out_layers(pipe),
        output_type="pil",
    )
    # With guidance > 1 on the Base model the pipeline runs true CFG; passing cached negative embeds avoids
    # running the text encoder for the empty negative prompt.
    if negative_cond is not None:
        kwargs["negative_prompt_embeds"] = negative_cond["prompt_embeds"].to(device=dev, dtype=dt)
    elif guidance is not None and guidance > 1 and not getattr(pipe.config, "is_distilled", False):
        # pipeline_flux2_klein.py: do_classifier_free_guidance -> encode_prompt("") runs the Qwen3 text encoder.
        te = getattr(pipe, "text_encoder", None)
        te_dev = next(te.parameters()).device if te is not None else None
        if te is None or te_dev != torch.device(dev):
            raise ValueError(
                "flux2_klein.generate: Base (non-distilled) model with guidance > 1 runs true CFG and needs "
                "negative embeddings. Pass negative_cond=encode_prompts(pipe, ['']) (cached while the text "
                f"encoder was on the device); text encoder is {'missing' if te is None else 'on ' + str(te_dev)}."
            )
    return pipe(**kwargs).images[0]
