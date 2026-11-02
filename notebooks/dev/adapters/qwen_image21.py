"""Qwen-Image 2.1 adapter.

Mirrors diffusers v0.42.1 examples/dreambooth/train_dreambooth_lora_qwenimage21.py:
  * images are RGBA (this VAE has 4 input channels); RGB inputs get an opaque alpha (= +1 in [-1,1]).
  * latents = (vae.encode(px).latent_dist.sample() - latents_mean) / latents_std, shape (B, z_dim, 1, h, w).
  * x_t = (1 - s) x0 + s noise ; target = noise - x0 ; transformer gets timestep = s (= timesteps / 1000).
  * packing is a plain spatial flatten (patch_size 1); img_shapes = [[(1, h, w)]];
    img_mask = [False * prompt_len, True * (h*w // 4)] ; output tail [-h*w:] is the target prediction.
  * LoRA targets default: to_k, to_q, to_v, to_out.0.
"""
import torch
import torch.nn.functional as F

from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

FAMILY = "qwen_image21"
PIPELINE_CLASS = QwenImage21Pipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 32  # vae_scale_factor (16) * 2: target latents are grouped into 2x2 vision slots
DEFAULT_STEPS = 40  # pipeline / training-script validation default (overridden by config.sample_sigmas if set)
DEFAULT_GUIDANCE = 1.0  # true_cfg_scale; the model is sampled without CFG (official validation uses 1.0)
LORA_TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def _processor_without_video(model_path: str):
    """Qwen3VLProcessor without its video sub-processor (that one needs torchvision; text prompts never use it)."""
    from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLProcessor

    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="processor")
    image_processor = Qwen2VLImageProcessor.from_pretrained(model_path, subfolder="processor")
    orig = Qwen3VLProcessor.check_argument_for_proper_class

    def _check(self, name, arg):
        if name == "video_processor" and arg is None:
            return None
        return orig(self, name, arg)

    Qwen3VLProcessor.check_argument_for_proper_class = _check
    try:
        return Qwen3VLProcessor(image_processor=image_processor, tokenizer=tokenizer, video_processor=None,
                                chat_template=tokenizer.chat_template)
    finally:
        Qwen3VLProcessor.check_argument_for_proper_class = orig


def load_pipeline(model_path: str, dtype: torch.dtype):
    try:
        import torchvision  # noqa: F401
        return QwenImage21Pipeline.from_pretrained(model_path, dtype=dtype)
    except ImportError:
        return QwenImage21Pipeline.from_pretrained(model_path, dtype=dtype,
                                                   processor=_processor_without_video(model_path))


def tiny_pipeline(dtype: torch.dtype):
    # Copied from tests/pipelines/qwenimage21/test_qwenimage21.py::get_dummy_components (v0.42.1)
    from transformers import (
        AutoTokenizer,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    z_dim = 8
    text_hidden_size = 16
    torch.manual_seed(0)
    transformer = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=z_dim,
        out_channels=z_dim,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=text_hidden_size,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )
    torch.manual_seed(0)
    vae = AutoencoderKLQwenImage21(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=z_dim,
        dim_mult=[1, 1, 1, 1, 1],
        num_res_blocks=1,
        attn_scales=[],
        temperal_downsample=[False, True, True, True],
        latents_mean=[0.0] * z_dim,
        latents_std=[1.0] * z_dim,
    )
    torch.manual_seed(0)
    scheduler = FlowMatchEulerDiscreteScheduler()
    torch.manual_seed(0)
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": text_hidden_size,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 16,
            "num_heads": 2,
            "out_hidden_size": text_hidden_size,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0],
        },
    )
    text_encoder = Qwen3VLForConditionalGeneration(config).eval()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    image_processor = Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=32 * 32, max_pixels=64 * 64
    )
    try:
        video_processor = Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2)
    except ImportError:
        # Video processor needs torchvision (absent in the CPU test env); text-only prompts never touch it, so
        # build the processor without it by skipping the class check for that one slot.
        video_processor = None
    if video_processor is not None:
        processor = Qwen3VLProcessor(
            image_processor=image_processor, tokenizer=tokenizer, video_processor=video_processor,
            chat_template=tokenizer.chat_template,
        )
    else:
        orig = Qwen3VLProcessor.check_argument_for_proper_class

        def _check(self, name, arg):
            if name == "video_processor" and arg is None:
                return None
            return orig(self, name, arg)

        Qwen3VLProcessor.check_argument_for_proper_class = _check
        try:
            processor = Qwen3VLProcessor(
                image_processor=image_processor, tokenizer=tokenizer, video_processor=None,
                chat_template=tokenizer.chat_template,
            )
        finally:
            Qwen3VLProcessor.check_argument_for_proper_class = orig
    pipe = QwenImage21Pipeline(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, processor=processor, transformer=transformer
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    out = []
    for p in prompts:
        # One prompt at a time -> no padding (left-padding would otherwise change encoder positions).
        emb, mask, _ = pipe.encode_prompt(prompt=[p], device=device)
        if mask is None:
            mask = torch.ones(emb.shape[:2], dtype=torch.long, device=emb.device)
        out.append(
            {
                "prompt_embeds": emb.to(dtype=dtype).cpu(),
                "prompt_embeds_mask": mask.to(torch.long).cpu(),
            }
        )
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    px = pixel_values.to(device=device, dtype=vae.dtype)
    if px.shape[1] == 3:  # official script converts every image to RGBA; opaque alpha = 1.0 after normalize
        px = torch.cat([px, torch.ones_like(px[:, :1])], dim=1)
    if px.ndim == 4:
        px = px.unsqueeze(2)  # (B, C, 1, H, W)
    lat = vae.encode(px).latent_dist.sample()
    lat = lat.float()  # normalise with fp32 statistics like the official script
    mean = torch.tensor(vae.config.latents_mean, device=lat.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=lat.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    lat = (lat - mean) / std
    return lat.cpu()


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    if x0.ndim == 4:
        x0, noise = x0.unsqueeze(2), noise.unsqueeze(2)
    bsz, ch, _, lh, lw = x0.shape
    vsf = pipe.vae_scale_factor
    wdt = next(p for p in denoiser.parameters()).dtype
    dev = x0.device

    s = sigma.to(device=dev, dtype=torch.float32).view(-1, 1, 1, 1, 1)
    x0f, nf = x0.float(), noise.float()
    noisy = ((1.0 - s) * x0f + s * nf).to(wdt)
    target = nf - x0f

    packed = QwenImage21Pipeline._pack_latents(noisy, bsz, ch, lh, lw)
    emb = cond["prompt_embeds"].to(device=dev, dtype=wdt)
    mask = cond.get("prompt_embeds_mask")
    if emb.shape[0] != bsz:
        emb = emb.expand(bsz, -1, -1)
        mask = mask.expand(bsz, -1) if mask is not None else None
    if mask is not None:
        mask = mask.to(dev)
        if mask.all():  # official: no mask when nothing is padded
            mask = None

    img_shapes = [[(1, lh, lw)]] * bsz
    img_mask = torch.cat(
        [
            torch.zeros(bsz, emb.shape[1], dtype=torch.bool, device=dev),
            torch.ones(bsz, (lh * lw) // 4, dtype=torch.bool, device=dev),
        ],
        dim=1,
    )
    pred = denoiser(
        hidden_states=packed,
        encoder_hidden_states=emb,
        encoder_hidden_states_mask=mask,
        timestep=sigma.to(device=dev, dtype=wdt),  # == timesteps / 1000 in the official script
        img_shapes=img_shapes,
        img_mask=img_mask,
        return_dict=False,
    )[0]
    pred = pred[:, -packed.shape[1]:]
    pred = QwenImage21Pipeline._unpack_latents(pred, lh * vsf, lw * vsf, vsf)
    return F.mse_loss(pred.float(), target, reduction="mean")


def save_lora(pipe, lora_state_dict, out_dir):
    import os

    QwenImage21Pipeline.save_lora_weights(out_dir, transformer_lora_layers=lora_state_dict)
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    dev = pipe._execution_device
    dt = pipe.transformer.dtype

    def prep(c):
        e = c["prompt_embeds"].to(dev, dt)
        m = c.get("prompt_embeds_mask")
        m = m.to(dev) if m is not None else None
        return e, m

    pe, pm = prep(prompt_cond)
    kw = dict(prompt_embeds=pe, prompt_embeds_mask=pm)
    if negative_cond is not None and guidance > 1.0:
        ne, nm = prep(negative_cond)
        kw.update(negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm)
    img = pipe(
        **kw,
        true_cfg_scale=guidance,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=generator,
        output_type="pil",
    ).images[0]
    if img.mode == "RGBA":
        from PIL import Image
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    return img.convert("RGB")
