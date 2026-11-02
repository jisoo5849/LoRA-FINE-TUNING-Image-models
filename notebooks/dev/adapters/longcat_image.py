"""LongCat-Image (meituan-longcat/LongCat-Image) adapter.

diffusers 0.42.1 ships no LongCat training script and LongCatImagePipeline has no LoRA loader mixin.
The reference used here is the official Meituan LoRA script:
  https://github.com/meituan-longcat/LongCat-Image/blob/main/train_examples/lora/train_lora.py
which trains against diffusers' LongCatImageTransformer2DModel:
  latents = (vae_latents - shift_factor) * scaling_factor
  x_t = (1 - s) * x0 + s * noise ; target = noise - x0
  transformer(pack(x_t), prompt_embeds, timestep=s (model multiplies by 1000 internally), img_ids, txt_ids, guidance=None)
  text: 512 padded tokens (prefix/suffix template stripped), no attention mask in the transformer
  txt_ids = text pos ids (modality 0), img_ids = image pos ids (modality 1) starting at (512, 512)
Real checkpoint (transformer/config.json): 10 double + 20 single blocks (NOT the class defaults 19/38), 24 heads x 128,
joint_attention_dim 3584, guidance_embeds false; ~6B params. LORA_TARGET_MODULES match 180 Linear modules there
(120 in transformer_blocks, 60 in single_transformer_blocks), same as the official target list.
"""
import os

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, LongCatImagePipeline, LongCatImageTransformer2DModel
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.loaders.lora_pipeline import CogView4LoraLoaderMixin
from diffusers.pipelines.longcat_image.pipeline_longcat_image import prepare_pos_ids, split_quotation


class LongCatImageLoraLoaderMixin(CogView4LoraLoaderMixin):
    """Single-transformer LoRA loader (diffusers has none for LongCat in 0.42.1).

    Also accepts the official Meituan PEFT checkpoints (adapter_model.safetensors with
    'base_model.model.' prefixed keys) by remapping them to 'transformer.'.
    """

    @classmethod
    def lora_state_dict(cls, pretrained_model_name_or_path_or_dict, **kwargs):
        if isinstance(pretrained_model_name_or_path_or_dict, str) and os.path.isdir(pretrained_model_name_or_path_or_dict):
            if "weight_name" not in kwargs and os.path.isfile(
                os.path.join(pretrained_model_name_or_path_or_dict, "adapter_model.safetensors")
            ) and not os.path.isfile(os.path.join(pretrained_model_name_or_path_or_dict, "pytorch_lora_weights.safetensors")):
                kwargs["weight_name"] = "adapter_model.safetensors"
        out = super().lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        sd, meta = (out if isinstance(out, tuple) else (out, None))
        if any(k.startswith("base_model.model.") for k in sd):
            sd = {"transformer." + k[len("base_model.model."):].replace(".default.", "."): v for k, v in sd.items()}
        return (sd, meta) if isinstance(out, tuple) else sd


class LongCatImageLoraPipeline(LongCatImageLoraLoaderMixin, LongCatImagePipeline):
    """LongCatImagePipeline + LoRA loading + support for prompt_embeds without prompt strings."""

    def encode_prompt(self, prompt=None, num_images_per_prompt=1, prompt_embeds=None):
        if prompt_embeds is not None:
            # upstream does len(prompt) even when prompt is None -> take batch from the embeds instead
            bs, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(bs * num_images_per_prompt, seq_len, -1)
            text_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=seq_len).to(self.device)
            return prompt_embeds.to(self.device), text_ids
        return super().encode_prompt(prompt=prompt, num_images_per_prompt=num_images_per_prompt)

    def _encode_prompt_on(self, prompt, device):
        """Copy of LongCatImagePipeline._encode_prompt (diffusers 0.42.1) with input_ids placed on `device`
        (the text encoder's device) instead of self.device, which returns the first non-CPU module and can
        differ from where the text encoder lives."""
        batch_all_tokens = []
        for each_prompt in prompt:
            all_tokens = []
            for clean_prompt_sub, matched in split_quotation(each_prompt):
                if matched:
                    for sub_word in clean_prompt_sub:
                        all_tokens.extend(self.tokenizer(sub_word, add_special_tokens=False)["input_ids"])
                else:
                    all_tokens.extend(self.tokenizer(clean_prompt_sub, add_special_tokens=False)["input_ids"])
            batch_all_tokens.append(all_tokens[: self.tokenizer_max_length])
        tok = self.tokenizer.pad(
            {"input_ids": batch_all_tokens}, max_length=self.tokenizer_max_length, padding="max_length",
            return_attention_mask=True, return_tensors="pt",
        )
        prefix = self.tokenizer(self.prompt_template_encode_prefix, add_special_tokens=False)["input_ids"]
        suffix = self.tokenizer(self.prompt_template_encode_suffix, add_special_tokens=False)["input_ids"]
        bs = tok.input_ids.size(0)
        ids_dt, m_dt = tok.input_ids.dtype, tok.attention_mask.dtype
        input_ids = torch.cat((torch.tensor(prefix, dtype=ids_dt).unsqueeze(0).expand(bs, -1), tok.input_ids,
                               torch.tensor(suffix, dtype=ids_dt).unsqueeze(0).expand(bs, -1)), dim=-1)
        attention_mask = torch.cat((torch.ones(bs, len(prefix), dtype=m_dt), tok.attention_mask,
                                    torch.ones(bs, len(suffix), dtype=m_dt)), dim=-1)
        out = self.text_encoder(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                output_hidden_states=True)
        return out.hidden_states[-1].detach()[:, len(prefix):-len(suffix), :]


FAMILY = "longcat_image"
PIPELINE_CLASS = LongCatImageLoraPipeline
DENOISER_ATTR = "transformer"
PREDICTION = "flow"
PIXEL_MULTIPLE = 16  # vae_scale_factor 8 * 2x2 packing
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.5
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


def _try_processor(path, **kw):
    # text_processor is only used by the optional prompt-rewrite step (disabled here). Qwen2VLProcessor needs
    # torchvision for its video/image sub-processors; without torchvision fall back to None.
    try:
        from transformers import Qwen2VLProcessor

        return Qwen2VLProcessor.from_pretrained(path, **kw)
    except Exception:
        return None


def load_pipeline(model_path: str, dtype: torch.dtype):
    sub = {"subfolder": "text_processor"} if not os.path.isdir(model_path) or os.path.isdir(
        os.path.join(model_path, "text_processor")) else {}
    proc = _try_processor(model_path, **sub)
    return PIPELINE_CLASS.from_pretrained(model_path, torch_dtype=dtype, text_processor=proc)


def tiny_pipeline(dtype: torch.dtype):
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

    repo = "hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration"
    torch.manual_seed(0)
    transformer = LongCatImageTransformer2DModel(
        patch_size=1, in_channels=64, num_layers=1, num_single_layers=1, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=16, pooled_projection_dim=16, axes_dims_rope=[4, 6, 6],
    )
    torch.manual_seed(0)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=["DownEncoderBlock2D"] * 4, up_block_types=["UpDecoderBlock2D"] * 4,
        block_out_channels=[8, 8, 8, 8], layers_per_block=1, latent_channels=16, norm_num_groups=4,
        use_quant_conv=False, use_post_quant_conv=False, shift_factor=0.1159, scaling_factor=0.3611,
    )
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096,
    )
    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 16, "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "max_position_embeddings": 2048,
            "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
            "rope_theta": 1000000.0,
        },
        vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2, "out_hidden_size": 16},
        hidden_size=16, vocab_size=152064,
        vision_end_token_id=151653, vision_start_token_id=151652, vision_token_id=151654,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration(config).eval()
    tokenizer = Qwen2Tokenizer.from_pretrained(repo)
    processor = _try_processor(repo)
    pipe = PIPELINE_CLASS(
        scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_processor=processor, transformer=transformer,
    )
    return pipe.to(dtype=dtype)


@torch.no_grad()
def encode_prompts(pipe, prompts, device, dtype):
    pipe.text_encoder.to(device)
    out = []
    for p in prompts:
        # pipeline's own tokenisation: quote-splitting, pad to 512, template prefix/suffix stripped
        embeds = pipe._encode_prompt_on([p], device)
        out.append({"prompt_embeds": embeds.to(dtype).cpu()})
    return out


@torch.no_grad()
def encode_images(pipe, pixel_values, device, dtype):
    vae = pipe.vae
    x = pixel_values.to(device=device, dtype=vae.dtype)
    lat = vae.encode(x).latent_dist.sample()
    lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
    return lat.float().cpu()


def forward_loss(pipe, denoiser, x0, cond, noise, sigma):
    B, C, h, w = x0.shape
    dtype = next(p for p in denoiser.parameters()).dtype
    device = x0.device
    s = sigma.to(device=device, dtype=x0.dtype).view(-1, 1, 1, 1)
    x_t = (1 - s) * x0 + s * noise
    packed = pipe._pack_latents(x_t.to(dtype), B, C, h, w)

    embeds = cond["prompt_embeds"].to(device=device, dtype=dtype)
    if embeds.shape[0] != B:
        embeds = embeds.expand(B, -1, -1)
    seq = embeds.shape[1]
    img_ids = prepare_pos_ids(modality_id=1, type="image", start=(seq, seq), height=h // 2, width=w // 2).to(device)
    txt_ids = prepare_pos_ids(modality_id=0, type="text", start=(0, 0), num_token=seq).to(device)

    pred = denoiser(
        hidden_states=packed, encoder_hidden_states=embeds,
        timestep=sigma.to(device=device, dtype=dtype),  # in [0,1]; model multiplies by 1000 internally
        img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False,
    )[0]
    pred = pipe._unpack_latents(pred, h * pipe.vae_scale_factor, w * pipe.vae_scale_factor, pipe.vae_scale_factor)
    target = noise - x0
    return torch.mean((pred.float() - target.float()) ** 2)


def save_lora(pipe, lora_state_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    PIPELINE_CLASS.save_lora_weights(out_dir, transformer_lora_layers=lora_state_dict, safe_serialization=True)
    return os.path.join(out_dir, "pytorch_lora_weights.safetensors")


@torch.no_grad()
def generate(pipe, prompt_cond, negative_cond, height, width, steps, guidance, generator):
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    pe = prompt_cond["prompt_embeds"].to(device, dtype)
    if negative_cond is not None:
        npe = negative_cond["prompt_embeds"].to(device, dtype)
    elif guidance > 1:
        # The pipeline's own unconditional branch is the encoded empty string (and the official training drops
        # prompts to ''), so zeros would be out of distribution. Text encoder is on CPU here -> require the cache.
        raise ValueError("longcat_image.generate: guidance > 1 needs negative_cond = encode_prompts(pipe, [''], ...)[0]")
    else:
        npe = None
    return pipe(
        prompt_embeds=pe, negative_prompt_embeds=npe, height=height, width=width,
        num_inference_steps=steps, guidance_scale=guidance, generator=generator,
        enable_prompt_rewrite=False, enable_cfg_renorm=True, output_type="pil",
    ).images[0]
