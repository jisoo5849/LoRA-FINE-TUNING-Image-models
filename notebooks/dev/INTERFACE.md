# Adapter interface (every family module must implement exactly this)

File: `C:\Users\shj04\.lvw\adapters\<family>.py`. Pure functions/attributes at module level. Python 3.14, diffusers 0.42.1, peft 0.21.2, transformers 5.22.1, torch 2.15 (CPU here; real runs are on one CUDA GPU with 72 GB, bf16).

```python
FAMILY: str                     # e.g. "qwen_image"
PIPELINE_CLASS: type            # diffusers pipeline class used for from_pretrained and inference
DENOISER_ATTR: str              # "transformer" or "unet"
PREDICTION: str                 # "flow" (target = noise - latents, x_t = (1-s)*x0 + s*noise) | "eps" | "v_prediction"
PIXEL_MULTIPLE: int             # height/width in pixels must be divisible by this
DEFAULT_STEPS: int              # inference steps for validation samples (official default)
DEFAULT_GUIDANCE: float         # guidance value for validation samples (official default for this model)
LORA_TARGET_MODULES: list[str]  # module-name suffixes for peft LoraConfig(target_modules=...), matching the official diffusers training script for this family

def load_pipeline(model_path: str, dtype: torch.dtype) -> DiffusionPipeline
def tiny_pipeline(dtype: torch.dtype) -> DiffusionPipeline
    # random-initialised, CPU-sized components with the SAME classes as the real model (configs shrunk);
    # tokenizers may be tiny ones downloaded from the HF hub "hf-internal-testing/..." repos used by diffusers tests.
def encode_prompts(pipe, prompts: list[str], device, dtype) -> list[dict[str, torch.Tensor]]
    # ONE dict per prompt, every tensor has leading batch dim 1 and lives on CPU (cacheable).
    # Includes everything the denoiser needs from text (embeds, masks, pooled, text_ids, ...).
def encode_images(pipe, pixel_values: torch.Tensor, device, dtype) -> torch.Tensor
    # pixel_values: (1,3,H,W) float in [-1,1]. Returns NORMALISED latents x0 on CPU, fp32,
    # in exactly the space the official training script adds noise in (apply shift/scale or latents_mean/std).
def forward_loss(pipe, denoiser, x0, cond: dict, noise, sigma) -> torch.Tensor
    # x0, noise: same shape (on device). sigma: (B,) float in (0,1) on device.
    # flow: x_t=(1-s)x0+s*noise, target=noise-x0 (or the sign convention the official script uses),
    # pass the timestep in the unit the denoiser expects (e.g. s or s*1000), pack/unpack latents as the official
    # script does, handle guidance embedding, img_shapes/ids, attention masks.
    # eps/v: timesteps = (sigma*(num_train_timesteps-1)).round().long(); use pipe.scheduler.add_noise / get_velocity.
    # Returns scalar fp32 MSE loss (weighting = uniform).
def save_lora(pipe, lora_state_dict: dict, out_dir: str) -> str
    # lora_state_dict = peft.get_peft_model_state_dict(denoiser). Save in the format pipe.load_lora_weights() accepts
    # (use PIPELINE_CLASS.save_lora_weights(..., <denoiser>_lora_layers=...) when it exists). Return the file path.
def generate(pipe, prompt_cond: dict, negative_cond: dict | None, height, width, steps, guidance, generator) -> PIL.Image.Image
    # inference through PIPELINE_CLASS.__call__ using CACHED embeddings (prompt_embeds=... etc.) so the text encoder
    # is not needed on GPU. If the pipeline cannot take precomputed embeds, document why and fall back to prompt strings.
```

Device contract used by the notebook (real GPU runs): during training only the denoiser is on CUDA (bf16, LoRA params fp32); text encoders and VAE are on CPU. encode_prompts/encode_images are called while text encoders/VAE ARE on the device. generate() is called with the denoiser and VAE on CUDA and text encoders still on CPU, conds already moved to CUDA — generate() must therefore pass `device`-consistent tensors and not rely on pipe._execution_device being CUDA if that is derived from a CPU module (cast/route explicitly if needed). forward_loss receives x0/noise/sigma/cond on CUDA in fp32; it must cast inputs to the denoiser dtype itself and return an fp32 loss.

Test with: `C:\Users\shj04\.lv\Scripts\python.exe C:\Users\shj04\.lvw\harness.py <family>` — must print `HARNESS OK <family>`.
