"""Wan2.2 TI2V-5B adapter: Wan-AI/Wan2.2-TI2V-5B-Diffusers (WanPipeline, expand_timesteps true, one denoiser).

Training math is the shared Wan code in wan_common.py; the TI2V parts mirror the expand_timesteps branches of
diffusers v0.42.1 pipeline_wan.py (T2V) and pipeline_wan_i2v.py (I2V) and the original wan/textimage2video.py:
  x0 = (vae.encode(px).latent_dist.mode() - latents_mean) / latents_std with the Wan2.2 VAE (z 48, 2x2 patchify,
  16x spatial, 4x temporal) ; x_t = (1 - s) * x0 + s * noise ; target = noise - x0 ;
  input = (1 - m) * x0 + m * x_t ; per-token timestep = (m[0, 0][:, ::2, ::2] * 1000 * s).flatten(), shape
  (1, f * h/2 * w/2), as the pipeline's (mask[0][0][:, ::2, ::2] * t).flatten() ;
  "t2v" (images F = 1 and clips): m = 1 everywhere -> input x_t, every token at 1000 * s, loss over all frames ;
  "i2v" (clips): m = 0 on latent frame 0 -> clean first frame with timestep 0 on its tokens, loss over latent frames
  1: only (DiffSynth / ai-toolkit: the clean frame carries no loss).
The I2V pipeline conditions on vae.encode(frame0) alone; the Wan VAE is causal in time, so that equals x0[:, :, :1].
Validation = official Wan2.2 wan/configs/wan_ti2v_5B.py: sample_steps 50, sample_shift 5.0 (the -Diffusers repo
ships 5.0 too), sample_guide_scale 5.0, sample_fps 24, frame_num 121, vae_stride (4, 16, 16) -> H, W multiples of 32.
generate() runs T2V without an image and I2V with one (WanImageToVideoPipeline over the same modules). A 5B LoRA
does not load into the A14B models (inner dim 3072 vs 5120).
"""

from diffusers import WanPipeline

try:  # dev: video_adapters/ on sys.path; notebook: lora_video_tools/ on sys.path or importable as a package
    import wan_common as W
except ImportError:
    from lora_video_tools import wan_common as W

FAMILY = "wan22_ti2v_5b"
PIPELINE_CLASS = WanPipeline
EXPERTS = ["transformer"]
EXPERT_TAGS = {"transformer": "main"}
CONDITIONING = "ti2v"
FPS = 24  # wan_ti2v_5B.py sample_fps
SPATIAL_FACTOR = 16  # vae_stride (4, 16, 16)
SPATIAL_MULTIPLE = 32  # 16 x patch 2
TEMPORAL_FACTOR = 4
TOKEN_DIV = 32
VAL_STEPS = 50
VAL_SHIFT = 5.0
VAL_GUIDANCE = (5.0, 5.0)
NEGATIVE_PROMPT = W.NEGATIVE_PROMPT
LORA_TARGETS = W.LORA_TARGETS
_SPEC = dict(two_experts=False, in_channels=48, z_dim=48, expand_timesteps=True)

encode_prompts = W.encode_prompts
lora_to_comfy = W.lora_to_comfy
save_lora = W.save_lora
load_lora = W.load_lora


def sigma_range(pipe, expert):
    return W.sigma_range(pipe, expert, EXPERTS)


def load_pipeline(model_path, dtype):
    # model_index.json of the -Diffusers repo: WanPipeline, expand_timesteps true, no transformer_2 / boundary_ratio;
    # rejects Wan2.1 and A14B folders (z 16, in 16 / 36, two experts) and z-48 folders without expand_timesteps
    return W.load_pipeline(model_path, dtype, PIPELINE_CLASS, FAMILY, **_SPEC)


def tiny_pipeline(dtype):
    # diffusers v0.42.1 tests/pipelines/wan/test_wan_22.py Wan225B components (VAE z 48, patch 2, 16x spatial,
    # transformer in/out 48, expand_timesteps); UniPC flow_shift 3.0 as the test, so the harness sees generate() set 5.0
    pipe = W.tiny_pipeline(PIPELINE_CLASS, dtype, in_channels=48, out_channels=48, z_dim=48, expand_timesteps=True)
    W.check_pipeline(pipe, FAMILY, **_SPEC)
    return pipe


def encode_video(pipe, frames, device, dtype, condition):  # dtype unused; no "cond": I2V conditions on x0[:, :, :1]
    return W.encode_video(pipe, frames, device, conditioning=CONDITIONING, multiple=SPATIAL_MULTIPLE,
                          condition=condition)


def forward_loss(pipe, denoiser, x0, cond, text, noise, sigma, mode):
    return W.flow_loss(denoiser, x0, noise, sigma, text, conditioning=CONDITIONING, mode=mode, cond=cond)


def generate(pipe, text, negative, *, height, width, num_frames, generator, image=None, steps=None, shift=None,
             guidance=None, device=None, offload=False):
    return W.generate(pipe, text, negative, height=height, width=width, num_frames=num_frames, generator=generator,
                      image=image, steps=VAL_STEPS if steps is None else steps,
                      shift=VAL_SHIFT if shift is None else shift,
                      guidance=VAL_GUIDANCE if guidance is None else guidance, device=device, offload=offload,
                      conditioning=CONDITIONING)
