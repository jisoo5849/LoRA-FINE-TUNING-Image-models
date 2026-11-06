"""Wan2.2 T2V-A14B adapter: Wan-AI/Wan2.2-T2V-A14B-Diffusers (WanPipeline with two denoisers).

Training math is the shared Wan code in wan_common.py (mirrors diffusers v0.42.1 pipeline_wan.py):
  x0 = (vae.encode(px).latent_dist.mode() - latents_mean) / latents_std ; x_t = (1 - s) * x0 + s * noise ;
  target = noise - x0 ; expert(hidden_states=x_t, timestep=1000 * s, encoder_hidden_states=UMT5 at 512 tokens).
Experts: `transformer` = original high_noise_model, used by the pipeline while t >= boundary_ratio * 1000 (0.875 in
model_index.json); `transformer_2` = low_noise_model below. Each expert trains its own LoRA on its sigma range only
(t = 1000 * s): high [0.875, 1], low [0, 0.875) as in musubi-tuner's T2V split; two files, the low one loads with
load_into_transformer_2. Text only; still images are F = 1 clips (WanPipeline accepts num_frames 1).
Validation = official Wan2.2 wan/configs/wan_t2v_A14B.py: sample_steps 40, sample_shift 12.0,
sample_guide_scale (3.0, 4.0) in (low, high) order -> diffusers guidance_scale 4.0 (high), guidance_scale_2 3.0
(low); shared_config sample_fps 16, frame_num 81. The -Diffusers repo ships UniPC flow_shift 3.0, so generate()
always sets the shift explicitly.
"""

from diffusers import WanPipeline

try:  # dev: video_adapters/ on sys.path; notebook: lora_video_tools/ on sys.path or importable as a package
    import wan_common as W
except ImportError:
    from lora_video_tools import wan_common as W

FAMILY = "wan22_t2v"
PIPELINE_CLASS = WanPipeline
EXPERTS = ["transformer", "transformer_2"]  # high noise first
EXPERT_TAGS = {"transformer": "high", "transformer_2": "low"}
CONDITIONING = "t2v"
FPS = 16
SPATIAL_FACTOR = 8  # vae_stride (4, 8, 8)
SPATIAL_MULTIPLE = 16  # 8 x patch 2
TEMPORAL_FACTOR = 4
TOKEN_DIV = 16
VAL_STEPS = 40
VAL_SHIFT = 12.0
VAL_GUIDANCE = (4.0, 3.0)  # (high-noise expert, low-noise expert)
NEGATIVE_PROMPT = W.NEGATIVE_PROMPT
LORA_TARGETS = W.LORA_TARGETS
BOUNDARY_RATIO = 0.875  # wan_t2v_A14B.py boundary; the loaded pipeline's own boundary_ratio is what sigma_range uses
_SPEC = dict(two_experts=True, in_channels=16, z_dim=16, expand_timesteps=False)

encode_prompts = W.encode_prompts
lora_to_comfy = W.lora_to_comfy
save_lora = W.save_lora
load_lora = W.load_lora


def sigma_range(pipe, expert):
    return W.sigma_range(pipe, expert, EXPERTS)


def load_pipeline(model_path, dtype):
    return W.load_pipeline(model_path, dtype, PIPELINE_CLASS, FAMILY, **_SPEC)


def tiny_pipeline(dtype):
    # diffusers v0.42.1 tests/pipelines/wan/test_wan_22.py Wan22 two-expert components (boundary 0.875)
    pipe = W.tiny_pipeline(PIPELINE_CLASS, dtype, two_experts=True, boundary_ratio=BOUNDARY_RATIO)
    W.check_pipeline(pipe, FAMILY, **_SPEC)
    return pipe


def encode_video(pipe, frames, device, dtype, condition):  # dtype unused: the VAE runs in its own dtype (fp32)
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
