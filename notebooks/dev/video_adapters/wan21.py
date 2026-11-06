"""Wan2.1 T2V adapter: Wan-AI/Wan2.1-T2V-1.3B-Diffusers or Wan-AI/Wan2.1-T2V-14B-Diffusers (WanPipeline, one denoiser).

Training math is the shared Wan code in wan_common.py (mirrors diffusers v0.42.1 pipeline_wan.py):
  x0 = (vae.encode(px).latent_dist.mode() - latents_mean) / latents_std ; x_t = (1 - s) * x0 + s * noise ;
  target = noise - x0 ; transformer(hidden_states=x_t, timestep=1000 * s, encoder_hidden_states=UMT5 at 512 tokens).
Text only (CONDITIONING "t2v"); still images are F = 1 clips (WanPipeline accepts num_frames 1). One expert, so
sigma_range is (0, 1) and the LoRA file loads into `transformer`.
Validation defaults = official Wan2.1 generate.py for --task t2v-1.3B and t2v-14B (wan/configs/wan_t2v_*.py hold no
sampling values): sample_steps 50, sample_shift 5.0, sample_guide_scale 5.0, sample_fps 16, frame_num 81. Other
documented values: the Wan2.1 README advises 1.3B to use guide 6 with shift 8-12 (generate(shift=8.0,
guidance=(6.0, 6.0))), and its diffusers snippet uses flow_shift 3.0 for 480P. The -Diffusers repos ship UniPC
flow_shift 3.0, so generate() always sets the shift explicitly.
"""

from diffusers import WanPipeline

try:  # dev: video_adapters/ on sys.path; notebook: lora_video_tools/ on sys.path or importable as a package
    import wan_common as W
except ImportError:
    from lora_video_tools import wan_common as W

FAMILY = "wan21"
PIPELINE_CLASS = WanPipeline
EXPERTS = ["transformer"]
EXPERT_TAGS = {"transformer": "main"}
CONDITIONING = "t2v"
FPS = 16  # shared_config.py sample_fps
SPATIAL_FACTOR = 8  # vae_stride (4, 8, 8)
SPATIAL_MULTIPLE = 16  # 8 x patch 2
TEMPORAL_FACTOR = 4
TOKEN_DIV = 16
VAL_STEPS = 50
VAL_SHIFT = 5.0
VAL_GUIDANCE = (5.0, 5.0)
NEGATIVE_PROMPT = W.NEGATIVE_PROMPT
LORA_TARGETS = W.LORA_TARGETS
_SPEC = dict(two_experts=False, in_channels=16, z_dim=16, expand_timesteps=False)

encode_prompts = W.encode_prompts
lora_to_comfy = W.lora_to_comfy
save_lora = W.save_lora
load_lora = W.load_lora


def sigma_range(pipe, expert):
    return W.sigma_range(pipe, expert, EXPERTS)


def load_pipeline(model_path, dtype):
    return W.load_pipeline(model_path, dtype, PIPELINE_CLASS, FAMILY, **_SPEC)


def tiny_pipeline(dtype):
    # diffusers v0.42.1 tests/pipelines/wan/test_wan.py components, but with the UniPC scheduler the real
    # Wan2.1-T2V-*-Diffusers repos ship (the test uses FlowMatchEulerDiscreteScheduler(shift=7.0)).
    pipe = W.tiny_pipeline(PIPELINE_CLASS, dtype)
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
