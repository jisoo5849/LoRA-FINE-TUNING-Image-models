"""Wan2.2 I2V-A14B adapter: Wan-AI/Wan2.2-I2V-A14B-Diffusers (WanImageToVideoPipeline with two denoisers; also
Index-AniSora V3.2 once converted to that layout).

Training math is the shared Wan code in wan_common.py (mirrors diffusers v0.42.1 pipeline_wan_i2v.py, the original
wan/image2video.py and the musubi-tuner / ai-toolkit / diffusion-pipe Wan2.2 I2V losses):
  x0 = (vae.encode(px).latent_dist.mode() - latents_mean) / latents_std ; x_t = (1 - s) * x0 + s * noise ;
  y = cat([mask4, (vae.encode(cat([frame0, zeros(F - 1)])).mode() - latents_mean) / latents_std]) exactly as
  prepare_latents (mask4 = pixel mask [1, 0, ..., 0] with frame 0 repeated 4x, viewed as 4 channels per latent frame) ;
  expert(hidden_states=cat([x_t, y]) (16 + 4 + 16 = 36 channels), timestep=1000 * s, encoder_hidden_states=UMT5 at
  512 tokens) ; loss = MSE(pred, noise - x0) over ALL latent frames, the first one included (as those trainers do).
No CLIP image tokens (transformer image_dim null; Wan2.1 I2V had them). Clips only: with F = 1 the condition already
holds x0, so forward_loss refuses single-latent-frame samples, and mode "t2v" altogether.
Experts: `transformer` = high_noise_model, used while t >= boundary_ratio * 1000 (0.9 in model_index.json), trains
sigma [0.9, 1]; `transformer_2` = low_noise_model, sigma [0, 0.9) (musubi-tuner's I2V split 900-1000 / 0-900). Two
files; the low one loads with load_into_transformer_2.
Validation = official Wan2.2 wan/configs/wan_i2v_A14B.py: sample_steps 40, sample_shift 5.0, sample_guide_scale
(3.5, 3.5) -> guidance_scale 3.5 (high), guidance_scale_2 3.5 (low); shared_config sample_fps 16, frame_num 81. The
-Diffusers repo ships UniPC flow_shift 3.0, so generate() always sets the shift. generate() needs the start image.
"""

from diffusers import WanImageToVideoPipeline

try:  # dev: video_adapters/ on sys.path; notebook: lora_video_tools/ on sys.path or importable as a package
    import wan_common as W
except ImportError:
    from lora_video_tools import wan_common as W

FAMILY = "wan22_i2v"
PIPELINE_CLASS = WanImageToVideoPipeline
EXPERTS = ["transformer", "transformer_2"]  # high noise first
EXPERT_TAGS = {"transformer": "high", "transformer_2": "low"}
CONDITIONING = "i2v_concat"
FPS = 16
SPATIAL_FACTOR = 8  # vae_stride (4, 8, 8)
SPATIAL_MULTIPLE = 16  # 8 x patch 2
TEMPORAL_FACTOR = 4
TOKEN_DIV = 16
VAL_STEPS = 40
VAL_SHIFT = 5.0
VAL_GUIDANCE = (3.5, 3.5)  # (high-noise expert, low-noise expert)
NEGATIVE_PROMPT = W.NEGATIVE_PROMPT
LORA_TARGETS = W.LORA_TARGETS
BOUNDARY_RATIO = 0.9  # wan_i2v_A14B.py boundary; the loaded pipeline's own boundary_ratio is what sigma_range uses
_SPEC = dict(two_experts=True, in_channels=36, z_dim=16, expand_timesteps=False)

encode_prompts = W.encode_prompts
lora_to_comfy = W.lora_to_comfy
save_lora = W.save_lora
load_lora = W.load_lora


def sigma_range(pipe, expert):
    return W.sigma_range(pipe, expert, EXPERTS)


def load_pipeline(model_path, dtype):
    # rejects T2V-A14B (in_channels 16), Wan2.1 I2V (one expert, image_dim / image_encoder) and TI2V-5B (z 48)
    return W.load_pipeline(model_path, dtype, PIPELINE_CLASS, FAMILY, **_SPEC)


def tiny_pipeline(dtype):
    # diffusers v0.42.1 tests/pipelines/wan/test_wan_22_image_to_video.py Wan22 two-expert I2V components (in 36,
    # out 16, no image encoder), boundary 0.9 as the real model_index.json (the test uses 0.875)
    pipe = W.tiny_pipeline(PIPELINE_CLASS, dtype, in_channels=36, two_experts=True, boundary_ratio=BOUNDARY_RATIO)
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
