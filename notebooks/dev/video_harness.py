"""CPU test for one Wan video adapter: python video_harness.py <family>   (contract: VIDEO_INTERFACE.md)

Runs the family on its tiny random pipeline (fp32, CPU; section 9 in bf16) and checks:
  1 encode_prompts: (1, 512, D) on CPU, rows after the real token count exactly zero; when diffusers reports no ftfy it
    raises ImportError naming `pip install ftfy`
  2 encode_video, F = 1, 5, 9: shapes; pixels and normalization equal the pipeline's own math; "i2v_concat" cond equals
    WanImageToVideoPipeline.prepare_latents bit for bit; "ti2v" first latent frame equals the I2V image latent
  3 forward_loss: finite fp32 scalar, gradients only on LoRA params of the called expert; a perfect predictor gives
    loss 0, x_t - s * target == x0 through UniPC's own flow_prediction formula, timestep 1000 * s (float) within 1 of
    the int64 timestep WanPipeline passes; i2v inputs (cond channels / clean first frame, per-token timesteps)
  4 sigma_range per expert tiles [0, 1] at pipe.config.boundary_ratio
  5 save_lora: ComfyUI names, exact inverse of diffusers' converter, fp32 + metadata; load_lora into a fresh tiny
    pipeline (low file with load_into_transformer_2) gives bit-identical denoiser outputs
  6 generate: uint8 (T, H, W, 3); per-step expert routing and timesteps of the requested shift; guidance per expert;
    scheduler restored; deterministic; image required / refused by conditioning; "ti2v": the per-token timestep
    tensor the pipeline passes (T2V and I2V) equals forward_loss's for the same latent shape at s = t / 1000;
    offload=True: device=None without CUDA is a plain call (same frames); on mocked CUDA (pipeline call, offload and
    hook removal mocked) enable_model_cpu_offload(cuda) -> call at the requested shift / guidance -> remove_all_hooks,
    also when the call raises, for device="cuda" and for device=None with CUDA available
  7 LORA_TARGETS scopes inject 10 / 8 / 6 block linears per block
  8 save_pretrained -> load_pipeline round trip; wrong-variant folders (expert count, in_channels, VAE z_dim 16 / 48,
    expand_timesteps, CLIP image_encoder, WanVACEPipeline class) raise ValueError in the JSON pre-check, which names
    the defect
  9 bf16 as the notebook trains (bf16 base, LoRA params cast to fp32): bf16 inputs reach the denoiser, finite fp32 loss,
    gradients only on the fp32 LoRA of the called expert; save_lora -> load_lora into a fresh bf16 pipeline: LoRA params
    fp32 and bit-equal to the trained ones, denoiser outputs and loss equal within 1e-6, generate() frames identical
    (control: a plain load_lora_weights gives a bf16-rounded adapter, which these checks would catch)
Prints "VIDEO HARNESS OK <family>". On Windows set VIDEO_HARNESS_TMP to a short folder (long paths break saving).
Optional dev-only check: when C:/Users/shj04/.lvr/cfk/scripts/comfysim.py (a local simulation of ComfyUI's LoRA key
map, not part of this repo) exists, the saved files also go through it (env COMFYSIM / COMFY_ROOT override paths).
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_wan_lora_to_diffusers
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from peft import LoraConfig, get_peft_model_state_dict
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionConfig, CLIPVisionModelWithProjection

sys.path.insert(0, str(Path(__file__).resolve().parent / "video_adapters"))
fam = sys.argv[1]
A = importlib.import_module(fam)
import wan_common as W  # noqa: E402

NAMES = ["FAMILY", "PIPELINE_CLASS", "EXPERTS", "EXPERT_TAGS", "CONDITIONING", "FPS", "SPATIAL_FACTOR",
         "SPATIAL_MULTIPLE", "TEMPORAL_FACTOR", "TOKEN_DIV", "VAL_STEPS", "VAL_SHIFT", "VAL_GUIDANCE",
         "NEGATIVE_PROMPT", "LORA_TARGETS", "sigma_range", "load_pipeline", "tiny_pipeline", "encode_prompts",
         "encode_video", "forward_loss", "lora_to_comfy", "save_lora", "load_lora", "generate"]
missing = [n for n in NAMES if not hasattr(A, n)]
assert not missing, f"missing {missing}"
assert A.FAMILY == fam and A.CONDITIONING in ("t2v", "i2v_concat", "ti2v"), (A.FAMILY, A.CONDITIONING)
assert A.EXPERTS in (["transformer"], ["transformer", "transformer_2"]) and set(A.EXPERT_TAGS) == set(A.EXPERTS)
assert A.TEMPORAL_FACTOR == 4 and A.SPATIAL_MULTIPLE == 2 * A.SPATIAL_FACTOR == A.TOKEN_DIV and len(A.VAL_GUIDANCE) == 2
assert A.NEGATIVE_PROMPT == W.NEGATIVE_PROMPT and set(A.LORA_TARGETS) == {"all", "attn", "kvo"}
MODES = {"t2v": ["t2v"], "i2v_concat": ["i2v"], "ti2v": ["t2v", "i2v"]}[A.CONDITIONING]
TWO = len(A.EXPERTS) == 2
dt, dev = torch.float32, torch.device("cpu")
SM, SF, TF = A.SPATIAL_MULTIPLE, A.SPATIAL_FACTOR, A.TEMPORAL_FACTOR
H, WD = 2 * SM, 3 * SM  # non-square
TMP = Path(tempfile.mkdtemp(prefix="vh_", dir=os.environ.get("VIDEO_HARNESS_TMP")))
rng = np.random.default_rng(0)
torch.manual_seed(0)


def frames(n):
    return rng.integers(0, 256, size=(n, H, WD, 3), dtype=np.uint8)


def on(d):
    return None if d is None else {k: v.to(dev) for k, v in d.items()}


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise ValueError")


def first_image(fr):
    return pipe.video_processor.preprocess(Image.fromarray(fr[0]), height=H, width=WD).to(dev, torch.float32)


pipe = A.tiny_pipeline(dt)
assert isinstance(pipe, A.PIPELINE_CLASS), type(pipe)
for comp in pipe.components.values():
    if isinstance(comp, torch.nn.Module):
        comp.requires_grad_(False)
experts = {e: getattr(pipe, e) for e in A.EXPERTS}
assert all(m is not None for m in experts.values()) and (TWO or pipe.transformer_2 is None)
C = pipe.vae.config.z_dim
IN_CH = experts["transformer"].config.in_channels
print(f"0 {fam}: {A.PIPELINE_CLASS.__name__}, experts {A.EXPERTS}, conditioning {A.CONDITIONING}, z {C}, in {IN_CH}")

# 1 ---------------------------------------------------------------------------------------- encode_prompts
prompts = ["nkgreen, full body, front view", "nkgreen &amp; friend,  walking   slowly, side view, starry night sky"]
conds = A.encode_prompts(pipe, prompts, dev, dt)
D = pipe.text_encoder.config.d_model
assert len(conds) == 2
for p, c in zip(prompts, conds):
    e = c["prompt_embeds"]
    assert e.shape == (1, W.TEXT_LEN, D) and e.device.type == "cpu" and e.dtype == dt, (e.shape, e.device, e.dtype)
    n_tok = len(pipe.tokenizer(prompt_clean(p), max_length=W.TEXT_LEN, truncation=True)["input_ids"])  # with </s>
    rows = e[0].abs().sum(-1)
    assert (rows[:n_tok] > 0).all() and (rows[n_tok:] == 0).all(), (n_tok, int((rows > 0).sum()))
    print(f"1 encode_prompts: {tuple(e.shape)}, {n_tok} real tokens, {W.TEXT_LEN - n_tok} zero rows")
neg = A.encode_prompts(pipe, [A.NEGATIVE_PROMPT], dev, dt)[0]
with mock.patch.object(W, "is_ftfy_available", lambda: False):  # what diffusers reports in a venv without ftfy
    try:
        A.encode_prompts(pipe, prompts[:1], dev, dt)
    except ImportError as ex:
        assert "pip install ftfy" in str(ex), ex
    else:
        raise AssertionError("encode_prompts ran although diffusers reports no ftfy")
print("1 encode_prompts without ftfy: ImportError naming 'pip install ftfy'")

# 2 ---------------------------------------------------------------------------------------- encode_video
vids = {}
for n in (1, 5, 9):
    fr = frames(n)
    out = A.encode_video(pipe, fr, dev, dt, True)
    x0, f = out["x0"], 1 + (n - 1) // TF
    assert x0.shape == (1, C, f, H // SF, WD // SF) and x0.dtype == torch.float32 and x0.device.type == "cpu", x0.shape
    px = pipe.video_processor.preprocess_video([Image.fromarray(a) for a in fr], height=H, width=WD)
    assert torch.equal(W.to_pixels(fr), px), "pixels differ from VideoProcessor.preprocess_video"
    z = retrieve_latents(pipe.vae.encode(px.to(dtype=pipe.vae.dtype)), sample_mode="argmax").float()
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, C, 1, 1, 1).to(dev, torch.float32)
    inv_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, C, 1, 1, 1).to(dev, torch.float32)
    assert torch.equal(x0, (z - mean) * inv_std), "x0 != prepare_latents normalization"
    std = torch.tensor(pipe.vae.config.latents_std).view(1, C, 1, 1, 1)
    assert torch.allclose(x0, (z - mean) / std, rtol=1e-6, atol=1e-6), "x0 != (z - mean) / std"
    assert torch.allclose(W.denormalize(pipe.vae, x0), z, rtol=1e-5, atol=1e-5), "denormalize is not the inverse"
    note = ""
    if A.CONDITIONING == "i2v_concat":
        _, ref = pipe.prepare_latents(first_image(fr), 1, C, H, WD, n, torch.float32, dev, None, None, None)
        assert out["cond"].shape == (1, 4 + C, f, H // SF, WD // SF) and torch.equal(out["cond"], ref)
        assert "cond" not in A.encode_video(pipe, fr, dev, dt, False)
        note = ", cond == prepare_latents (bit-exact)"
    else:
        assert "cond" not in out, "only i2v_concat returns cond"
    if A.CONDITIONING == "ti2v":
        _, lc, _ = W.as_i2v(pipe).prepare_latents(first_image(fr), 1, C, H, WD, n, torch.float32, dev, None, None,
                                                  None)
        d = (x0[:, :, :1] - lc).abs().max().item()
        assert d < 1e-4, d
        note = f", first latent frame vs I2V image latent max diff {d:.1e}"
    vids[n] = (fr, out)
    print(f"2 encode_video F={n}: x0 {tuple(x0.shape)} == (z - mean) * (1 / std) of the pipeline{note}")

# 3 ---------------------------------------------------------------------------------------- forward_loss
cfg_all = LoraConfig(r=4, lora_alpha=4, init_lora_weights=True, target_modules=A.LORA_TARGETS["all"])
for den in experts.values():
    den.add_adapter(cfg_all)
    for n_, p in den.named_parameters():
        p.requires_grad_("lora_" in n_)
opt = torch.optim.Adam([p for den in experts.values() for p in den.parameters() if p.requires_grad], lr=1e-2)
text = on(conds[1])
out9 = vids[9][1]
x9 = out9["x0"].to(dev)
c9 = on(out9) if A.CONDITIONING == "i2v_concat" else None

for e, den in experts.items():
    lo, hi = A.sigma_range(pipe, e)
    for mode in MODES:
        for s in (lo + 1e-3, (lo + hi) / 2, hi - 1e-3):
            loss = A.forward_loss(pipe, den, x9, c9, text, torch.randn_like(x9), torch.tensor([s]), mode)
            assert loss.ndim == 0 and loss.dtype == torch.float32 and torch.isfinite(loss), loss
            loss.backward()
            stray = [n_ for m in experts.values() for n_, p in m.named_parameters() if p.grad is not None
                     and ("lora_" not in n_ or m is not den)]
            assert not stray, f"gradients outside the LoRA of {e}: {stray[:3]}"
            assert sum(p.grad.abs().sum().item() for p in den.parameters() if p.grad is not None) > 0, "no LoRA grad"
            opt.step()
            opt.zero_grad(set_to_none=True)
        print(f"3 forward_loss {e} ({A.EXPERT_TAGS[e]}) mode {mode}: sigma {lo:.3f}..{hi:.3f} loss {loss.item():.4f}, "
              "grads only on its LoRA")
if A.CONDITIONING != "i2v_concat":  # images: F = 1 clips in T2V mode
    x1 = vids[1][1]["x0"].to(dev)
    l1 = A.forward_loss(pipe, experts[A.EXPERTS[-1]], x1, None, text, torch.randn_like(x1), torch.tensor([0.3]), "t2v")
    assert torch.isfinite(l1)
    print(f"3 forward_loss F=1 image (t2v mode): loss {l1.item():.4f}")
t0, half = experts["transformer"], torch.tensor([0.5])
for mode in {"t2v", "i2v"} - set(MODES):
    raises(A.forward_loss, pipe, t0, x9, c9, text, torch.randn_like(x9), half, mode)
if "i2v" in MODES:  # an image (one latent frame) cannot train I2V
    x1 = vids[1][1]["x0"].to(dev)
    c1 = on(vids[1][1]) if A.CONDITIONING == "i2v_concat" else None
    raises(A.forward_loss, pipe, t0, x1, c1, text, torch.randn_like(x1), half, "i2v")
if A.CONDITIONING == "i2v_concat":  # cond is required
    raises(A.forward_loss, pipe, t0, x9, None, text, torch.randn_like(x9), half, "i2v")


class Oracle(torch.nn.Module):
    """Stand-in denoiser: records its inputs and returns `out` (the exact velocity noise - x0)."""

    def __init__(self, out, config):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(()))
        self.out, self.config, self.seen = out, config, []

    def forward(self, hidden_states, timestep, encoder_hidden_states, return_dict=False, **kw):
        self.seen.append((hidden_states.detach(), timestep.detach(), encoder_hidden_states.detach()))
        return (self.out.clone(),)


ref = W.with_flow_shift(pipe.scheduler, A.VAL_SHIFT)
ref.set_timesteps(A.VAL_STEPS)
fx, fh, fw = x9.shape[2:]
tok = fh // 2 * (fw // 2)
for i in (0, A.VAL_STEPS // 2, A.VAL_STEPS - 1):
    sig = ref.sigmas[i].reshape(1)
    for mode in MODES:
        noise = torch.randn_like(x9)
        v = noise - x9
        if mode == "i2v" and A.CONDITIONING == "ti2v":
            v[:, :, 0] = 1e3  # clean frame is outside the loss: anything there must not count
        orc = Oracle(v, experts["transformer"].config)
        loss = A.forward_loss(pipe, orc, x9, c9, text, noise, sig, mode)
        assert len(orc.seen) == 1 and loss.item() == 0.0, (len(orc.seen), loss.item())
        hs, ts, enc = orc.seen[0]
        assert hs.shape[1] == IN_CH and torch.equal(enc, text["prompt_embeds"].to(dt))
        x_t = hs[:, :C]
        ref._step_index = i
        x0_rec = ref.convert_model_output(noise - x9, sample=x_t)  # UniPC flow_prediction: sample - sigma * v
        sl = slice(1, None) if (mode == "i2v" and A.CONDITIONING == "ti2v") else slice(None)
        assert torch.allclose(x0_rec[:, :, sl], x9[:, :, sl], atol=1e-5), "x_t - sigma * target != x0"
        assert ts.dtype.is_floating_point, ts.dtype
        t_pipe = float(ref.timesteps[i])  # what WanPipeline passes at this step (int64)
        if ts.ndim == 1:
            assert torch.allclose(ts, 1000 * sig) and abs(ts.item() - t_pipe) < 1, (ts, t_pipe)
        else:  # ti2v: per-token (b, f * h/2 * w/2)
            want = torch.full((1, fx * tok), 1000 * sig.item())
            if mode == "i2v":
                want[:, :tok] = 0
            assert ts.shape == want.shape and torch.allclose(ts, want) and abs(ts.max().item() - t_pipe) < 1
        if A.CONDITIONING == "i2v_concat":
            assert torch.equal(hs[:, C:], c9["cond"]), "condition channels differ"
        if mode == "i2v" and A.CONDITIONING == "ti2v":
            assert torch.equal(hs[:, :, 0], x9[:, :, 0]), "first latent frame is not the clean x0"
print(f"3 perfect predictor: loss 0, x_t - s * (noise - x0) == x0 via UniPC flow_prediction, timestep 1000 * s "
      f"(float) within 1 of WanPipeline's int64 t at steps 0/{A.VAL_STEPS // 2}/{A.VAL_STEPS - 1} "
      f"(shift {A.VAL_SHIFT})")

# 4 ---------------------------------------------------------------------------------------- sigma_range
ranges = {e: A.sigma_range(pipe, e) for e in A.EXPERTS}
if TWO:  # the pipeline runs `transformer` while t >= boundary_ratio * 1000 (checked against real calls in 6)
    b = pipe.config.boundary_ratio
    assert ranges["transformer"] == (b, 1.0) and ranges["transformer_2"] == (0.0, b), ranges
else:
    assert ranges["transformer"] == (0.0, 1.0), ranges
raises(A.sigma_range, pipe, "unet")
print(f"4 sigma_range: {ranges}")

# 5 ---------------------------------------------------------------------------------------- save / load LoRA
paths, n_mod = {}, {}
for e, den in experts.items():
    sd = get_peft_model_state_dict(den)
    comfy = A.lora_to_comfy(sd)
    n_mod[e] = sum(k.endswith("lora_A.weight") for k in sd)
    assert len(comfy) == len(sd) == 2 * n_mod[e] == 2 * 10 * den.config.num_layers, (len(comfy), len(sd))
    assert all(k.startswith("diffusion_model.blocks.") and ".lora_" in k for k in comfy)
    assert not any(k.endswith(".alpha") for k in comfy)
    back = _convert_non_diffusers_wan_lora_to_diffusers(dict(comfy))
    assert set(back) == {"transformer." + k for k in sd} and all(torch.equal(back["transformer." + k], v)
                                                                  for k, v in sd.items()), "not the exact inverse"
    path = A.save_lora(sd, str(TMP / f"{fam}_{A.EXPERT_TAGS[e]}.safetensors"), {"family": fam, "expert": e, "rank": 4})
    with safe_open(path, "pt") as fh_:
        meta = fh_.metadata()
        assert set(fh_.keys()) == set(comfy) and meta["family"] == fam and meta["rank"] == "4"
        assert all(fh_.get_tensor(k).dtype == torch.float32 for k in fh_.keys())
    paths[e] = path
    print(f"5 save_lora {e}: {len(comfy)} keys e.g. {sorted(comfy)[0]}, exact inverse of diffusers' converter")
raises(A.lora_to_comfy, {"blocks.0.norm2.lora_A.weight": torch.zeros(1, 1)})
raises(A.lora_to_comfy, {"proj_out.lora_A.weight": torch.zeros(1, 1)})
raises(A.lora_to_comfy, {"blocks.0.attn1.to_q.lora_A.weight": torch.zeros(1, 1)})  # lora_B missing

sim_py = Path(os.environ.get("COMFYSIM", "C:/Users/shj04/.lvr/cfk/scripts/comfysim.py"))
comfy_root = Path(os.environ.get("COMFY_ROOT", "C:/Users/shj04/.lvr/cfk/dl/src/ComfyUI-master"))
if sim_py.is_file() and sim_py.with_name("wanmap.py").is_file() and comfy_root.is_dir():
    sys.path.insert(0, str(sim_py.parent))
    cs, wm = importlib.import_module("comfysim"), importlib.import_module("wanmap")
    sim = cs.ComfySim(str(comfy_root))
    for e, den in experts.items():
        model_sd = {}
        for k, v in den.state_dict().items():
            if "lora_" not in k:
                try:
                    model_sd["diffusion_model." + wm.wan_param_to_comfy(k.replace(".base_layer", ""))] = tuple(v.shape)
                except KeyError:
                    pass
        with safe_open(paths[e], "pt") as fh_:
            shapes = {k: tuple(fh_.get_slice(k).get_shape()) for k in fh_.keys()}
        patches, not_loaded = sim.load(shapes, sim.key_map(sim.make_model("WAN21", model_sd)))
        bad = cs.check_shapes(patches, model_sd)
        assert len(patches) == n_mod[e] and not not_loaded and not bad, (len(patches), not_loaded[:3], bad[:3])
        print(f"5 comfysim (dev only) {e}: {len(patches)}/{n_mod[e]} modules patched, 0 'lora key not loaded'")
else:
    print("5 comfysim not found: ComfyUI key-map check skipped")

pipe2 = A.tiny_pipeline(dt)
for e in A.EXPERTS:
    A.load_lora(pipe2, paths[e], e, f"niki_{A.EXPERT_TAGS[e]}")
if not TWO:
    raises(A.load_lora, pipe2, paths["transformer"], "transformer_2", "x")
probe = torch.randn(1, IN_CH, *x9.shape[2:])
with torch.no_grad():
    for e, den in experts.items():
        den2 = getattr(pipe2, e)
        assert list(den2.peft_config) == [f"niki_{A.EXPERT_TAGS[e]}"], list(den2.peft_config)
        lo, hi = A.sigma_range(pipe, e)
        kw = dict(hidden_states=probe, timestep=torch.tensor([500.0 * (lo + hi)]), encoder_hidden_states=text[
            "prompt_embeds"], return_dict=False)
        den.eval()
        den2.eval()
        a, b2 = den(**kw)[0], den2(**kw)[0]
        den.disable_adapters()
        base = den(**kw)[0]
        den.enable_adapters()
        assert torch.equal(a, b2), (a - b2).abs().max()
        effect = (a - base).abs().max().item()
        assert effect > 0, "LoRA has no effect"
        print(f"5 load_lora {e}: outputs bit-identical to the trained adapter (LoRA effect {effect:.2e})")

# 6 ---------------------------------------------------------------------------------------- generate
img = Image.fromarray(frames(1)[0])
sched0, shift0 = pipe2.scheduler, pipe2.scheduler.config.flow_shift


def gen(steps=None, shift=None, guidance=None, image=None, nf=5, offload=False, device=dev):
    seen = []
    hooks = [getattr(pipe2, e).register_forward_pre_hook(
        lambda m, a, k, e=e: seen.append((e, k["timestep"].detach().float())), with_kwargs=True) for e in A.EXPERTS]
    try:
        vid = A.generate(pipe2, conds[0], neg, height=H, width=WD, num_frames=nf,
                         generator=torch.Generator("cpu").manual_seed(7), image=image, steps=steps, shift=shift,
                         guidance=guidance, device=device, offload=offload)
    finally:
        for hk in hooks:
            hk.remove()
    steps, shift = steps or A.VAL_STEPS, shift or A.VAL_SHIFT
    g = guidance or A.VAL_GUIDANCE
    assert isinstance(vid, np.ndarray) and vid.dtype == np.uint8 and vid.shape == (nf, H, WD, 3), (vid.dtype,
                                                                                                    vid.shape)
    assert pipe2.scheduler is sched0 and pipe2.scheduler.config.flow_shift == shift0, "scheduler not restored"
    r = UniPCMultistepScheduler.from_config(sched0.config, flow_shift=shift)
    r.set_timesteps(steps)
    per = 2 if g[0] > 1 else 1  # cond + uncond
    assert len(seen) == per * steps, (len(seen), per, steps)
    for k, (e, ts) in enumerate(seen):
        st = k // per
        t_ref, s_ref = float(r.timesteps[st]), r.sigmas[st].item()
        assert ts.max().item() == t_ref and abs(t_ref - 1000 * s_ref) < 1, (k, ts.max(), t_ref)
        want = "transformer" if (not TWO or t_ref >= pipe2.config.boundary_ratio * 1000) else "transformer_2"
        lo, hi = ranges[e]
        assert e == want and lo <= s_ref <= hi and (e != "transformer_2" or s_ref < hi), (k, e, want, t_ref, s_ref)
    if A.CONDITIONING == "ti2v":  # the pipeline's per-token timesteps == forward_loss's tensor at s = t / 1000
        z = torch.zeros(1, C, 1 + (nf - 1) // TF, H // SF, WD // SF)
        for e, ts in seen:
            orc = Oracle(z, experts[e].config)
            mode = "t2v" if image is None else "i2v"
            A.forward_loss(pipe2, orc, z, None, conds[0], z, ts.max().reshape(1) / 1000, mode)
            assert orc.seen[0][1].shape == ts.shape and torch.allclose(orc.seen[0][1], ts, atol=1e-3), ts
    if image is None or isinstance(pipe2, WanImageToVideoPipeline):  # generate() ran pipe2 itself
        assert pipe2._guidance_scale == g[0] and (not TWO or pipe2._guidance_scale_2 == g[1])
    return vid, [e for e, _ in seen]


modes_gen = {"t2v": [None], "i2v_concat": [img], "ti2v": [None, img]}[A.CONDITIONING]
for im in modes_gen:
    vid, called = gen(steps=4, image=im)
    vid2, _ = gen(steps=4, image=im)
    assert np.array_equal(vid, vid2), "generate is not deterministic for a fixed generator"
    names = sorted(set(called))
    assert names == sorted(A.EXPERTS), f"experts called {names}"
    print(f"6 generate ({'i2v' if im is not None else 't2v'}): {vid.shape} {vid.dtype}, 4 steps, shift {A.VAL_SHIFT}, "
          f"guidance {A.VAL_GUIDANCE}: calls {[called.count(e) for e in A.EXPERTS]} per expert {A.EXPERTS}, "
          f"scheduler restored (flow_shift {shift0})"
          + (", per-token timesteps == forward_loss's at s = t / 1000" if A.CONDITIONING == "ti2v" else ""))
vid3, _ = gen(steps=3, shift=5.0, guidance=(2.0, 2.0), image=modes_gen[0], nf=1)
vid4, _ = gen(steps=3, shift=5.0, guidance=(2.0, 2.0), image=modes_gen[0], nf=1, offload=True)  # CPU: plain call
assert np.array_equal(vid3, vid4)
if not torch.cuda.is_available():  # device=None + offload=True without CUDA: a plain call where the modules are
    vid5, _ = gen(steps=3, shift=5.0, guidance=(2.0, 2.0), image=modes_gen[0], nf=1, offload=True, device=None)
    assert np.array_equal(vid3, vid5)
if A.CONDITIONING == "t2v":
    raises(A.generate, pipe2, conds[0], neg, height=H, width=WD, num_frames=5, generator=None, image=img, steps=2)
if A.CONDITIONING == "i2v_concat":
    raises(A.generate, pipe2, conds[0], neg, height=H, width=WD, num_frames=5, generator=None, steps=2)
raises(A.generate, pipe2, conds[0], None, height=H, width=WD, num_frames=5, generator=None, image=modes_gen[0], steps=2,
       guidance=(3.0, 3.0))
if TWO:
    raises(A.generate, pipe2, conds[0], neg, height=H, width=WD, num_frames=5, generator=None, steps=2,
           image=modes_gen[0], guidance=(1.0, 3.0))
print("6 generate: custom steps/shift/guidance, F=1, offload=True on CPU (device=cpu and device=None), argument checks")


class OnCuda:
    """A cached embedding 'moved to CUDA' on this CPU-only torch: generate() only calls .to() before the pipeline."""

    def to(self, *a, **k):
        return self


def mocked_cuda(image, device, boom=False, cuda=True):
    """generate(offload=True) with the pipeline call, enable_model_cpu_offload and remove_all_hooks mocked and
    torch.cuda.is_available() -> cuda; returns the call log."""
    target = WanImageToVideoPipeline if image is not None else type(pipe2)  # the class generate() runs
    log = []

    def call(self, *a, **k):
        log.append(("call", self.scheduler.config.flow_shift, k.get("guidance_scale"), k.get("guidance_scale_2")))
        if boom:
            raise RuntimeError("boom")
        return SimpleNamespace(frames=[np.zeros((1, H, WD, 3), np.float32)])

    with mock.patch.object(target, "__call__", call), \
            mock.patch.object(target, "enable_model_cpu_offload",
                              lambda self, gpu_id=None, device=None: log.append(("offload", str(device)))), \
            mock.patch.object(target, "remove_all_hooks", lambda self: log.append(("remove",))), \
            mock.patch.object(torch.cuda, "is_available", lambda: cuda):
        try:
            A.generate(pipe2, {"prompt_embeds": OnCuda()}, {"prompt_embeds": OnCuda()}, height=H, width=WD,
                       num_frames=1, generator=None, image=image, steps=2, device=device, offload=True)
        except RuntimeError as ex:
            assert boom and str(ex) == "boom", ex
        else:
            assert not boom, "the exception from the pipeline call was swallowed"
    assert pipe2.scheduler is sched0 and pipe2.scheduler.config.flow_shift == shift0, "scheduler not restored"
    return log


g_call = ("call", A.VAL_SHIFT, A.VAL_GUIDANCE[0], A.VAL_GUIDANCE[1] if TWO else None)
for im in modes_gen:
    for device, boom in (("cuda", False), ("cuda", True), (None, False), (None, True)):
        log = mocked_cuda(im, device, boom)
        assert log == [("offload", "cuda"), g_call, ("remove",)], (device, boom, log)
    assert mocked_cuda(im, None, cuda=False) == [g_call], "no CUDA: offload must be a plain call"
    assert mocked_cuda(im, "cpu") == [g_call], "device=cpu: offload must be a plain call"
print(f"6 generate offload on mocked CUDA ({'/'.join('i2v' if im is not None else 't2v' for im in modes_gen)}): "
      f"enable_model_cpu_offload(cuda) -> call {g_call[1:]} -> remove_all_hooks, also on an exception, for "
      "device='cuda' and device=None; plain call without CUDA or with device='cpu'; scheduler restored")

# 7 ---------------------------------------------------------------------------------------- LoRA scopes
L = experts["transformer"].config.num_layers
for scope, targets in A.LORA_TARGETS.items():
    m = type(experts["transformer"]).from_config(experts["transformer"].config)
    m.add_adapter(LoraConfig(r=4, lora_alpha=4, target_modules=targets))
    wrapped = sorted({n_.split(".lora_A")[0] for n_, _ in m.named_parameters() if ".lora_A." in n_})
    want = {"all": 10, "attn": 8, "kvo": 6}[scope] * L
    assert len(wrapped) == want and all(n_.startswith("blocks.") for n_ in wrapped), (scope, len(wrapped), want)
    if scope != "all":
        assert not any(".ffn." in n_ for n_ in wrapped)
    if scope == "kvo":
        assert not any(n_.endswith("to_q") for n_ in wrapped)
    print(f"7 LORA_TARGETS[{scope!r}]: {len(wrapped)} block linears ({want // L} per block x {L})")

# 8 ---------------------------------------------------------------------------------------- load_pipeline
fresh = A.tiny_pipeline(dt)
fresh.save_pretrained(TMP / "good")
p3 = A.load_pipeline(str(TMP / "good"), dt)
assert isinstance(p3, A.PIPELINE_CLASS) and p3.vae.dtype == torch.float32
for e in A.EXPERTS:
    sa, sb = getattr(fresh, e).state_dict(), getattr(p3, e).state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa), e
comps = dict(fresh.components)
tx = fresh.transformer
cfg_bad = {**dict(tx.config), "in_channels": 36 if tx.config.in_channels != 36 else 16}
br, xt = fresh.config.boundary_ratio, fresh.config.expand_timesteps
extra = dict(expand_timesteps=xt)
torch.manual_seed(0)  # Wan2.1 I2V layout: CLIP image encoder (diffusers tests/pipelines/wan/test_wan_image_to_video.py)
clip = CLIPVisionModelWithProjection(CLIPVisionConfig(hidden_size=4, projection_dim=4, num_hidden_layers=2,
                                                      num_attention_heads=2, image_size=32, intermediate_size=16,
                                                      patch_size=1))
no_img = {k: v for k, v in comps.items() if k not in ("image_encoder", "image_processor")}
z_other = 48 if C == 16 else 16  # the other Wan VAE: Wan2.2 TI2V-5B (z 48) / Wan2.1 (z 16)
wrong = {  # name: (pipeline, or None = the good folder with an edited model_index.json; what the message must name)
    "experts": (type(fresh)(**{**comps, "transformer_2": None}, boundary_ratio=None, **extra) if TWO else
                type(fresh)(**{**comps, "transformer_2": type(tx).from_config(tx.config)}, boundary_ratio=0.875,
                            **extra), "needs two experts" if TWO else "needs one transformer"),
    "in_channels": (type(fresh)(**{**comps, "transformer": type(tx).from_config(cfg_bad),
                                   "transformer_2": type(tx).from_config(cfg_bad) if TWO else None},
                                boundary_ratio=br, **extra), "in_channels="),
    "z_dim": (type(fresh)(**{**comps, "vae": W.tiny_pipeline(type(fresh), dt, z_dim=z_other).vae}, boundary_ratio=br,
                          **extra), f"vae.z_dim={z_other}"),
    "expand_timesteps": (type(fresh)(**comps, boundary_ratio=br, expand_timesteps=not xt), "expand_timesteps="),
    "image_encoder": (WanImageToVideoPipeline(**no_img, image_encoder=clip,
                                              image_processor=CLIPImageProcessor(crop_size=32, size=32),
                                              boundary_ratio=br, **extra),
                      "image_encoder=CLIPVisionModelWithProjection"),
    "class_name": (None, "pipeline is WanVACEPipeline"),
}
for name, (bad, want) in wrong.items():
    folder = TMP / f"bad_{name}"
    if bad is not None:
        bad.save_pretrained(folder)
    else:
        shutil.copytree(TMP / "good", folder)
        mi = folder / "model_index.json"
        mi.write_text(json.dumps({**json.loads(mi.read_text(encoding="utf-8")), "_class_name": "WanVACEPipeline"}),
                      encoding="utf-8")
    msg = raises(A.load_pipeline, str(folder), dt)
    found = msg.split("variant: ", 1)[-1]
    # the JSON pre-check names the folder (before any weights load); the new cases name exactly one defect
    assert str(folder) in msg and want in found, (name, msg)
    assert name in ("experts", "in_channels") or "; " not in found, (name, msg)
    print(f"8 load_pipeline wrong variant ({name}): ValueError ...{found[:140]}")
print("8 save_pretrained -> load_pipeline: same class, VAE fp32, expert weights identical")

# 9 ---------------------------------------------------------------------------------------- bf16 (notebook layout)
bf = torch.bfloat16
pb = A.tiny_pipeline(bf)
for comp in pb.components.values():
    if isinstance(comp, torch.nn.Module):
        comp.requires_grad_(False)
enc_b = A.encode_video(pb, vids[9][0], dev, bf, True)
text_b = on(A.encode_prompts(pb, [prompts[1]], dev, bf)[0])
neg_b = A.encode_prompts(pb, [A.NEGATIVE_PROMPT], dev, bf)[0]
assert enc_b["x0"].dtype == torch.float32 and text_b["prompt_embeds"].dtype == bf, text_b["prompt_embeds"].dtype
xb = enc_b["x0"].to(dev)
cb = on(enc_b) if A.CONDITIONING == "i2v_concat" else None
exp_b = {e: getattr(pb, e) for e in A.EXPERTS}
for den in exp_b.values():
    den.add_adapter(cfg_all)
    for n_, p in den.named_parameters():
        p.requires_grad_("lora_" in n_)
        if "lora_" in n_:
            p.data = p.data.float()  # as the notebook: LoRA params fp32 on the bf16 base
    assert den.dtype == bf, den.dtype
opt_b = torch.optim.Adam([p for den in exp_b.values() for p in den.parameters() if p.requires_grad], lr=1e-2)
for e, den in exp_b.items():
    seen = []
    hk = den.register_forward_pre_hook(
        lambda m, a, k: seen.append((k["hidden_states"].dtype, k["encoder_hidden_states"].dtype)), with_kwargs=True)
    lo, hi = A.sigma_range(pb, e)
    for mode in MODES:
        for s in (lo + 1e-3, (lo + hi) / 2, hi - 1e-3):
            loss = A.forward_loss(pb, den, xb, cb, text_b, torch.randn_like(xb), torch.tensor([s]), mode)
            assert loss.ndim == 0 and loss.dtype == torch.float32 and torch.isfinite(loss), loss
            loss.backward()
            grads = [(n_, p.grad.dtype, m is den) for m in exp_b.values() for n_, p in m.named_parameters()
                     if p.grad is not None]
            assert grads and all("lora_" in n_ and d == torch.float32 and own for n_, d, own in grads), grads[:3]
            opt_b.step()
            opt_b.zero_grad(set_to_none=True)
    hk.remove()
    assert set(seen) == {(bf, bf)}, set(seen)
    print(f"9 bf16 forward_loss {e} ({A.EXPERT_TAGS[e]}): bf16 hidden_states and text reach the denoiser, fp32 loss "
          f"{loss.item():.4f}, grads only on its fp32 LoRA")

pb2 = A.tiny_pipeline(bf)  # A.load_lora: the file's fp32 values written back
pb3 = A.tiny_pipeline(bf)  # control: plain load_lora_weights, the adapter cast to the bf16 base dtype
probe_b = torch.randn(1, IN_CH, *xb.shape[2:]).to(bf)
for e, den in exp_b.items():
    tag = A.EXPERT_TAGS[e]
    sd = get_peft_model_state_dict(den)
    path = A.save_lora(sd, str(TMP / f"{fam}_bf16_{tag}.safetensors"), {"family": fam, "expert": e})
    A.load_lora(pb2, path, e, f"bf_{tag}")
    pb3.load_lora_weights(path, adapter_name=f"plain_{tag}", load_into_transformer_2=(e == "transformer_2"))
    den2, den3 = getattr(pb2, e), getattr(pb3, e)
    sd2 = get_peft_model_state_dict(den2, adapter_name=f"bf_{tag}")
    sd3 = get_peft_model_state_dict(den3, adapter_name=f"plain_{tag}")
    assert sd.keys() == sd2.keys() == sd3.keys() and all(v.dtype == torch.float32 for v in sd.values())
    assert all(sd2[k].dtype == torch.float32 and torch.equal(sd2[k], sd[k]) for k in sd), "reloaded LoRA != trained"
    w_plain = max((sd3[k].float() - sd[k]).abs().max().item() for k in sd)
    assert w_plain > 0, "control: a plain load_lora_weights should have rounded the fp32 LoRA to bf16"
    lo, hi = A.sigma_range(pb, e)
    sig = torch.tensor([(lo + hi) / 2])
    noise = torch.randn_like(xb)
    kw = dict(hidden_states=probe_b, timestep=1000 * sig, encoder_hidden_states=text_b["prompt_embeds"],
              return_dict=False)
    with torch.no_grad():
        for m in (den, den2, den3):
            m.eval()
        a, b2, c3 = den(**kw)[0], den2(**kw)[0], den3(**kw)[0]
        den.disable_adapters()
        base = den(**kw)[0]
        den.enable_adapters()
        l1 = A.forward_loss(pb, den, xb, cb, text_b, noise, sig, MODES[-1])
        l2 = A.forward_loss(pb2, den2, xb, cb, text_b, noise, sig, MODES[-1])
    d_out, d_plain = (a.float() - b2.float()).abs().max().item(), (a.float() - c3.float()).abs().max().item()
    d_loss, effect = abs(l1.item() - l2.item()), (a.float() - base.float()).abs().max().item()
    assert a.dtype == bf and effect > 0 and d_out <= 1e-6 and d_loss <= 1e-6, (d_out, d_loss, effect)
    print(f"9 bf16 save_lora -> load_lora {e}: LoRA params fp32 and bit-equal to the trained ones, output diff "
          f"{d_out:.1e}, loss diff {d_loss:.1e} (LoRA effect {effect:.2e}; control plain load_lora_weights: weights "
          f"off by {w_plain:.1e}, output by {d_plain:.1e})")
kw_g = dict(height=H, width=WD, num_frames=5, image=modes_gen[0], steps=2, device=dev)
v_tr = A.generate(pb, text_b, neg_b, generator=torch.Generator("cpu").manual_seed(3), **kw_g)
v_re = A.generate(pb2, text_b, neg_b, generator=torch.Generator("cpu").manual_seed(3), **kw_g)
assert v_tr.dtype == np.uint8 and v_tr.shape == (5, H, WD, 3) and np.array_equal(v_tr, v_re), "reloaded != trained"
print(f"9 bf16 generate: trained and reloaded LoRA give identical frames {v_tr.shape}")

shutil.rmtree(TMP, ignore_errors=True)
print(f"VIDEO HARNESS OK {fam}")
