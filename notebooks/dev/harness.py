"""Common CPU test for one adapter: python harness.py <family>"""
import importlib
import sys
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict

sys.path.insert(0, str(Path(__file__).parent / "adapters"))
fam = sys.argv[1]
A = importlib.import_module(fam)
for name in ["FAMILY", "PIPELINE_CLASS", "DENOISER_ATTR", "PREDICTION", "PIXEL_MULTIPLE", "DEFAULT_STEPS",
             "DEFAULT_GUIDANCE", "LORA_TARGET_MODULES", "load_pipeline", "tiny_pipeline", "encode_prompts",
             "encode_images", "forward_loss", "save_lora", "generate"]:
    assert hasattr(A, name), f"missing {name}"
assert A.PREDICTION in ("flow", "eps", "v_prediction")

torch.manual_seed(0)
dt = torch.float32
dev = torch.device("cpu")
pipe = A.tiny_pipeline(dt)
assert isinstance(pipe, A.PIPELINE_CLASS), type(pipe)
den = getattr(pipe, A.DENOISER_ATTR)
for p in pipe.components.values():
    if isinstance(p, torch.nn.Module):
        p.requires_grad_(False)

# two prompts of different length -> one dict each, batch dim 1, CPU
conds = A.encode_prompts(pipe, ["nkgreen, full body, front view", "nkgreen, close-up, side view, teary sad look, starry night sky"], dev, dt)
assert len(conds) == 2 and all(isinstance(c, dict) for c in conds)
for c in conds:
    for k, v in c.items():
        if torch.is_tensor(v):
            assert v.shape[0] == 1 and v.device.type == "cpu", (k, v.shape, v.device)

# two non-square sizes on the pixel grid
m = A.PIXEL_MULTIPLE
for (H, W) in [(2 * m, 3 * m), (3 * m, 2 * m)]:
    px = torch.rand(1, 3, H, W) * 2 - 1
    x0 = A.encode_images(pipe, px, dev, dt)
    assert x0.device.type == "cpu" and x0.dtype == torch.float32 and torch.isfinite(x0).all(), (x0.dtype, x0.device)

    cfg = LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=A.LORA_TARGET_MODULES)
    if not getattr(den, "_hf_peft_config_loaded", False):
        den.add_adapter(cfg)
    n_lora = sum(p.numel() for n, p in den.named_parameters() if "lora_" in n)
    assert n_lora > 0, "no LoRA params injected"
    for n, p in den.named_parameters():
        p.requires_grad_("lora_" in n)

    noise = torch.randn_like(x0)
    sigma = torch.tensor([0.37])
    loss = A.forward_loss(pipe, den, x0, {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in conds[1].items()}, noise, sigma)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    loss.backward()
    g = sum(p.grad.abs().sum().item() for n, p in den.named_parameters() if p.grad is not None)
    assert g > 0, "no gradient reached LoRA"
    den.zero_grad(set_to_none=True)
    print(f"  size {H}x{W}: latents {tuple(x0.shape)} loss {loss.item():.4f}")

# sigma extremes stay finite
for s in (0.001, 0.999):
    l2 = A.forward_loss(pipe, den, x0, conds[0], torch.randn_like(x0), torch.tensor([s]))
    assert torch.isfinite(l2)

with tempfile.TemporaryDirectory() as d:
    sd = get_peft_model_state_dict(den)
    path = A.save_lora(pipe, sd, d)
    assert Path(path).exists(), path
    pipe2 = A.tiny_pipeline(dt)
    pipe2.load_lora_weights(path if Path(path).is_file() else d)
    den2 = getattr(pipe2, A.DENOISER_ATTR)
    names2 = [n for n, _ in den2.named_parameters() if "lora_" in n]
    assert len(names2) > 0, "LoRA did not load back"

    gen = torch.Generator("cpu").manual_seed(1)
    img = A.generate(pipe2, conds[0], conds[1], 2 * m, 2 * m, 2, A.DEFAULT_GUIDANCE, gen)
    assert img.size == (2 * m, 2 * m), img.size
print(f"HARNESS OK {fam}")
