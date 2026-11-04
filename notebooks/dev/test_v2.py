"""v1·v2 노트북을 시험 모드(작은 무작위 모델)로 계열마다 돌려, v2 가 환경변수만 바꿔 모든 계열에서 도는지 확인한다.

  python test_v2.py <짧은 작업 폴더> [계열 ...] [--device cpu|cuda]

- 계열마다 v1 1번, v2 4번(attn+off, attn_all+dop, attn_mlp+off, all+dop)을 새 커널로 실행한다. 커널은 환경변수 KN (기본 python3).
- 확인하는 것
  - 모든 실행이 끝까지 돌고 summary.json 을 남기는지
  - v2 attn+off 의 단계별 손실이 v1 과 같은지
  - 범위마다 정해 둔 층 묶음이 모두 LoRA 에 들어갔는지
  - 학습 뒤 LoRA B 가 0 이 아닌지 (마지막 블록의 글 쪽 출력 층은 모델이 결과를 버리므로 제외)
  - 보존 손실이 dop 이면 1단계 0, 마지막 단계 0 보다 큼, off 이면 기록 없음
- 먼저 v2 의 0-2 설정 셀만 따로 실행해, 이전 결과 폴더가 있으면 멈추는지·RUN_NAME 기본값·Turbo 경고를 본다.
- Windows 는 경로가 길면 LoRA 저장이 실패하므로 작업 폴더를 짧게 준다 (예: C:\\Users\\<사용자>\\.lvt).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
NB_V1, NB_V2 = HERE.parent / "lora_training.ipynb", HERE.parent / "lora_training_v2.ipynb"
FAMILIES = ["qwen_image", "qwen_image21", "flux2_klein", "z_image", "krea2", "flux1", "chroma", "longcat_image",
            "sdxl", "sd3"]
V2_RUNS = [("attn", "off"), ("attn_all", "dop"), ("attn_mlp", "off"), ("all", "dop")]
SCOPE_GROUPS = {"attn": [], "attn_all": ["attn_extra"], "attn_mlp": ["attn_extra", "mlp"], "all": ["attn_extra", "mlp", "mod"]}
DEFINED = {   # 계열마다 v2_parts.LAYER_GROUPS 에 정해 둔 묶음 (작은 시험 모델에 모두 있는 것)
    "qwen_image": {"attn_extra", "mlp", "mod"}, "qwen_image21": {"mlp"}, "flux2_klein": {"attn_extra", "mlp"},
    "z_image": {"mlp", "mod"}, "krea2": {"mlp"}, "flux1": {"mlp", "mod"}, "chroma": {"mlp"},
    "longcat_image": {"mlp", "mod"}, "sdxl": {"mlp"}, "sd3": {"mlp", "mod"},   # sd3 의 attn2 는 SD3.5 Medium 에만 있음
}
# 마지막 블록에서 출력이 버려지는 글 쪽 층 (Qwen-Image 의 add_q_proj·to_add_out·txt_mlp, SD3 의 add_q_proj).
# FLUX 계열은 글 쪽이 한 갈래 블록으로 이어지므로 해당 없음
DEAD_FAMILIES = {"qwen_image", "sd3"}
DEAD = re.compile(r"transformer_blocks\.(\d+)\.(attn\.add_q_proj|attn\.to_add_out|txt_mlp\.net\.(0\.proj|2))\.")


def check_setup_cell(work):
    """v2 의 0-2 설정 셀만 실행한다 (모델을 불러오지 않음). 문제 목록을 돌려준다."""
    import nbformat
    from types import MappingProxyType

    src = next(c.source for c in nbformat.read(NB_V2, as_version=4).cells
               if c.cell_type == "code" and "USER = MappingProxyType" in c.source)
    used, empty = work / "setup_used", work / "setup_empty"
    (used / "nkgreen_v2").mkdir(parents=True, exist_ok=True)
    (used / "nkgreen_v2" / "summary.json").write_text("{}", encoding="utf-8")
    empty.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in os.environ.items() if not k.startswith("LORA_")}

    def run_cell(**env):
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(base, **env)
        ns = dict(os=os, re=re, json=json, Path=Path, MappingProxyType=MappingProxyType)
        try:
            exec(src, ns)
            return ns, None
        except Exception as e:
            return ns, e
        finally:
            os.environ.clear()
            os.environ.update(saved)

    problems = []
    _, err = run_cell(LORA_OUTPUT_DIR=str(used))
    if not isinstance(err, FileExistsError):
        problems.append(f"이전 결과 폴더가 있는데 멈추지 않음 ({err!r})")
    ns, err = run_cell(LORA_OUTPUT_DIR=str(used), LORA_RUN_NAME="other_run")
    if err:
        problems.append(f"새 RUN_NAME 에서 오류 ({err!r})")
    ns, err = run_cell(LORA_OUTPUT_DIR=str(empty))
    if err or ns["USER"]["RUN_NAME"] != "nkgreen_v2":
        problems.append(f"RUN_NAME 기본값이 nkgreen_v2 가 아님 ({err!r})")
    else:
        notes = ns["model_notes"]
        for fam, path in [("qwen_image21", "/home/mgs/models/diffusers/Qwen-Image-2.1-Turbo"),
                          ("z_image", "/home/mgs/models/diffusers/Z-Image-Turbo"), ("sdxl", "stabilityai/sdxl-turbo")]:
            if not any("Turbo" in n for n in notes(fam, path)):
                problems.append(f"Turbo 경고 없음: {fam} {path}")
        if any("Turbo" in n for n in notes("z_image", "/home/mgs/models/diffusers/Z-Image")):
            problems.append("base 모델에 Turbo 경고가 나옴")
    return problems


def run(work, nb, fam, tag, device, env_extra):
    wd = work / tag
    shutil.rmtree(wd, ignore_errors=True)        # 앞선 시험 결과를 잘못 읽지 않게 매번 비운다
    wd.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LORA_")}
    env.update(NB=str(nb), PYTHONIOENCODING="utf-8", **env_extra)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "run_nb.py"), fam, device, "x"], cwd=wd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = any(l.startswith(f"{fam} {device} OK") for l in r.stdout.splitlines())
    outs = [p for p in (wd / "test_runs" / f"run_{fam}_{device}" / "outputs").glob("*") if p.is_dir()]
    out = outs[0] if len(outs) == 1 else None
    return ok and out is not None, out, time.time() - t0, (r.stdout + r.stderr)[-600:]


def check_v2(fam, scope, pres, out):
    problems = []
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    log = pd.read_csv(out / "train_log.csv")
    want = {"base"} | (set(SCOPE_GROUPS[scope]) & DEFINED[fam])
    if set(s["lora_groups"]) != want:
        problems.append(f"묶음 {sorted(s['lora_groups'])} != {sorted(want)}")
    if (s["notebook"], s["target_scope"], s["preservation"]) != ("v2", scope, pres):
        problems.append("summary 의 v2 값이 다름")
    sd = load_file(next(p for p in sorted(out.glob("checkpoint-*"))[-1].glob("*.safetensors")))
    bs = {k: v for k, v in sd.items() if "lora_B" in k or "lora.up" in k}
    last = max((int(m.group(1)) for k in bs for m in [re.search(r"(?<!single_)transformer_blocks\.(\d+)\.", k)] if m),
               default=-1)
    zero = [k for k, v in bs.items() if float(v.abs().sum()) == 0]
    bad = [k for k in zero if not (fam in DEAD_FAMILIES and (m := DEAD.search(k)) and int(m.group(1)) == last)]
    if bad:
        problems.append(f"학습되지 않은 LoRA {len(bad)}개 (예: {bad[0]})")
    if pres == "dop":
        if not (log.pres_loss.iloc[0] == 0 and log.pres_loss.iloc[-1] > 0):
            problems.append(f"보존 손실 이상: {log.pres_loss.tolist()}")
    elif log.pres_loss.notna().any():
        problems.append("off 인데 보존 손실 기록이 있음")
    return problems, len(bs), len(zero)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("families", nargs="*", default=FAMILIES)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    setup_problems = check_setup_cell(a.work)
    print("v2 0-2 설정 셀", "OK" if not setup_problems else "FAIL " + "; ".join(setup_problems), flush=True)
    failed = len(setup_problems)
    for fam in a.families:
        ok1, out1, sec, tail = run(a.work, NB_V1, fam, f"v1_{fam}", a.device, {})
        print(f"{fam:<14} v1              {'OK ' if ok1 else 'FAIL'} {sec:4.0f}s" + ("" if ok1 else f"\n{tail}"), flush=True)
        failed += not ok1
        for scope, pres in V2_RUNS:
            ok, out, sec, tail = run(a.work, NB_V2, fam, f"v2_{fam}_{scope}_{pres}", a.device,
                                     dict(LORA_TARGET_SCOPE=scope, LORA_PRESERVATION=pres))
            note = ""
            if ok:
                problems, n_b, n_zero = check_v2(fam, scope, pres, out)
                if scope == "attn" and pres == "off" and ok1:
                    l1 = pd.read_csv(out1 / "train_log.csv").loss.tolist()
                    l2 = pd.read_csv(out / "train_log.csv").loss.tolist()
                    tol = 1e-6 if a.device == "cpu" else 1e-2     # GPU bf16 커널은 실행마다 끝자리가 조금 다를 수 있음
                    if len(l1) != len(l2) or any(abs(x - y) > tol * max(1.0, abs(x)) for x, y in zip(l1, l2)):
                        problems.append(f"v1 과 손실 다름 {l1} / {l2}")
                    else:
                        note = " | v1 과 손실 같음"
                ok = not problems
                note = f" | LoRA {n_b}층 (출력 버려지는 층 {n_zero})" + note + ("" if ok else " | " + "; ".join(problems))
            else:
                note = f"\n{tail}"
            print(f"{fam:<14} v2 {scope:<8} {pres:<4} {'OK ' if ok else 'FAIL'} {sec:4.0f}s{note}", flush=True)
            failed += not ok
    print("ALL OK" if not failed else f"{failed}건 실패")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
