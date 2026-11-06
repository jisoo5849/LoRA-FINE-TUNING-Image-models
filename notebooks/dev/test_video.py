"""영상 노트북을 시험 모드(작은 무작위 모델 + 문제를 넣은 합성 데이터)로 계열마다 끝까지 돌린다.

  python test_video.py <짧은 작업 폴더> [계열 ...] [--device cpu|cuda]

- 계열마다 노트북을 새 커널로 실행한다. 커널은 환경변수 KN (기본 python3).
- 확인하는 것
  - 노트북이 끝까지 돌고 'NOTEBOOK TEST OK <계열>' 을 출력하는지 (EDA 의 합성 데이터 확인 포함)
  - summary.json 이 있고, 학습한 전문가마다 체크포인트 파일이 있는지
  - LoRA 파일의 키가 모두 ComfyUI 형식(diffusion_model.blocks.)이고 lora_A/lora_B 짝이 맞는지
  - 학습 뒤 LoRA B 가 0 이 아닌지, next_run.json 이 run_many.py 형식인지
- Windows 는 경로가 길면 저장이 실패하므로 작업 폴더를 짧게 준다 (예: C:\\Users\\<사용자>\\.lvt).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "lora_training_video.ipynb"
FAMILIES = ["wan22_t2v", "wan22_i2v", "wan22_ti2v_5b", "wan21"]


def run(work, fam, device):
    wd = work / f"video_{fam}_{device}"
    shutil.rmtree(wd, ignore_errors=True)
    wd.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("LORA_")}
    env.update(NB=str(NB), PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "run_nb.py"), fam, device, "x"], cwd=wd, env=env,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = any(line.startswith(f"{fam} {device} OK") for line in r.stdout.splitlines())
    outs = [p for p in (wd / "test_runs" / f"run_{fam}_{device}" / "outputs_video").glob("*") if p.is_dir()]
    return ok, (outs[0] if len(outs) == 1 else None), time.time() - t0, (r.stdout + r.stderr)[-1500:]


def check(fam, out):
    problems = []
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    for tag, ck in s["checkpoints"].items():
        if not ck:
            problems.append(f"{tag} 체크포인트 없음"); continue
        sd = load_file(ck[max(ck, key=int)])
        bad = [k for k in sd if not k.startswith("diffusion_model.blocks.")]
        a = {k[:-len(".lora_A.weight")] for k in sd if k.endswith(".lora_A.weight")}
        b = {k[:-len(".lora_B.weight")] for k in sd if k.endswith(".lora_B.weight")}
        if bad or a != b or not a:
            problems.append(f"{tag} 키 형식 문제 (예: {bad[:1]}, A {len(a)} / B {len(b)})")
        zero = [k for k, v in sd.items() if k.endswith(".lora_B.weight") and float(v.abs().sum()) == 0]
        if zero:
            problems.append(f"{tag} 학습되지 않은 LoRA {len(zero)}개 (예: {zero[0]})")
    plan = json.loads((out / "next_run.json").read_text(encoding="utf-8"))
    if not (isinstance(plan.get("runs"), list) and plan["runs"] and plan["runs"][0].get("family") == fam):
        problems.append("next_run.json 형식이 run_many.py 와 다름")
    tags = [t for t, _ in s["expert_plan"]]
    return problems, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("families", nargs="*", default=FAMILIES)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    failed = 0
    for fam in a.families:
        ok, out, sec, tail = run(a.work, fam, a.device)
        note = ""
        if ok and out is not None:
            problems, tags = check(fam, out)
            ok = not problems
            note = f" | 전문가 {tags}" + ("" if ok else " | " + "; ".join(problems))
        else:
            ok, note = False, "\n" + tail
        print(f"{fam:<14} {a.device} {'OK ' if ok else 'FAIL'} {sec:5.0f}s{note}", flush=True)
        failed += not ok
    print("ALL OK" if not failed else f"{failed}건 실패")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
