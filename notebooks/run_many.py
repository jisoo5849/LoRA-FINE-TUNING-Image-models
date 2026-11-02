"""여러 LoRA 학습을 차례로 실행한다. 목록의 실행마다 lora_training.ipynb 를 새 커널로 한 번씩 돌린다.

  python run_many.py <목록.json> [--dry-run] [--test] [--test-device cpu|cuda] [--kernel python3] [--stop-on-error]

목록 파일 예 (run_many_example.json). 상대 경로는 목록 파일이 있는 폴더 기준이다.

  {
    "defaults": {"dataset_dir": "dataset/nkgreen", "trigger": "nkgreen", "validation_file": "", "output_dir": "outputs"},
    "runs": [
      {"family": "flux2_klein", "model_path": "/srv/models/diffusers/FLUX.2-klein-base-4B", "run_name": "nkgreen_klein4b_v1"},
      {"family": "qwen_image", "model_path": "/srv/models/diffusers/Qwen-Image-2512", "run_name": "nkgreen_qwen2512_v1"}
    ]
  }

- 실행마다 값이 노트북 USER 값(LORA_* 환경변수)이 되고, defaults 는 실행에 없는 값을 채운다.
- 시작하기 전에 목록 전체를 검사한다 (계열 이름, 빠진 값, 실행 이름 중복, 데이터셋·모델·검증 파일 경로).
- <output_dir>/<run_name>/summary.json 이 있으면 끝난 실행으로 보고 건너뛴다. 다시 학습하려면 run_name 을 바꾼다 (예: _v2).
- 폴더는 있는데 summary.json 이 없으면(중단·실패) 그 폴더를 <run_name>.incomplete-<시각> 으로 옮기고 처음부터 다시 한다.
- 실패한 실행이 있어도 다음 실행으로 넘어간다 (--stop-on-error 면 멈춘다).
- 실행한 노트북은 <실행 폴더>/executed.ipynb, 실행마다 한 줄 기록은 <output_dir>/run_many_log.csv 에 남는다.
- --test 는 노트북 시험 모드(작은 무작위 모델, 가짜 데이터셋)로 목록의 계열이 이 환경에서 도는지 빠르게 본다.
- 학습용 가상환경의 python 으로 실행한다 (기본 커널 python3 이 그 환경을 쓴다).
- 메인 서버에서는 허브에 학습 작업이 생기기 전까지 관리자 '대기열 멈춤' 상태에서만 돌린다 (계획서 P2).
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
KEYS = {  # 목록 파일의 키 → 노트북 환경변수
    "family": "LORA_MODEL_FAMILY",
    "model_path": "LORA_MODEL_PATH",
    "dataset_dir": "LORA_DATASET_DIR",
    "output_dir": "LORA_OUTPUT_DIR",
    "trigger": "LORA_TRIGGER",
    "run_name": "LORA_RUN_NAME",
    "validation_file": "LORA_VALIDATION_FILE",
}
REQUIRED = ["family", "model_path", "dataset_dir", "trigger", "run_name"]
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
HF_REPO = re.compile(r"[\w.-]+/[\w.-]+")   # 노트북 check_model_path 와 같은 규칙
ANSI = re.compile(r"\x1b\[[0-9;]*m")       # 커널 오류 메시지의 색 코드
LOG_FIELDS = ["started", "mode", "run_name", "family", "model_path", "status", "total_minutes", "train_minutes",
              "peak_gpu_gb", "lora_file_mb", "final_train_loss", "best_val_step", "note"]


def supported_families(nb):
    """노트북의 어댑터 셀(%%writefile lora_adapters/<계열>.py)에서 지원 계열을 읽는다."""
    return {m.group(1) for c in nb.cells if c.cell_type == "code"
            for m in [re.match(r"%%writefile lora_adapters/(\w+)\.py", c.source)] if m}


def resolve(base, p):
    q = Path(p).expanduser()
    return (q if q.is_absolute() else base / q).resolve()


def check_model_path(base, mp):
    """로컬 diffusers 형식 폴더면 절대 경로로, Hugging Face 저장소 이름이면 그대로 돌려준다."""
    p = resolve(base, mp)
    if p.exists():
        if not (p / "model_index.json").is_file():
            return None, f"diffusers 형식 폴더가 아닙니다 (model_index.json 없음): {p}"
        return str(p), None
    if Path(mp).expanduser().is_absolute() or mp.startswith((".", "~")) or not HF_REPO.fullmatch(mp):
        return None, f"모델 경로가 없습니다: {mp}"
    return mp, None


def load_plan(path, fams, test):
    """목록 파일을 실행 목록으로 바꾼다. 문제는 모두 모아 보여 주고, 하나라도 있으면 아무것도 실행하지 않는다."""
    try:
        plan = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        sys.exit(f"목록 파일을 읽을 수 없습니다: {path} ({e})")
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        sys.exit('목록 파일에는 비어 있지 않은 "runs" 목록이 있어야 합니다.')
    base, errors, runs, seen = path.parent, [], [], set()
    defaults = plan.get("defaults", {})
    if not isinstance(defaults, dict):
        errors.append("defaults: {...} 형태여야 합니다")
        defaults = {}
    errors += [f"defaults: 모르는 키 '{k}'" for k in defaults if k not in KEYS]
    for i, item in enumerate(plan["runs"]):
        at = f"runs[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{at}: {{...}} 형태여야 합니다")
            continue
        errors += [f"{at}: 모르는 키 '{k}'" for k in item if k not in KEYS]
        r = {"output_dir": "outputs", "validation_file": "", **defaults, **item}
        wrong = [k for k in KEYS if k in r and not isinstance(r[k], str)]
        missing = [k for k in REQUIRED if k not in wrong and not str(r.get(k, "")).strip()]
        if wrong or missing:
            if wrong:
                errors.append(f"{at}: 문자열이어야 하는 값: {', '.join(wrong)}")
            if missing:
                errors.append(f"{at}: 빠진 값: {', '.join(missing)}")
            continue
        if r["family"] not in fams:
            errors.append(f"{at}: 노트북이 지원하지 않는 계열 '{r['family']}' (지원: {', '.join(sorted(fams))})")
        if not RUN_NAME.fullmatch(r["run_name"]):
            errors.append(f"{at}: run_name 은 영문·숫자로 시작하고 영문·숫자·. _ - 만 씁니다: {r['run_name']}")
        out_dir = resolve(base, r["output_dir"])
        if (out_dir, r["run_name"]) in seen:
            errors.append(f"{at}: 같은 output_dir 에 같은 run_name 이 이미 있습니다: {r['run_name']}")
        seen.add((out_dir, r["run_name"]))
        env = {"LORA_MODEL_FAMILY": r["family"], "LORA_MODEL_PATH": r["model_path"],
               "LORA_DATASET_DIR": str(resolve(base, r["dataset_dir"])), "LORA_OUTPUT_DIR": str(out_dir),
               "LORA_TRIGGER": r["trigger"], "LORA_RUN_NAME": r["run_name"], "LORA_VALIDATION_FILE": ""}
        if r["validation_file"].strip():
            vf = resolve(base, r["validation_file"])
            if not vf.is_file():
                errors.append(f"{at}: 검증 프롬프트 파일이 없습니다: {vf}")
            env["LORA_VALIDATION_FILE"] = str(vf)
        if not test:   # 시험 모드는 작은 무작위 모델과 가짜 데이터셋을 쓰므로 경로를 보지 않는다
            if not Path(env["LORA_DATASET_DIR"]).is_dir():
                errors.append(f"{at}: 데이터셋 폴더가 없습니다: {env['LORA_DATASET_DIR']}")
            mp, err = check_model_path(base, r["model_path"])
            if err:
                errors.append(f"{at}: {err}")
            else:
                env["LORA_MODEL_PATH"] = mp
        runs.append(dict(name=r["run_name"], family=r["family"], out_dir=out_dir, env=env))
    if errors:
        print("목록 파일에 문제가 있어 아무것도 실행하지 않았습니다.")
        for e in errors:
            print("  -", e)
        sys.exit(2)
    return runs


def prepare(run, dry_run):
    """끝난 실행은 건너뛰고, 끝나지 않은 폴더는 옆으로 옮긴다. ('skip' 또는 'run', 옮긴 경로) 를 돌려준다."""
    d = run["out_dir"] / run["name"]
    if (d / "summary.json").is_file():
        return "skip", None
    if d.exists():
        dest = d.with_name(f"{run['name']}.incomplete-{datetime.now():%Y%m%d-%H%M%S}")
        if not dry_run:
            d.rename(dest)
        return "run", dest
    return "run", None


def make_client(nb, **kw):
    """셀 출력(print, 진행 막대)을 실행 중에 터미널에도 보여 주는 NotebookClient"""
    from nbclient import NotebookClient

    class EchoClient(NotebookClient):
        def output(self, outs, msg, *args, **kwargs):
            if msg.get("msg_type") == "stream":
                out = sys.stderr if msg["content"].get("name") == "stderr" else sys.stdout
                out.write(msg["content"].get("text", ""))
                out.flush()
            return super().output(outs, msg, *args, **kwargs)

    return EchoClient(nb, **kw)


def append_log(path, row):
    """실행마다 한 줄. 새 파일이면 머리줄을 쓰고, 엑셀에서 한글이 깨지지 않게 BOM 을 붙인다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def describe(s):
    return (f"학습 {fmt(s.get('train_minutes'), '.1f')}분, 최대 GPU {fmt(s.get('peak_gpu_gb'), '.1f')}GB, "
            f"LoRA {fmt(s.get('lora_file_mb'), '.1f')}MB, 마지막 손실 {fmt(s.get('final_train_loss'), '.4f')}, "
            f"검증 손실 최저 단계 {s.get('best_val_step', '-')}")


def run_one(run, nb_path, args):
    """노트북을 새 커널로 한 번 실행하고 (상태, 설명) 을 돌려준다."""
    run_dir = run["out_dir"] / (run["name"] + (f"_test_{run['family']}" if args.test else ""))
    env = dict(run["env"])
    if args.test:
        env.update(LORA_NB_TEST=run["family"], LORA_NB_TEST_DEVICE=args.test_device)
    for k in [k for k in os.environ if k.startswith("LORA_")]:   # 앞 실행의 값이 남지 않게
        del os.environ[k]
    os.environ.update(env)   # 커널은 이 프로세스의 환경변수를 물려받는다
    nb = nbformat.read(nb_path, as_version=4)
    started, t0 = datetime.now(), time.time()
    status, note, summary = "실패", "", {}
    try:
        make_client(nb, timeout=None, kernel_name=args.kernel,
                    resources={"metadata": {"path": str(nb_path.parent)}}).execute()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        status, note = "완료", describe(summary)
    except KeyboardInterrupt:
        status, note = "중단", "사용자가 멈춤"
        raise
    except Exception as e:
        lines = ANSI.sub("", str(e)).strip().splitlines()
        note = f"{type(e).__name__}: {lines[-1][:300] if lines else ''}"
    finally:
        run_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, run_dir / "executed.ipynb")
        append_log(run["out_dir"] / "run_many_log.csv", dict(
            started=f"{started:%Y-%m-%d %H:%M:%S}", mode="시험" if args.test else "학습", run_name=run["name"],
            family=run["family"], model_path=env["LORA_MODEL_PATH"], status=status,
            total_minutes=round((time.time() - t0) / 60, 1), train_minutes=summary.get("train_minutes"),
            peak_gpu_gb=summary.get("peak_gpu_gb"), lora_file_mb=summary.get("lora_file_mb"),
            final_train_loss=summary.get("final_train_loss"), best_val_step=summary.get("best_val_step"), note=note))
    return status, note


def main():
    ap = argparse.ArgumentParser(description="여러 LoRA 학습을 차례로 실행한다 (자세한 설명은 파일 맨 위)")
    ap.add_argument("plan", type=Path, help="실행 목록 JSON 파일")
    ap.add_argument("--notebook", type=Path, default=HERE / "lora_training.ipynb", help="실행할 노트북")
    ap.add_argument("--dry-run", action="store_true", help="검사하고 실행 계획만 보여 준다")
    ap.add_argument("--test", action="store_true", help="노트북 시험 모드(작은 무작위 모델, 가짜 데이터셋)로 돌린다")
    ap.add_argument("--test-device", choices=["cpu", "cuda"], default="cpu", help="시험 모드 장치 (기본 cpu)")
    ap.add_argument("--kernel", default="python3", help="노트북을 실행할 Jupyter 커널 (기본 python3)")
    ap.add_argument("--stop-on-error", action="store_true", help="실패하면 남은 실행을 하지 않는다")
    args = ap.parse_args()

    nb_path = args.notebook.resolve()
    fams = supported_families(nbformat.read(nb_path, as_version=4))
    runs = load_plan(args.plan.resolve(), fams, args.test)
    print(f"{'시험' if args.test else '학습'} {len(runs)}건 | 노트북 {nb_path} | 커널 {args.kernel}"
          + (" | --dry-run: 실행하지 않음" if args.dry_run else ""))
    if not args.test and not args.dry_run:
        print("메인 서버에서는 관리자 '대기열 멈춤' 상태에서만 돌립니다 (계획서 P2).")
    results = []
    try:
        for n, run in enumerate(runs, 1):
            head = f"[{n}/{len(runs)}] {run['name']} ({run['family']})"
            action, moved = ("run", None) if args.test else prepare(run, args.dry_run)
            if action == "skip":
                print(f"{head}: summary.json 이 있어 건너뜀")
                results.append((run, "건너뜀", ""))
                continue
            if moved:
                print(f"{head}: 끝나지 않은 이전 폴더를 {moved.name} 으로 {'옮길 예정' if args.dry_run else '옮김'}")
            if args.dry_run:
                print(f"{head}: 실행 예정")
                for k, v in run["env"].items():
                    print(f"      {k}={v}")
                results.append((run, "실행 예정", ""))
                continue
            print(f"\n{head}: 시작 {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
            status, note = run_one(run, nb_path, args)
            print(f"{head}: {status}  {note}", flush=True)
            results.append((run, status, note))
            if status != "완료" and args.stop_on_error:
                print("--stop-on-error: 남은 실행을 하지 않습니다.")
                break
    except KeyboardInterrupt:
        print("\n사용자가 멈췄습니다. 실행 중이던 폴더는 다음에 돌릴 때 .incomplete 로 옮겨집니다.")
        sys.exit(130)
    print("\n결과")
    width = max(len(r["name"]) for r, _, _ in results)
    for r, status, note in results:
        print(f"  {status:<5} {r['name']:<{width}}  {r['family']:<13} {note}")
    sys.exit(1 if any(status == "실패" for _, status, _ in results) else 0)


if __name__ == "__main__":
    main()
