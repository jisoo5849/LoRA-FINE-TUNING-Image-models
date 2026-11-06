"""영상 노트북을 '사용자 방식'으로 시험한다: 설정 5개(TRIGGER, MODEL_PATH, RUN_NAME, DATASET_DIR, TARGET_SCOPE)만 주고 실행.

  python test_video_user.py <짧은 작업 폴더> [계열 ...] [--device cpu|cuda]

- test_video.py(시험 모드)는 모델·데이터를 노트북 안에서 만들어 쓰므로, 사용자가 실제로 지나는 경로를 따로 확인한다.
  - 작은 무작위 모델을 diffusers 형식 폴더로 저장해 MODEL_PATH 로 준다 → 0-2 의 계열 자동 인식, 0-4 의 load_pipeline
  - 데이터셋 폴더를 DATASET_DIR 로 준다 → 폴더 확인, 캡션에서 CLASS_WORD 자동 선택
  - TARGET_SCOPE 는 계열마다 다르게 준다 (이미지 v2 이름 attn_mlp 포함)
  - 학습 크기만 LORA_NB_SMOKE=1 로 시험 실행처럼 줄인다 (사용자는 쓰지 않는 개발용 값)
- 확인하는 것
  - 끝까지 돌고 summary.json 의 계열·범위·CLASS_WORD 가 맞는지, LoRA 파일 키가 ComfyUI 형식이고 범위에 맞는 층만 있는지
  - next_run.json 이 run_many.py 형식이고 경로가 절대 경로인지
  - 이미지만 있는 데이터(니키 캡션 형식, 1프레임 mp4 포함)로 T2V-A14B 가 저노이즈 LoRA 만 전체 단계로 만드는지
  - 이미지 20장 + 클립 1개에서 검증용으로 이미지만 빼고 클립은 고노이즈 학습에 남는지, CLASS_WORD 를 캡션에서 고르는지
  - next_run.json 의 target_scope 가 이번 실행과 같은지
  - 같은 RUN_NAME 으로 다시 돌리면 0-2 에서 멈추는지, I2V 에 이미지만 주면 모델을 불러오기 전에 멈추는지,
    없는 MODEL_PATH·DATASET_DIR, 틀린 TARGET_SCOPE, 규칙에 맞지 않는 RUN_NAME, 계열이 다른 MODEL_FAMILY 를 바로 알려 주는지
- 커널은 환경변수 KN (기본 python3). HF_HOME 에 hf-internal-testing/tiny-random-t5 가 있어야 한다 (HF_HUB_OFFLINE=1 권장).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = HERE.parent / "lora_training_video.ipynb"
FAMILIES = ["wan22_t2v", "wan22_i2v", "wan22_ti2v_5b", "wan21"]
FOLDER = {"wan22_t2v": "Wan2.2-T2V-A14B-Diffusers", "wan22_i2v": "Wan2.2-I2V-A14B-Diffusers",
          "wan22_ti2v_5b": "Wan2.2-TI2V-5B-Diffusers", "wan21": "Wan2.1-T2V-1.3B-Diffusers"}
SCOPE = {"wan22_t2v": ("all", "all"), "wan22_i2v": ("attn", "attn"), "wan22_ti2v_5b": ("kvo", "kvo"),
         "wan21": ("attn_mlp", "all")}           # (USER 에 주는 값, 노트북이 쓰는 값)


def child(wd):
    """새 커널로 노트북을 실행하고 첫 오류를 한 줄로 알린다 (부모가 환경변수를 정해 준다)."""
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    nb = nbformat.read(str(NB), as_version=4)
    err = ""
    try:
        NotebookClient(nb, timeout=1800, kernel_name=os.environ.get("KN", "python3"),
                       resources={"metadata": {"path": wd}}).execute()
    except CellExecutionError as e:
        err = f"{e.ename}: {e.evalue}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    nbformat.write(nb, str(Path(wd) / "executed.ipynb"))
    print(("CHILD-ERR " + err.replace("\n", " ")[:600]) if err else "CHILD-OK")


def run_nb(wd, settings, device):
    env = {k: v for k, v in os.environ.items() if not k.startswith("LORA_")}
    env.update({f"LORA_{k}": v for k, v in settings.items()})
    env.update(LORA_NB_SMOKE="1", LORA_NB_TEST_DEVICE=device, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-X", "utf8", __file__, "--child", str(wd)], env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    line = next((x for x in r.stdout.splitlines() if x.startswith("CHILD-")), "CHILD-ERR (no result) " + r.stderr[-600:])
    return line, time.time() - t0


def make_models(root, families):
    import torch
    sys.path.insert(0, str(HERE / "video_adapters"))
    import importlib
    out = {}
    for fam in families:
        d = root / FOLDER[fam]
        if not (d / "model_index.json").exists():
            importlib.import_module(fam).tiny_pipeline(torch.float32).save_pretrained(str(d))
        out[fam] = d
    return out


def make_datasets(root):
    sys.path.insert(0, str(HERE / "video_tools"))
    import numpy as np
    from PIL import Image
    import video_io as V
    mixed = root / "data_mixed"
    if not mixed.exists():
        V.make_test_dataset(mixed, "nkq7")          # 이미지 + 문제가 있는 클립 (회전·VFR·검은 구간 …)
    rng = np.random.default_rng(0)

    def still(i):
        a = np.full((96, 80, 3), 235, np.uint8)
        y, x = rng.integers(10, 50, 2)
        a[y:y + 40, x:x + 24] = rng.integers(0, 255, 3, dtype=np.uint8)
        return a

    # 니키 데이터처럼 이미지와 캡션만. 캡션은 실제 니키 캡션 형식(종류 낱말·대명사 없음) → CLASS_WORD 는 person.
    # 1프레임짜리 mp4 도 하나 넣는다 (정지 그림으로 읽어야 한다).
    images = root / "data_images"
    if not images.exists():
        images.mkdir(parents=True)
        for i in range(6):
            Image.fromarray(still(i)).save(images / f"niki_{i:02d}.png")
            (images / f"niki_{i:02d}.txt").write_text(
                f"nkq7, full body, three-quarter view, standing, pose {i}, plain light gray background", encoding="utf-8")
        V.save_mp4(images / "still_00.mp4", still(9)[None], 16)
        (images / "still_00.txt").write_text("nkq7, upper body, front view, gentle smile", encoding="utf-8")
    # 이미지 20장 + 클립 1개: 검증용으로 빼는 파일이 하나뿐인 클립을 가져가면 안 된다. 캡션의 her → CLASS_WORD girl.
    mix1 = root / "data_mix1"
    if not mix1.exists():
        mix1.mkdir(parents=True)
        for i in range(20):
            Image.fromarray(still(i)).save(mix1 / f"niki_{i:02d}.png")
            (mix1 / f"niki_{i:02d}.txt").write_text(f"nkq7, full body, her hands behind her back, pose {i}", encoding="utf-8")
        fr = np.full((24, 96, 96, 3), 235, np.uint8)
        for t in range(24):
            fr[t, 30:60, 4 + 3 * t:28 + 3 * t] = (200, 40, 60)
        V.save_mp4(mix1 / "walk_01.mp4", fr, 16)
        (mix1 / "walk_01.txt").write_text("nkq7, full body, static camera, she walks to the right", encoding="utf-8")
    return mixed, images, mix1


def check_run(out, fam, scope_used, expect_tags, expect_class=None, expect_steps=None):
    from safetensors.torch import load_file
    problems = []
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    if s.get("family") != fam or not s.get("family_auto"):
        problems.append(f"계열 {s.get('family')} (자동 {s.get('family_auto')}), 기대 {fam}")
    if s.get("target_scope") != scope_used:
        problems.append(f"범위 {s.get('target_scope')}, 기대 {scope_used}")
    if expect_class and s.get("class_word") != expect_class:
        problems.append(f"CLASS_WORD {s.get('class_word')}, 기대 {expect_class}")
    tags = [t for t, _ in s["expert_plan"]]
    if tags != expect_tags:
        problems.append(f"전문가 {tags}, 기대 {expect_tags}")
    if expect_steps is not None and [n for _, n in s["expert_plan"]] != expect_steps:
        problems.append(f"단계 {s['expert_plan']}, 기대 {expect_steps}")
    for tag, ck in s["checkpoints"].items():
        if not ck:
            problems.append(f"{tag} 체크포인트 없음")
            continue
        sd = load_file(ck[max(ck, key=int)])
        mods = {k.rsplit(".lora_", 1)[0] for k in sd}
        if any(not k.startswith("diffusion_model.blocks.") for k in sd):
            problems.append(f"{tag}: ComfyUI 형식이 아닌 키")
        kinds = {m.split(".", 3)[3] for m in mods}   # self_attn.q, ffn.0, ...
        want = {"all": {"self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k",
                        "cross_attn.v", "cross_attn.o", "ffn.0", "ffn.2"},
                "attn": {"self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.q", "cross_attn.k",
                         "cross_attn.v", "cross_attn.o"},
                "kvo": {"self_attn.k", "self_attn.v", "self_attn.o", "cross_attn.k", "cross_attn.v", "cross_attn.o"}}[scope_used]
        if kinds != want:
            problems.append(f"{tag}: 층 {sorted(kinds)} != 범위 {scope_used} {sorted(want)}")
        if any(k.endswith("lora_B.weight") and float(v.abs().sum()) == 0 for k, v in sd.items()):
            problems.append(f"{tag}: 학습되지 않은 LoRA B")
    plan = json.loads((out / "next_run.json").read_text(encoding="utf-8"))
    run = (plan.get("runs") or [{}])[0]
    if run.get("family") != "auto":                  # 사용자가 계열을 정하지 않았으니 다음 실행도 auto
        problems.append(f"next_run family {run.get('family')} (기대 auto)")
    if run.get("target_scope") != scope_used:        # 다음 실행도 같은 범위로 학습해야 한다
        problems.append(f"next_run target_scope {run.get('target_scope')}, 기대 {scope_used}")
    for k in ("dataset_dir", "output_dir"):
        if not Path(str(run.get(k, ""))).is_absolute():
            problems.append(f"next_run {k} 가 절대 경로가 아님: {run.get(k)}")
    return problems, s


def main():
    if "--child" in sys.argv:            # run_nb 가 부른 자식 프로세스: 노트북만 실행
        return child(sys.argv[sys.argv.index("--child") + 1])
    ap = argparse.ArgumentParser()
    ap.add_argument("work", type=Path)
    ap.add_argument("families", nargs="*", default=FAMILIES)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    models = make_models(work / "models", a.families)
    mixed, images, mix1 = make_datasets(work)
    failed = 0

    def report(name, ok, sec, note=""):
        nonlocal failed
        failed += not ok
        print(f"{name:<34} {a.device} {'OK ' if ok else 'FAIL'} {sec:5.0f}s {note}", flush=True)

    for fam in a.families:
        given, used = SCOPE[fam]
        wd = work / f"u_{fam}_{a.device}"
        shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir()
        st = dict(TRIGGER="nkq7", MODEL_PATH=str(models[fam]), RUN_NAME=f"smoke_{fam}_v1", DATASET_DIR=str(mixed),
                  TARGET_SCOPE=given)
        line, sec = run_nb(wd, st, a.device)
        tags_want = ["low", "high"] if fam in ("wan22_t2v", "wan22_i2v") else ["main"]
        if line != "CHILD-OK":
            report(f"{fam} 전체 실행", False, sec, line)
            continue
        problems, tags = check_run(wd / "outputs_video" / st["RUN_NAME"], fam, used, tags_want)
        report(f"{fam} 전체 실행 (범위 {given})", not problems, sec, "; ".join(problems))
        if fam == a.families[0]:   # 같은 RUN_NAME 으로 다시 → 0-2 에서 멈춰야 한다
            line, sec = run_nb(wd, st, a.device)
            report("  같은 RUN_NAME 다시 실행", line.startswith("CHILD-ERR FileExistsError"), sec, line[:160])

    if "wan22_t2v" in a.families:
        # 이미지만 (니키 데이터 형식 + 1프레임 mp4) → 저노이즈만 전체 단계(시험 6), CLASS_WORD person
        wd = work / f"u_images_{a.device}"
        shutil.rmtree(wd, ignore_errors=True)
        wd.mkdir()
        st = dict(TRIGGER="nkq7", MODEL_PATH=str(models["wan22_t2v"]), RUN_NAME="smoke_images_v1",
                  DATASET_DIR=str(images), TARGET_SCOPE="all")
        line, sec = run_nb(wd, st, a.device)
        if line != "CHILD-OK":
            report("wan22_t2v 이미지만", False, sec, line)
        else:
            problems, _ = check_run(wd / "outputs_video" / st["RUN_NAME"], "wan22_t2v", "all", ["low"], "person", [6])
            report("wan22_t2v 이미지만 (저노이즈만)", not problems, sec, "; ".join(problems))
        # 이미지 20장 + 클립 1개 → 클립은 학습에 남고(고노이즈 학습), 검증용으로는 이미지 2장만 뺀다
        wd3 = work / f"u_mix1_{a.device}"
        shutil.rmtree(wd3, ignore_errors=True)
        wd3.mkdir()
        st3 = dict(st, RUN_NAME="smoke_mix1_v1", DATASET_DIR=str(mix1))
        line, sec = run_nb(wd3, st3, a.device)
        if line != "CHILD-OK":
            report("wan22_t2v 이미지 20 + 클립 1", False, sec, line)
        else:
            problems, sm = check_run(wd3 / "outputs_video" / st3["RUN_NAME"], "wan22_t2v", "all", ["low", "high"], "girl")
            held = sm.get("holdout_files") or []
            if len(held) != 2 or any(not h.endswith(".png") for h in held) or not sm.get("val_heldout"):
                problems.append(f"검증용 파일 {held} (기대: 이미지 2장), val_heldout {sm.get('val_heldout')}")
            report("wan22_t2v 이미지 20 + 클립 1", not problems, sec, "; ".join(problems))
        bad = [("없는 MODEL_PATH", dict(st, MODEL_PATH=str(work / "no_such_model")), "CHILD-ERR FileNotFoundError: MODEL_PATH"),
               ("없는 DATASET_DIR", dict(st, DATASET_DIR=str(work / "no_such_data")), "CHILD-ERR FileNotFoundError: DATASET_DIR"),
               ("틀린 TARGET_SCOPE", dict(st, TARGET_SCOPE="mlp"), "CHILD-ERR ValueError: TARGET_SCOPE"),
               ("규칙에 맞지 않는 RUN_NAME", dict(st, RUN_NAME="니키 v1"), "CHILD-ERR ValueError: RUN_NAME"),
               ("계열이 다른 MODEL_FAMILY", dict(st, MODEL_FAMILY="wan21"), "CHILD-ERR ValueError: MODEL_FAMILY")]
        if "wan22_i2v" in a.families:
            bad.append(("I2V 에 이미지만", dict(st, MODEL_PATH=str(models["wan22_i2v"])), "CHILD-ERR ValueError: wan22_i2v 는 클립"))
        for name, s2, want in bad:
            wd2 = work / f"u_bad_{a.device}"
            shutil.rmtree(wd2, ignore_errors=True)
            wd2.mkdir()
            line, sec = run_nb(wd2, dict(s2, RUN_NAME="smoke_bad_v1") if s2["RUN_NAME"] == st["RUN_NAME"] else s2, a.device)
            report("  " + name, line.startswith(want), sec, line[:200])
    print("USER MODE ALL OK" if not failed else f"USER MODE {failed}건 실패")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
