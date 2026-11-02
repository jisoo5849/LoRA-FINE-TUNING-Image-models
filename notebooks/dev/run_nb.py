import os, sys, time, nbformat
from nbclient import NotebookClient
fam, dev, py = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["LORA_NB_TEST"] = fam; os.environ["LORA_NB_TEST_DEVICE"] = dev
nb = nbformat.read(os.environ.get("NB", "../lora_training.ipynb"), as_version=4)
wd = f"test_runs/run_{fam}_{dev}"; os.makedirs(wd, exist_ok=True)
t = time.time()
try:
    NotebookClient(nb, timeout=1800, kernel_name=os.environ.get("KN","python3"), resources={"metadata": {"path": wd}}).execute()
    out = "".join(o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "stream")
    ok = f"NOTEBOOK TEST OK {fam}" in out
    print(fam, dev, "OK" if ok else "NO-MARKER", f"{time.time()-t:.0f}s")
except Exception as e:
    print(fam, dev, "FAIL", f"{time.time()-t:.0f}s", str(e)[-1500:])
finally:
    nbformat.write(nb, f"{wd}/executed.ipynb")
