"""Constrói o .ipynb a partir de um script com marcadores '# %%' e o executa via jupyter_client.

Uso:
    python tools/nbtools.py build tools/notebook_src.py notebook_sereno_calor.ipynb
    python tools/nbtools.py run notebook_sereno_calor.ipynb
(Equivalente a `jupyter nbconvert --execute --inplace`, quando nbconvert estiver disponível.)
"""
import json
import re
import sys
from pathlib import Path


def build(src: str, dst: str):
    text = Path(src).read_text(encoding="utf-8")
    partes = re.split(r"^# %%(.*)$", text, flags=re.M)
    cells = []
    for i in range(1, len(partes), 2):
        tag, body = partes[i].strip(), partes[i + 1].strip("\n")
        if "[markdown]" in tag:
            linhas = [re.sub(r"^# ?", "", l) for l in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": "\n".join(linhas).splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": body.splitlines(keepends=True)})
    nb = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for k, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{k:03d}"
    Path(dst).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def run(path: str, timeout: int = 1800):
    from jupyter_client.manager import start_new_kernel
    p = Path(path)
    nb = json.loads(p.read_text(encoding="utf-8"))
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(p.parent.resolve()))
    erros = 0
    try:
        n = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            n += 1
            outs = []
            msg_id = kc.execute("".join(cell["source"]))
            while True:
                msg = kc.get_iopub_msg(timeout=timeout)
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "status" and c["execution_state"] == "idle":
                    break
                if t == "stream":
                    if outs and outs[-1]["output_type"] == "stream" and outs[-1]["name"] == c["name"]:
                        outs[-1]["text"] += c["text"]
                    else:
                        outs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif t in ("execute_result", "display_data"):
                    o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                    if t == "execute_result":
                        o["execution_count"] = n
                    outs.append(o)
                elif t == "error":
                    erros += 1
                    outs.append({"output_type": "error", "ename": c["ename"],
                                 "evalue": c["evalue"], "traceback": c["traceback"]})
                    print("ERRO na célula", n, c["ename"], c["evalue"], file=sys.stderr)
            for o in outs:
                if o["output_type"] == "stream":
                    o["text"] = o["text"].splitlines(keepends=True)
            cell["outputs"], cell["execution_count"] = outs, n
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Executado: {n} células de código, {erros} erro(s).")
    return erros


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2], sys.argv[3])
    else:
        sys.exit(1 if run(sys.argv[2]) else 0)
