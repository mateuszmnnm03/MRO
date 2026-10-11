"""Wykonuje notebook w lokalnym środowisku i zapisuje wyniki po każdej komórce."""
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
os.environ["JUPYTER_PATH"] = str(root / ".venv" / "share" / "jupyter")
os.environ["PYTHONIOENCODING"] = "utf-8"

import nbformat
from nbclient import NotebookClient

path = root / "lab2.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)


def save_progress(cell, cell_index, **kwargs):
    nbformat.write(notebook, path)
    print(f"Zapisano komorke {cell_index}", flush=True)
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            print(output.get("text", ""), end="", flush=True)


client = NotebookClient(
    notebook, timeout=1800, kernel_name="lab2-python312",
    resources={"metadata": {"path": str(root)}},
    on_cell_executed=save_progress,
)
try:
    client.execute()
finally:
    nbformat.write(notebook, path)

nbformat.validate(notebook)
assert all(
    cell.get("execution_count") is not None
    for cell in notebook.cells if cell.cell_type == "code"
)
assert not any(
    output.output_type == "error"
    for cell in notebook.cells if cell.cell_type == "code" for output in cell.outputs
)
print("Notebook wykonany w calosci i poprawny strukturalnie.")
