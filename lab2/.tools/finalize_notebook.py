"""Uaktualnia opis po wizualnej kontroli, zachowując wykonane wyniki kodu."""
import copy
import subprocess
import sys
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
path = root / "lab2.ipynb"
executed = nbformat.read(path, as_version=4)
template_path = root / ".tools" / "template.ipynb"
subprocess.run([sys.executable, str(root / ".tools" / "build_notebook.py"), str(template_path)], cwd=root, check=True)
final = nbformat.read(template_path, as_version=4)
old_code = {cell.source: cell for cell in executed.cells if cell.cell_type == "code"}
for cell in final.cells:
    if cell.cell_type == "code":
        previous = old_code[cell.source]
        cell.outputs = copy.deepcopy(previous.outputs)
        cell.execution_count = previous.execution_count
        # Pomijamy jedynie techniczny postęp pobierania (setki linii), zachowując wyniki.
        if "train_mnist = MNIST" in cell.source:
            text = "".join(output.get("text", "") for output in cell.outputs if output.output_type == "stream")
            marker = text.find("Pełne dane:")
            if marker != -1:
                cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text="MNIST pobrany do / odczytany z katalogu data/.\n" + text[marker:])]
nbformat.validate(final)
assert all(cell.execution_count is not None for cell in final.cells if cell.cell_type == "code")
assert not any(output.output_type == "error" for cell in final.cells if cell.cell_type == "code" for output in cell.outputs)
assert sum("image/png" in output.get("data", {}) for cell in final.cells if cell.cell_type == "code" for output in cell.outputs) == 6
nbformat.write(final, path)
print(f"Gotowe: {len(final.cells)} komorki, wszystkie komorki kodu wykonane, 6 wykresow w notebooku.")
