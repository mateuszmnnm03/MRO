"""Uruchom notebook i zapisz jego komórki wraz z wynikami."""
import asyncio
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def main():
    if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    path = Path(__file__).resolve().parent / 'Laboratorium_4.ipynb'
    notebook = nbformat.read(path, as_version=4)

    def completed(cell, cell_index, execute_reply):
        nbformat.write(notebook, path)
        print(f'\nKomórka {cell_index + 1}/{len(notebook.cells)} gotowa.', flush=True)

    with tempfile.TemporaryDirectory(prefix='lab4-kernel-') as runtime:
        client = ProgressClient(notebook, timeout=1200, kernel_name=notebook.metadata.kernelspec.name,
                                resources={'metadata': {'path': str(path.parent)}},
                                on_cell_executed=completed)
        client.create_kernel_manager()
        client.km.connection_file = str(Path(runtime) / 'kernel.json')
        try:
            client.execute()
        finally:
            nbformat.write(notebook, path)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            assert cell.execution_count is not None, i
            assert not any(o.output_type == 'error' for o in cell.outputs), i
    print('Notebook wykonany i zapisany bez błędów.')


if __name__ == '__main__':
    main()
