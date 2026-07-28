from pathlib import Path

import nbformat
from nbclient import NotebookClient


OUTPUT_DIR = Path(__file__).resolve().parent
source = OUTPUT_DIR / "correct_dst_k10_statistical_analysis.ipynb"
target = OUTPUT_DIR / "correct_dst_k10_statistical_analysis.executed.ipynb"
notebook = nbformat.read(source, as_version=4)
client = NotebookClient(
    notebook,
    timeout=None,
    kernel_name="MIL",
    resources={"metadata": {"path": str(OUTPUT_DIR)}},
    allow_errors=False,
)
client.execute()
nbformat.write(notebook, target)
print(target)
