"""Run the notebook in this Python environment, without modifying user kernels."""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "01_shg_walkthrough.ipynb"
notebook = nbformat.read(path, as_version=4)
with TemporaryDirectory(prefix="nii2-kernel-") as directory:
    kernel = Path(directory) / "nii2-project"
    kernel.mkdir()
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [
                    sys.executable,
                    "-m",
                    "ipykernel_launcher",
                    "-f",
                    "{connection_file}",
                ],
                "display_name": "NiI2 project verification",
                "language": "python",
            }
        )
    )
    manager = KernelManager(
        kernel_name="nii2-project",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[directory]),
    )
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=180,
        resources={"metadata": {"path": str(root / "notebooks")}},
    )
    client.execute()
nbformat.write(notebook, path)
print(
    f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {path}"
)
