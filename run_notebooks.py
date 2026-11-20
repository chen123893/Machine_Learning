"""Execute project notebooks in order, each in a fresh local Python kernel."""

import argparse
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import nbformat
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager
from ipykernel.kernelspec import install
from nbclient import NotebookClient
from tempfile import TemporaryDirectory


def run(start, end):
    root = Path(__file__).resolve().parent
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    with TemporaryDirectory() as kernel_prefix:
        os.environ.setdefault("IPYTHONDIR", str(Path(kernel_prefix) / "ipython"))
        os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(Path(kernel_prefix) / "runtime"))
        name = f"loan-review-{uuid4().hex}"
        install(kernel_name=name, prefix=kernel_prefix)
        specs = KernelSpecManager(kernel_dirs=[str(Path(kernel_prefix) / "share/jupyter/kernels")])
        for path in sorted((root / "notebooks").glob("0[1-8]_*.ipynb")):
            if not start <= int(path.name[:2]) <= end:
                continue
            print(f"Running {path.name}", flush=True)
            timer = perf_counter()
            notebook = nbformat.read(path, as_version=4)
            manager = AsyncKernelManager(kernel_name=name, kernel_spec_manager=specs)
            client = NotebookClient(notebook, km=manager,
                                    timeout=5400, resources={"metadata": {"path": str(root)}})
            client.execute(cleanup_kc=True)
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"Completed {path.name} in {perf_counter() - timer:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--end", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("--start must be at or before --end")
    run(args.start, args.end)
