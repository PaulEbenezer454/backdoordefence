# Stage 0/1 environment snapshot

- Inspected 2026-10-04 on Windows 11, Python 3.14.6, project-local `.venv`.
- Installed and exercised Torch 2.11.0+cpu, torchvision 0.26.0+cpu, Flower 1.39.0.
- Scientific runtime pins: NumPy 2.5.2, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, Matplotlib 3.11.2, Seaborn 0.13.2, PyYAML 6.0.3, pytest 9.1.1.
- NVIDIA RTX 3050 exists on host, but the project environment is CPU-only (`torch.cuda.is_available()` false); all reported development runs used CPU.
- Python 3.11 was not installed. No user installation was needed for the centralized and Flower core stages.
- MNIST was downloaded to `data/` only for explicit experiments. Unit tests use synthetic values.
- Flower simulation is implemented in-process: local simulated clients produce Flower `FitRes` objects and the installed Flower `FedAvg` Strategy performs its weighted aggregation. Flower's Ray-backed distributed simulation and RPC transport are not used because Ray is not part of this local execution path.
- Dependency file records the versions validated for development. Install the Torch pair from the official CPU wheel index shown in `requirements.txt`; select the appropriate official index for a CUDA build on other machines.
