# [SigKit](https://github.com/users/IsaiahHarvi/projects/5)

[![Version](https://img.shields.io/github/v/release/IsaiahHarvi/SigKit.svg)](https://github.com/IsaiahHarvi/SigKit/releases)
[![Tests Passing](https://img.shields.io/github/actions/workflow/status/IsaiahHarvi/SigKit/test.yaml)](https://github.com/IsaiahHarvi/SigKit/actions?query=workflow%3Apy-test)
[![GitHub Contributors](https://img.shields.io/github/contributors/IsaiahHarvi/SigKit.svg)](https://github.com/IsaiahHarvi/SigKit/graphs/contributors)
[![Issues](https://img.shields.io/github/issues/IsaiahHarvi/SigKit.svg)](https://github.com/IsaiahHarvi/SigKit/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/IsaiahHarvi/SigKit.svg)](https://github.com/IsaiahHarvi/SigKit/pulls)


**SigKit** is a modular digital signal‐processing toolkit built on top of NumPy. It has additive impairments to simulate OTA impairments. We include a full Machine Learning toolkit with methods for PyTorch designed for training Modulation Classification Models that are capable of generalizing over the air (OTA). It provides:

- **Core abstractions** (`Signal`, `Impairment`, `Modem`, …) for working in complex baseband
- **NumPy ipairments,  PyTorch Transforms & tools** (AWGN, fading, filtering, SNR & BER calculators)
- **PyTorch Transforms** so you can drop signal operations straight into `Compose`
- **PyTorch Lightning Pipeline** includes our pretrained model and methods for retraining
- **(WIP) GNURadio Blocks** wrapping our tools
- **Synthetic data generators** & `torch.utils.data.Dataset` classes

---

## 🚀 Getting Started

### Try the example notebook
A quick way to explore SigKit is to run the Jupyter notebook in:
```
examples/notebooks/basic.ipynb
```

The notebook is a guide that covers:
- Generating a signal with a Modem
- Adding an Impairment like AWGN
- Calculating Signal Metrics
- Visualizing the waveform
> Be sure to restart your notebook's kernel after installing the package.

### Classification
There is also an example notebook for running the pretrained model.
```
examples/notebooks/classification.ipynb
```
> Pull down LFS objects first with `git lfs pull`


## Installing SigKit

### Package Installation
##### Installing the [pypi](https://pypi.org/project/sigkit/) package
```bash
pip install sigkit
```

##### Installing from source (recommended for ML tasks)
```bash
git clone https://github.com/IsaiahHarvi/SigKit.git
cd SigKit
pip install -e .
```

##### Sanity Check for source installs
You can be gauranteeed your installation is sound by running `pytest` without failure from the root of the repository.

### (Optional) DevContainer for VS Code

If you use VS Code and would prefer to isolate your SigKit installation, we’ve provided a DevContainer in the repository.
It is designed to be OS agnostic but it is confirmed to support: `Ubunutu >22.04, ARM MacOS, Windows`

To setup:
1. Install the **Remote – Containers** extension in VS Code.
2. Clone, open the project, and run `chmod +x .devcontainer/setup.sh`.
3. Run **Reopen in Container** from the VSCode console.

Inside the container you’ll have all dependencies installed and SigKit ready to run.

---

## State of Implementations

The below table tracks major planned features. Smaller features like Signal Impairments and Torch Transforms are not tracked here.

Feature | Implemented | Torch Transform
--------|-------------|----------------
**_Architecture_**    |  -   | -
Rust bindings         |  No  | -
**_GNURadio_**        |  -   | -
ClassificationBlock   |  No  | -
Modulator             |  No  | -
**Modems**            |  -   | -
PSK                   | Yes  | -
FSK                   | Yes  | -
QAM                   | WIP  | -
OFDM                  | WIP  | -
MSK                   | No   | -
ASK                   | No   | -
**Machine Learning**  | -    | -
Training Pipeline     | Yes  | -
Pretrained Model      | Yes (WIP) | -
Procedural Dataset    | Yes  | -
File Dataset          | No  | -
**Signal Impairments**| -    | -
AWGN                  | Yes  | ✅
Frequency Shift       | Yes  | ✅
Phase Shift           | Yes  | ✅
Polyphase Resampling  | No   | ❌
Rayleigh Fading       | No   | ❌
Random Resampling     | No   | ❌
IQ Imbalance          | No   | ❌
Time Shift            | No   | ❌

---

## 🛠️ Development

* **Tests**:

  ```bash
  pytest
  ```
* **Lint & Format**:

  ```bash
  ruff check . --fix
  ruff format .
  ```
* **Update docs**:

  ```bash
  # Install the optional dependencies [docs] or:
  pip install sphinx piccolo_theme tomli

  ./docs/gen.sh
  ```
---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
