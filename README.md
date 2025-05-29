# SigKit

> NOTE: Not all features are available yet, this project is actively under development but has a clear roadmap.

**SigKit** is a modular signal‐processing toolkit built on top of NumPy, with PyTorch integration for training Modulation Classifiers capable of generalizing OTA. It provides:

- **Core abstractions** (`Signal`, `Impairment`, `Modem`, …) for working in complex baseband
- **Pure-NumPy transforms & impairments** (AWGN, fading, filtering, SNR & BER calculators)
- **PyTorch methods** so you can drop signal operations straight into `nn.Sequential`
- **Synthetic data generators** & `torch.utils.data.Dataset` classes for quick ML prototyping

---

## 🚀 Getting Started

### 1. Try the example notebook
A quick way to explore SigKit is to run the Jupyter notebook in:
```
examples/notebooks/basic/modulator\_demo.ipynb
```

It walks through:
- Generating a signal with a Modem
- Adding an Impairment like AWGN
- Calculating Signal Metrics
- Visualizing the waveform

### 2. Local installation

```bash
git clone https://github.com/IsaiahHarvi/SigKit.git
cd SigKit
pip install -e .
```

### 2.1 (Optional) DevContainer for VS Code

If you use VS Code, we’ve provided a DevContainer configuration:

1. Install the **Remote – Containers** extension in VS Code.
2. Clone and open the project and choose **Reopen in Container** from the VSCode console.
3. Inside the container you’ll have all dependencies installed and SigKit ready to run.

### 3. Sanity Check
You can be gauranteeed your installation is sound by running `pytest` without failure from the root of the repository.

---

## 📦 Features

* **Core** (`src/sigkit/core`):
  ‣ `Signal` container, `SignalDataset` interface, utility functions (SNR, BER, etc.)
* **Impairments** (`src/sigkit/impairments`):
  ‣ Methods to simulate Over-the-air and digital effects on waveforms
* **Transforms** (`src/sigkit/transforms`):
  ‣ PyTorch `nn.Module` implementations for **Impairments**
* **Modems** (`src/sigkit/modem`):
  ‣ Implementations of various Modems (FSK, QAM, OFDM, etc.)
* **Datasets** (`src/sigkit/datasets`):
  ‣ `torch.utils.data.Dataset` bindings
* **Metrics** (`src/sigkit/metrics`):
  ‣ SNR, BER, Waveform visualiations, etc.

---

## 🛠️ Development

* **Tests**:

  ```bash
  pytest
  ```
* **Lint & Format**:

  ```bash
  ruff check .
  ruff format .
  ```
* **Update docs**:

  ```bash
  # Install the optional dependencies [docs] or:
  pip install sphinx piccolo_theme
  ./docs/gen.sh
  ```
---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.
