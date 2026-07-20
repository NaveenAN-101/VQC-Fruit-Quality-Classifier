# 🍎 Fruit Quality Classifier — Hybrid Quantum-Classical VQC

A hybrid quantum-classical image classifier that grades fruit quality (**good** / **moderate** / **bad**) using a CNN feature extractor feeding into a variational quantum circuit, simulated via [PennyLane](https://pennylane.ai/) and trained with PyTorch.

**🔗 Live demo:** [vqc-fruit-quality-classifier.onrender.com](https://vqc-fruit-quality-classifier.onrender.com/)

> Note: hosted on Render's free tier, so the first request after a period of inactivity may take a little longer while the instance wakes up.



---

## How it works

```
Fruit image (128×128 RGB)
        │
        ▼
┌───────────────────┐
│   CNN Featurizer   │   3 conv+ReLU blocks → global avg pool → linear → 8 features
└───────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Amplitude Embedding       │   8 classical features loaded as normalized
│  (3 qubits)                │   amplitudes across 3 qubits
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  Strongly Entangling       │   4 trainable variational layers
│  Layers (variational)      │
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  PauliZ expectation ×3     │   → good / moderate / bad
└───────────────────────────┘
```

The quantum circuit runs as a **classical simulation** on CPU via PennyLane's `default.qubit` device — no quantum hardware is required to train or run this model. Gradients flow end-to-end from the loss through the quantum layer into the CNN, using PennyLane's `TorchLayer` to make the circuit a standard differentiable PyTorch module.

## Dataset

[FruitNet: Indian Fruits Dataset with Quality](https://www.kaggle.com/datasets/shashwatwork/fruitnet-indian-fruits-dataset-with-quality) (Kaggle), containing ~19,500 images of 6 Indian fruits, each labeled by quality:

| Folder | Mapped label |
|---|---|
| `Good Quality_Fruits` | `good` |
| `Mixed Qualit_Fruits` | `moderate` |
| `Bad Quality_Fruits` | `bad` |

Split 70% train / 15% val / 15% test.

## Results

Best model — CNN + 3-qubit AmplitudeEmbedding VQC, trained with square-root-dampened class-weighted loss to address class imbalance (`moderate` is only ~6% of the data):

**Test accuracy: 93.48%**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| good | 0.95 | 0.95 | 0.95 | 1741 |
| moderate | 0.67 | 0.80 | 0.73 | 167 |
| bad | 0.95 | 0.93 | 0.94 | 1021 |
| **accuracy** | | | **0.93** | 2929 |
| macro avg | 0.86 | 0.89 | 0.87 | 2929 |
| weighted avg | 0.94 | 0.93 | 0.94 | 2929 |

Class-weighted loss was key to `moderate`'s recall — an unweighted baseline scored only 0.38 recall on this class, since it's heavily underrepresented and easy for the model to default to `good`.

## Project structure

```
.
├── AI_proj_amplitude_emb.ipynb   # Training notebook (Kaggle/Colab): data loading,
│                                  # model definition, training loop, evaluation
├── app.py                        # FastAPI inference server + web UI
├── requirements.txt               # Runtime dependencies
├── .python-version                # Pins the Python version for Render's build
├── vqc_amplitude_final.pt        # Trained model checkpoint (weights + metadata)
└── README.md
```

## Running locally

```bash
git clone https://github.com/NaveenAN-101/VQC-Fruit-Quality-Classifier.git
cd VQC-Fruit-Quality-Classifier

pip install -r requirements.txt

python app.py
```

Then open `http://localhost:7860` in your browser.

> Make sure `vqc_amplitude_final.pt` is present in the same directory as `app.py` — the app reads the model architecture (qubit/layer/feature counts) directly from the checkpoint's saved config, so it doesn't need to be told the model shape separately.

## Training your own model

Open `AI_proj_amplitude_emb.ipynb` in Kaggle or Colab, attach the FruitNet dataset, and run top to bottom. The notebook:
- Trains the hybrid CNN + VQC model with Adam and class-weighted cross-entropy loss
- Saves the best checkpoint by validation accuracy
- Evaluates on the held-out test set and bundles weights + `test_acc` + `confusion_matrix` + `model_config` into a final checkpoint
- Includes a Gradio-based quick-demo cell for local testing

## Tech stack

- **PyTorch** — model definition, training, autograd
- **PennyLane** — quantum circuit simulation (`default.qubit`) and PyTorch integration (`qml.qnn.TorchLayer`)
- **torchvision** — image loading and transforms
- **FastAPI** + **uvicorn** — inference server
- **Render** — deployment (Docker-free, Python web service)

## Credits

Built by **Naveen AN**.
