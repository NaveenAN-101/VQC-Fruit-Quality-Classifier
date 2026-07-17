"""
Fruit Quality Classifier — Hybrid Quantum-Classical VQC
Serves the trained model (CNN featurizer -> 3-qubit AmplitudeEmbedding VQC) via a Gradio UI.

Usage:
    1. Place your trained checkpoint (vqc_amplitude_final.pt) in the same folder as this file,
       or set CHECKPOINT_PATH to wherever it lives.
    2. pip install -r requirements.txt
    3. python app.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import gradio as gr
import torchvision as tv
from torchvision import transforms

print("1. Starting app...", flush=True)

CHECKPOINT_PATH = "vqc_amplitude_final.pt"

# This model never benefits from a GPU: the CNN is tiny and the quantum circuit runs as a
# CPU-based classical simulation via PennyLane's default.qubit. Being explicit here avoids
# any confusion about what hardware this actually needs when deploying (CPU-basic is enough).
device = torch.device("cpu")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found at '{CHECKPOINT_PATH}'. Make sure vqc_amplitude_final.pt "
        f"is uploaded alongside this app.py file."
    )

print("2. Loading checkpoint...", flush=True)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
print("3. Checkpoint loaded", flush=True)

class_names = checkpoint["class_names"]
sd = checkpoint["model_state_dict"]
cfg = checkpoint.get("model_config", {})

# Prefer the config saved with the checkpoint; fall back to reading it straight off the
# weight shapes if an older checkpoint doesn't have model_config (keeps this app portable
# across your different training runs without hardcoding architecture numbers).
if cfg:
    n_qubits = cfg["n_qubits"]
    n_layers = cfg["n_layers"]
    n_features = cfg["n_features"]
else:
    q_key = [k for k in sd if k.endswith("weights") and sd[k].dim() == 3][0]
    n_layers, n_qubits, _ = sd[q_key].shape
    head_key = [k for k in sd if k.endswith("head.weight")][0]
    n_features = sd[head_key].shape[0]

print("4. Configuration loaded", flush=True)

print(
    f"Loaded model: n_qubits={n_qubits}, "
    f"n_layers={n_layers}, "
    f"n_features={n_features}, "
    f"classes={class_names}, "
    f"test_acc={checkpoint.get('test_acc', 'N/A')}",
    flush=True
)


class Featurizer(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))


print("5. Creating quantum device...", flush=True)
dev = qml.device("default.qubit", wires=n_qubits)
print("6. Quantum device created", flush=True)


@qml.qnode(dev, interface="torch")
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(
        features=inputs,
        wires=range(n_qubits),
        normalize=True,
        pad_with=0.0
    )
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(len(class_names))]


weight_shapes = {"weights": (n_layers, n_qubits, 3)}

print("7. Creating TorchLayer...", flush=True)
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
print("8. TorchLayer created", flush=True)


class HybridVQC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe = Featurizer(out_dim=n_features)
        self.q = qlayer

    def forward(self, x):
        return self.q(self.fe(x))


print("9. Creating model...", flush=True)
model = HybridVQC().to(device)
print("10. Model created", flush=True)

model.load_state_dict(sd)
print("11. Weights loaded", flush=True)

model.eval()
print("12. Model ready", flush=True)

transform = transforms.Compose([
    transforms.Resize((128, 128))
])


def predict(image):
    if image is None:
        return None
    try:
        img = image.convert("RGB")
        x = transform(tv.transforms.functional.to_tensor(img)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze()

        return {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

    except Exception as e:
        # Surface a clean message in the UI instead of a raw traceback
        gr.Warning(f"Couldn't process this image: {e}")
        return None


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a fruit image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Quality"),
    title="🍎 Fruit Quality Classifier",
    description=(
        "Hybrid quantum-classical model: CNN feature extractor -> "
        "3-qubit AmplitudeEmbedding variational quantum circuit "
        "(simulated via PennyLane) -> good / moderate / bad prediction."
    ),
)

demo.queue()  # handle concurrent requests gracefully instead of them piling up and timing out


if __name__ == "__main__":
    print("13. Launching Gradio...", flush=True)

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )

    print("14. Gradio launched", flush=True)