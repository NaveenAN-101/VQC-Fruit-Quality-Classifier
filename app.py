import os
import threading
import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import torchvision as tv
from torchvision import transforms

CHECKPOINT_PATH = "vqc_amplitude_final.pt"
device = torch.device("cpu")

model = None
class_names = None
model_ready = threading.Event()
load_error = None


def load_model():
    global model, class_names, load_error
    try:
        print("Loading checkpoint...", flush=True)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        cn = checkpoint["class_names"]
        sd = checkpoint["model_state_dict"]
        cfg = checkpoint.get("model_config", {})

        if cfg:
            n_qubits, n_layers, n_features = cfg["n_qubits"], cfg["n_layers"], cfg["n_features"]
        else:
            q_key = [k for k in sd if k.endswith("weights") and sd[k].dim() == 3][0]
            n_layers, n_qubits, _ = sd[q_key].shape
            head_key = [k for k in sd if k.endswith("head.weight")][0]
            n_features = sd[head_key].shape[0]

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

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(n_qubits), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(len(cn))]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        class HybridVQC(nn.Module):
            def __init__(self):
                super().__init__()
                self.fe = Featurizer(out_dim=n_features)
                self.q = qlayer

            def forward(self, x):
                return self.q(self.fe(x))

        m = HybridVQC().to(device)
        m.load_state_dict(sd)
        m.eval()

        global model, class_names
        model = m
        class_names = cn
        model_ready.set()
        print("Model ready", flush=True)
    except Exception as e:
        load_error = str(e)
        print(f"Model failed to load: {e}", flush=True)


threading.Thread(target=load_model, daemon=True).start()

transform = transforms.Compose([transforms.Resize((128, 128))])


def predict(image):
    if load_error:
        return {"error - check logs": 1.0}
    if not model_ready.is_set():
        return {"model still loading, try again shortly": 1.0}
    if image is None:
        return None
    img = image.convert("RGB")
    x = transform(tv.transforms.functional.to_tensor(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze()
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a fruit image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted Quality"),
    title="🍎 Fruit Quality Classifier",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Binding to port {port}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)