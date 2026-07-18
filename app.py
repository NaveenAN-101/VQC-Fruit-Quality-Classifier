"""
Fruit Quality Classifier — FastAPI version
Opens the HTTP port immediately on startup, loads the (slow-to-import) torch/pennylane
model in a background thread afterward. This avoids host platforms (like Render) timing
out their port-scan while heavy ML imports are still happening.
"""

import os
import io
import threading

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI()

CHECKPOINT_PATH = "vqc_amplitude_final.pt"

# --- Shared state, filled in by the background loader ---
model = None
class_names = None
model_ready = threading.Event()
load_error = None


def load_model():
    """Runs in a background thread so the server can accept connections immediately."""
    global model, class_names, load_error
    try:
        print("Importing torch/pennylane...", flush=True)
        import torch
        import torch.nn as nn
        import pennylane as qml
        import torchvision as tv

        device = torch.device("cpu")

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

        print("Building quantum circuit...", flush=True)
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

        # Stash everything the /predict route needs, module-level, after success
        global _torch, _F, _tv, _transforms
        import torch.nn.functional as F
        from torchvision import transforms
        _torch, _F, _tv, _transforms = torch, F, tv, transforms

        model = m
        class_names = cn
        model_ready.set()
        print("✅ Model ready", flush=True)

    except Exception as e:
        load_error = str(e)
        print(f"❌ Model failed to load: {e}", flush=True)


threading.Thread(target=load_model, daemon=True).start()


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fruit Quality Classifier</title>
        <style>
            body { font-family: sans-serif; max-width: 480px; margin: 60px auto; text-align: center; }
            #result { margin-top: 20px; font-size: 18px; }
            .bar-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; text-align: left; }
            .bar-label { width: 80px; }
            .bar-track { flex: 1; background: #eee; border-radius: 4px; overflow: hidden; height: 16px; }
            .bar-fill { background: #2d9e6f; height: 100%; }
        </style>
    </head>
    <body>
        <h2>🍎 Fruit Quality Classifier</h2>
        <p>Hybrid quantum-classical model (CNN + AmplitudeEmbedding VQC)</p>
        <input type="file" id="fileInput" accept="image/*">
        <div id="result"></div>

        <script>
            document.getElementById('fileInput').addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                document.getElementById('result').innerText = 'Predicting...';

                const formData = new FormData();
                formData.append('file', file);

                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();

                if (data.error) {
                    document.getElementById('result').innerText = data.error;
                    return;
                }

                let html = '';
                for (const [label, prob] of Object.entries(data)) {
                    const pct = (prob * 100).toFixed(1);
                    html += `<div class="bar-row">
                        <div class="bar-label">${label}</div>
                        <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                        <div>${pct}%</div>
                    </div>`;
                }
                document.getElementById('result').innerHTML = html;
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
def health():
    """Useful for checking load status separately from the port scan."""
    if load_error:
        return JSONResponse({"status": "error", "detail": load_error}, status_code=500)
    if model_ready.is_set():
        return {"status": "ready"}
    return {"status": "loading"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if load_error:
        return JSONResponse({"error": f"Model failed to load: {load_error}"}, status_code=500)
    if not model_ready.is_set():
        return JSONResponse({"error": "Model still loading, try again in a moment"}, status_code=503)

    from PIL import Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    transform = _transforms.Compose([_transforms.Resize((128, 128))])
    x = transform(_tv.transforms.functional.to_tensor(image)).unsqueeze(0)

    with _torch.no_grad():
        logits = model(x)
        probs = _F.softmax(logits, dim=1).squeeze()

    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port} (model loads in background)...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)