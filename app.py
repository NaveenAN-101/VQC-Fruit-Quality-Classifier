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
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fruit Quality Classifier</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap" rel="stylesheet">
    <style>
        :root {
            --paper: #F3F6F1;
            --paper-raised: #FFFFFF;
            --ink: #16211A;
            --ink-soft: #4C5A50;
            --indigo: #5B4FE8;
            --indigo-soft: #EDEBFC;
            --good: #3FA34D;
            --moderate: #E3A73B;
            --bad: #D64550;
            --line: #DCE3D8;
            --radius: 14px;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: var(--paper);
            color: var(--ink);
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 32px 16px;
        }
        .card {
            width: 100%;
            max-width: 460px;
        }
        .eyebrow {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--indigo);
            margin: 0 0 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .eyebrow::before {
            content: "";
            width: 6px; height: 6px;
            background: var(--indigo);
            border-radius: 50%;
            display: inline-block;
        }
        h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 30px;
            letter-spacing: -0.01em;
            margin: 0 0 6px;
            line-height: 1.1;
        }
        .subtitle {
            color: var(--ink-soft);
            font-size: 14px;
            line-height: 1.5;
            margin: 0 0 28px;
        }
        .subtitle b { color: var(--ink); font-weight: 600; }

        .dropzone {
            background: var(--paper-raised);
            border: 1.5px dashed var(--line);
            border-radius: var(--radius);
            padding: 36px 20px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.15s ease, background 0.15s ease;
            position: relative;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: var(--indigo);
            background: var(--indigo-soft);
        }
        .dropzone:focus-visible {
            outline: 2px solid var(--indigo);
            outline-offset: 2px;
        }
        .dz-icon {
            font-size: 26px;
            margin-bottom: 10px;
            display: block;
        }
        .dz-label {
            font-weight: 600;
            font-size: 14.5px;
            margin-bottom: 4px;
        }
        .dz-hint {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11.5px;
            color: var(--ink-soft);
        }
        #fileInput { display: none; }

        .preview-wrap {
            display: none;
            margin-top: 18px;
            gap: 16px;
            background: var(--paper-raised);
            border: 1px solid var(--line);
            border-radius: var(--radius);
            padding: 14px;
            align-items: center;
        }
        .preview-wrap.show { display: flex; }
        #previewImg {
            width: 64px; height: 64px;
            object-fit: cover;
            border-radius: 8px;
            flex-shrink: 0;
        }
        .preview-name {
            font-size: 13px;
            font-weight: 500;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .change-link {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: var(--indigo);
            cursor: pointer;
            background: none; border: none;
            padding: 0; margin-top: 4px;
        }

        /* --- Circuit loading animation --- */
        .circuit {
            display: none;
            margin-top: 22px;
            background: var(--ink);
            border-radius: var(--radius);
            padding: 22px 20px;
        }
        .circuit.show { display: block; }
        .circuit-caption {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: #A9B8AE;
            letter-spacing: 0.04em;
            margin-bottom: 14px;
        }
        .wire {
            position: relative;
            height: 2px;
            background: #33422F;
            margin: 16px 0;
            border-radius: 2px;
            overflow: visible;
        }
        .wire::before {
            content: "q" attr(data-q) "⟩";
            position: absolute;
            left: -26px; top: 50%;
            transform: translateY(-50%);
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px;
            color: #6F8272;
        }
        .pulse {
            position: absolute;
            top: 50%; left: 0;
            width: 10px; height: 10px;
            background: var(--indigo);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 12px 2px rgba(91,79,232,0.7);
            animation: travel 1.6s ease-in-out infinite;
        }
        .wire:nth-child(2) .pulse { animation-delay: 0.15s; }
        .wire:nth-child(3) .pulse { animation-delay: 0.3s; }
        @keyframes travel {
            0% { left: 0%; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { left: 100%; opacity: 0; }
        }

        button.analyze {
            width: 100%;
            margin-top: 18px;
            background: var(--ink);
            color: var(--paper);
            border: none;
            border-radius: var(--radius);
            padding: 14px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 14.5px;
            letter-spacing: 0.01em;
            cursor: pointer;
            transition: transform 0.1s ease, background 0.15s ease;
        }
        button.analyze:hover { background: var(--indigo); }
        button.analyze:active { transform: scale(0.98); }
        button.analyze:disabled { background: var(--line); color: var(--ink-soft); cursor: not-allowed; }

        /* --- Result --- */
        .result {
            display: none;
            margin-top: 22px;
        }
        .result.show { display: block; }
        .stamp {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 16px;
            border-radius: 999px;
            border: 2px solid currentColor;
            transform: rotate(-1.5deg);
            margin-bottom: 18px;
        }
        .stamp.good { color: var(--good); }
        .stamp.moderate { color: var(--moderate); }
        .stamp.bad { color: var(--bad); }
        .stamp-dot { width: 9px; height: 9px; border-radius: 50%; background: currentColor; }

        .bar-row { margin-bottom: 12px; }
        .bar-top {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 5px;
        }
        .bar-name { font-weight: 500; text-transform: capitalize; }
        .bar-pct { font-family: 'IBM Plex Mono', monospace; color: var(--ink-soft); }
        .bar-track {
            background: var(--line);
            border-radius: 6px;
            height: 8px;
            overflow: hidden;
        }
        .bar-fill {
            height: 100%;
            border-radius: 6px;
            width: 0%;
            transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .bar-fill.good { background: var(--good); }
        .bar-fill.moderate { background: var(--moderate); }
        .bar-fill.bad { background: var(--bad); }

        .status-note {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11.5px;
            color: var(--ink-soft);
            margin-top: 14px;
            text-align: center;
        }
        .error-note { color: var(--bad); }

        footer {
            margin-top: 30px;
            font-size: 11.5px;
            color: var(--ink-soft);
            text-align: center;
            line-height: 1.6;
        }
        footer code {
            font-family: 'IBM Plex Mono', monospace;
            background: var(--indigo-soft);
            color: var(--indigo);
            padding: 1px 5px;
            border-radius: 4px;
        }
    </style>
    </head>
    <body>
        <div class="card">
            <p class="eyebrow">Quantum-classical inference</p>
            <h1>Fruit Quality Classifier</h1>
            <p class="subtitle">Upload a photo of a fruit and a <b>CNN feature extractor</b> feeds 8 amplitudes into a <b>3-qubit variational circuit</b>, simulated live, to grade it as good, moderate, or bad.</p>

            <div class="dropzone" id="dropzone" tabindex="0" role="button" aria-label="Upload a fruit image">
                <span class="dz-icon">🍊</span>
                <div class="dz-label">Drop a fruit photo, or click to browse</div>
                <div class="dz-hint">JPG or PNG</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="preview-wrap" id="previewWrap">
                <img id="previewImg" alt="Preview">
                <div>
                    <div class="preview-name" id="previewName"></div>
                    <button class="change-link" id="changeBtn">choose a different photo</button>
                </div>
            </div>

            <button class="analyze" id="analyzeBtn" disabled>Analyze fruit</button>

            <div class="circuit" id="circuit">
                <div class="circuit-caption">reading amplitudes across 3 qubits...</div>
                <div class="wire" data-q="0"><div class="pulse"></div></div>
                <div class="wire" data-q="1"><div class="pulse"></div></div>
                <div class="wire" data-q="2"><div class="pulse"></div></div>
            </div>

            <div class="result" id="result"></div>
            <div class="status-note" id="statusNote"></div>

            <footer>
                Model config auto-detected from checkpoint &middot; runs as a <code>default.qubit</code> simulation, no quantum hardware required
            </footer>
        </div>

        <script>
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('fileInput');
            const previewWrap = document.getElementById('previewWrap');
            const previewImg = document.getElementById('previewImg');
            const previewName = document.getElementById('previewName');
            const changeBtn = document.getElementById('changeBtn');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const circuit = document.getElementById('circuit');
            const resultEl = document.getElementById('result');
            const statusNote = document.getElementById('statusNote');

            let selectedFile = null;

            const classColors = { good: 'good', moderate: 'moderate', bad: 'bad' };
            const classEmoji = { good: '✓', moderate: '~', bad: '✕' };

            dropzone.addEventListener('click', () => fileInput.click());
            dropzone.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
            });
            dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
            dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
            dropzone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
                if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
            });
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length) handleFile(e.target.files[0]);
            });
            changeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                fileInput.click();
            });

            function handleFile(file) {
                selectedFile = file;
                previewImg.src = URL.createObjectURL(file);
                previewName.textContent = file.name;
                previewWrap.classList.add('show');
                dropzone.style.display = 'none';
                analyzeBtn.disabled = false;
                resultEl.classList.remove('show');
                statusNote.textContent = '';
            }

            analyzeBtn.addEventListener('click', async () => {
                if (!selectedFile) return;
                resultEl.classList.remove('show');
                statusNote.textContent = '';
                circuit.classList.add('show');
                analyzeBtn.disabled = true;

                const formData = new FormData();
                formData.append('file', selectedFile);

                try {
                    const res = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    circuit.classList.remove('show');
                    analyzeBtn.disabled = false;

                    if (data.error) {
                        statusNote.textContent = data.error;
                        statusNote.classList.add('error-note');
                        return;
                    }
                    statusNote.classList.remove('error-note');
                    renderResult(data);
                } catch (err) {
                    circuit.classList.remove('show');
                    analyzeBtn.disabled = false;
                    statusNote.textContent = 'Connection error — try again.';
                    statusNote.classList.add('error-note');
                }
            });

            function renderResult(probs) {
                const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
                const [topLabel, topProb] = entries[0];
                const cls = classColors[topLabel] || 'good';

                let html = `<div class="stamp ${cls}"><span class="stamp-dot"></span>${topLabel} &middot; ${(topProb*100).toFixed(0)}%</div>`;
                for (const [label, prob] of entries) {
                    const pct = (prob * 100).toFixed(1);
                    const c = classColors[label] || 'good';
                    html += `
                        <div class="bar-row">
                            <div class="bar-top">
                                <span class="bar-name">${label}</span>
                                <span class="bar-pct">${pct}%</span>
                            </div>
                            <div class="bar-track"><div class="bar-fill ${c}" style="width:0%" data-target="${pct}"></div></div>
                        </div>`;
                }
                resultEl.innerHTML = html;
                resultEl.classList.add('show');

                requestAnimationFrame(() => {
                    resultEl.querySelectorAll('.bar-fill').forEach(el => {
                        el.style.width = el.dataset.target + '%';
                    });
                });
            }
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
        return JSONResponse({"error": "The model failed to start up. Check the server logs."}, status_code=500)
    if not model_ready.is_set():
        return JSONResponse({"error": "Still warming up the circuit — try again in a few seconds."}, status_code=503)

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