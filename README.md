# ForensicLens — Image Forgery Detector (Flask App)

A local Flask web app for pixel-level image forgery detection using your trained UNet + EfficientNet-B4 model from IMD2020.

---

## Project Structure

```
forgery_detector/
├── app.py                  ← Flask backend
├── requirements.txt
├── models/
│   └── unet_effb4_imd2020_best.pth   ← ⬅ PUT YOUR MODEL HERE
├── templates/
│   └── index.html          ← Frontend UI
└── uploads/                ← Auto-created, temp files
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> If you're on CPU only, install the CPU build of PyTorch first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

### 2. Place your model

Copy your trained `.pth` file into the `models/` folder:

```
models/unet_effb4_imd2020_best.pth
```

Or set a custom path via environment variable:

```bash
export MODEL_PATH=/path/to/your/model.pth
```

### 3. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## How it works

1. You drag & drop or browse to upload an image (JPG/PNG/WEBP/BMP)
2. The image is sent to Flask via POST `/predict`
3. Flask runs the UNet model inference:
   - Resizes to 320×320, normalizes with ImageNet stats
   - Sigmoid threshold at 0.42 for binary mask
4. Returns:
   - **Original image** (base64)
   - **Probability heatmap** (inferno colormap)
   - **Tampered overlay** (red highlights on forged regions)
   - Verdict: `STRONG FORGERY / SUSPICIOUS / LIKELY AUTHENTIC`

---

## Notes

- Everything runs **100% locally** — no data is sent anywhere
- GPU is used automatically if available (CUDA)
- The model must match the architecture: `smp.Unet(encoder_name="efficientnet-b4", in_channels=3, classes=1, activation="sigmoid")`
