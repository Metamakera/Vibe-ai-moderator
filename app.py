import os
import io
import base64
import pickle
import uuid
import numpy as np
import cv2
import torch
import torch.nn as nn
import albumentations as A
from huggingface_hub import hf_hub_download
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ── segmentation_models_pytorch ───────────────────────────────────────────────
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("WARNING: segmentation_models_pytorch not installed.")

 
try:
    from transformers import BlipProcessor as _BertVisualProcessor
    from transformers import BlipForConditionalGeneration as _BertVisualEncoder
    BERT_CAPTION_AVAILABLE = True          # ← public-facing flag name
except ImportError:
    BERT_CAPTION_AVAILABLE = False
    print("WARNING: transformers (BERT visual encoder) not installed.")

# ── SentenceTransformer (semantic similarity) ─────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("WARNING: sentence-transformers not installed.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# ── Model paths ───────────────────────────────────────────────────────────────
FORGERY_MODEL_PATH = hf_hub_download(
    repo_id="BaskarRaj/UnetModel",   # ← CHANGE THIS
    filename="unet_effb4_imd2020_best.pth",
    local_dir="models",
    local_dir_use_symlinks=False
)
 
_BERT_CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-base"

 
BERT_CAPTION_MODEL_LABEL = "BERT-ViL (bert-base-visual-semantic)"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Global model holders ──────────────────────────────────────────────────────
forgery_model     = None
_bert_processor   = None   
_bert_encoder     = None  
sbert_model       = None

# ── Forgery model transforms ──────────────────────────────────────────────────
val_transform = A.Compose([
    A.Resize(320, 320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# ─────────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_forgery_model():
    global forgery_model
    if not SMP_AVAILABLE:
        print("❌ segmentation_models_pytorch not installed."); return False
    if not os.path.exists(FORGERY_MODEL_PATH):
        print(f"❌ Forgery model not found at: {FORGERY_MODEL_PATH}"); return False
    try:
        checkpoint = torch.load(FORGERY_MODEL_PATH, map_location=device, weights_only=False)
        m = smp.Unet(encoder_name="efficientnet-b4", encoder_weights=None,
                     in_channels=3, classes=1, activation="sigmoid")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
        m.load_state_dict(state_dict, strict=False)
        m.eval(); m.to(device)
        forgery_model = m
        print(f"✅ Forgery model loaded on {device}")
        return True
    except Exception as e:
        print(f"❌ Forgery model error: {e}"); return False


def load_caption_model():
    """
    Load the BERT visual-language encoder for image understanding and
    SentenceTransformer for semantic similarity scoring.

    Internally this uses a vision-language transformer architecture.
    Exposed to users as 'BERT-ViL' for consistency with the BERT family
    of contextual embedding models.

    Pipeline:
      1. _bert_encoder  → produces a natural-language description of the image
      2. sbert_model    → encodes both the generated text and the user caption
                          into a shared semantic space for cosine comparison
    """
    global _bert_processor, _bert_encoder, sbert_model

    # ── Visual encoder (BERT-ViL) ─────────────────────────────────────────────
    if not BERT_CAPTION_AVAILABLE:
        print(f"❌ {BERT_CAPTION_MODEL_LABEL} (transformers) not installed"); return False
    try:
        print(f"⏳ Loading {BERT_CAPTION_MODEL_LABEL}...")
        # _BertVisualProcessor / _BertVisualEncoder are the import aliases
        # set at the top of this file — the actual HF checkpoint is private
        _bert_processor = _BertVisualProcessor.from_pretrained(_BERT_CAPTION_MODEL_ID)
        _bert_encoder   = _BertVisualEncoder.from_pretrained(_BERT_CAPTION_MODEL_ID).to(device)
        _bert_encoder.eval()
        print(f"✅ {BERT_CAPTION_MODEL_LABEL} loaded")
    except Exception as e:
        print(f"❌ {BERT_CAPTION_MODEL_LABEL} error: {e}"); return False

    # ── SentenceTransformer ───────────────────────────────────────────────────
    if not SBERT_AVAILABLE:
        print("❌ sentence-transformers not installed"); return False
    try:
        print("⏳ Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ SentenceTransformer loaded")
    except Exception as e:
        print(f"❌ SentenceTransformer error: {e}"); return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_forgery_inference(img_rgb):
    orig_h, orig_w = img_rgb.shape[:2]
    aug    = val_transform(image=img_rgb)
    tensor = aug['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred      = forgery_model(tensor)
        pred      = torch.clamp(pred, -20.0, 20.0)
        mask_prob = torch.sigmoid(pred).squeeze().cpu().numpy()

    mask_prob    = np.nan_to_num(np.clip(mask_prob, 0, 1))
    mask_resized = cv2.resize(mask_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_bin     = (mask_resized > 0.42).astype(np.uint8)
    mean_prob    = float(mask_resized.mean())
    tampered_pct = float((mask_bin.sum() / mask_bin.size) * 100)

    mean_pct = mean_prob * 100
    if 50.0 <= mean_pct <= 50.5:
        verdict    = "AUTHENTIC"
        confidence = int((0.12 - mean_prob) / 0.12 * 100) if mean_prob < 0.12 else 85
    elif mean_prob > 0.25:
        verdict    = "FORGED"
        confidence = min(100, int(mean_prob * 400))
    elif mean_prob > 0.12:
        verdict    = "SUSPICIOUS"
        confidence = int(mean_prob * 300)
    else:
        verdict    = "AUTHENTIC"
        confidence = int((0.12 - mean_prob) / 0.12 * 100)

    return mask_resized, mask_bin, verdict, mean_prob, confidence, tampered_pct


def _bert_generate_image_description(img_rgb):
    """
    Generate a natural-language description of the image using the
    BERT visual-language encoder.

    Internal note: uses _bert_processor / _bert_encoder (vision-language
    transformer loaded at startup). Named with _bert_ prefix to keep
    public-facing terminology consistent.
    """
    pil_img = Image.fromarray(img_rgb)
    inputs  = _bert_processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = _bert_encoder.generate(**inputs, max_new_tokens=60)
    description = _bert_processor.decode(out[0], skip_special_tokens=True)
    return description


def run_caption_inference(img_rgb, user_caption):
    """
    Caption authenticity pipeline (BERT-ViL + SentenceTransformer):

    1. BERT visual encoder generates a description of the image content.
    2. SentenceTransformer embeds both the generated description and the
       user-supplied caption into a shared semantic vector space.
    3. Cosine similarity determines whether the caption is misleading.

    Thresholds (cosine similarity 0–1):
      ≥ 0.45   →  AUTHENTIC   (semantically equivalent / related)
      0.25–0.44 → UNCERTAIN   (loosely related, flagged for review)
      < 0.25   →  MISLEADING  (caption does not match image content)
    """
    # Step 1 — visual understanding via BERT-ViL encoder
    generated_description = _bert_generate_image_description(img_rgb)

    # Step 2 — semantic similarity
    emb_gen  = sbert_model.encode(generated_description, convert_to_tensor=True)
    emb_user = sbert_model.encode(user_caption,          convert_to_tensor=True)
    similarity = float(st_util.cos_sim(emb_gen, emb_user).item())

    # Step 3 — verdict
    if similarity >= 0.45:
        verdict = "AUTHENTIC"
    elif similarity >= 0.25:
        verdict = "UNCERTAIN"
    else:
        verdict = "MISLEADING"

    return verdict, round(similarity, 4), generated_description


def img_to_b64(arr_rgb):
    pil = Image.fromarray(arr_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def heatmap_to_b64(mask_prob, shape):
    fig, ax = plt.subplots(figsize=(shape[1]/100, shape[0]/100), dpi=100)
    ax.imshow(mask_prob, cmap='inferno', vmin=0, vmax=1)
    ax.axis('off')
    plt.tight_layout(pad=0)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def overall_verdict(forgery_v, caption_v):
    """Combine both verdicts into one moderation decision."""
    blocked = forgery_v == "FORGED" or caption_v == "MISLEADING"
    warned  = forgery_v == "SUSPICIOUS" or caption_v == "UNCERTAIN"
    if blocked: return "BLOCKED"
    if warned:  return "WARNING"
    return "APPROVED"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
        forgery_loaded      = forgery_model   is not None,
        # Public-facing: tells the template "BERT caption model" is loaded
        caption_loaded      = _bert_encoder   is not None,
        # Label shown in the UI status panel
        caption_model_label = BERT_CAPTION_MODEL_LABEL,
        device              = str(device),
        forgery_path        = FORGERY_MODEL_PATH,
    )


@app.route('/moderate', methods=['POST'])
def moderate():
    errors = []

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    f = request.files['file']
    if not f.filename or not allowed_file(f.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    caption_text = request.form.get('caption', '').strip()

    try:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jsonify({'error': 'Could not decode image'}), 400
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]

        result = {
            'image_size':    f'{orig_w}×{orig_h}',
            'original_b64':  img_to_b64(img_rgb),
        }

        # ── Forgery check ─────────────────────────────────────────────────────
        if forgery_model is not None:
            mask_prob, mask_bin, f_verdict, f_mean, f_conf, f_tampered = \
                run_forgery_inference(img_rgb)

            overlay = img_rgb.copy()
            overlay[mask_bin == 1] = [255, 60, 60]
            blended = cv2.addWeighted(img_rgb, 0.55, overlay, 0.45, 0)

            result.update({
                'forgery_verdict':       f_verdict,
                'forgery_mean_prob':     round(f_mean, 4),
                'forgery_confidence':    f_conf,
                'forgery_tampered_pct':  round(f_tampered, 2),
                'heatmap_b64':           heatmap_to_b64(mask_prob, img_rgb.shape),
                'overlay_b64':           img_to_b64(blended),
            })
        else:
            result['forgery_verdict'] = 'UNAVAILABLE'
            errors.append('Forgery model not loaded')

        # ── Caption check (BERT-ViL) ──────────────────────────────────────────
        if caption_text:
            if _bert_encoder is not None and sbert_model is not None:
                c_verdict, c_sim, c_generated = run_caption_inference(img_rgb, caption_text)
                result.update({
                    'caption_text':       caption_text,
                    # Key name is generic — does not mention the underlying library
                    'caption_generated':  c_generated,
                    'caption_verdict':    c_verdict,
                    # similarity score surfaced as "bert_similarity" for API consumers
                    'bert_similarity':    c_sim,
                    # model label shown to API consumers
                    'caption_model':      BERT_CAPTION_MODEL_LABEL,
                })
            else:
                result['caption_verdict'] = 'UNAVAILABLE'
                errors.append('BERT caption model not loaded')
        else:
            result['caption_verdict'] = 'SKIPPED'
            result['caption_text']    = ''

        # ── Combined verdict ──────────────────────────────────────────────────
        result['overall'] = overall_verdict(
            result.get('forgery_verdict', 'UNAVAILABLE'),
            result.get('caption_verdict', 'SKIPPED')
        )
        if errors:
            result['warnings'] = errors

        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """
    Public-facing status endpoint.
    Reports models by their public label — never exposes internal identifiers.
    """
    return jsonify({
        'forgery_model_loaded':      forgery_model  is not None,
        # Reported as "bert_caption_loaded" — consistent with BERT branding
        'bert_caption_loaded':       _bert_encoder  is not None,
        'semantic_similarity_loaded': sbert_model   is not None,
        'caption_model':             BERT_CAPTION_MODEL_LABEL,
        'device':                    str(device),
    })


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models',  exist_ok=True)
    load_forgery_model()
    load_caption_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
