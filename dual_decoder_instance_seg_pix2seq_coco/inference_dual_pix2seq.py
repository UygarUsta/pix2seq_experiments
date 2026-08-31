"""
inference_dual_pix2seq.py
=========================
Eğitilmiş dual-decoder modelini görüntü/klasör üzerinde çalıştırır.

Ön işleme eğitimdeki val yoluyla birebir aynı: pad_to_square (sol-üst, gri) ->
IMG_SIZE'a resize -> ImageNet normalize. Tahminler orijinal çözünürlüğe geri
ölçeklenir.
"""

import os
import json
import time
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm

from dual_decoder_pix2seq import (
    DualPix2SeqModel, dual_ar_decode,
    VOCAB_SIZE, IMG_SIZE, ID_TO_LABEL, INFER_SCORE_THRESH, MAX_DETS,
    VAL_IMG_DIR, CKPT_PREFIX,
)

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PALETTE = [
    (255, 64, 64), (64, 200, 255), (120, 255, 120), (255, 200, 64),
    (200, 120, 255), (64, 255, 220), (255, 128, 200), (180, 180, 255),
]


# ── Ön işleme ────────────────────────────────────────────────────────────────
def pad_to_square_infer(img):
    w, h = img.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), (128, 128, 128))
    canvas.paste(img, (0, 0))
    return canvas, m


def resolve_image_path(image_path):
    if not os.path.exists(image_path) or image_path.endswith(".json"):
        base = os.path.splitext(image_path)[0]
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(base + ext):
                return base + ext
    return image_path


def preprocess_image(image_path):
    """(tensor[1,3,H,W], padded_pil, orig_w, orig_h, max_dim)"""
    original = Image.open(image_path).convert("RGB")
    ow, oh = original.size
    padded, max_dim = pad_to_square_infer(original)
    resized = TF.resize(padded, list(IMG_SIZE))
    return _NORMALIZE(resized).unsqueeze(0), padded, ow, oh, max_dim


# ── Görselleştirme ───────────────────────────────────────────────────────────
def draw_predictions(padded_img, predictions, orig_w, orig_h, draw_box=True,
                     mask_alpha=0.35):
    base = padded_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    for i, pred in enumerate(predictions):
        color = PALETTE[i % len(PALETTE)]
        od.polygon([tuple(p) for p in pred["polygon"]],
                   fill=color + (int(255 * mask_alpha),), outline=color + (255,))

    result = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(result)

    for i, pred in enumerate(predictions):
        color = PALETTE[i % len(PALETTE)]
        pts = [tuple(p) for p in pred["polygon"]]
        draw.line(pts + [pts[0]], fill=color, width=3)

        x0, y0, x1, y1 = pred["box"]
        if draw_box:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        tag = f'{pred["label"]} {pred["score"]:.2f}'
        draw.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(tag) + 8, y0], fill=color)
        draw.text((x0 + 4, max(0, y0 - 15)), tag, fill="black")

    return result.crop((0, 0, orig_w, orig_h))


def load_model(model_path, device):
    model = DualPix2SeqModel(vocab_size=VOCAB_SIZE).to(device)
    obj = torch.load(model_path, map_location=device)
    state = obj.get("model", obj) if isinstance(obj, dict) and "model" in obj else obj
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v
             for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ── Batch inference ──────────────────────────────────────────────────────────
@torch.no_grad()
def predict_batch(model, image_paths, device, score_thresh=INFER_SCORE_THRESH,
                  max_dets=MAX_DETS):
    tensors, metas = [], []
    for path in image_paths:
        p = resolve_image_path(path)
        if not os.path.exists(p):
            print(f"  Atlandı (bulunamadı): {p}")
            continue
        t, padded, ow, oh, md = preprocess_image(p)
        tensors.append(t)
        metas.append({"path": p, "padded": padded, "w": ow, "h": oh, "max_dim": md})

    if not tensors:
        return []

    batch = torch.cat(tensors, 0).to(device)
    raw = dual_ar_decode(model, batch, device, score_thresh=score_thresh,
                         max_dets=max_dets)

    results = []
    for i, meta in enumerate(metas):
        sx = meta["max_dim"] / float(IMG_SIZE[1])
        sy = meta["max_dim"] / float(IMG_SIZE[0])
        scaled = []
        for r in raw[i]:
            poly = r["polygon"].copy()
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            scaled.append({
                "label": ID_TO_LABEL.get(r["label"], "unknown"),
                "box": [r["box"][0] * sx, r["box"][1] * sy,
                        r["box"][2] * sx, r["box"][3] * sy],
                "polygon": poly.tolist(),
                "score": r["score"],
            })
        results.append({**meta, "predictions": scaled})
    return results


def predict_folder(image_paths, model, device, batch_size=8, save_json=False,
                   out_dir="test_output", score_thresh=INFER_SCORE_THRESH):
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch Inference"):
        for r in predict_batch(model, image_paths[i:i + batch_size], device,
                               score_thresh=score_thresh):
            img = draw_predictions(r["padded"], r["predictions"], r["w"], r["h"])
            out_path = os.path.join(out_dir, os.path.basename(r["path"]))
            img.save(out_path)

            if save_json:
                shapes = [{"label": p["label"], "points": p["polygon"],
                           "group_id": None, "shape_type": "polygon",
                           "flags": {"score": round(float(p["score"]), 4)}}
                          for p in r["predictions"]]
                with open(os.path.splitext(out_path)[0] + ".json", "w",
                          encoding="utf-8") as f:
                    json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                               "imagePath": os.path.basename(r["path"]),
                               "imageHeight": r["h"], "imageWidth": r["w"]},
                              f, ensure_ascii=False, indent=2)


def benchmark(model, image_path, device, runs=5, score_thresh=INFER_SCORE_THRESH):
    t, *_ = preprocess_image(resolve_image_path(image_path))
    img = t.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    dual_ar_decode(model, img, device, score_thresh=score_thresh)   # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        dual_ar_decode(model, img, device, score_thresh=score_thresh)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"  Gecikme: {(time.perf_counter() - t0) / runs * 1000:7.1f} ms/görüntü")


# ── Giriş noktası ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_IMG_PATH = ""
    MODEL_PATH    = f"{CKPT_PREFIX}_best_map.pth"
    FOLDER_PATH   = VAL_IMG_DIR       # COCO val2017; boş bırakırsan atlanır
    N_FOLDER      = 50                # ilk N görüntü (tüm val2017 uzun sürer)
    SCORE_THRESH  = 0.5               # görselleştirme eşiği; mAP çok daha düşüğünü ister

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference cihazı: {device}")
    model = load_model(MODEL_PATH, device)

    if TEST_IMG_PATH and os.path.exists(TEST_IMG_PATH):
        res = predict_batch(model, [TEST_IMG_PATH], device, score_thresh=SCORE_THRESH)
        if res:
            r = res[0]
            draw_predictions(r["padded"], r["predictions"], r["w"], r["h"]) \
                .save("dual_single_infer_result.jpg")
            print(f"Tek görüntü sonucu ({len(r['predictions'])} nesne) "
                  f"-> dual_single_infer_result.jpg")
        print("\nGecikme testi:")
        benchmark(model, TEST_IMG_PATH, device, score_thresh=SCORE_THRESH)

    if FOLDER_PATH and os.path.isdir(FOLDER_PATH):
        paths = sorted(p for p in glob(os.path.join(FOLDER_PATH, "*"))
                       if not p.endswith(".json"))[:N_FOLDER]
        print(f"\nKlasör inference: {FOLDER_PATH} ({len(paths)} görüntü)")
        predict_folder(paths, model, device, batch_size=8, save_json=True,
                       out_dir="test_output_val", score_thresh=SCORE_THRESH)
