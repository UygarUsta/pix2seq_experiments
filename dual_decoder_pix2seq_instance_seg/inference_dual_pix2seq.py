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
    VOCAB_SIZE, IMG_SIZE, ID_TO_LABEL, INFER_SCORE_THRESH
)

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PALETTE = [
    (255, 64, 64), (64, 200, 255), (120, 255, 120), (255, 200, 64),
    (200, 120, 255), (64, 255, 220), (255, 128, 200), (180, 180, 255),
]


# ── Preprocessing & Resolution Helpers ─────────────────────────────────────────

def pad_to_square_infer(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    new_img.paste(img, (0, 0))
    return new_img, max_dim


def resolve_image_path(image_path):
    if not os.path.exists(image_path) or image_path.endswith(".json"):
        base = os.path.splitext(image_path)[0]
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(base + ext):
                return base + ext
    return image_path


def preprocess_image(image_path):
    """Returns (tensor[1,3,H,W], padded_pil, orig_w, orig_h, max_dim)."""
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size
    padded_img, max_dim = pad_to_square_infer(original_img)

    resized = TF.resize(padded_img, IMG_SIZE)
    img_tensor = _NORMALIZE(resized).unsqueeze(0)
    return img_tensor, padded_img, orig_w, orig_h, max_dim


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_predictions(padded_img, predictions, orig_w, orig_h, draw_box=True, mask_alpha=0.35):
    base = padded_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    for i, pred in enumerate(predictions):
        color = PALETTE[i % len(PALETTE)]
        pts = [tuple(p) for p in pred["polygon"]]
        od.polygon(pts, fill=color + (int(255 * mask_alpha),), outline=color + (255,))

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
    state = torch.load(model_path, map_location=device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ── Batch Inference Logic ─────────────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, image_paths, device, score_thresh=INFER_SCORE_THRESH):
    """
    Batches image tensors into [B, 3, H, W], executes Decoder 1 for boxes,
    then executes Decoder 2 in parallel across all surviving objects.

    Decoder 1 now emits a fixed MAX_OBJECTS slots and scores each one against
    the noise class; score_thresh decides how many reach Decoder 2.
    """
    tensors, metas = [], []
    for path in image_paths:
        p = resolve_image_path(path)
        if not os.path.exists(p):
            print(f"  Skipped (not found): {p}")
            continue
        t, padded, ow, oh, md = preprocess_image(p)
        tensors.append(t)
        metas.append({"path": p, "padded": padded, "w": ow, "h": oh, "max_dim": md})

    if not tensors:
        return []

    batch = torch.cat(tensors, dim=0).to(device)
    batch_raw_preds = dual_ar_decode(model, batch, device, score_thresh=score_thresh)

    results = []
    for i, meta in enumerate(metas):
        raw_instances = batch_raw_preds[i]
        sx = meta["max_dim"] / float(IMG_SIZE[1])
        sy = meta["max_dim"] / float(IMG_SIZE[0])

        scaled_predictions = []
        for r in raw_instances:
            poly = r["polygon"].copy()
            poly[:, 0] *= sx
            poly[:, 1] *= sy

            box = [r["box"][0] * sx, r["box"][1] * sy, r["box"][2] * sx, r["box"][3] * sy]
            label_str = ID_TO_LABEL.get(r["label"], "unknown")

            scaled_predictions.append({
                "label": label_str,
                "box": box,
                "polygon": poly.tolist(),
                "score": r["score"],
            })

        results.append({**meta, "predictions": scaled_predictions})

    return results


def predict_folder(image_paths, model, device, batch_size=8, save_json=False,
                   out_dir="test_output", score_thresh=INFER_SCORE_THRESH):
    os.makedirs(out_dir, exist_ok=True)

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Batch Inference"):
        chunk = image_paths[i:i + batch_size]
        batch_results = predict_batch(model, chunk, device, score_thresh=score_thresh)

        for r in batch_results:
            result_img = draw_predictions(r["padded"], r["predictions"], r["w"], r["h"])
            out_img_path = os.path.join(out_dir, os.path.basename(r["path"]))
            result_img.save(out_img_path)

            if save_json:
                shapes = [{
                    "label": p["label"],
                    "points": p["polygon"],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {"score": round(float(p["score"]), 4)},
                } for p in r["predictions"]]
                
                json_path = os.path.splitext(out_img_path)[0] + ".json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": shapes,
                        "imagePath": os.path.basename(r["path"]),
                        "imageHeight": r["h"],
                        "imageWidth": r["w"]
                    }, f, ensure_ascii=False, indent=2)


def benchmark(model, image_path, device, runs=5):
    path = resolve_image_path(image_path)
    t, _, _, _, _ = preprocess_image(path)
    img = t.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()
    dual_ar_decode(model, img, device)  # Warmup
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        dual_ar_decode(model, img, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    dt = (time.perf_counter() - t0) / runs
    print(f"  Dual-Decoder Inference Latency: {dt * 1000:7.1f} ms/img")


# ── Execution Entrypoint ──────────────────────────────────────────────────────

if __name__ == "__main__":
    TEST_IMG_PATH  = "" #"/home/uygarusta/datasets/card_merged_datasets/merged_datasets/ruhsat_1ece4a35-2228-4f86-b6de-b80d3c066745.jpg"
    MODEL_PATH     = "dual_pix2seq_best_map.pth"
    DATA_DIR       = "/mnt/d/Datasets/person.v2i.coco-segmentation/train"
    VAL_SPLIT_PATH = "val_split.json"
    FOLDER_PATH    = ""  # Set path if running inference on an arbitrary folder
    SCORE_THRESH   = 0.5  # visualisation cutoff; evaluation should use a much lower one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    model = load_model(MODEL_PATH, device)

    # 1. Single Image Test
    if os.path.exists(TEST_IMG_PATH):
        single_result = predict_batch(model, [TEST_IMG_PATH], device, score_thresh=SCORE_THRESH)
        if single_result:
            r = single_result[0]
            img = draw_predictions(r["padded"], r["predictions"], r["w"], r["h"])
            img.save("dual_single_infer_result.jpg")
            print(f"Single image result saved ({len(r['predictions'])} detected) -> 'dual_single_infer_result.jpg'")

        print("\nLatency Test:")
        benchmark(model, TEST_IMG_PATH, device)

    # 2. Arbitrary Folder Inference
    if FOLDER_PATH and os.path.exists(FOLDER_PATH):
        folder_paths = [p for p in glob(os.path.join(FOLDER_PATH, "*")) if not p.endswith(".json")]
        print(f"\nRunning batch inference on folder: {FOLDER_PATH} ({len(folder_paths)} images)")
        predict_folder(folder_paths, model, device, batch_size=8, save_json=True,
                       out_dir="test_output_folder", score_thresh=SCORE_THRESH)

    # 3. Validation Set Inference
    if VAL_SPLIT_PATH and os.path.exists(VAL_SPLIT_PATH):
        print(f"\nRunning batch inference on validation split: {VAL_SPLIT_PATH}")
        with open(VAL_SPLIT_PATH) as f:
            val_filenames = json.load(f)["filenames"]
        val_paths = [os.path.join(DATA_DIR, f) for f in val_filenames]
        predict_folder(val_paths, model, device, batch_size=8, save_json=True,
                       out_dir="test_output_val", score_thresh=SCORE_THRESH)