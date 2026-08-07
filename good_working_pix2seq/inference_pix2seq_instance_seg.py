import os
import json
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw

from training_pix2seq_instance_seg import (
    Pix2SeqModel, ar_decode_batch, seq_to_instances,
    VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, MAX_SEQ_LEN,
    TOKENS_PER_OBJ, NUM_POLY_PTS,
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
    w, h    = img.size
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
    """→ (img_tensor[1,3,H,W], padded_img, orig_w, orig_h, max_dim)"""
    original_img        = Image.open(image_path).convert("RGB")
    orig_w, orig_h      = original_img.size
    padded_img, max_dim = pad_to_square_infer(original_img)

    resized    = TF.resize(padded_img, IMG_SIZE)
    img_tensor = _NORMALIZE(resized).unsqueeze(0)
    return img_tensor, padded_img, orig_w, orig_h, max_dim


# ── Token → tahmin ───────────────────────────────────────────────────────────

def tokens_to_predictions(tokens, max_dim, scores=None):
    """
    Tokenları padded-square uzayına ([0, max_dim]) ölçeklenmiş instance'lara çevirir.
    Her tahmin: {"label", "box":[x0,y0,x1,y1], "polygon":[[x,y],...], "score"}
    """
    img_h, img_w = IMG_SIZE
    boxes, polys, labels, confs = seq_to_instances(tokens, scores)

    sx = max_dim / float(img_w)
    sy = max_dim / float(img_h)

    predictions = []
    for box, poly, lab, conf in zip(boxes, polys, labels, confs):
        label = ID_TO_LABEL.get(lab)
        if label is None or label == "noise":
            continue

        scaled_poly = poly.copy()
        scaled_poly[:, 0] *= sx
        scaled_poly[:, 1] *= sy

        predictions.append({
            "label":   label,
            "box":     [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy],
            "polygon": scaled_poly.tolist(),
            "score":   float(conf),
        })
    return predictions


def predictions_to_masks(predictions, width, height):
    """Her tahmin için [H, W] uint8 binary maske (0/1). cv2 varsa onu kullanır."""
    masks = []
    for pred in predictions:
        m = Image.new("L", (width, height), 0)
        ImageDraw.Draw(m).polygon([tuple(p) for p in pred["polygon"]], fill=1)
        masks.append(np.array(m, dtype=np.uint8))
    return masks


# ── Çizim ────────────────────────────────────────────────────────────────────

def draw_predictions(padded_img, predictions, orig_w, orig_h,
                     draw_box=True, mask_alpha=0.35):
    """
    Poligonları padded uzayda çizer (koordinat dönüşümü gerektirmez), sonra
    orijinal boyuta kırpar. Maske yarı saydam doldurulur.
    """
    base    = padded_img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)

    for i, pred in enumerate(predictions):
        color = PALETTE[i % len(PALETTE)]
        pts   = [tuple(p) for p in pred["polygon"]]
        od.polygon(pts, fill=color + (int(255 * mask_alpha),), outline=color + (255,))

    result = Image.alpha_composite(base, overlay).convert("RGB")
    draw   = ImageDraw.Draw(result)

    for i, pred in enumerate(predictions):
        color = PALETTE[i % len(PALETTE)]
        pts   = [tuple(p) for p in pred["polygon"]]
        draw.line(pts + [pts[0]], fill=color, width=3)

        x0, y0, x1, y1 = pred["box"]
        if draw_box:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=1)

        tag = f'{pred["label"]} {pred["score"]:.2f}'
        draw.rectangle([x0, max(0, y0 - 16), x0 + 8 * len(tag) + 8, y0], fill=color)
        draw.text((x0 + 4, max(0, y0 - 15)), tag, fill="black")

    return result.crop((0, 0, orig_w, orig_h))


# ── Model yükleme ────────────────────────────────────────────────────────────

def load_model(model_path, device):
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN).to(device)
    state = torch.load(model_path, map_location=device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


# ── Çıkarım (tek görüntü / batch) ────────────────────────────────────────────

@torch.no_grad()
def predict_batch(model, image_paths, device, use_cache=True):
    """
    Birden fazla görüntüyü TEK forward'da decode eder. KV cache + batching birlikte
    tek görüntü döngüsüne göre çok daha hızlı: encoder bir kez, decoder adım başına
    yalnızca son tokenı işler.
    """
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

    batch = torch.cat(tensors, dim=0).to(device)
    seqs, scores = ar_decode_batch(model, batch, MAX_SEQ_LEN, device, use_cache=use_cache)

    results = []
    for i, meta in enumerate(metas):
        toks = seqs[i].tolist()
        if EOS_TOKEN in toks:
            toks = toks[:toks.index(EOS_TOKEN)]
        toks = [t for t in toks if t not in (BOS_TOKEN, PAD_TOKEN)]

        preds = tokens_to_predictions(toks, meta["max_dim"], scores[i].float().cpu().numpy())
        results.append({**meta, "predictions": preds})
    return results


def predict_and_draw(image_path, model_path=None, model=None, device=None,
                     folder_mode=False, save_json=False, use_cache=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_model(model_path, device)

    out = predict_batch(model, [image_path], device, use_cache=use_cache)
    if not out:
        return None
    r = out[0]

    result_img = draw_predictions(r["padded"], r["predictions"], r["w"], r["h"])

    if folder_mode:
        os.makedirs("test_output", exist_ok=True)
        output_path = f"test_output/{os.path.basename(r['path'])}"
    else:
        output_path = "inference_result.jpg"

    result_img.save(output_path)

    if save_json:
        # LabelMe uyumlu çıktı — mevcut anotasyon araçlarınıza geri beslenebilir
        shapes = [{
            "label": p["label"], "points": p["polygon"],
            "group_id": None, "shape_type": "polygon", "flags": {},
        } for p in r["predictions"]]
        with open(os.path.splitext(output_path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                       "imagePath": os.path.basename(r["path"]),
                       "imageHeight": r["h"], "imageWidth": r["w"]},
                      f, ensure_ascii=False, indent=2)

    print(f"{len(r['predictions'])} instance → '{output_path}'")
    return r["predictions"]


def predict_folder(image_paths, model, device, batch_size=8,
                   save_json=False, use_cache=True):
    from tqdm import tqdm
    os.makedirs("test_output", exist_ok=True)

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inference"):
        chunk = image_paths[i:i + batch_size]
        for r in predict_batch(model, chunk, device, use_cache=use_cache):
            img = draw_predictions(r["padded"], r["predictions"], r["w"], r["h"])
            out_path = f"test_output/{os.path.basename(r['path'])}"
            img.save(out_path)

            if save_json:
                shapes = [{
                    "label": p["label"], "points": p["polygon"],
                    "group_id": None, "shape_type": "polygon", "flags": {},
                } for p in r["predictions"]]
                with open(os.path.splitext(out_path)[0] + ".json", "w", encoding="utf-8") as f:
                    json.dump({"version": "5.0.1", "flags": {}, "shapes": shapes,
                               "imagePath": os.path.basename(r["path"]),
                               "imageHeight": r["h"], "imageWidth": r["w"]},
                              f, ensure_ascii=False, indent=2)


def benchmark(model, image_path, device, runs=5):
    """KV cache açık/kapalı hız karşılaştırması."""
    path = resolve_image_path(image_path)
    t, _, _, _, _ = preprocess_image(path)
    img = t.to(device)

    for use_cache in (False, True):
        if device.type == "cuda":
            torch.cuda.synchronize()
        ar_decode_batch(model, img, MAX_SEQ_LEN, device, use_cache=use_cache)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(runs):
            ar_decode_batch(model, img, MAX_SEQ_LEN, device, use_cache=use_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / runs
        print(f"  KV cache {'ON ' if use_cache else 'OFF'}: {dt*1000:7.1f} ms/img")


if __name__ == "__main__":
    TEST_IMAGE_PATH = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/ruhsat_1ece4a35-2228-4f86-b6de-b80d3c066745.jpg"
    MODEL_PATH      = "pix2seq_instseg_best_map.pth"
    DATA_DIR        = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference {device} cihazında yapılıyor...")

    model = load_model(MODEL_PATH, device)

    predict_and_draw(TEST_IMAGE_PATH, model=model, device=device, save_json=True)

    print("\nHız testi:")
    benchmark(model, TEST_IMAGE_PATH, device)

    FOLDER_PATH = ""
    if FOLDER_PATH:
        from glob import glob
        paths = [p for p in glob(os.path.join(FOLDER_PATH, "*"))
                 if not p.endswith(".json")]
        predict_folder(paths, model, device, batch_size=8, save_json=True)

    VAL_JSON = "val_split.json"
    if VAL_JSON and os.path.exists(VAL_JSON):
        print("VAL_JSON active!")
        with open(VAL_JSON) as f:
            val_filenames = json.load(f)["filenames"]
        paths = [os.path.join(DATA_DIR, f) for f in val_filenames]
        predict_folder(paths, model, device, batch_size=8, save_json=True)
