import os
import json
import time
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw

from dual_decoder_pix2seq import (
    DualPix2SeqModel, dual_ar_decode,
    VOCAB_SIZE, IMG_SIZE, ID_TO_LABEL
)

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PALETTE = [
    (255, 64, 64), (64, 200, 255), (120, 255, 120), (255, 200, 64),
    (200, 120, 255), (64, 255, 220), (255, 128, 200), (180, 180, 255),
]


def pad_to_square_infer(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    new_img.paste(img, (0, 0))
    return new_img, max_dim


def preprocess_image(image_path):
    original_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = original_img.size
    padded_img, max_dim = pad_to_square_infer(original_img)

    resized = TF.resize(padded_img, IMG_SIZE)
    img_tensor = _NORMALIZE(resized).unsqueeze(0)
    return img_tensor, padded_img, orig_w, orig_h, max_dim


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


@torch.no_grad()
def predict_and_draw(image_path, model, device, save_json=False, output_path="dual_infer_result.jpg"):
    img_tensor, padded_img, orig_w, orig_h, max_dim = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    # Autoregressive Box Decode -> Parallel Instance Mask Decode
    raw_results = dual_ar_decode(model, img_tensor, device)[0]

    sx = max_dim / float(IMG_SIZE[1])
    sy = max_dim / float(IMG_SIZE[0])

    scaled_predictions = []
    for r in raw_results:
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

    result_img = draw_predictions(padded_img, scaled_predictions, orig_w, orig_h)
    result_img.save(output_path)

    if save_json:
        shapes = [{
            "label": p["label"],
            "points": p["polygon"],
            "group_id": None,
            "shape_type": "polygon",
            "flags": {},
        } for p in scaled_predictions]
        
        json_path = os.path.splitext(output_path)[0] + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "5.0.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": os.path.basename(image_path),
                "imageHeight": orig_h,
                "imageWidth": orig_w
            }, f, ensure_ascii=False, indent=2)

    print(f"Decoded {len(scaled_predictions)} instances -> '{output_path}'")
    return scaled_predictions


if __name__ == "__main__":
    TEST_IMG   = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/ruhsat_1ece4a35-2228-4f86-b6de-b80d3c066745.jpg"
    MODEL_PATH = "dual_pix2seq_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")

    if os.path.exists(MODEL_PATH) and os.path.exists(TEST_IMG):
        model = load_model(MODEL_PATH, device)
        predict_and_draw(TEST_IMG, model, device, save_json=True)
    else:
        print("Check MODEL_PATH and TEST_IMG paths.")
