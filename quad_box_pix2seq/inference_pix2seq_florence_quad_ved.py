import os
import json
import torch
from pix2seq_florence_quad_ved import predict, draw_predictions, SAVE_BEST_MAP
from transformers import VisionEncoderDecoderModel

IMG_ROOT = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
IMG_NAME = "0ff4c9b5-d7e3-4b43-9cfc-a06193b8cb97.jpeg"
IMG_PATH = os.path.join(IMG_ROOT, IMG_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionEncoderDecoderModel.from_pretrained(SAVE_BEST_MAP).to(device).eval()

img, preds = predict(model, IMG_PATH, device=device)
draw_predictions(img, preds, "ornek_pred_ved.jpg")

INFERENCE_SAVE_DIR = "output_images_ved"
os.makedirs(INFERENCE_SAVE_DIR, exist_ok=True)
with open("val_split.json", "r") as file:
    VAL_IMAGES = json.load(file)["filenames"]

for val in VAL_IMAGES:
    file_path = os.path.join(IMG_ROOT, val)
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
        img_path = file_path.replace(".json", ext)
        if os.path.exists(img_path):
            img, preds = predict(model, img_path, device=device)
            draw_predictions(img, preds, f"{INFERENCE_SAVE_DIR}/{os.path.basename(img_path)}.jpg")
            break
