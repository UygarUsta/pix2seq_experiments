import torch
import os
import json
from pix2seq_florence_quad import predict, draw_predictions, Pix2SeqModel, VOCAB_SIZE, MAX_SEQ_LEN

IMG_ROOT = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
IMG_NAME = "0ff4c9b5-d7e3-4b43-9cfc-a06193b8cb97.jpeg"
IMG_PATH = os.path.join(IMG_ROOT,IMG_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN).to(device)
model.load_state_dict(torch.load("pix2seq_florence_best_map.pth", map_location=device))

img, preds = predict(model, IMG_PATH, device=device)
draw_predictions(img, preds, "ornek_pred.jpg")

INFERENCE_SAVE_DIR = "output_images"
os.makedirs(INFERENCE_SAVE_DIR, exist_ok=True)
JSON_DIR = "val_split.json"
with open(JSON_DIR, "r") as file:
    VAL_IMAGES = json.load(file)["filenames"]

for val in VAL_IMAGES:
    file_path = os.path.join(IMG_ROOT, val)
    for exts in [".jpg",".JPG",".png",".PNG",".jpeg",".JPEG"]:
        IMG_PATH = file_path.replace(".json", exts)
        if os.path.exists(IMG_PATH):
            img, preds = predict(model, IMG_PATH, device=device)
            draw_predictions(img, preds, f"{INFERENCE_SAVE_DIR}/{os.path.basename(IMG_PATH)}.jpg")
                

    
    
