import os
import json
import time
import numpy as np
import torch
from pix2seq_florence_quad import load_image
from pix2seq_florence_quad_hier import (Pix2SeqHierModel, predict_hier, draw_predictions,
                                        load_checkpoint)
from gpu_preprocess import GpuPreprocessor, predict_hier_gpu

IMG_ROOT = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
IMG_NAME = "0ff4c9b5-d7e3-4b43-9cfc-a06193b8cb97.jpeg"
IMG_PATH = os.path.join(IMG_ROOT, IMG_NAME)
CKPT = "pix2seq_hier_best_map.pth"
if os.path.exists(CKPT.replace(".pth", "_draft.pth")):        # taslak head'li sürüm varsa
    CKPT = CKPT.replace(".pth", "_draft.pth")
ACCEPT_TOL = 2          # 0 = birebir greedy; 2 = benchmark'taki hızlı mod (mAP ~aynı)
USE_GPU_PRE = True      # True: nvJPEG decode + resize GPU'da; False: eski CPU yolu
SAVE_IMAGES = True      # çizim/kaydetme ölçüme dahil değil

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Checkpoint: {CKPT} | accept_tol={ACCEPT_TOL} | ön-işleme: {'GPU' if USE_GPU_PRE else 'CPU'}")
model = Pix2SeqHierModel().to(device)
load_checkpoint(model, CKPT)
model.eval()
pre = GpuPreprocessor(device) if USE_GPU_PRE else None


def run(path, timings=None):
    if USE_GPU_PRE:
        preds, _ = predict_hier_gpu(model, path, pre, accept_tol=ACCEPT_TOL, timings=timings)
        return preds
    _, preds = predict_hier(model, path, device=device, accept_tol=ACCEPT_TOL, timings=timings)
    return preds


# Isınma: CUDA graph yakalama + nvJPEG başlatma -> ölçüme dahil edilmez
preds = run(IMG_PATH)
draw_predictions(load_image(IMG_PATH), preds, "ornek_pred_hier.jpg")

INFERENCE_SAVE_DIR = "output_images_hier"
os.makedirs(INFERENCE_SAVE_DIR, exist_ok=True)
with open("val_split.json", "r") as file:
    VAL_IMAGES = json.load(file)["filenames"]

timings, totals = {}, []
for val in VAL_IMAGES:
    file_path = os.path.join(IMG_ROOT, val)
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]:
        path = file_path.replace(".json", ext)
        if os.path.exists(path):
            t0 = time.perf_counter()
            preds = run(path, timings)
            totals.append(time.perf_counter() - t0)
            if SAVE_IMAGES:
                draw_predictions(load_image(path), preds,
                                 f"{INFERENCE_SAVE_DIR}/{os.path.basename(path)}.jpg")
            break

if totals:
    pre_t, mdl = np.array(timings["pre"]) * 1e3, np.array(timings["model"]) * 1e3
    tot = np.array(totals) * 1e3
    print(f"{len(tot)} görüntü | toplam ort. {tot.mean():.1f} ms ({1000 / tot.mean():.1f} FPS)")
    print(f"  ön-işleme : ort {pre_t.mean():.1f} ms | p95 {np.percentile(pre_t, 95):.1f}")
    print(f"  model     : ort {mdl.mean():.1f} ms | p95 {np.percentile(mdl, 95):.1f}")
    if pre is not None:
        print(f"  decode    : {pre.stats}")