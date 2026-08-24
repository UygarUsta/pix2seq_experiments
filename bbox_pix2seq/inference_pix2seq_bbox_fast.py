"""
inference_pix2seq_bbox_fast.py
==============================
Orijinal inference_pix2seq_bbox.py'nin hizlandirilmis surumu.

Orijinaldeki EN BUYUK sorun:
    run_autoregressive() her adimda  model(img_tensor, tgt_seq)  cagiriyordu.
    Bu, ENCODER'i her token icin bastan calistirmak demek.
    max_seq_len = 1 + 5*150 + 1 = 752  ->  goruntu basina 752 encoder forward'i.
    Burada encoder 1 kez calisiyor.

Diger iyilestirmeler:
  - KV-cache'li decode (Pix2SeqKVDecoder)
  - Batch'li cikarim (goruntu goruntu degil)
  - DataLoader + num_workers ile paralel goruntu okuma/resize
  - fp16 autocast + channels_last + TF32
  - Cizim/kaydetme opsiyonel (draw=False ise sadece tahmin doner)
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm

from training_pix2seq_bbox import (
    Pix2SeqModel, VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, TOKENS_PER_OBJ,
    NOISE_CLASS_TOKEN, NUM_CLASSES, NUM_CLASS_SLOTS, NOISE_FILL_TO_MAX,
    INFER_SCORE_THRESH,
)
# NOT: pix2seq_fast_decode.Pix2SeqKVDecoder artık kullanılmıyor. O decoder EOS'ta
# duruyor ve kısıtsız argmax yapıyordu; sequence augmentation ile eğitilen modelde
# EOS hiç üretilmediği için erken durma mantığı anlamsız, kısıtsız argmax ise 5'li
# hizalamayı bozabiliyor. Aynı KV-cache hızı bbox_eval_utils.ar_decode_batch
# içinde, kısıtlı ve sabit uzunlukta çözen sürümde mevcut.
from bbox_eval_utils_pix2seq import ar_decode_batch, seq_to_boxes

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------- #
#  Yardimcilar
# --------------------------------------------------------------------------- #
def dequantize(token, max_size, num_bins):
    return (token / (num_bins - 1)) * max_size


def resolve_image_path(image_path):
    if not os.path.exists(image_path) or image_path.endswith(".json"):
        base = os.path.splitext(image_path)[0]
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff",
                    ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(base + ext):
                return base + ext
    return image_path


def pad_to_square_infer(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    new_img.paste(img, (0, 0))
    return new_img, max_dim


def tokens_to_predictions(tokens, max_dim, tok_scores=None, noise_scores=None,
                          score_thresh=INFER_SCORE_THRESH):
    """
    Koordinatlar padded-square uzayinda. Noise slotlari atlanir; skor
    p(sinif)/(p(sinif)+p(noise)) olarak hesaplanir (kalibre objectness).
    """
    img_h_model, img_w_model = IMG_SIZE
    predictions = []
    for i in range(0, len(tokens) - (len(tokens) % 5), 5):
        t = tokens[i:i + 5]
        if t[4] == NOISE_CLASS_TOKEN:          # model "burada nesne yok" dedi
            continue
        if not all(0 <= t[j] < NUM_BINS for j in range(4)):
            continue
        if not (NUM_BINS <= t[4] < NUM_BINS + NUM_CLASSES):
            continue
        label = ID_TO_LABEL.get(t[4] - NUM_BINS)
        if label is None:
            continue

        if tok_scores is None:
            score = 1.0
        elif noise_scores is not None:
            p_cls = float(tok_scores[i + 4]); p_noi = float(noise_scores[i + 4])
            score = p_cls / max(p_cls + p_noi, 1e-6)
        else:
            score = float(np.prod(tok_scores[i:i + 5]))
        if score < score_thresh:
            continue

        sx, sy = max_dim / img_w_model, max_dim / img_h_model
        x_min = dequantize(t[0], img_w_model, NUM_BINS) * sx
        y_min = dequantize(t[1], img_h_model, NUM_BINS) * sy
        x_max = dequantize(t[2], img_w_model, NUM_BINS) * sx
        y_max = dequantize(t[3], img_h_model, NUM_BINS) * sy
        if x_max <= x_min or y_max <= y_min:
            continue
        predictions.append({"label": label, "box": [x_min, y_min, x_max, y_max],
                            "score": score})
    return predictions


def draw_predictions(padded_img, predictions, orig_w, orig_h):
    result = padded_img.copy()
    draw = ImageDraw.Draw(result)
    for pred in predictions:
        x_min, y_min, x_max, y_max = pred["box"]
        tag = f'{pred["label"]} {pred.get("score", 1.0):.2f}'
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.rectangle([x_min, max(0, y_min - 15), x_min + 8 * len(tag) + 6, y_min], fill="red")
        draw.text((x_min + 2, max(0, y_min - 15)), tag, fill="white")
    return result.crop((0, 0, orig_w, orig_h))


# --------------------------------------------------------------------------- #
#  Dataset: goruntu okuma/pad/resize/normalize islerini worker'lara dagit
# --------------------------------------------------------------------------- #
class InferenceDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.norm = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = resolve_image_path(self.paths[idx])
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return (torch.zeros(3, *IMG_SIZE), "", 0, 0, 1)

        orig_w, orig_h = img.size
        padded, max_dim = pad_to_square_infer(img)
        small = TF.resize(padded, IMG_SIZE)
        tensor = self.norm(transforms.functional.to_tensor(small))
        return tensor, path, orig_w, orig_h, max_dim


def _collate(batch):
    tensors = torch.stack([b[0] for b in batch])
    meta = [(b[1], b[2], b[3], b[4]) for b in batch]
    return tensors, meta


# --------------------------------------------------------------------------- #
#  Model yukleme
# --------------------------------------------------------------------------- #
def load_model(model_path, device, max_seq_len):
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
    sd = torch.load(model_path, map_location=device)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# --------------------------------------------------------------------------- #
#  Toplu cikarim
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_folder(paths, model_path=None, model=None, device=None,
                   batch_size=16, num_workers=8, draw=True,
                   out_dir="test_output", max_objects_infer=None,
                   amp_dtype=torch.float16, score_thresh=INFER_SCORE_THRESH):
    """
    paths            : goruntu yollari listesi
    max_objects_infer: ARTIK KULLANILMIYOR. Sequence augmentation ile dizi
                       uzunlugu zaten sabit (MAX_OBJECTS slot) ve EOS yok;
                       kisaltmak dogrudan recall kaybi demek. Dizi uzunlugunu
                       dusurmek istiyorsan MAX_OBJECTS'i dusurup yeniden egit.
    score_thresh     : p(sinif)/(p(sinif)+p(noise)) esigi. mAP icin dusuk tut
                       (0.05), gorsellestirme icin 0.5 civari.
    draw=False       : sadece tahminleri dondurur, disk yazmaz (cok daha hizli)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 1 + (TOKENS_PER_OBJ * MAX_OBJECTS)
    if model is None:
        model = load_model(model_path, device, max_seq_len)

    loader = DataLoader(
        InferenceDataset(paths), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=_collate,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    if draw:
        os.makedirs(out_dir, exist_ok=True)

    results = {}
    for tensors, meta in tqdm(loader, desc="inference"):
        tensors = tensors.to(device, non_blocking=True)
        if device.type == "cuda":
            tensors = tensors.contiguous(memory_format=torch.channels_last)

        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda" and amp_dtype is not None):
            seqs, scs, nps = ar_decode_batch(model, tensors, max_seq_len, device)
        seqs_l = seqs.cpu().tolist()
        scs_l  = scs.float().cpu().numpy()
        nps_l  = nps.float().cpu().numpy()

        for b, (path, ow, oh, max_dim) in enumerate(meta):
            if not path:
                continue
            toks = seqs_l[b]
            # sabit uzunlukta cozumde EOS/PAD uretilmez; eski checkpointler icin dursun
            if EOS_TOKEN in toks:
                toks = toks[:toks.index(EOS_TOKEN)]
            preds = tokens_to_predictions(toks, max_dim, scs_l[b], nps_l[b],
                                          score_thresh=score_thresh)
            results[path] = preds

            if draw:
                img = Image.open(path).convert("RGB")
                padded, _ = pad_to_square_infer(img)
                out = draw_predictions(padded, preds, ow, oh)
                out.save(os.path.join(out_dir, os.path.basename(path)))

    return results


@torch.no_grad()
def predict_single(image_path, model_path=None, model=None, device=None,
                   out_path="inference_result.jpg", draw=True,
                   max_objects_infer=None, amp_dtype=torch.float16,
                   score_thresh=INFER_SCORE_THRESH):
    """Tek goruntu icin. Encoder 1 kez calisir (orijinalde 752 kez calisiyordu)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 1 + (TOKENS_PER_OBJ * MAX_OBJECTS)
    if model is None:
        model = load_model(model_path, device, max_seq_len)

    path = resolve_image_path(image_path)
    img = Image.open(path).convert("RGB")
    ow, oh = img.size
    padded, max_dim = pad_to_square_infer(img)
    norm = transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    t = norm(transforms.functional.to_tensor(TF.resize(padded, IMG_SIZE)))
    t = t.unsqueeze(0).to(device)

    seqs, scs, nps = ar_decode_batch(model, t, max_seq_len, device)
    toks = seqs[0].cpu().tolist()
    if EOS_TOKEN in toks:
        toks = toks[:toks.index(EOS_TOKEN)]

    preds = tokens_to_predictions(toks, max_dim,
                                  scs[0].float().cpu().numpy(),
                                  nps[0].float().cpu().numpy(),
                                  score_thresh=score_thresh)
    if draw:
        draw_predictions(padded, preds, ow, oh).save(out_path)
        print(f"Kaydedildi: {out_path}")
    return preds


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    MODEL_PATH = "pix2seq_best_map.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_len = 1 + (TOKENS_PER_OBJ * MAX_OBJECTS)
    model = load_model(MODEL_PATH, device, max_seq_len)

    # --- (istege bagli) hizli/dogru mu kontrolu -----------------------------
    # from pix2seq_fast_decode import verify_against_naive
    # dummy = torch.randn(2, 3, *IMG_SIZE, device=device)
    # verify_against_naive(model, dummy, max_seq_len, device,
    #                      BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, amp_dtype=None)

    VAL_JSON = "val_split.json"
    DATA_DIR = "/mnt/d/Datasets/license-plate.v2i.coco-segmentation/train/"
    if VAL_JSON and os.path.exists(VAL_JSON):
        with open(VAL_JSON) as f:
            names = json.load(f)["filenames"]
        paths = [os.path.join(DATA_DIR, n) for n in names]
        predict_folder(paths, model=model, device=device,
                       batch_size=16, num_workers=8, draw=True,
                       score_thresh=0.5)   # gorsellestirme esigi
