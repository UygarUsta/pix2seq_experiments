"""
infer_pix2seq_paper.py
======================
train_pix2seq_paper.py ile eğitilen checkpointler için çıkarım.

Eski inference_pix2seq_bbox_fast.py'yi KULLANMA — o dosya
training_pix2seq_bbox'tan IMG_SIZE=(512,512) alıyor ve model 640'ta eğitildiyse
hem `img_pos_emb` boyutu tutmaz hem de koordinat ölçekleri sessizce kayar.

Koordinat zinciri (tek kritik nokta)
------------------------------------
    orijinal (w,h) -> pad_to_square (max_dim, sol-üst) -> resize IMG_SIZE -> bin

Geri dönüşte IMG_SIZE tamamen sadeleşiyor:
    x_orig = bin / (NUM_BINS-1) * IMG_W * (max_dim / IMG_W)
           = bin / (NUM_BINS-1) * max_dim
Yani ölçek doğrudan max_dim. Padding sağda/altta olduğu için sonuçta (w,h)'ye
kırpmak yeterli.

Kullanım
--------
    # val split üzerinde, kutuları çiz
    python infer_pix2seq_paper.py --model p2s_paper_best_map.pth --val-split

    # klasör, sadece JSON çıktı
    python infer_pix2seq_paper.py --model ... --images /path/to/imgs \\
        --no-draw --save-json preds.json

    # tek görüntü
    python infer_pix2seq_paper.py --model ... --image foo.jpg --score-thresh 0.5
"""

import os
import json
import glob
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.ops import batched_nms
from PIL import Image, ImageDraw
from tqdm import tqdm

from train_pix2seq_paper import (
    Pix2SeqModel, ar_decode_batch,
    IMG_SIZE, IMG_H, IMG_W, MAX_OBJECTS, MAX_SEQ_LEN,
    NUM_BINS, ID_TO_LABEL,
)

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
#  Girdi hazırlama — eğitimdekiyle BİREBİR aynı olmalı
# --------------------------------------------------------------------------- #
def pad_to_square_infer(img, fill=(128, 128, 128)):
    w, h = img.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), fill)
    canvas.paste(img, (0, 0))          # eğitimdeki gibi sol-üst
    return canvas, m


def resolve_image_path(p):
    if os.path.exists(p) and not p.endswith(".json"):
        return p
    base = os.path.splitext(p)[0]
    for ext in IMG_EXTS + tuple(e.upper() for e in IMG_EXTS):
        if os.path.exists(base + ext):
            return base + ext
    return p


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
            return torch.zeros(3, IMG_H, IMG_W), "", 0, 0, 1
        w, h = img.size
        padded, max_dim = pad_to_square_infer(img)
        t = self.norm(TF.to_tensor(TF.resize(padded, list(IMG_SIZE))))
        return t, path, w, h, max_dim


def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            [(b[1], b[2], b[3], b[4]) for b in batch])


# --------------------------------------------------------------------------- #
#  Model yükleme (çözünürlük değişmişse pos-emb interpolasyonu ile)
# --------------------------------------------------------------------------- #
def interpolate_pos_emb(pe, new_tokens):
    """[1, N, D] -> [1, new_tokens, D]. Kare grid varsayar."""
    n_old, dim = pe.shape[1], pe.shape[2]
    g_old, g_new = int(round(n_old ** 0.5)), int(round(new_tokens ** 0.5))
    if g_old == g_new:
        return pe
    x = pe.reshape(1, g_old, g_old, dim).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(g_new, g_new), mode="bicubic", align_corners=False)
    print(f"  img_pos_emb interpolasyonu: {g_old}x{g_old} -> {g_new}x{g_new}")
    return x.permute(0, 2, 3, 1).reshape(1, g_new * g_new, dim)


def load_model(path, device, dilated_c5=False):
    model = Pix2SeqModel(max_seq_len=MAX_SEQ_LEN, img_size=IMG_SIZE,
                         drop_path=0.0, dilated_c5=dilated_c5).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}

    want = model.img_pos_emb.shape[1]
    if "img_pos_emb" in sd and sd["img_pos_emb"].shape[1] != want:
        sd["img_pos_emb"] = interpolate_pos_emb(sd["img_pos_emb"], want)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [uyarı] eksik anahtar: {missing}")
    if unexpected:
        print(f"  [uyarı] fazla anahtar: {unexpected}")
    model.eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# --------------------------------------------------------------------------- #
#  Slot -> tahmin
# --------------------------------------------------------------------------- #
def slots_to_preds(coord, cls_id, noise_won, conf, max_dim, orig_w, orig_h,
                   score_thresh, nms_iou, keep_noise_slots):
    """
    Tek görüntünün 100 slotunu tahmin listesine çevirir.
    conf = p(sınıf) / (p(sınıf) + p(noise)) — kalibre objectness.
    """
    boxes = coord.float() * (max_dim / (NUM_BINS - 1))     # -> orijinal piksel
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not keep_noise_slots:
        keep &= ~noise_won
    keep &= conf >= score_thresh

    b, s, l = boxes[keep], conf[keep], cls_id[keep]

    if nms_iou is not None and b.numel():
        idx = batched_nms(b, s, l, nms_iou)
        b, s, l = b[idx], s[idx], l[idx]

    # padding sağda/altta -> orijinal kadraja kırp
    b[:, 0::2] = b[:, 0::2].clamp(0, orig_w)
    b[:, 1::2] = b[:, 1::2].clamp(0, orig_h)
    ok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b, s, l = b[ok], s[ok], l[ok]

    order = s.argsort(descending=True)
    return [{"label": ID_TO_LABEL[int(l[i])],
             "class_id": int(l[i]),
             "box": [round(float(v), 2) for v in b[i]],
             "score": round(float(s[i]), 4)}
            for i in order]


_PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
            "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9a6324",
            "#800000", "#aaffc3", "#808000", "#000075"]


def draw_predictions(img, preds):
    out = img.copy()
    d = ImageDraw.Draw(out)
    for p in preds:
        x0, y0, x1, y1 = p["box"]
        color = _PALETTE[p["class_id"] % len(_PALETTE)]
        tag = f'{p["label"]} {p["score"]:.2f}'
        d.rectangle([x0, y0, x1, y1], outline=color, width=3)
        tw = 7 * len(tag) + 6
        d.rectangle([x0, max(0, y0 - 14), x0 + tw, y0], fill=color)
        d.text((x0 + 3, max(0, y0 - 14)), tag, fill="white")
    return out


# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(paths, model, device, batch_size=16, num_workers=8,
            score_thresh=0.5, nms_iou=0.6, keep_noise_slots=False,
            draw=True, out_dir="infer_out", amp_dtype=torch.float16):
    loader = DataLoader(InferenceDataset(paths), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True,
                        collate_fn=_collate,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
    if draw:
        os.makedirs(out_dir, exist_ok=True)

    results = {}
    n_pred = 0
    for tensors, meta in tqdm(loader, desc="inference"):
        tensors = tensors.to(device, non_blocking=True)
        if device.type == "cuda":
            tensors = tensors.contiguous(memory_format=torch.channels_last)

        use_amp = device.type == "cuda" and amp_dtype is not None
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.float32, enabled=use_amp):
            coord, cls_id, noise_won, conf = ar_decode_batch(model, tensors)

        for b, (path, w, h, max_dim) in enumerate(meta):
            if not path:
                continue
            preds = slots_to_preds(coord[b], cls_id[b], noise_won[b], conf[b],
                                   max_dim, w, h, score_thresh, nms_iou,
                                   keep_noise_slots)
            results[path] = preds
            n_pred += len(preds)
            if draw:
                img = Image.open(path).convert("RGB")
                draw_predictions(img, preds).save(
                    os.path.join(out_dir, os.path.basename(path)), quality=92)

    print(f"{len(results)} görüntü | toplam {n_pred} kutu "
          f"| ort {n_pred/max(len(results),1):.2f}/görüntü")
    return results


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="p2s_paper_best_map.pth")
    ap.add_argument("--image", help="tek görüntü")
    ap.add_argument("--images", help="klasör")
    ap.add_argument("--val-split", action="store_true",
                    help="val_split.json'daki dosyalar üzerinde çalış")
    ap.add_argument("--split-path", default="val_split.json")
    ap.add_argument("--data-dir",
                    default="/home/uygarusta/Oriented-Centernet/"
                            "centernet_ciou_iou_aware_pl/coco_dataset/train/")
    ap.add_argument("--out-dir", default="infer_out")
    ap.add_argument("--save-json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.5,
                    help="görselleştirme için 0.5; mAP/analiz için 0.0")
    ap.add_argument("--nms-iou", type=float, default=0.6,
                    help="negatif verirsen NMS kapanır")
    ap.add_argument("--keep-noise-slots", action="store_true",
                    help="argmax'i noise olan slotları da tahmin say")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dilated-c5", action="store_true",
                    help="checkpoint stride-16 ile eğitildiyse")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.image:
        paths = [args.image]
    elif args.images:
        paths = sorted(p for p in glob.glob(os.path.join(args.images, "*"))
                       if p.lower().endswith(IMG_EXTS))
    elif args.val_split:
        with open(args.split_path) as f:
            names = json.load(f)["filenames"]
        paths = [os.path.join(args.data_dir, n) for n in names]
    else:
        ap.error("--image, --images veya --val-split ver")

    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        ap.error("görüntü bulunamadı")

    print(f"model: {args.model} | img: {IMG_SIZE} | slot: {MAX_OBJECTS} "
          f"| {len(paths)} görüntü")
    model = load_model(args.model, device, dilated_c5=args.dilated_c5)

    results = predict(
        paths, model, device,
        batch_size=args.batch_size, num_workers=args.num_workers,
        score_thresh=args.score_thresh,
        nms_iou=None if args.nms_iou < 0 else args.nms_iou,
        keep_noise_slots=args.keep_noise_slots,
        draw=not args.no_draw, out_dir=args.out_dir,
        amp_dtype=None if args.fp32 else torch.float16,
    )

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"JSON -> {args.save_json}")
    if not args.no_draw:
        print(f"görseller -> {args.out_dir}/")


if __name__ == "__main__":
    main()
