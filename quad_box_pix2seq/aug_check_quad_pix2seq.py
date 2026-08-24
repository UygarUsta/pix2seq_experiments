"""
Quad pix2seq — sequence-augmentation görsel kontrolü.

Çizim `records` ara yapısından DEĞİL, dataset'in döndürdüğü token dizisinden
(seq_in) geri çözülerek yapılır. Yani modelin gerçekten gördüğü şey
görselleşir: quantize kaybı, slot sıralaması, noise'un kuyrukta durması,
köşe sırası — hepsi bu yolda doğrulanmış olur.

Renkler:
    palet renkleri + dolu çizgi : gerçek GT quad
    kırmızı nokta / sarı nokta  : 1. ve 2. köşe (kanonik başlangıç + yön kontrolü)
    cyan ince dikdörtgen        : quadın HBB zarfı (referans)
    kırmızı + kesikli çizgi     : sentetik noise quad

Kullanım:
    python aug_check_quad_pix2seq.py                    # 24 örnek -> aug_check/
    python aug_check_quad_pix2seq.py --n 60 --seed 3
    python aug_check_quad_pix2seq.py --val              # noise çıkmamalı (kontrol)
    python aug_check_quad_pix2seq.py --stats-only       # sadece sayısal rapor
"""

import os
import math
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from training_pix2seq import (
    Pix2SeqDataset, decode_sequence, verify_sequence, dataset_object_stats,
    quad_iou_fast, _poly_area,
    JSON_DIR, IMG_DIR, IMG_SIZE, MAX_OBJECTS, LABEL_TO_ID,
    NOISE_FILL_TO_MAX, NOISE_IOU_REJECT,
    train_transform, val_transform,
)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

PALETTE = ["#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231",
           "#48f90a", "#92cc17", "#3ddb86", "#1a9334", "#00c2ff"]
NOISE_COLOR = "#ff2d2d"


def tensor_to_pil(img_t):
    arr = (img_t.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def dashed_polygon(d, pts, fill, width=2, dash=9):
    ring = list(pts) + [pts[0]]
    for (x0, y0), (x1, y1) in zip(ring[:-1], ring[1:]):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg < 1e-6:
            continue
        n = max(int(seg // dash), 1)
        for k in range(0, n, 2):
            t0, t1 = k / n, min((k + 1) / n, 1.0)
            d.line([x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0,
                    x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1], fill=fill, width=width)


def render(img_t, seq_in, max_objects):
    img = tensor_to_pil(img_t)
    d   = ImageDraw.Draw(img)
    W, H = img.size

    instances = decode_sequence(seq_in, W, H, keep_noise=True)
    reals = [np.asarray(p, dtype=np.float64) for p, _, nz in instances if not nz]
    n_gt = n_noise = 0
    ious = []

    # önce noise (altta kalsın), sonra GT
    for poly, _label, is_noise in instances:
        if not is_noise:
            continue
        n_noise += 1
        q = np.asarray(poly, dtype=np.float64)
        best = max((quad_iou_fast(q, r) for r in reals), default=0.0)
        ious.append(best)
        dashed_polygon(d, poly, NOISE_COLOR, width=2)
        # gerçeğe fazla yaklaşan noise'u işaretle (elemeye rağmen sınırda kalanlar)
        tag = "noise" if best < 0.25 else f"noise {best:.2f}"
        d.text((poly[0][0] + 5, poly[0][1] - 12), tag, fill=NOISE_COLOR)

    for poly, label, is_noise in instances:
        if is_noise:
            continue
        n_gt += 1
        color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        d.rectangle([min(xs), min(ys), max(xs), max(ys)], outline="#00e5ff", width=1)
        d.line([c for p in poly for c in p] + list(poly[0]), fill=color, width=3)
        x0, y0 = poly[0]; x1, y1 = poly[1]
        d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill="red")
        d.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill="yellow")
        d.text((x0 + 7, y0 - 13), label, fill=color)

    d.rectangle([0, 0, W, 18], fill="#141414")
    d.text((5, 4), f"GT: {n_gt}", fill="#48f90a")
    d.text((70, 4), f"noise: {n_noise}", fill=NOISE_COLOR)
    slot_txt = f"slot: {n_gt + n_noise}/{max_objects}"
    d.text((165, 4), slot_txt, fill="#bbbbbb" if n_gt + n_noise == max_objects else "#ffb21d")
    return img, n_gt, n_noise, ious


def aspect(q):
    q = np.asarray(q, dtype=np.float64)
    w = q[:, 0].max() - q[:, 0].min()
    h = q[:, 1].max() - q[:, 1].min()
    return w / max(h, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default=JSON_DIR)
    ap.add_argument("--img-dir",  default=IMG_DIR)
    ap.add_argument("--out",      default="aug_check")
    ap.add_argument("--n",        type=int, default=24)
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--val",      action="store_true",
                    help="is_train=False ile çalıştır — hiç noise çıkmamalı")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="görsel kaydetme, sadece sayısal rapor")
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ds = Pix2SeqDataset(args.json_dir, args.img_dir,
                        transform=val_transform if args.val else train_transform,
                        is_train=not args.val)
    ds.json_files = sorted(ds.json_files)
    print(f"Dataset: {len(ds)} örnek | is_train={not args.val} | "
          f"fill_to_max={NOISE_FILL_TO_MAX} | MAX_OBJECTS={MAX_OBJECTS}")

    counts = dataset_object_stats(ds)
    p99 = float(np.percentile(counts, 99))
    if MAX_OBJECTS > 2.5 * max(p99, 1):
        print(f"  [uyarı] MAX_OBJECTS ({MAX_OBJECTS}) gerçek dağılıma göre çok büyük "
              f"(p99={p99:.0f}). Kuyruk noise ile dolduğu için görüntü başına "
              f"~{MAX_OBJECTS - counts.mean():.0f} noise üretilecek ve dizi gereksiz uzun olacak.")
    if MAX_OBJECTS < counts.max():
        print(f"  [uyarı] MAX_OBJECTS ({MAX_OBJECTS}) < verideki max nesne ({counts.max()}). "
              f"EOS kaldırıldığı için bu, sert bir recall tavanı demek.")

    if not args.stats_only:
        os.makedirs(args.out, exist_ok=True)

    n = min(args.n, len(ds))
    idxs = random.sample(range(len(ds)), n)

    tot_gt = tot_noise = 0
    all_ious, noise_ar, real_ar, slot_full = [], [], [], 0

    for i, idx in enumerate(idxs):
        img_t, seq_in, seq_tgt = ds[idx]
        if not args.no_verify:
            verify_sequence(seq_in, seq_tgt)

        img, n_gt, n_noise, ious = render(img_t, seq_in, MAX_OBJECTS)
        all_ious += ious
        slot_full += int(n_gt + n_noise == MAX_OBJECTS)

        for poly, _l, nz in decode_sequence(seq_in, IMG_SIZE[1], IMG_SIZE[0], keep_noise=True):
            (noise_ar if nz else real_ar).append(aspect(poly))

        if not args.stats_only:
            stem = os.path.splitext(ds.json_files[idx])[0][:40]
            img.save(os.path.join(args.out, f"{i:03d}_{stem}_gt{n_gt}_noise{n_noise}.jpg"),
                     quality=92)
        tot_gt += n_gt; tot_noise += n_noise

    print(f"\n{n} örnek incelendi" + ("" if args.stats_only else f" -> {args.out}/"))
    print(f"  gerçek quad : {tot_gt} (ort. {tot_gt / max(n,1):.2f}/görüntü)")
    print(f"  noise quad  : {tot_noise} (ort. {tot_noise / max(n,1):.2f}/görüntü)")
    if NOISE_FILL_TO_MAX and not args.val:
        print(f"  dolu slot   : {slot_full}/{n} görüntüde tam {MAX_OBJECTS}/{MAX_OBJECTS}")

    if all_ious:
        a = np.asarray(all_ious)
        print(f"  noise-gerçek IoU : ort {a.mean():.3f} | p95 {np.percentile(a,95):.3f} | "
              f"max {a.max():.3f} (eleme eşiği {NOISE_IOU_REJECT})")
        if a.max() > NOISE_IOU_REJECT + 1e-6:
            print("  [uyarı] eşiğin üstünde noise var — IoU elemesi çalışmıyor")
        if a.mean() < 0.02:
            print("  [uyarı] noise'lar gerçek nesnelerden çok uzak; zor negatif üretmiyorsun. "
                  "NOISE_JITTER_FRAC'i artır.")
    if noise_ar and real_ar:
        print(f"  en/boy oranı     : noise {np.mean(noise_ar):.2f} vs gerçek {np.mean(real_ar):.2f} "
              "(yakın olmalı; uzaksa model 'garip şekil = noise' kısayolunu öğrenir)")
    if args.val and tot_noise:
        print("  [HATA] val modunda noise üretildi — is_train bayrağı kontrol edilmeli")
    if not args.no_verify:
        print("  dizi kontrolü    : OK (noise kuyrukta, hedefte koordinatlar PAD, hiza doğru)")


if __name__ == "__main__":
    main()
