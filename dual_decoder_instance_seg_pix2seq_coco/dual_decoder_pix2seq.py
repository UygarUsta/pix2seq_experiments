"""
dual_decoder_pix2seq.py
=======================
Dual-decoder Pix2Seq instance segmentation, COCO sürümü.

  Decoder 1  ->  [x0, y0, x1, y1, cls] x MAX_OBJECTS      (kutu + sınıf)
  Decoder 2  ->  [BOS, kutu, cls, poligon..., EOS]        (maske = poligon)

train_pix2seq_paper.py'deki reçeteye hizalandı:
  backbone         ResNet18            ->  ResNet50 (ImageNet ön-eğitimli, ayrı düşük lr)
  transformer enc  yok                 ->  6 katman (görüntü token'ları üzerinde self-attn)
  decoder          4 katman            ->  6 katman (her iki decoder da)
  çözünürlük       512                 ->  640
  MAX_OBJECTS      10                  ->  100 (paper)
  augmentasyon     ShiftScaleRotate    ->  LSJ (RandomScale 0.1-2.0 + pad + crop) + HFlip
  nesne sırası     %50 permütasyon     ->  her gösterimde rastgele (PERMUTE_PROB=1.0)
  weight decay     1e-4                ->  0.05 (norm/bias/embedding hariç)
  stochastic depth yok                 ->  0.1 (encoder + iki decoder)
  label smoothing  0.1                 ->  0.0 (paper düz CE)
  lr schedule      OneCycle            ->  warmup + lineer düşüş
  efektif batch    16                  ->  16 x 4 = 64
  grad clip        1.0                 ->  0.1
  decode           naif AR             ->  KV-cache'li AR (500 adım artık makul)

Veri: standart COCO instance segmentation.
    COCO_ROOT/
        train2017/ *.jpg
        val2017/   *.jpg
        annotations/instances_train2017.json
        annotations/instances_val2017.json

İlk çalıştırmada annotation json'ı taranıp kompakt bir npz index'e yazılır
(cocoidx_*.npz). Sonraki koşular onu okur; ayrıca index düz numpy dizileri
olduğu için DataLoader worker'ları fork ettiğinde kopyalanmaz.

Kullanım:
    python dual_decoder_pix2seq.py
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import json
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
COCO_ROOT     = "/mnt/d/Datasets/coco"
TRAIN_IMG_DIR = os.path.join(COCO_ROOT, "train2017")
VAL_IMG_DIR   = os.path.join(COCO_ROOT, "val2017")
TRAIN_ANN     = os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
VAL_ANN       = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
CACHE_DIR     = "."                    # cocoidx_*.npz buraya yazılır
CKPT_PREFIX   = "dual_p2s_coco"

IMG_SIZE      = (640, 640)             # paper 1333; bütçeye göre 640
NUM_BINS      = 500                    # 640/500 -> ~1.28 piksel/bin
MAX_OBJECTS   = 100                    # paper 100 -> kutu dizisi 500 token
NUM_POLY_PTS  = 16                     # poligon köşe sayısı (maske çözünürlüğü tavanı)

BATCH_SIZE    = 16
GRAD_ACCUM    = 4                      # efektif batch 64 (paper 256)
EPOCHS        = 50                     # COCO 118k -> epoch başına ~7.4k iterasyon
BASE_LR       = 5e-4
BACKBONE_LR   = 5e-5
WEIGHT_DECAY  = 0.05
WARMUP_FRAC   = 0.03                   # paper: 300 epoch'ta 10 -> %3
GRAD_CLIP     = 0.1
DROP_PATH     = 0.1
DROPOUT       = 0.1
LABEL_SMOOTH  = 0.0
MASK_LOSS_W   = 2.0                    # toplam loss = box + w * mask
PERMUTE_PROB  = 1.0
SCALE_RANGE   = (0.1, 2.0)             # LSJ
USE_BF16      = True
NUM_WORKERS   = 8

HIDDEN_DIM    = 256
NHEADS        = 8
ENC_LAYERS    = 6
DEC_LAYERS    = 6                      # her iki decoder için

EVAL_EVERY      = 2
EVAL_MAX_IMAGES = 500                  # tam val2017 (5k) maske mAP'i RAM yer; bkz. eval utils
INFER_SCORE_THRESH = 0.05              # düşük tut: mAP sıralamayı sever
MAX_DETS        = 100                  # COCO standardı
DEC2_CHUNK      = 128                  # decoder 2'nin aynı anda işlediği instance sayısı

# Bellek notu: 640 + ResNet50 + 6L encoder + 2x6L decoder, batch 16 bf16 ile
# yaklaşık 18-22 GB. OOM alırsan BATCH_SIZE=8 / GRAD_ACCUM=8 yap; efektif batch
# aynı kalır, sadece adım süresi uzar.

# ─────────────────────────────────────────────────────────────────────────────
# COCO SINIFLARI (80 sınıf, kategori id'leri süreksiz: 1..90)
# ─────────────────────────────────────────────────────────────────────────────
COCO_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (5, "airplane"),
    (6, "bus"), (7, "train"), (8, "truck"), (9, "boat"), (10, "traffic light"),
    (11, "fire hydrant"), (13, "stop sign"), (14, "parking meter"), (15, "bench"),
    (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"), (20, "sheep"),
    (21, "cow"), (22, "elephant"), (23, "bear"), (24, "zebra"), (25, "giraffe"),
    (27, "backpack"), (28, "umbrella"), (31, "handbag"), (32, "tie"),
    (33, "suitcase"), (34, "frisbee"), (35, "skis"), (36, "snowboard"),
    (37, "sports ball"), (38, "kite"), (39, "baseball bat"), (40, "baseball glove"),
    (41, "skateboard"), (42, "surfboard"), (43, "tennis racket"), (44, "bottle"),
    (46, "wine glass"), (47, "cup"), (48, "fork"), (49, "knife"), (50, "spoon"),
    (51, "bowl"), (52, "banana"), (53, "apple"), (54, "sandwich"), (55, "orange"),
    (56, "broccoli"), (57, "carrot"), (58, "hot dog"), (59, "pizza"), (60, "donut"),
    (61, "cake"), (62, "chair"), (63, "couch"), (64, "potted plant"), (65, "bed"),
    (67, "dining table"), (70, "toilet"), (72, "tv"), (73, "laptop"), (74, "mouse"),
    (75, "remote"), (76, "keyboard"), (77, "cell phone"), (78, "microwave"),
    (79, "oven"), (80, "toaster"), (81, "sink"), (82, "refrigerator"), (84, "book"),
    (85, "clock"), (86, "vase"), (87, "scissors"), (88, "teddy bear"),
    (89, "hair drier"), (90, "toothbrush"),
]
CATID_TO_ID = {cid: i for i, (cid, _) in enumerate(COCO_CATEGORIES)}
ID_TO_LABEL = {i: name for i, (_, name) in enumerate(COCO_CATEGORIES)}
LABEL_TO_ID = {name: i for i, name in ID_TO_LABEL.items()}
NUM_CLASSES = len(COCO_CATEGORIES)                # 80

# ── Token uzayı ──────────────────────────────────────────────────────────────
NOISE_CLASS_ID    = NUM_CLASSES                   # 80
NOISE_CLASS_TOKEN = NUM_BINS + NOISE_CLASS_ID     # 580
NUM_CLASS_SLOTS   = NUM_CLASSES + 1               # 81

VOCAB_SIZE = NUM_BINS + NUM_CLASS_SLOTS + 3       # 584
BOS_TOKEN  = VOCAB_SIZE - 3                       # 581
EOS_TOKEN  = VOCAB_SIZE - 2                       # 582 (sadece decoder 2)
PAD_TOKEN  = VOCAB_SIZE - 1                       # 583

BOX_TOKENS_PER_OBJ  = 5
POLY_TOKENS_PER_OBJ = 2 * NUM_POLY_PTS            # 32
MAX_BOX_SEQ_LEN     = 1 + BOX_TOKENS_PER_OBJ * MAX_OBJECTS               # 501
MAX_MASK_SEQ_LEN    = 1 + BOX_TOKENS_PER_OBJ + POLY_TOKENS_PER_OBJ + 1   # 39

# Kutu dizisinde EOS yok: kuyruk noise nesneleriyle doldurulur, dolayısıyla
# decode her zaman tam MAX_OBJECTS * 5 adım sürer ve chunk'lar asla kaymaz.
NOISE_FILL_TO_MAX = True
NOISE_IOU_REJECT  = 0.5

MIN_SIDE      = 2.0
MIN_AREA      = 16.0
MIN_AREA_KEEP = 0.2      # paper MIN_VISIBLE_FRAC: LSJ crop'undan sonra görünürlük eşiği
MIN_ANN_AREA  = 16.0     # index kurulurken atılan minik annotation eşiği

IMG_H, IMG_W = IMG_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRİ & TOKEN YARDIMCILARI
# ─────────────────────────────────────────────────────────────────────────────
def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def clip_polygon(poly, w, h):
    """Sutherland-Hodgman: poligonu [0,w]x[0,h] kadrajına kırpar."""
    def _clip_edge(pts, inside_fn, isect_fn):
        if not pts:
            return []
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            c_in, p_in = inside_fn(cur), inside_fn(prev)
            if c_in:
                if not p_in:
                    out.append(isect_fn(prev, cur))
                out.append(cur)
            elif p_in:
                out.append(isect_fn(prev, cur))
        return out

    def _isect(p, q, axis, val):
        d = q[axis] - p[axis]
        t = 0.0 if abs(d) < 1e-9 else (val - p[axis]) / d
        return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))

    pts = [(float(p[0]), float(p[1])) for p in poly]
    edges = [
        (lambda p: p[0] >= 0.0, lambda p, q: _isect(p, q, 0, 0.0)),
        (lambda p: p[0] <= w,   lambda p, q: _isect(p, q, 0, w)),
        (lambda p: p[1] >= 0.0, lambda p, q: _isect(p, q, 1, 0.0)),
        (lambda p: p[1] <= h,   lambda p, q: _isect(p, q, 1, h)),
    ]
    for inside_fn, isect_fn in edges:
        pts = _clip_edge(pts, inside_fn, isect_fn)
        if len(pts) < 3:
            return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def canonicalize_polygon(poly):
    """Sarım yönünü ve başlangıç köşesini sabitler - HFlip sonrası şart."""
    poly = np.asarray(poly, dtype=np.float32)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    start = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])
    return np.roll(poly, -start, axis=0)


def resample_polygon(poly, K):
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) < 3:
        return None
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-6:
        return None
    out = []
    for t in np.linspace(0.0, total, K, endpoint=False):
        i = min(int(np.searchsorted(cum, t, side="right")) - 1, len(seg) - 1)
        i = max(i, 0)
        r = (t - cum[i]) / max(float(seg[i]), 1e-6)
        out.append(closed[i] + r * (closed[i + 1] - closed[i]))
    return np.stack(out).astype(np.float32)


def quantize(coordinate, max_size, num_bins):
    v = coordinate / max_size
    v = max(0.0, min(1.0, v))
    return int(v * (num_bins - 1))


def dequantize(token, max_size, num_bins):
    return (token / (num_bins - 1)) * max_size


def pad_to_square(image, fill_color=(128, 128, 128)):
    """Sol-üste yapıştırır: annotation koordinatları değişmez."""
    w, h = image.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), fill_color)
    canvas.paste(image, (0, 0))
    return canvas, m


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASYON — Large Scale Jittering (paper Appendix B)
# ─────────────────────────────────────────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=(SCALE_RANGE[0] - 1.0, SCALE_RANGE[1] - 1.0), p=1.0),
    A.PadIfNeeded(min_height=IMG_H, min_width=IMG_W, border_mode=0, fill=(128, 128, 128)),
    A.RandomCrop(height=IMG_H, width=IMG_W, p=1.0),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

val_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


# ─────────────────────────────────────────────────────────────────────────────
# NOISE KUTULARI (vektörize) — paper'ın sequence augmentation'ı
# ─────────────────────────────────────────────────────────────────────────────
def _iou_matrix(a, b):
    """a:[N,4], b:[M,4] -> [N,M]"""
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix1 - ix0, 0.0, None)
    ih = np.clip(iy1 - iy0, 0.0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-6)


def box_iou(a, b):
    return float(_iou_matrix(np.asarray([a], np.float32),
                             np.asarray([b], np.float32))[0, 0])


def _sample_candidates(real, img_w, img_h, n):
    """%60 gerçek kutudan jitter (zor negatif), %40 tamamen rastgele arka plan."""
    boxes = np.empty((n, 4), dtype=np.float32)
    from_real = np.random.rand(n) < 0.6 if len(real) else np.zeros(n, dtype=bool)

    k = int(from_real.sum())
    if k:
        idx = np.random.randint(0, len(real), size=k)
        r = real[idx]
        bw = np.maximum(r[:, 2] - r[:, 0], MIN_SIDE)
        bh = np.maximum(r[:, 3] - r[:, 1], MIN_SIDE)
        cx = (r[:, 0] + r[:, 2]) * 0.5 + np.random.uniform(-0.7, 0.7, k) * bw
        cy = (r[:, 1] + r[:, 3]) * 0.5 + np.random.uniform(-0.7, 0.7, k) * bh
        sw = bw * np.random.uniform(0.5, 1.6, k)
        sh = bh * np.random.uniform(0.5, 1.6, k)
        boxes[from_real] = np.stack([cx - sw / 2, cy - sh / 2,
                                     cx + sw / 2, cy + sh / 2], 1)

    m = n - k
    if m:
        w = np.random.uniform(20.0, img_w * 0.5, m)
        h = np.random.uniform(20.0, img_h * 0.5, m)
        x0 = np.random.uniform(0.0, np.maximum(img_w - w, 1.0))
        y0 = np.random.uniform(0.0, np.maximum(img_h - h, 1.0))
        boxes[~from_real] = np.stack([x0, y0, x0 + w, y0 + h], 1)

    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, img_w - MIN_SIDE)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, img_h - MIN_SIDE)
    boxes[:, 2] = np.clip(boxes[:, 2], boxes[:, 0] + MIN_SIDE, img_w)
    boxes[:, 3] = np.clip(boxes[:, 3], boxes[:, 1] + MIN_SIDE, img_h)
    return boxes


def generate_synthetic_boxes(real_boxes, img_w, img_h, count, max_iou=NOISE_IOU_REJECT):
    """
    Gerçek bir nesnenin üstüne düşen aday reddedilir - aksi hâlde modele
    "bu doğru nesne aslında noise" diye öğretilirdi.
    """
    if count <= 0:
        return []
    real = np.asarray(real_boxes, dtype=np.float32).reshape(-1, 4)
    kept = []
    for _ in range(8):
        need = count - len(kept)
        if need <= 0:
            break
        cand = _sample_candidates(real, img_w, img_h, need * 3)
        if len(real):
            cand = cand[_iou_matrix(cand, real).max(1) <= max_iou]
        kept.extend(map(tuple, cand[:need].tolist()))
    return kept[:count]


# ─────────────────────────────────────────────────────────────────────────────
# COCO INDEX  (kompakt numpy dizileri -> fork-safe, hızlı açılış)
# ─────────────────────────────────────────────────────────────────────────────
def _largest_part(seg):
    """COCO segmentation birden çok parçalı olabilir; en büyük alanlıyı alırız."""
    best, best_a = None, 0.0
    for part in seg:
        if not isinstance(part, list) or len(part) < 6:
            continue
        p = np.asarray(part, dtype=np.float32).reshape(-1, 2)
        a = abs(polygon_area(p))
        if a > best_a:
            best, best_a = p, a
    return best, best_a


def build_coco_index(ann_file, cache_dir=CACHE_DIR, min_area=MIN_ANN_AREA,
                     keep_empty=True):
    tag = os.path.basename(ann_file).replace(".json", "")
    cache = os.path.join(cache_dir, f"cocoidx_{tag}_a{int(min_area)}_e{int(keep_empty)}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}

    print(f"COCO index kuruluyor: {ann_file}  (bir kez, sonra {cache} okunur)")
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    file_of = {im["id"]: im["file_name"] for im in data["images"]}
    per_img = {}
    n_crowd = n_rle = n_small = 0
    for a in data["annotations"]:
        if a.get("iscrowd", 0):
            n_crowd += 1
            continue
        seg = a.get("segmentation")
        if not isinstance(seg, list):      # RLE -> poligon değil, atlanır
            n_rle += 1
            continue
        cid = CATID_TO_ID.get(a["category_id"])
        if cid is None:
            continue
        part, area = _largest_part(seg)
        if part is None or area < min_area:
            n_small += 1
            continue
        per_img.setdefault(a["image_id"], []).append((cid, part))
    del data

    img_ids = sorted(file_of.keys()) if keep_empty else sorted(per_img.keys())

    names, img_start, img_cnt = [], [], []
    ann_off, ann_len, ann_lab, chunks = [], [], [], []
    cursor = 0
    for iid in img_ids:
        anns = per_img.get(iid, [])
        names.append(file_of[iid])
        img_start.append(len(ann_off))
        img_cnt.append(len(anns))
        for cid, part in anns:
            ann_off.append(cursor)
            ann_len.append(len(part))
            ann_lab.append(cid)
            chunks.append(part)
            cursor += len(part)

    idx = {
        "names":     np.array(names),
        "img_start": np.asarray(img_start, dtype=np.int64),
        "img_cnt":   np.asarray(img_cnt,   dtype=np.int32),
        "ann_off":   np.asarray(ann_off,   dtype=np.int64),
        "ann_len":   np.asarray(ann_len,   dtype=np.int32),
        "ann_lab":   np.asarray(ann_lab,   dtype=np.int16),
        "xy": (np.concatenate(chunks, 0).astype(np.float32)
               if chunks else np.zeros((0, 2), np.float32)),
    }
    np.savez_compressed(cache, **idx)
    print(f"  görüntü {len(names)} | nesne {len(ann_off)} | "
          f"atlanan: crowd {n_crowd}, rle {n_rle}, küçük {n_small}")
    return idx


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class CocoDualPix2SeqDataset(Dataset):
    """
    Hizalanmış (input, target) çiftleri döndürür - eğitim döngüsü KAYDIRMAZ.

    box_in  = [BOS, x0,y0,x1,y1,c, ...,  nx0,ny0,nx1,ny1,NOISE, ...]
    box_tgt = [     x0,y0,x1,y1,c, ...,  PAD,PAD,PAD,PAD,NOISE, ...]

    Noise koordinatları hedefte PAD ("n/a", loss ağırlığı 0) — paper Eq.1'deki
    w_j = 1[y_j != "n/a"]. Noise nesneleri her zaman KUYRUKTA durur; model bu
    sayede MAX_OBJECTS slotunun her birinde skorlanabilir, durmak için EOS'a
    ihtiyaç kalmaz.
    """

    def __init__(self, ann_file, img_dir, transform, is_train,
                 max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 permute_prob=PERMUTE_PROB, noise_fill_to_max=NOISE_FILL_TO_MAX,
                 noise_iou_reject=NOISE_IOU_REJECT, keep_empty=True,
                 cache_dir=CACHE_DIR):
        idx = build_coco_index(ann_file, cache_dir=cache_dir, keep_empty=keep_empty)
        self.names     = idx["names"]
        self.img_start = idx["img_start"]
        self.img_cnt   = idx["img_cnt"]
        self.ann_off   = idx["ann_off"]
        self.ann_len   = idx["ann_len"]
        self.ann_lab   = idx["ann_lab"]
        self.xy        = idx["xy"]

        self.img_dir     = img_dir
        self.transform   = transform
        self.is_train    = is_train
        self.max_objects = max_objects
        self.img_size    = img_size
        # Noise ve permütasyon SADECE is_train'e bağlı: val_transform da None
        # değil, ona bakarak karar vermek val GT'sine noise sızdırırdı.
        self.permute_prob      = permute_prob if is_train else 0.0
        self.noise_fill_to_max = noise_fill_to_max
        self.noise_iou_reject  = noise_iou_reject

    def __len__(self):
        return len(self.names)

    def _raw_shapes(self, i):
        s, c = int(self.img_start[i]), int(self.img_cnt[i])
        polys, labels = [], []
        for j in range(s, s + c):
            o, l = int(self.ann_off[j]), int(self.ann_len[j])
            polys.append(self.xy[o:o + l])          # view, kopya yok
            labels.append(int(self.ann_lab[j]))
        return polys, labels

    def _records_from_points(self, new_points, counts, labels, img_w, img_h):
        records, cursor = [], 0
        for cnt, lab in zip(counts, labels):
            poly = np.asarray(new_points[cursor:cursor + cnt], dtype=np.float32)
            cursor += cnt
            if len(poly) < 3:
                continue

            area_before = abs(polygon_area(poly))
            poly = clip_polygon(poly, float(img_w), float(img_h))
            if len(poly) < 3:
                continue

            # LSJ'de ölçek 2x'e kadar çıkıyor; kadrajın çoğu dışında kalan nesneyi
            # kırpıp tutmak "kenarda hep şu sınıf var" diye yanlış sinyal verir.
            area_after = abs(polygon_area(poly))
            if area_after < MIN_AREA or area_after < MIN_AREA_KEEP * max(area_before, 1e-6):
                continue

            xs, ys = poly[:, 0], poly[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
                continue

            poly = canonicalize_polygon(poly)
            poly = resample_polygon(poly, NUM_POLY_PTS)
            if poly is None:
                continue

            records.append({"class_id": lab, "box": (x0, y0, x1, y1),
                            "poly": poly, "is_noise": False})
        return records

    def __getitem__(self, i):
        img_path = os.path.join(self.img_dir, str(self.names[i]))
        try:
            image = Image.open(img_path).convert("RGB")
            image, _ = pad_to_square(image)
            image_np = np.array(image)
        except Exception as e:
            print(f"HATA: {img_path} okunamadı: {e}")
            image_np = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        polys, labels = self._raw_shapes(i)
        counts, all_points = [], []
        for p in polys:
            counts.append(len(p))
            all_points.extend([(float(a), float(b)) for a, b in p])

        aug = self.transform(image=image_np, keypoints=all_points)
        image_tensor = aug["image"]
        new_points = [(float(p[0]), float(p[1])) for p in aug["keypoints"]]

        records = self._records_from_points(new_points, counts, labels, IMG_W, IMG_H)

        # ── 1. Sıra: paper -> her gösterimde rastgele ────────────────────────
        if self.permute_prob > 0 and random.random() < self.permute_prob:
            random.shuffle(records)
        else:
            records.sort(key=lambda r: (r["box"][1], r["box"][0]))
        records = records[:self.max_objects]
        num_valid_objs = len(records)          # sadece gerçek nesneler

        # ── 2. Noise nesneleri KUYRUĞA ──────────────────────────────────────
        if self.is_train:
            n_noise = (self.max_objects - num_valid_objs) if self.noise_fill_to_max else 0
            for sbox in generate_synthetic_boxes([r["box"] for r in records],
                                                 IMG_W, IMG_H, n_noise,
                                                 self.noise_iou_reject):
                records.append({"class_id": NOISE_CLASS_ID, "box": sbox,
                                "poly": None, "is_noise": True})

        # ── 3. Hizalanmış diziler ───────────────────────────────────────────
        box_in_toks, box_tgt_toks = [], []
        mask_in_seqs, mask_tgt_seqs = [], []

        for rec in records:
            x0, y0, x1, y1 = rec["box"]
            q = [quantize(x0, IMG_W, NUM_BINS), quantize(y0, IMG_H, NUM_BINS),
                 quantize(x1, IMG_W, NUM_BINS), quantize(y1, IMG_H, NUM_BINS)]

            if rec["is_noise"]:
                box_in_toks.extend(q + [NOISE_CLASS_TOKEN])
                box_tgt_toks.extend([PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN])
                mask_in_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
                mask_tgt_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
            else:
                c = NUM_BINS + rec["class_id"]
                box_in_toks.extend(q + [c])
                box_tgt_toks.extend(q + [c])

                poly_tokens = []
                for px, py in rec["poly"]:
                    poly_tokens.append(quantize(float(px), IMG_W, NUM_BINS))
                    poly_tokens.append(quantize(float(py), IMG_H, NUM_BINS))

                m_in = [BOS_TOKEN] + q + [c] + poly_tokens
                m_tgt = q + [c] + poly_tokens + [EOS_TOKEN]
                m_in = (m_in + [PAD_TOKEN] * MAX_MASK_SEQ_LEN)[:MAX_MASK_SEQ_LEN]
                m_tgt = (m_tgt + [PAD_TOKEN] * MAX_MASK_SEQ_LEN)[:MAX_MASK_SEQ_LEN]
                mask_in_seqs.append(m_in)
                mask_tgt_seqs.append(m_tgt)

        box_in = ([BOS_TOKEN] + box_in_toks + [PAD_TOKEN] * MAX_BOX_SEQ_LEN)[:MAX_BOX_SEQ_LEN]
        box_tgt = (box_tgt_toks + [PAD_TOKEN] * MAX_BOX_SEQ_LEN)[:MAX_BOX_SEQ_LEN]

        while len(mask_in_seqs) < self.max_objects:
            mask_in_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
            mask_tgt_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)

        return (image_tensor,
                torch.tensor(box_in, dtype=torch.long),
                torch.tensor(box_tgt, dtype=torch.long),
                torch.tensor(mask_in_seqs, dtype=torch.long),
                torch.tensor(mask_tgt_seqs, dtype=torch.long),
                torch.tensor(num_valid_objs, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC DEPTH
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Residual dalını örnek başına tamamen düşürür (paper: stochastic depth %10).
    nn.Transformer*Layer'da dropout1/2/3 tam olarak dalın çıkışında durur, o
    yüzden onları bununla değiştirmek dalın TAMAMINI atlamak demek.
    Eval'de kimlik fonksiyonu -> KV-cache decode etkilenmez.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"


def apply_stochastic_depth(stack, p_max):
    """Derinlikle lineer artan drop oranı (timm/ViT konvansiyonu)."""
    layers = list(stack.layers)
    n = len(layers)
    for i, layer in enumerate(layers):
        p = p_max * i / max(n - 1, 1)
        for name in ("dropout1", "dropout2", "dropout3"):
            if hasattr(layer, name):
                setattr(layer, name, DropPath(p))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def causal_mask(size, device):
    """
    Bool nedensellik maskesi (True = bakılamaz). generate_square_subsequent_mask
    float döndürüyor; onu bool key_padding_mask ile birlikte kullanmak PyTorch'ta
    "mismatched mask" uyarısı üretiyor.
    """
    return torch.ones(size, size, dtype=torch.bool, device=device).triu(1)


class DualPix2SeqModel(nn.Module):
    """
    ResNet50 -> 1x1 proj -> 6 katman transformer encoder -> iki ayrı 6 katman
    decoder. Embedding ve çıkış başı iki decoder arasında paylaşılır (aynı
    token uzayı, daha az parametre).
    """

    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, nheads=NHEADS,
                 num_encoder_layers=ENC_LAYERS, num_decoder_layers=DEC_LAYERS,
                 img_size=IMG_SIZE, dropout=DROPOUT, drop_path=DROP_PATH,
                 norm_first=False, dilated_c5=False):
        super().__init__()
        # dilated_c5=True -> çıktı stride 32 yerine 16 (paper'ın DC5 varyantı).
        # AP_small için en büyük tek kaldıraç, ama encoder self-attention maliyeti
        # 16x (400 -> 1600 token). Bütçe varsa aç.
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT,
            replace_stride_with_dilation=[False, False, dilated_c5],
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        stride = 16 if dilated_c5 else 32
        gh, gw = img_size[0] // stride, img_size[1] // stride
        self.img_pos_emb = nn.Parameter(torch.zeros(1, gh * gw, hidden_dim))
        nn.init.trunc_normal_(self.img_pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_encoder_layers,
            norm=nn.LayerNorm(hidden_dim) if norm_first else None,
            enable_nested_tensor=False)

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD_TOKEN].zero_()

        self.seq_pos_encoding = PositionalEncoding(
            hidden_dim, max_len=max(MAX_BOX_SEQ_LEN, MAX_MASK_SEQ_LEN))
        self.emb_dropout = nn.Dropout(dropout)

        def _mk_decoder():
            layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, batch_first=True, norm_first=norm_first)
            return nn.TransformerDecoder(
                layer, num_layers=num_decoder_layers,
                norm=nn.LayerNorm(hidden_dim) if norm_first else None)

        self.box_decoder = _mk_decoder()
        self.mask_decoder = _mk_decoder()
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        if drop_path > 0:
            apply_stochastic_depth(self.encoder, drop_path)
            apply_stochastic_depth(self.box_decoder, drop_path)
            apply_stochastic_depth(self.mask_decoder, drop_path)

    # ── encoder ──────────────────────────────────────────────────────────────
    def encode(self, images):
        f = self.enc_proj(self.backbone(images))
        return self.encoder(f.flatten(2).permute(0, 2, 1) + self.img_pos_emb)

    # ── decoder 1 ────────────────────────────────────────────────────────────
    def forward_boxes(self, memory, box_seq):
        mask = causal_mask(box_seq.size(1), box_seq.device)
        emb = self.emb_dropout(self.seq_pos_encoding(self.embedding(box_seq)))
        out = self.box_decoder(tgt=emb, memory=memory, tgt_mask=mask,
                               tgt_key_padding_mask=(box_seq == PAD_TOKEN))
        return self.fc_out(out)

    # ── decoder 2 ────────────────────────────────────────────────────────────
    def forward_masks(self, memory, mask_seqs):
        """
        Sadece gerçekten dizi taşıyan nesne slotlarını çalıştırır. Boş slotlar
        ve noise nesneleri hem inputta hem hedefte tamamen PAD, o yüzden onların
        logitleri loss'ta zaten kullanılmıyor — MAX_OBJECTS=100'de batch'in
        %90'ından fazlası boş, hepsini hesaplamak saf israf olurdu.

        Dönüş: (sub_logits [M,S,V], sel [M]) — sel, flat (b*N+obj) indeksleri.
        """
        B, N, S = mask_seqs.shape
        flat = mask_seqs.reshape(B * N, S)
        valid = (flat != PAD_TOKEN).any(dim=1)
        sel = valid.nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            return None, sel

        owner = torch.div(sel, N, rounding_mode="floor")
        sub_seqs = flat.index_select(0, sel)
        sub_mem = memory.index_select(0, owner)

        mask = causal_mask(S, mask_seqs.device)
        emb = self.emb_dropout(self.seq_pos_encoding(self.embedding(sub_seqs)))
        out = self.mask_decoder(tgt=emb, memory=sub_mem, tgt_mask=mask,
                                tgt_key_padding_mask=(sub_seqs == PAD_TOKEN))
        return self.fc_out(out), sel

    def forward(self, images, box_in, mask_in):
        memory = self.encode(images)
        box_logits = self.forward_boxes(memory, box_in)
        mask_logits, sel = self.forward_masks(memory, mask_in)
        return box_logits, mask_logits, sel


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE'Lİ DECODE
# ─────────────────────────────────────────────────────────────────────────────
class KVCacheDecoder:
    """
    nn.TransformerDecoder için adım adım decode. Cross-attention K/V'si memory'den
    bir kez hesaplanır, self-attention K/V'si biriktirilir. Model eval modunda
    olmalı (dropout/DropPath kimlik) — residual'lar o varsayımla sadeleştirildi.
    """

    def __init__(self, decoder, memory):
        self.layers = decoder.layers
        self.norm = decoder.norm
        self.self_k = [None] * len(self.layers)
        self.self_v = [None] * len(self.layers)
        self.cross_k, self.cross_v = [], []
        for layer in self.layers:
            a = layer.multihead_attn
            D = a.embed_dim
            W, b = a.in_proj_weight, a.in_proj_bias
            bk = None if b is None else b[D:2 * D]
            bv = None if b is None else b[2 * D:]
            self.cross_k.append(self._split(F.linear(memory, W[D:2 * D], bk), a.num_heads))
            self.cross_v.append(self._split(F.linear(memory, W[2 * D:], bv), a.num_heads))

    @staticmethod
    def _split(x, nh):
        B, L, D = x.shape
        return x.view(B, L, nh, D // nh).transpose(1, 2)

    @staticmethod
    def _merge(x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).reshape(B, L, H * hd)

    def _self_attn(self, layer, i, x, is_causal):
        a = layer.self_attn
        D = a.embed_dim
        q, k, v = F.linear(x, a.in_proj_weight, a.in_proj_bias).split(D, dim=-1)
        q, k, v = (self._split(t, a.num_heads) for t in (q, k, v))
        self.self_k[i] = k if self.self_k[i] is None else torch.cat([self.self_k[i], k], 2)
        self.self_v[i] = v if self.self_v[i] is None else torch.cat([self.self_v[i], v], 2)
        o = F.scaled_dot_product_attention(q, self.self_k[i], self.self_v[i],
                                           is_causal=is_causal)
        return a.out_proj(self._merge(o))

    def _cross_attn(self, layer, i, x):
        a = layer.multihead_attn
        D = a.embed_dim
        bq = None if a.in_proj_bias is None else a.in_proj_bias[:D]
        q = self._split(F.linear(x, a.in_proj_weight[:D], bq), a.num_heads)
        o = F.scaled_dot_product_attention(q, self.cross_k[i], self.cross_v[i])
        return a.out_proj(self._merge(o))

    @staticmethod
    def _ff(layer, x):
        act = layer.activation
        if isinstance(act, str):
            act = F.relu if act == "relu" else F.gelu
        return layer.linear2(layer.dropout(act(layer.linear1(x))))

    def step(self, x, is_causal=False):
        """is_causal sadece prefill'de (cache boş, q_len == k_len) kullanılabilir."""
        for i, layer in enumerate(self.layers):
            if getattr(layer, "norm_first", False):
                x = x + self._self_attn(layer, i, layer.norm1(x), is_causal)
                x = x + self._cross_attn(layer, i, layer.norm2(x))
                x = x + self._ff(layer, layer.norm3(x))
            else:
                x = layer.norm1(x + self._self_attn(layer, i, x, is_causal))
                x = layer.norm2(x + self._cross_attn(layer, i, x))
                x = layer.norm3(x + self._ff(layer, x))
        return x if self.norm is None else self.norm(x)


@torch.no_grad()
def _decode_polygons(model, inst_memory, prefixes):
    """
    prefixes: [N, 6] = [BOS, x0, y0, x1, y1, cls]
    Sabit uzunlukta 2*NUM_POLY_PTS koordinat token'ı üretir (EOS beklenmez, bu
    yüzden dizi asla kısa kalmaz). Dönüş: [N, 2*NUM_POLY_PTS]
    """
    cache = KVCacheDecoder(model.mask_decoder, inst_memory)
    pe = model.seq_pos_encoding.pe
    L = prefixes.size(1)

    emb = model.embedding(prefixes) + pe[:, :L]
    h = cache.step(emb, is_causal=True)[:, -1]

    toks = []
    for t in range(POLY_TOKENS_PER_OBJ):
        nxt = model.fc_out(h).float()[:, :NUM_BINS].argmax(-1)   # koordinat binine kısıtla
        toks.append(nxt)
        if t == POLY_TOKENS_PER_OBJ - 1:
            break
        e = model.embedding(nxt).unsqueeze(1) + pe[:, L + t:L + t + 1]
        h = cache.step(e)[:, -1]
    return torch.stack(toks, 1)


@torch.no_grad()
def dual_ar_decode(model, images, device=None, score_thresh=INFER_SCORE_THRESH,
                   max_dets=MAX_DETS, drop_noise_argmax=False):
    """
    Decoder 1: sabit uzunlukta kısıtlı argmax decode (MAX_OBJECTS slot).
      * koordinat adımları [0, NUM_BINS) ile sınırlı
      * sınıf adımları sınıf slotlarıyla sınırlı (gerçek sınıflar + noise)
      * EOS yok -> chunk'lar asla kayamaz
      * skor = p(en iyi gerçek sınıf) / (p(gerçek) + p(noise))

    Decoder 2: hayatta kalan her instance için poligon, DEC2_CHUNK'lık gruplar
    hâlinde ve KV-cache ile.

    score_thresh düşük olursa daha çok düşük güvenli kutu kalır; mAP bunu sever
    (kötüler zaten alta sıralanır), gecikme sevmez (her instance bir decoder-2
    rollout'u demek).
    """
    model.eval()
    device = device or images.device
    B = images.size(0)

    memory = model.encode(images)
    cache = KVCacheDecoder(model.box_decoder, memory)
    pe = model.seq_pos_encoding.pe

    cur = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    coord = torch.zeros(B, MAX_OBJECTS, 4, dtype=torch.long, device=device)
    cls_id = torch.zeros(B, MAX_OBJECTS, dtype=torch.long, device=device)
    conf = torch.zeros(B, MAX_OBJECTS, device=device)
    noise_won = torch.zeros(B, MAX_OBJECTS, dtype=torch.bool, device=device)
    lo, hi = NUM_BINS, NUM_BINS + NUM_CLASS_SLOTS

    for step in range(BOX_TOKENS_PER_OBJ * MAX_OBJECTS):
        emb = model.embedding(cur) + pe[:, step:step + 1]
        probs = model.fc_out(cache.step(emb)[:, -1]).float().softmax(-1)
        obj, pos = divmod(step, BOX_TOKENS_PER_OBJ)

        if pos == BOX_TOKENS_PER_OBJ - 1:
            p_real, best = probs[:, lo:lo + NUM_CLASSES].max(-1)
            p_noise = probs[:, NOISE_CLASS_TOKEN]
            cls_id[:, obj] = best
            conf[:, obj] = p_real / (p_real + p_noise).clamp_min(1e-6)
            # Eğitimde noise slotlarının inputu NOISE token'ı taşıyor, o yüzden
            # geri beslenen şey modelin gerçekten seçtiği sınıf olmalı.
            nxt = probs[:, lo:hi].argmax(-1) + lo
            noise_won[:, obj] = (nxt == NOISE_CLASS_TOKEN)
        else:
            nxt = probs[:, :NUM_BINS].argmax(-1)
            coord[:, obj, pos] = nxt
        cur = nxt.unsqueeze(1)

    # ── Hayatta kalan instance'ları topla ────────────────────────────────────
    sx = IMG_SIZE[1] / (NUM_BINS - 1)
    sy = IMG_SIZE[0] / (NUM_BINS - 1)
    boxes_px = coord.float() * torch.tensor([sx, sy, sx, sy], device=device)

    keep = (boxes_px[:, :, 2] > boxes_px[:, :, 0]) & (boxes_px[:, :, 3] > boxes_px[:, :, 1])
    keep &= conf >= score_thresh
    if drop_noise_argmax:
        keep &= ~noise_won

    all_inst, flat_prefix, flat_owner = [], [], []
    for b in range(B):
        idxs = keep[b].nonzero(as_tuple=True)[0]
        if idxs.numel() > max_dets:                      # skora göre ilk max_dets
            top = conf[b, idxs].topk(max_dets).indices
            idxs = idxs[top]
        inst = []
        for s in idxs.tolist():
            inst.append({"box": boxes_px[b, s].tolist(),
                         "label": int(cls_id[b, s]),
                         "score": float(conf[b, s])})
            flat_prefix.append([BOS_TOKEN] + coord[b, s].tolist()
                               + [NUM_BINS + int(cls_id[b, s])])
            flat_owner.append(b)
        all_inst.append(inst)

    if not flat_prefix:
        return [[] for _ in range(B)]

    # ── Decoder 2, chunk'lar hâlinde ─────────────────────────────────────────
    prefixes = torch.tensor(flat_prefix, dtype=torch.long, device=device)
    owners = torch.tensor(flat_owner, dtype=torch.long, device=device)

    poly_chunks = []
    for s in range(0, prefixes.size(0), DEC2_CHUNK):
        p = prefixes[s:s + DEC2_CHUNK]
        m = memory.index_select(0, owners[s:s + DEC2_CHUNK])
        poly_chunks.append(_decode_polygons(model, m, p))
    poly_toks = torch.cat(poly_chunks, 0).cpu().numpy()

    out = [[] for _ in range(B)]
    cursor = 0
    for b in range(B):
        for inst in all_inst[b]:
            t = poly_toks[cursor]
            cursor += 1
            poly = np.stack([t[0::2].astype(np.float32) * sx,
                             t[1::2].astype(np.float32) * sy], axis=1)
            out[b].append({"box": inst["box"], "polygon": poly,
                           "label": inst["label"], "score": inst["score"]})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EĞİTİM
# ─────────────────────────────────────────────────────────────────────────────
def build_optimizer(model):
    """
    wd 0.05 ile norm/bias/embedding/pos-emb ayrımı artık önemli — 1e-4'te fark
    edilmezdi. Backbone ayrı ve düşük lr (ImageNet ön-eğitimli).
    """
    decay, no_decay, bb_decay, bb_no_decay = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n.startswith("backbone")
        skip = p.ndim <= 1 or "img_pos_emb" in n or "embedding" in n
        if is_bb:
            (bb_no_decay if skip else bb_decay).append(p)
        else:
            (no_decay if skip else decay).append(p)

    opt = torch.optim.AdamW([
        {"params": bb_decay,    "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": bb_no_decay, "lr": BACKBONE_LR, "weight_decay": 0.0},
        {"params": decay,       "lr": BASE_LR,     "weight_decay": WEIGHT_DECAY},
        {"params": no_decay,    "lr": BASE_LR,     "weight_decay": 0.0},
    ])
    assert sum(len(g["params"]) for g in opt.param_groups) == \
        sum(1 for p in model.parameters() if p.requires_grad), "parametre kaçağı"
    return opt


def compute_losses(model, criterion, images, box_in, box_tgt, mask_in, mask_tgt):
    box_logits, mask_logits, sel = model(images, box_in, mask_in)
    loss_box = criterion(box_logits.reshape(-1, VOCAB_SIZE), box_tgt.reshape(-1))

    if sel.numel() == 0:
        loss_mask = box_logits.new_zeros(())
    else:
        B, N, S = mask_tgt.shape
        tgt = mask_tgt.reshape(B * N, S).index_select(0, sel)
        loss_mask = criterion(mask_logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
    return loss_box, loss_mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = USE_BF16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"Dual-Decoder Pix2Seq (COCO) | device={device} | img={IMG_SIZE} | "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}={BATCH_SIZE * GRAD_ACCUM} | "
          f"enc={ENC_LAYERS}L dec={DEC_LAYERS}Lx2 | vocab={VOCAB_SIZE}")

    train_ds = CocoDualPix2SeqDataset(TRAIN_ANN, TRAIN_IMG_DIR, train_transform,
                                      is_train=True)
    val_ds = CocoDualPix2SeqDataset(VAL_ANN, VAL_IMG_DIR, val_transform,
                                    is_train=False)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              prefetch_factor=2, drop_last=True,
                              persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            prefetch_factor=2, persistent_workers=NUM_WORKERS > 0)

    model = DualPix2SeqModel().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=LABEL_SMOOTH)
    optimizer = build_optimizer(model)

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    assert steps_per_epoch > 0, (
        f"train_loader'da {len(train_loader)} batch var, GRAD_ACCUM={GRAD_ACCUM} "
        "ile bir optimizer adımı bile atılamıyor - BATCH_SIZE/GRAD_ACCUM'u küçült.")
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(int(total_steps * WARMUP_FRAC), 1)

    def lr_lambda(step):                       # paper: warmup + lineer düşüş
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - prog)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    from instance_seg_eval_utils_dual_pix2seq import evaluate_dual

    best_val_loss = float("inf")
    best_segm_map = 0.0
    history = []
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="ep")

    for epoch in epoch_bar:
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        run, run_b, run_m, nb = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(train_loader, desc=f"  Train {epoch + 1}/{EPOCHS}", leave=False, unit="b")
        for i, (images, box_in, box_tgt, mask_in, mask_tgt, _) in enumerate(bar):
            images = images.to(device, non_blocking=True).float()
            box_in = box_in.to(device, non_blocking=True)
            box_tgt = box_tgt.to(device, non_blocking=True)
            mask_in = mask_in.to(device, non_blocking=True)
            mask_tgt = mask_tgt.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_enabled):
                loss_box, loss_mask = compute_losses(model, criterion, images,
                                                     box_in, box_tgt, mask_in, mask_tgt)
                loss = loss_box + MASK_LOSS_W * loss_mask

            (loss / GRAD_ACCUM).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(box=f"{loss_box.item():.3f}",
                                mask=f"{loss_mask.item():.3f}", gnorm=f"{gn:.2f}")

            run += loss.item(); run_b += loss_box.item(); run_m += loss_mask.item(); nb += 1

        train_loss = run / max(nb, 1)

        # ── Val loss ─────────────────────────────────────────────────────────
        model.eval()
        vloss, nv = 0.0, 0
        with torch.no_grad():
            for images, box_in, box_tgt, mask_in, mask_tgt, _ in tqdm(
                    val_loader, desc=f"  Val {epoch + 1}", leave=False, unit="b"):
                images = images.to(device, non_blocking=True).float()
                box_in, box_tgt = box_in.to(device), box_tgt.to(device)
                mask_in, mask_tgt = mask_in.to(device), mask_tgt.to(device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_enabled):
                    lb, lm = compute_losses(model, criterion, images, box_in,
                                            box_tgt, mask_in, mask_tgt)
                    vloss += (lb + MASK_LOSS_W * lm).item()
                nv += 1
        val_loss = vloss / max(nv, 1)

        lr_now = optimizer.param_groups[2]["lr"]
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              lr=f"{lr_now:.2e}", best_map=f"{best_segm_map:.4f}")

        torch.save({"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch + 1}, f"{CKPT_PREFIX}_last.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{CKPT_PREFIX}_best_loss.pth")

        # ── mAP ──────────────────────────────────────────────────────────────
        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            res = evaluate_dual(model, val_loader, device=device,
                                max_images=EVAL_MAX_IMAGES, amp_dtype=amp_dtype)
            history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss,
                            "train_box": run_b / max(nb, 1), "train_mask": run_m / max(nb, 1),
                            "lr": lr_now, **res})
            tqdm.write(
                f"  ep {epoch + 1:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr_now:.2e} | segm mAP {res['segm_map']:.4f} @50 {res['segm_map_50']:.4f} "
                f"| bbox mAP {res['bbox_map']:.4f} | pred/img {res['preds_per_img']:.1f}")

            if res["segm_map"] > best_segm_map:
                best_segm_map = res["segm_map"]
                torch.save(model.state_dict(), f"{CKPT_PREFIX}_best_map.pth")
                tqdm.write(f"       ✓ yeni en iyi segm mAP {best_segm_map:.4f}")

            with open(f"{CKPT_PREFIX}_history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"\nBitti. En iyi segm mAP: {best_segm_map:.4f} -> {CKPT_PREFIX}_best_map.pth")
