import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np 
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split
from tqdm import tqdm 
import random
from PIL import ImageDraw

# --- YAPILANDIRMA (CONFIG) ---
# Kendi yollarına göre burayı güncelleyebilirsin
JSON_DIR       = "/mnt/d/Datasets/license-plate.v2i.coco-segmentation/train"
IMG_DIR        = "/mnt/d/Datasets/license-plate.v2i.coco-segmentation/train"
NUM_BINS = 500 
MAX_OBJECTS = 10
BATCH_SIZE = 16
EPOCHS = 350
LEARNING_RATE = 3e-4
IMG_SIZE = (512, 512)
VAL_SPLIT_PATH = "val_split.json"

LABEL_TO_ID = {
    "my_pla2": 0
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_CLASSES = len(LABEL_TO_ID)          # SADECE gerçek sınıflar

# ── Sequence augmentation ────────────────────────────────────────────────────
# DİKKAT: "noise" LABEL_TO_ID'ye EKLENMEZ. Eklenirse NUM_CLASSES kayar, per-class
# AP'ye sahte bir sınıf girer ve annotasyonda "noise" etiketli bir shape gerçek
# GT sayılır. Ayrı bir token olarak durur.
NOISE_CLASS_ID     = NUM_CLASSES                   # 1
NOISE_CLASS_TOKEN  = NUM_BINS + NOISE_CLASS_ID     # 501
NUM_CLASS_SLOTS    = NUM_CLASSES + 1               # gerçek + noise

# Görüntü başına noise sayısı serbest bir parametre DEĞİL: (MAX_OBJECTS - gerçek).
# Bu yüzden MAX_OBJECTS'i gerçek dağılıma göre ayarla — dataset_object_stats()
# p99'u söyler, MAX_OBJECTS ~ p99 + pay olmalı.
NOISE_FILL_TO_MAX  = True
NOISE_MAX_PER_IMG  = None   # normalde None; MAX_OBJECTS'i küçültemiyorsan sert sınır
NOISE_RATIO        = 0.5    # NOISE_FILL_TO_MAX=False iken gerçek nesne başına oran
NOISE_IOU_REJECT   = 0.5    # gerçek kutuyla bu IoU'yu aşan sentetik kutu elenir
NOISE_JITTER_FRAC  = 0.7    # noise'un ne kadarı gerçek kutu bozularak üretilir
INFER_SCORE_THRESH = 0.05

VOCAB_SIZE = NUM_BINS + NUM_CLASS_SLOTS + 3        # 505
BOS_TOKEN = VOCAB_SIZE - 3                         # 502
EOS_TOKEN = VOCAB_SIZE - 2                         # 503 (fill_to_max iken kullanılmaz)
PAD_TOKEN = VOCAB_SIZE - 1                         # 504
TOKENS_PER_OBJ = 5
MIN_SIDE = 2.0

train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(std_range=(0.05,0.1),p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.2),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(
    format='xy',
    remove_invisible=False
))

def pad_to_square(image, fill_color=(128, 128, 128)):
    """Resmin en-boy oranını bozmadan, en uzun kenarı baz alarak kareye tamamlar."""
    w, h = image.size
    max_dim = max(w, h)
    
    # Gri renkte boş bir kare oluştur (Padding)
    new_image = Image.new("RGB", (max_dim, max_dim), fill_color)
    
    # Orijinal resmi sol üst köşeye yapıştır
    new_image.paste(image, (0, 0))
    return new_image, max_dim

def quantize(coordinate, max_size, num_bins):
    normalized = coordinate / max_size
    normalized = max(0.0, min(1.0, normalized))
    return int(normalized * (num_bins - 1))


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION KATMANI — poligon / segmentasyon maskesi / bbox hepsini kabul eder
# ─────────────────────────────────────────────────────────────────────────────
# Bu model kutu üretiyor ama annotasyon kutu olmak zorunda değil. Desteklenenler:
#
#   1) labelme  : {"shapes": [{"label", "points", "shape_type"}]}
#                 polygon (N>=3), rectangle (2 nokta), circle (merkez+kenar),
#                 quad (4 nokta) — hepsi kabul
#   2) COCO     : {"annotations": [{"category_id"/"label", "segmentation", "bbox"}]}
#                 segmentation: poligon listesi VEYA RLE dict (pycocotools varsa)
#                 segmentation yoksa bbox [x, y, w, h] kullanılır
#   3) maske    : {"maskPath": "..."} — ikili ya da instance-id kodlu PNG.
#                 cv2 ile bağlantılı bileşenler bulunup konturlara çevrilir.
#
# ÖNEMLİ: poligon noktaları augmentasyona keypoint olarak girer, kutu ANCAK
# augmentasyondan SONRA min/max ile hesaplanır. Bu, döndürme/perspektif sonrası
# kutunun gerçekten sıkı olmasını sağlar — sadece 4 köşe taşınsaydı döndürülmüş
# bir nesnenin kutusu gereksiz şişerdi. Maskeli veride bu fark ciddi.

MAX_PTS_PER_SHAPE = 32     # keypoint sayısını sınırla; kutu için zarf yeterli
COCO_CATEGORY_MAP = {}     # {category_id: "label"} — gerekiyorsa doldur


def _subsample_points(pts, max_pts=MAX_PTS_PER_SHAPE):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) <= max_pts:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_pts).round().astype(int)
    return pts[idx]


def _corners(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _mask_to_points(mask, min_area=16):
    """İkili/instance maskeden nesne başına kontur noktaları."""
    try:
        import cv2
    except ImportError:
        raise RuntimeError("Maske annotasyonu için opencv-python gerekli.")

    out = []
    ids = [i for i in np.unique(mask) if i != 0]
    binary_only = len(ids) == 1 and ids[0] in (1, 255)

    for inst_id in ids:
        m = (mask == inst_id).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if binary_only:
            # ikili maske: her bağlantılı bileşen ayrı bir nesne
            for c in cnts:
                if cv2.contourArea(c) >= min_area:
                    out.append(_subsample_points(c.reshape(-1, 2)))
        else:
            pts = np.concatenate([c.reshape(-1, 2) for c in cnts], axis=0) if cnts else None
            if pts is not None and len(pts) >= 3:
                out.append(_subsample_points(pts))
    return out


def _seg_to_points(seg, height=None, width=None):
    """COCO segmentation -> [N,2] nokta dizisi (çok parçalı instance birleştirilir)."""
    if isinstance(seg, dict):                      # RLE
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        rle = seg
        if isinstance(rle.get("counts"), list) and height and width:
            rle = mask_utils.frPyObjects(rle, height, width)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.max(axis=2)
        parts = _mask_to_points((m > 0).astype(np.uint8) * 1)
        if not parts:
            return None
        return _subsample_points(np.concatenate(parts, axis=0))

    if isinstance(seg, (list, tuple)) and len(seg) > 0:
        # [[x1,y1,x2,y2,...], ...] — tüm parçalar tek nesnenin zarfına katkı verir
        polys = seg if isinstance(seg[0], (list, tuple)) else [seg]
        pts = []
        for poly in polys:
            arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            if len(arr) >= 3:
                pts.append(_subsample_points(arr, max(8, MAX_PTS_PER_SHAPE // max(len(polys), 1))))
        if pts:
            return np.concatenate(pts, axis=0)
    return None


def _resolve_label(ann, categories):
    for key in ("label", "category_name", "name"):
        if ann.get(key) in LABEL_TO_ID:
            return ann[key]
    cid = ann.get("category_id")
    if cid is not None:
        if cid in COCO_CATEGORY_MAP and COCO_CATEGORY_MAP[cid] in LABEL_TO_ID:
            return COCO_CATEGORY_MAP[cid]
        name = categories.get(cid)
        if name in LABEL_TO_ID:
            return name
    if NUM_CLASSES == 1:                 # tek sınıflı veri: eşleştirmeye gerek yok
        return next(iter(LABEL_TO_ID))
    return None


def load_shapes(item, img_dir=None):
    """
    Hangi formatta olursa olsun -> [{"label": str, "points": np.ndarray[N,2]}]
    Noktalar ORİJİNAL (kareye padlenmiş) görüntü uzayındadır.
    """
    shapes = []

    # ── 1) labelme ───────────────────────────────────────────────────────────
    for sh in item.get("shapes", []):
        label = sh.get("label")
        if label not in LABEL_TO_ID:
            continue
        pts = np.asarray(sh.get("points", []), dtype=np.float32).reshape(-1, 2)
        st  = sh.get("shape_type", "polygon")

        if st == "rectangle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            pts = _corners(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        elif st == "circle" and len(pts) == 2:
            (cx, cy), (ex, ey) = pts
            r = float(np.hypot(ex - cx, ey - cy))
            pts = _corners(cx - r, cy - r, cx + r, cy + r)
        elif len(pts) < 3:
            continue

        shapes.append({"label": label, "points": _subsample_points(pts)})

    # ── 2) COCO ──────────────────────────────────────────────────────────────
    cats = {c["id"]: c.get("name") for c in item.get("categories", []) if "id" in c}
    H = item.get("imageHeight") or item.get("height")
    W = item.get("imageWidth") or item.get("width")
    for ann in item.get("annotations", []):
        label = _resolve_label(ann, cats)
        if label is None:
            continue
        pts = _seg_to_points(ann.get("segmentation"), H, W)
        if pts is None and ann.get("bbox") is not None:
            x, y, w, h = [float(v) for v in ann["bbox"]]      # COCO: xywh
            pts = _corners(x, y, x + w, y + h)
        if pts is None or len(pts) < 3:
            continue
        shapes.append({"label": label, "points": _subsample_points(pts)})

    # ── 3) Maske dosyası ─────────────────────────────────────────────────────
    mask_path = item.get("maskPath") or item.get("mask")
    if mask_path and img_dir:
        full = mask_path if os.path.isabs(mask_path) else os.path.join(img_dir, mask_path)
        if os.path.exists(full):
            mask = np.array(Image.open(full))
            if mask.ndim == 3:
                mask = mask[..., 0]
            label = item.get("maskLabel") or (next(iter(LABEL_TO_ID)) if NUM_CLASSES == 1 else None)
            if label in LABEL_TO_ID:
                for pts in _mask_to_points(mask):
                    shapes.append({"label": label, "points": pts})

    return shapes


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE AUGMENTATION — sentetik kutular
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(union, 1e-6)


def generate_noise_boxes(real_boxes, img_w, img_h, count,
                         max_iou=NOISE_IOU_REJECT, jitter_frac=NOISE_JITTER_FRAC):
    """
    Tip A: gerçek bir kutuyu ölçekle + kaydır -> zor negatif (asıl öğretici olan).
    Tip B: rastgele arka plan kutusu. Boyut dağılımı, varsa gerçek kutulardan
           örneklenir; yoksa geniş bir aralıktan.
    Gerçek bir kutuyla IoU > max_iou olan aday elenir: aksi hâlde modele
    "doğru tespit = noise" öğretmiş oluruz.
    """
    out, attempts = [], 0
    sizes = (np.array([[b[2] - b[0], b[3] - b[1]] for b in real_boxes])
             if real_boxes else None)

    while len(out) < count and attempts < max(count, 1) * 30:
        attempts += 1

        if real_boxes and random.random() < jitter_frac:
            rx0, ry0, rx1, ry1 = random.choice(real_boxes)
            bw, bh = max(rx1 - rx0, MIN_SIDE), max(ry1 - ry0, MIN_SIDE)
            cx, cy = (rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5
            sw = bw * np.random.uniform(0.5, 1.6)
            sh = bh * np.random.uniform(0.5, 1.6)
            cx += np.random.uniform(-0.7 * bw, 0.7 * bw)
            cy += np.random.uniform(-0.7 * bh, 0.7 * bh)
            x0, y0, x1, y1 = cx - sw / 2, cy - sh / 2, cx + sw / 2, cy + sh / 2
        else:
            if sizes is not None:
                w, h = sizes[np.random.randint(len(sizes))] * np.random.uniform(0.6, 1.5, size=2)
            else:
                w = np.random.uniform(0.05, 0.6) * img_w
                h = np.random.uniform(0.05, 0.6) * img_h
            x0 = np.random.uniform(0, max(img_w - w, 1.0))
            y0 = np.random.uniform(0, max(img_h - h, 1.0))
            x1, y1 = x0 + w, y0 + h

        cand = (float(x0), float(y0), float(x1), float(y1))
        # kırpmak yerine ele: kırpma kutuyu kenara yapıştırıp yapay bir
        # "kenarda hep noise var" sinyali üretir
        if cand[0] < -2 or cand[1] < -2 or cand[2] > img_w + 2 or cand[3] > img_h + 2:
            continue
        if (cand[2] - cand[0]) < MIN_SIDE or (cand[3] - cand[1]) < MIN_SIDE:
            continue
        if any(box_iou(cand, rb) > max_iou for rb in real_boxes):
            continue
        out.append(cand)

    return out


# --- POZİSYONEL KODLAMA ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
    



class Pix2SeqDataset(Dataset):
    """
    HİZALANMIŞ (input, target) çifti döndürür — eğitim döngüsü ARTIK kaydırma yapmaz.

        seq_in  = [BOS, x0, y0, x1, y1, cls, ...,  nx0, ny0, nx1, ny1, NOISE, ...]
        seq_tgt = [     x0, y0, x1, y1, cls, ...,  PAD, PAD, PAD, PAD, NOISE, ...]

    Noise objesinin koordinatları hedefte PAD ("n/a"), yalnızca SINIF tokenı
    öğrenilir. Tek tensör + shift ile bu mümkün değil: aynı pozisyon girdide
    gerçek koordinat, hedefte PAD olmak zorunda.

    Annotasyon formatı serbest (labelme / COCO segmentation / RLE / maske PNG);
    bkz. load_shapes(). Kutu, poligon noktaları augmentasyondan geçtikten SONRA
    min/max ile hesaplanır.
    """

    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 transform=train_transform, is_train=True, permute_prob=0.0,
                 mosaic_prob=0.0,
                 noise_fill_to_max=NOISE_FILL_TO_MAX, noise_ratio=NOISE_RATIO,
                 noise_iou_reject=NOISE_IOU_REJECT):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_objects = max_objects
        self.max_seq_len = 1 + (TOKENS_PER_OBJ * max_objects)   # EOS yok
        self.transform = transform
        # noise/permutation `transform is not None` ile değil is_train ile kontrol
        # edilir (val_transform da None değil — öyle olsaydı val hedefine noise sızardı)
        self.is_train          = is_train
        self.permute_prob      = permute_prob
        self.mosaic_prob       = mosaic_prob
        self.noise_fill_to_max = noise_fill_to_max
        self.noise_ratio       = noise_ratio
        self.noise_iou_reject  = noise_iou_reject

    def __len__(self):
        return len(self.json_files)

    def _read_item(self, idx):
        json_path = os.path.join(self.json_dir, self.json_files[idx])
        with open(json_path, 'r', encoding='utf-8') as f:
            item = json.load(f)

        img_name = item.get("imagePath") or item.get("file_name")
        if img_name:
            img_path = os.path.join(self.img_dir, os.path.basename(img_name))
        else:
            base = self.json_files[idx].replace('.json', '')
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                candidate = os.path.join(self.img_dir, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            else:
                img_path = os.path.join(self.img_dir, base + '.jpg')
        return item, img_path

    def __getitem__(self, idx):
        if self.is_train and self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            image_np, shapes = self._mosaic(idx)
        else:
            item, img_path = self._read_item(idx)
            try:
                image = Image.open(img_path).convert("RGB")
                image, _ = pad_to_square(image)
                image_np = np.array(image)
            except Exception as e:
                print(f"HATA: {img_path} okunamadı. Hata: {e}")
                image_np = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            shapes = load_shapes(item, self.img_dir)

        # yukarıdan aşağı, soldan sağa deterministik sıra
        shapes.sort(key=lambda s: (float(s["points"][:, 1].min()),
                                   float(s["points"][:, 0].min())))

        # ── Augmentation: TÜM poligon noktaları keypoint olarak taşınır ──────
        pt_counts, all_points = [], []
        for sh in shapes:
            pts = sh["points"]
            pt_counts.append(len(pts))
            all_points.extend([(float(p[0]), float(p[1])) for p in pts])

        if self.transform is not None:
            augmented    = self.transform(image=image_np, keypoints=all_points)
            image_tensor = augmented['image']
            new_points   = [(float(p[0]), float(p[1])) for p in augmented['keypoints']]
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            new_points   = all_points

        img_h, img_w = self.img_size[0], self.img_size[1]

        # ── 1. Gerçek kutular (augmentasyondan SONRA zarf) ───────────────────
        records, cursor = [], 0
        for sh, cnt in zip(shapes, pt_counts):
            pts = np.asarray(new_points[cursor:cursor + cnt], dtype=np.float32)
            cursor += cnt
            if len(pts) < 2:
                continue

            x_min = max(0.0, float(pts[:, 0].min())); x_max = min(float(img_w), float(pts[:, 0].max()))
            y_min = max(0.0, float(pts[:, 1].min())); y_max = min(float(img_h), float(pts[:, 1].max()))
            if (x_max - x_min) < MIN_SIDE or (y_max - y_min) < MIN_SIDE:
                continue

            records.append({"box": (x_min, y_min, x_max, y_max),
                            "class_id": LABEL_TO_ID[sh["label"]],
                            "is_noise": False})

        if self.is_train and self.permute_prob > 0 and random.random() < self.permute_prob:
            random.shuffle(records)
        records = records[:self.max_objects]
        n_real = len(records)

        # ── 2. Sentetik noise objeleri — KUYRUĞA eklenir ─────────────────────
        if self.is_train:
            real_boxes = [r["box"] for r in records]
            if self.noise_fill_to_max:
                n_noise = self.max_objects - n_real
            else:
                n_noise = int(round(n_real * self.noise_ratio))
            n_noise = min(n_noise, self.max_objects - n_real)
            if NOISE_MAX_PER_IMG is not None:
                n_noise = min(n_noise, NOISE_MAX_PER_IMG)
            if n_noise > 0:
                for nb in generate_noise_boxes(real_boxes, img_w, img_h, n_noise,
                                               self.noise_iou_reject):
                    records.append({"box": nb, "class_id": NOISE_CLASS_ID, "is_noise": True})

        # ── 3. Hizalanmış input / target dizileri ────────────────────────────
        toks_in, toks_tgt = [], []
        for rec in records:
            x0, y0, x1, y1 = rec["box"]
            q = [quantize(x0, img_w, NUM_BINS), quantize(y0, img_h, NUM_BINS),
                 quantize(x1, img_w, NUM_BINS), quantize(y1, img_h, NUM_BINS)]
            if rec["is_noise"]:
                toks_in.extend(q + [NOISE_CLASS_TOKEN])
                toks_tgt.extend([PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN])
            else:
                cls_tok = NUM_BINS + rec["class_id"]
                toks_in.extend(q + [cls_tok])
                toks_tgt.extend(q + [cls_tok])

        # NOISE_FILL_TO_MAX iken dizi her zaman tam MAX_OBJECTS slot uzunluğunda,
        # dolayısıyla EOS'a gerek yok ve konmuyor: model "dur" kararını EOS ile
        # değil, her slotta noise sınıfını seçerek veriyor. Erken EOS yüzünden
        # nesne kaçırma problemi böylece ortadan kalkıyor.
        L = self.max_seq_len
        tail = [] if self.noise_fill_to_max else [EOS_TOKEN]
        seq_in  = ([BOS_TOKEN] + toks_in)[:L]
        seq_tgt = (toks_tgt + tail)[:L]
        seq_in  = seq_in  + [PAD_TOKEN] * (L - len(seq_in))
        seq_tgt = seq_tgt + [PAD_TOKEN] * (L - len(seq_tgt))

        return (image_tensor,
                torch.tensor(seq_in,  dtype=torch.long),
                torch.tensor(seq_tgt, dtype=torch.long))

    def _mosaic(self, idx):
        """4 görüntüyü 2x2 birleştirir. Returns (image_np, shapes)."""
        h, w    = self.img_size
        indices = [idx] + random.sample(range(len(self.json_files)), 3)

        mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
        out_shapes = []

        tiles = [(0, 0, w // 2, h // 2), (w // 2, 0, w // 2, h // 2),
                 (0, h // 2, w // 2, h // 2), (w // 2, h // 2, w // 2, h // 2)]

        for i, (x_off, y_off, tw, th) in zip(indices, tiles):
            item, img_path = self._read_item(i)
            try:
                img = Image.open(img_path).convert("RGB")
                img, max_dim = pad_to_square(img)
                img = img.resize((tw, th), Image.BILINEAR)
            except Exception:
                img = Image.fromarray(np.full((th, tw, 3), 114, dtype=np.uint8))
                max_dim = max(tw, th)

            mosaic_img[y_off:y_off + th, x_off:x_off + tw] = np.array(img)

            for sh in load_shapes(item, self.img_dir):
                pts = sh["points"].copy()
                pts[:, 0] = pts[:, 0] / max_dim * tw + x_off
                pts[:, 1] = pts[:, 1] / max_dim * th + y_off
                out_shapes.append({"label": sh["label"], "points": pts})

        return mosaic_img, out_shapes


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION KONTROLÜ
# ─────────────────────────────────────────────────────────────────────────────

def decode_seq_boxes(seq, img_w, img_h, keep_noise=True):
    """Token dizisinden kutuları geri çözer -> [(box, label, is_noise)]"""
    toks = seq.tolist() if torch.is_tensor(seq) else list(seq)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    out = []
    for i in range(0, (len(toks) // TOKENS_PER_OBJ) * TOKENS_PER_OBJ, TOKENS_PER_OBJ):
        c = toks[i:i + TOKENS_PER_OBJ]
        cls_tok = c[4]
        if cls_tok in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break
        is_noise = cls_tok == NOISE_CLASS_TOKEN
        if not is_noise and not (NUM_BINS <= cls_tok < NUM_BINS + NUM_CLASSES):
            break
        if any(not (0 <= t < NUM_BINS) for t in c[:4]):
            continue                       # hedef dizisinde noise: koordinatlar PAD
        if is_noise and not keep_noise:
            continue
        box = [c[0] / (NUM_BINS - 1) * img_w, c[1] / (NUM_BINS - 1) * img_h,
               c[2] / (NUM_BINS - 1) * img_w, c[3] / (NUM_BINS - 1) * img_h]
        label = "noise" if is_noise else ID_TO_LABEL[cls_tok - NUM_BINS]
        out.append((box, label, is_noise))
    return out


def verify_sequence(seq_in, seq_tgt, expected_gt=None):
    """Bozuk örnekte AssertionError atar."""
    ti = seq_in.tolist() if torch.is_tensor(seq_in) else list(seq_in)
    tt = seq_tgt.tolist() if torch.is_tensor(seq_tgt) else list(seq_tgt)
    assert ti[0] == BOS_TOKEN and len(ti) == len(tt)

    body_in, body_tgt = ti[1:], tt[:-1]
    n_gt = n_noise = 0
    seen_noise = False
    for i in range(0, (len(body_in) // TOKENS_PER_OBJ) * TOKENS_PER_OBJ, TOKENS_PER_OBJ):
        cin, ctgt = body_in[i:i + 5], body_tgt[i:i + 5]
        if cin[4] == PAD_TOKEN:
            assert all(t in (PAD_TOKEN, EOS_TOKEN) for t in ctgt), "PAD slotunda hedef kirli"
            continue
        if cin[4] == NOISE_CLASS_TOKEN:
            n_noise += 1; seen_noise = True
            assert ctgt == [PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN], f"noise hedefi bozuk: {ctgt}"
            assert all(0 <= t < NUM_BINS for t in cin[:4]), "noise girdisi gerçek bin olmalı"
        else:
            n_gt += 1
            assert not seen_noise, "noise gerçek nesnelerin arasına girmiş (kuyrukta olmalı)"
            assert cin == ctgt, "gerçek nesnede input/target hizası bozuk"
    if expected_gt is not None:
        assert n_gt == expected_gt
    return n_gt, n_noise


def _dashed_rect(d, box, fill, width=2, dash=9):
    x0, y0, x1, y1 = box
    for (ax, ay, bx, by) in [(x0, y0, x1, y0), (x1, y0, x1, y1),
                             (x1, y1, x0, y1), (x0, y1, x0, y0)]:
        seg = math.hypot(bx - ax, by - ay)
        n = max(int(seg // dash), 1)
        for k in range(0, n, 2):
            t0, t1 = k / n, min((k + 1) / n, 1.0)
            d.line([ax + (bx - ax) * t0, ay + (by - ay) * t0,
                    ax + (bx - ax) * t1, ay + (by - ay) * t1], fill=fill, width=width)


def visualize_augmentations(dataset, out_dir="aug_check", n=20, verify=True):
    """
    Yeşil dolu çizgi : gerçek GT kutusu
    Kırmızı kesikli  : sentetik noise kutusu
    Çizim, dataset'in döndürdüğü token dizisinden geri çözülerek yapılır —
    yani modelin gerçekten gördüğü şey görselleşir.
    """
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    tot_gt = tot_noise = 0
    ious = []

    for k in range(n):
        idx = k % len(dataset)
        img_t, seq_in, seq_tgt = dataset[idx]
        img = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        W, H = img.size

        inst = decode_seq_boxes(seq_in, W, H, keep_noise=True)
        reals = [b for b, _l, nz in inst if not nz]
        n_gt = n_noise = 0

        for box, label, is_noise in inst:
            if is_noise:
                n_noise += 1
                best = max((box_iou(tuple(box), tuple(r)) for r in reals), default=0.0)
                ious.append(best)
                _dashed_rect(d, box, "#ff2d2d", width=2)
                d.text((box[0] + 4, max(0, box[1] - 12)),
                       "noise" if best < 0.25 else f"noise {best:.2f}", fill="#ff2d2d")
            else:
                n_gt += 1
                d.rectangle(box, outline="#48f90a", width=3)
                d.text((box[0] + 4, max(0, box[1] - 12)), label, fill="#48f90a")

        if verify:
            verify_sequence(seq_in, seq_tgt, expected_gt=n_gt)

        d.rectangle([0, 0, W, 18], fill="#141414")
        d.text((5, 4), f"GT: {n_gt}", fill="#48f90a")
        d.text((70, 4), f"noise: {n_noise}", fill="#ff2d2d")
        d.text((165, 4), f"slot: {n_gt + n_noise}/{dataset.max_objects}",
               fill="#bbbbbb" if n_gt + n_noise == dataset.max_objects else "#ffb21d")

        stem = os.path.splitext(dataset.json_files[idx])[0][:40]
        img.save(os.path.join(out_dir, f"aug_check_{k:02d}_{stem}_gt{n_gt}_noise{n_noise}.jpg"),
                 quality=92)
        tot_gt += n_gt; tot_noise += n_noise

    print(f"{n} görsel kaydedildi -> {out_dir}/")
    print(f"  gerçek kutu: {tot_gt} (ort. {tot_gt/max(n,1):.2f}) | "
          f"noise: {tot_noise} (ort. {tot_noise/max(n,1):.2f})")
    if ious:
        a = np.asarray(ious)
        print(f"  noise-gerçek IoU: ort {a.mean():.3f} | p95 {np.percentile(a,95):.3f} | "
              f"max {a.max():.3f} (eşik {NOISE_IOU_REJECT})")
    if verify:
        print("  dizi kontrolü: OK (noise kuyrukta, hedefte koordinatlar PAD, hiza doğru)")


def dataset_object_stats(dataset, n=None):
    """MAX_OBJECTS'i doğru boyutlamak için nesne sayısı dağılımı."""
    n = n or len(dataset)
    counts = []
    for i in range(min(n, len(dataset))):
        item, _ = dataset._read_item(i)
        counts.append(len(load_shapes(item, dataset.img_dir)))
    c = np.asarray(counts)
    print(f"Nesne/görüntü — ort {c.mean():.2f} | medyan {np.median(c):.0f} | "
          f"p99 {np.percentile(c, 99):.0f} | max {c.max()} | MAX_OBJECTS={MAX_OBJECTS}")
    if MAX_OBJECTS < c.max():
        print(f"  [uyarı] MAX_OBJECTS ({MAX_OBJECTS}) < max nesne ({c.max()}). "
              "EOS kaldırıldığı için bu sert bir recall tavanı demek.")
    return c

# --- MODEL ---
class Pix2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, nheads=8, num_layers=4, max_seq_len=200):
        super().__init__()
        # Özellik Çıkarıcı (Encoder)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])     
        self.enc_proj = nn.Conv2d(512, hidden_dim, kernel_size=1) #resnet18
        #self.enc_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
        # Resim özellikleri için konum kodlaması
        grid_h = IMG_SIZE[0] // 32
        grid_w = IMG_SIZE[1] // 32
        self.pos_emb = nn.Parameter(torch.randn(1, grid_h * grid_w, hidden_dim)) 
        
        # Hedef Dizi (Sequence) Modellemesi
        #self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(0.1)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, tgt_seq):
        features = self.encoder(images) 
        features = self.enc_proj(features) 
        
        batch_size = features.size(0)
        #memory = features.view(batch_size, -1, features.size(2) * features.size(3)).permute(0, 2, 1)
        memory = features.flatten(2).permute(0, 2, 1)
        memory = memory + self.pos_emb 
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(images.device)
        
        tgt_emb = self.embedding(tgt_seq) 
        tgt_emb = self.seq_pos_encoding(tgt_emb)
        tgt_emb = self.emb_dropout(tgt_emb)
        
        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim {device} üzerinde başlıyor...")
    USE_BF16 = True
    amp_enabled = USE_BF16 and device.type == "cuda"
    amp_dtype   = torch.bfloat16 if amp_enabled else torch.float32  

    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    
    if os.path.exists(VAL_SPLIT_PATH):
        print(f"Mevcut val split yükleniyor: {VAL_SPLIT_PATH}")
        with open(VAL_SPLIT_PATH) as f:
            saved = json.load(f)
        val_files_set = set(saved["filenames"])
        train_files   = [f for f in all_json_files if f not in val_files_set]
        val_files     = [f for f in all_json_files if f in val_files_set]
    else:
        print(f"Val split bulunamadı, yeni oluşturuluyor → {VAL_SPLIT_PATH}")
        random.shuffle(all_json_files)
        train_size    = int(0.9 * len(all_json_files))
        
        train_files   = all_json_files[:train_size]
        val_files     = all_json_files[train_size:]
        
        with open(VAL_SPLIT_PATH, "w") as f:
            json.dump({"filenames": val_files}, f, indent=2)
        print(f"Val split kaydedildi ({len(val_files)} dosya)")

    train_dataset            = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform,
                                              is_train=True)
    train_dataset.json_files = train_files

    # is_train=False -> val hedeflerinde noise YOK
    val_dataset              = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform,
                                              is_train=False)
    val_dataset.json_files   = val_files

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=2)
    val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    max_seq_len = train_dataset.max_seq_len
 
    model     = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)
 
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),         'lr': 3e-5},
        {'params': model.enc_proj.parameters(),        'lr': LEARNING_RATE},
        {'params': [model.pos_emb],                    'lr': LEARNING_RATE},
        {'params': model.embedding.parameters(),       'lr': LEARNING_RATE},
        {'params': model.seq_pos_encoding.parameters(),'lr': LEARNING_RATE},
        {'params': model.decoder.parameters(),         'lr': LEARNING_RATE},
        {'params': model.fc_out.parameters(),          'lr': LEARNING_RATE},
    ], weight_decay=1e-4)
 
    # FIX: total_steps based on train_dataloader, not the old dataloader
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[3e-5, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE,
                LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.05
    )

    dataset_object_stats(train_dataset)
    visualize_augmentations(train_dataset, out_dir="aug_check", n=20)

    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)
 
    best_val_loss = float('inf')
 
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
    best_map = 0.0

    from bbox_eval_utils_pix2seq import evaluate 
    EVAL_EVERY = 2 

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
 
        train_bar = tqdm(train_dataloader, desc=f"  Train {epoch+1}/{EPOCHS}",
                         leave=False, unit="batch")
 
        # dataset zaten hizalı (input, target) veriyor — KAYDIRMA YOK
        for images, decoder_input, decoder_target in train_bar:
            images         = images.to(device).float()
            decoder_input  = decoder_input.to(device)
            decoder_target = decoder_target.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(images, decoder_input)
                loss   = criterion(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
 
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
 
        avg_train_loss = train_loss / len(train_dataloader)
 
        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
 
        val_bar = tqdm(val_dataloader, desc=f"  Val   {epoch+1}/{EPOCHS}",
                       leave=False, unit="batch")
 
        with torch.no_grad():
            for images, decoder_input, decoder_target in val_bar:
                images         = images.to(device).float()
                decoder_input  = decoder_input.to(device)
                decoder_target = decoder_target.to(device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(images, decoder_input)
                    loss   = criterion(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))
 
                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")
 
        avg_val_loss = val_loss / len(val_dataloader)
 
        # ── Logging ────────────────────────────────────────────────────────
        epoch_bar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            best=f"{best_val_loss:.4f}"
        )
 
        # ── Checkpoint ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pix2seq_best.pth")
            tqdm.write(f"  ✓ Epoch {epoch+1:3d} — new best val loss: {avg_val_loss:.4f} → saved pix2seq_best.pth")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch > EPOCHS - 20:
            mAP, mAP50 = evaluate(model, val_dataloader, max_seq_len, device)
            tqdm.write(f"  epoch {epoch+1:3d} | val_loss {avg_val_loss:.4f} "
                    f"| mAP {mAP:.4f} | mAP@50 {mAP50:.4f}")
            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), "pix2seq_best_map.pth")
 
        torch.save(model.state_dict(), "pix2seq_model_v2.pth")
    print(f"\nEğitim tamamlandı.")
    print(f"  Son model  : pix2seq_model_v2.pth")
    print(f"  En iyi model: pix2seq_best.pth  (val loss: {best_val_loss:.4f})")