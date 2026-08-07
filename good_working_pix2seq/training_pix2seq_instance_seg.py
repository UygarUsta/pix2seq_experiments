import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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
JSON_DIR       = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/"
IMG_DIR        = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/"
NUM_BINS       = 500
MAX_OBJECTS    = 10
BATCH_SIZE     = 16
EPOCHS         = 350
LEARNING_RATE  = 3e-4
IMG_SIZE       = (512, 512)       # (H, W)
VAL_SPLIT_PATH = "val_split.json"

LABEL_TO_ID = {
    "card": 0
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_CLASSES = len(LABEL_TO_ID)
VOCAB_SIZE  = NUM_BINS + NUM_CLASSES + 3
BOS_TOKEN   = VOCAB_SIZE - 3
EOS_TOKEN   = VOCAB_SIZE - 2
PAD_TOKEN   = VOCAB_SIZE - 1

# --- INSTANCE SEGMENTATION TOKEN ŞEMASI ---
# Nesne başına: [x_min, y_min, x_max, y_max, class, px0, py0, px1, py1, ... pxK-1, pyK-1]
# Kutu önce geliyor: coarse-to-fine sinyal + bbox mAP'i kaybetmiyoruz.
NUM_POLY_PTS   = 16
TOKENS_PER_OBJ = 5 + 2 * NUM_POLY_PTS       # 37
MAX_SEQ_LEN    = 1 + TOKENS_PER_OBJ * MAX_OBJECTS + 1

# Geometri filtreleri (augmentation sonrası, model uzayında piksel cinsinden)
MIN_SIDE      = 2.0
MIN_AREA      = 16.0
MIN_AREA_KEEP = 0.15        # kırpma sonrası alanın en az bu oranı kalmalı


# ─────────────────────────────────────────────────────────────────────────────
# POLİGON YARDIMCILARI
# ─────────────────────────────────────────────────────────────────────────────

def polygon_area(poly):
    """İşaretli alan (shoelace). İşaret dönüş yönünü verir."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def clip_polygon(poly, w, h):
    """
    Sutherland-Hodgman ile poligonu [0,w]x[0,h] dikdörtgenine kırpar.
    Nokta bazlı clamp'ten farklı olarak maskeyi bozmaz.
    """
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
    """
    AR decoder'a deterministik hedef vermek için:
      1) dönüş yönünü sabitle (işaretli alan > 0),
      2) başlangıcı en üstteki (eşitlikte en soldaki) köşeye kaydır.
    Bu iki adım olmadan aynı maske için birden çok "doğru" sekans olur ve model yakınsamaz.
    """
    poly = np.asarray(poly, dtype=np.float32)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    start = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])   # önce y, eşitlikte x
    return np.roll(poly, -start, axis=0)


def resample_polygon(poly, K):
    """Kontur üzerinde yay-uzunluğuna göre eşit aralıklı K nokta. poly[0]'dan başlar."""
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) < 3:
        return None
    closed = np.vstack([poly, poly[:1]])
    seg    = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum    = np.concatenate([[0.0], np.cumsum(seg)])
    total  = float(cum[-1])
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
    normalized = coordinate / max_size
    normalized = max(0.0, min(1.0, normalized))
    return int(normalized * (num_bins - 1))


def dequantize(token, max_size, num_bins):
    return (token / (num_bins - 1)) * max_size


def seq_to_instances(tokens, tok_scores=None):
    """
    Token listesini model uzayındaki (IMG_SIZE) instance'lara çevirir.
    Dönen: boxes [N,4] xyxy, polys [N,K,2], labels [N], confs [N]
    """
    img_h, img_w = IMG_SIZE
    boxes, polys, labels, confs = [], [], [], []

    n = len(tokens) - (len(tokens) % TOKENS_PER_OBJ)
    for i in range(0, n, TOKENS_PER_OBJ):
        t = tokens[i:i + TOKENS_PER_OBJ]

        if not all(0 <= v < NUM_BINS for v in t[:4]):
            continue
        if not (NUM_BINS <= t[4] < NUM_BINS + NUM_CLASSES):
            continue
        coords = t[5:]
        if not all(0 <= v < NUM_BINS for v in coords):
            continue

        x0 = dequantize(t[0], img_w, NUM_BINS)
        y0 = dequantize(t[1], img_h, NUM_BINS)
        x1 = dequantize(t[2], img_w, NUM_BINS)
        y1 = dequantize(t[3], img_h, NUM_BINS)
        if x1 <= x0 or y1 <= y0:
            continue

        poly = np.array(
            [[dequantize(coords[k], img_w, NUM_BINS),
              dequantize(coords[k + 1], img_h, NUM_BINS)]
             for k in range(0, 2 * NUM_POLY_PTS, 2)],
            dtype=np.float32,
        )
        if abs(polygon_area(poly)) < MIN_AREA:
            continue

        boxes.append([x0, y0, x1, y1])
        polys.append(poly)
        labels.append(t[4] - NUM_BINS)

        if tok_scores is None:
            confs.append(1.0)
        else:
            # DİKKAT: 37 tokenın çarpımı sayısal olarak çöker (0.9^37 ≈ 0.02) ve
            # mAP sıralamasını bozar. Skoru sadece kutu+sınıf tokenlarından alıyoruz.
            confs.append(float(np.prod(tok_scores[i:i + 5])))

    return boxes, polys, labels, confs


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# DEĞİŞENLER (bbox sürümüne göre):
#   * Artık shape başına SABİT 4 değil, DEĞİŞKEN sayıda orijinal köşe geçiyor.
#   * remove_invisible=False zorunlu — nokta sayısı korunmazsa shape eşleşmesi kayar.
#   * Kanonikleştirme + resample augmentation SONRASINA taşındı (aşağıdaki nota bak).
#   * Kadraj dışına taşan poligonlar nokta bazlı clamp yerine gerçek kırpma ile ele alınıyor.
# Geometrik/fotometrik operasyonların kendisi bilerek DEĞİŞTİRİLMEDİ ki mevcut
# bbox baseline'ınızla (mAP ~0.51-0.53) karşılaştırma anlamlı kalsın.

train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
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
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def pad_to_square(image, fill_color=(128, 128, 128)):
    """Resmin en-boy oranını bozmadan, en uzun kenarı baz alarak kareye tamamlar."""
    w, h = image.size
    max_dim = max(w, h)
    new_image = Image.new("RGB", (max_dim, max_dim), fill_color)
    new_image.paste(image, (0, 0))
    return new_image, max_dim


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
    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS,
                 img_size=IMG_SIZE, transform=train_transform):
        self.json_dir    = json_dir
        self.img_dir     = img_dir
        self.img_size    = img_size
        self.max_objects = max_objects
        self.json_files  = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_seq_len = 1 + (TOKENS_PER_OBJ * max_objects) + 1
        self.transform   = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):

        # ── Mosaic (şu an kapalı) ─────────────────────────────────────────────
        if self.transform is not None and random.random() < 0.0:   # < 0.3
            image_np, all_points, pt_counts, valid_shapes = self._mosaic(idx)
        else:
            # ── Normal loading ────────────────────────────────────────────────
            json_path = os.path.join(self.json_dir, self.json_files[idx])
            with open(json_path, 'r', encoding='utf-8') as f:
                item = json.load(f)

            img_name = item.get("imagePath")
            if img_name:
                img_path = os.path.join(self.img_dir, img_name)
            else:
                base = self.json_files[idx].replace('.json', '')
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    candidate = os.path.join(self.img_dir, base + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                else:
                    img_path = os.path.join(self.img_dir, base + '.jpg')

            try:
                image = Image.open(img_path).convert("RGB")
                image, _ = pad_to_square(image)
                image_np = np.array(image)
            except Exception as e:
                print(f"HATA: {img_path} okunamadı. Hata: {e}")
                image_np = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            shapes = item.get("shapes", [])

            # DEĞİŞTİ: len(points) == 4 filtresi kalktı, >= 3 oldu.
            valid_shapes = [s for s in shapes
                            if s.get("label") in LABEL_TO_ID
                            and len(s.get("points", [])) >= 3]

            # DEĞİŞTİ: sıralama artık augmentation SONRASI koordinatlara göre yapılıyor
            # (aşağıda), çünkü rotate/perspective okuma sırasını değiştirebiliyor.
            all_points, pt_counts = [], []
            for shape in valid_shapes:
                pts = [(float(p[0]), float(p[1])) for p in shape["points"]]
                all_points.extend(pts)
                pt_counts.append(len(pts))

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.transform is not None:
            augmented    = self.transform(image=image_np, keypoints=all_points)
            image_tensor = augmented['image']
            new_points   = [(float(p[0]), float(p[1])) for p in augmented['keypoints']]
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            new_points   = all_points

        img_h, img_w = self.img_size[0], self.img_size[1]

        # ── Poligon post-processing ───────────────────────────────────────────
        # Sıra ÖNEMLİ: kırp → geçerlilik → kanonikleştir → resample.
        # Kanonikleştirme augmentation sonrasında yapılmalı; aksi halde 15° dönmüş
        # bir kartta "başlangıç noktası" görüntüden belirlenemez ve hedef belirsizleşir.
        records = []
        cursor  = 0
        for shape, cnt in zip(valid_shapes, pt_counts):
            poly = np.asarray(new_points[cursor:cursor + cnt], dtype=np.float32)
            cursor += cnt
            if len(poly) < 3:
                continue

            area_before = abs(polygon_area(poly))
            poly = clip_polygon(poly, float(img_w), float(img_h))
            if len(poly) < 3:
                continue

            area_after = abs(polygon_area(poly))
            if area_after < MIN_AREA or area_after < MIN_AREA_KEEP * max(area_before, 1e-6):
                continue

            xs, ys = poly[:, 0], poly[:, 1]
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            if (x_max - x_min) < MIN_SIDE or (y_max - y_min) < MIN_SIDE:
                continue

            poly = canonicalize_polygon(poly)
            poly = resample_polygon(poly, NUM_POLY_PTS)
            if poly is None:
                continue

            records.append({
                "class_id": LABEL_TO_ID[shape["label"]],
                "box":      (x_min, y_min, x_max, y_max),
                "poly":     poly,
            })

        # okuma sırası: yukarıdan aşağı, sonra soldan sağa (transform sonrası koordinatlarda)
        records.sort(key=lambda r: (r["box"][1], r["box"][0]))
        records = records[:self.max_objects]

        # ── Tokenizasyon ──────────────────────────────────────────────────────
        sequence = [BOS_TOKEN]
        for rec in records:
            x_min, y_min, x_max, y_max = rec["box"]
            sequence.extend([
                quantize(x_min, img_w, NUM_BINS),
                quantize(y_min, img_h, NUM_BINS),
                quantize(x_max, img_w, NUM_BINS),
                quantize(y_max, img_h, NUM_BINS),
                NUM_BINS + rec["class_id"],
            ])
            for px, py in rec["poly"]:
                sequence.append(quantize(float(px), img_w, NUM_BINS))
                sequence.append(quantize(float(py), img_h, NUM_BINS))

        sequence.append(EOS_TOKEN)

        if len(sequence) < self.max_seq_len:
            sequence.extend([PAD_TOKEN] * (self.max_seq_len - len(sequence)))
        else:
            sequence = sequence[:self.max_seq_len - 1] + [EOS_TOKEN]

        return image_tensor, torch.tensor(sequence, dtype=torch.long)

    def _mosaic(self, idx):
        """4 resmi 2x2 mozaikte birleştirir. (image_np, all_points, pt_counts, valid_shapes)."""
        h, w    = self.img_size
        indices = [idx] + random.sample(range(len(self.json_files)), 3)

        mosaic_img   = np.full((h, w, 3), 114, dtype=np.uint8)
        all_points   = []
        pt_counts    = []
        valid_shapes = []

        tiles = [
            (0,    0,    w // 2, h // 2),
            (w // 2, 0,  w // 2, h // 2),
            (0,    h // 2, w // 2, h // 2),
            (w // 2, h // 2, w // 2, h // 2),
        ]

        for i, (x_off, y_off, tw, th) in zip(indices, tiles):
            json_path = os.path.join(self.json_dir, self.json_files[i])
            with open(json_path, 'r', encoding='utf-8') as f:
                item = json.load(f)

            img_name = item.get("imagePath", self.json_files[i].replace('.json', '.jpg'))
            try:
                img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
                img, max_dim = pad_to_square(img)
                img = img.resize((tw, th), Image.BILINEAR)
            except Exception:
                img = Image.fromarray(np.full((th, tw, 3), 114, dtype=np.uint8))
                max_dim = max(tw, th)

            mosaic_img[y_off:y_off + th, x_off:x_off + tw] = np.array(img)

            for shape in item.get("shapes", []):
                if shape.get("label") in LABEL_TO_ID and len(shape.get("points", [])) >= 3:
                    valid_shapes.append(shape)
                    pts = shape["points"]
                    pt_counts.append(len(pts))
                    for px, py in pts:
                        sx = (px / max_dim) * tw + x_off
                        sy = (py / max_dim) * th + y_off
                        all_points.append((sx, sy))

        return mosaic_img, all_points, pt_counts, valid_shapes


# --- MODEL ---
class Pix2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, nheads=8, num_layers=4, max_seq_len=MAX_SEQ_LEN):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder  = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        grid_h = IMG_SIZE[0] // 32
        grid_w = IMG_SIZE[1] // 32
        self.pos_emb = nn.Parameter(torch.randn(1, grid_h * grid_w, hidden_dim))

        self.embedding        = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout      = nn.Dropout(0.1)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out  = nn.Linear(hidden_dim, vocab_size)

    def encode(self, images):
        features = self.enc_proj(self.encoder(images))
        memory   = features.flatten(2).permute(0, 2, 1)
        return memory + self.pos_emb

    def forward(self, images, tgt_seq):
        memory   = self.encode(images)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(images.device)

        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.seq_pos_encoding(tgt_emb)
        tgt_emb = self.emb_dropout(tgt_emb)

        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE'Lİ AUTOREGRESSIVE DECODE
# ─────────────────────────────────────────────────────────────────────────────
# nn.TransformerDecoder incremental decode desteklemiyor: her adımda tüm sekansı
# baştan işliyor → O(L^2) katman çağrısı. L=52'de sorun değildi, L=372'de eval
# süresini ~50x şişiriyor. Aşağıdaki sınıf AYNI ağırlıkları kullanarak adım adım
# decode eder; state_dict formatı değişmez, ayrı eğitim gerekmez.

class KVCacheDecoder:
    """nn.TransformerDecoder ağırlıkları üzerinde incremental (cache'li) decode."""

    def __init__(self, model, memory):
        self.layers = model.decoder.layers
        self.norm   = model.decoder.norm
        self.memory = memory

        self.self_k = [None] * len(self.layers)
        self.self_v = [None] * len(self.layers)
        self.cross_k, self.cross_v = [], []

        for layer in self.layers:
            attn = layer.multihead_attn
            D    = attn.embed_dim
            W, b = attn.in_proj_weight, attn.in_proj_bias
            bk   = None if b is None else b[D:2 * D]
            bv   = None if b is None else b[2 * D:]
            k = F.linear(memory, W[D:2 * D], bk)
            v = F.linear(memory, W[2 * D:],  bv)
            self.cross_k.append(self._split_heads(k, attn.num_heads))
            self.cross_v.append(self._split_heads(v, attn.num_heads))

    @staticmethod
    def _split_heads(x, nheads):
        B, L, D = x.shape
        return x.view(B, L, nheads, D // nheads).transpose(1, 2)     # [B, H, L, hd]

    @staticmethod
    def _merge_heads(x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).reshape(B, L, H * hd)

    def _self_attn_step(self, layer, i, x_t):
        attn = layer.self_attn
        D    = attn.embed_dim
        qkv  = F.linear(x_t, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.split(D, dim=-1)

        q = self._split_heads(q, attn.num_heads)
        k = self._split_heads(k, attn.num_heads)
        v = self._split_heads(v, attn.num_heads)

        # cache yalnızca geçmişi tuttuğu için causal mask'e gerek yok
        self.self_k[i] = k if self.self_k[i] is None else torch.cat([self.self_k[i], k], dim=2)
        self.self_v[i] = v if self.self_v[i] is None else torch.cat([self.self_v[i], v], dim=2)

        o = F.scaled_dot_product_attention(q, self.self_k[i], self.self_v[i])
        return attn.out_proj(self._merge_heads(o))

    def _cross_attn_step(self, layer, i, x_t):
        attn = layer.multihead_attn
        D    = attn.embed_dim
        bq   = None if attn.in_proj_bias is None else attn.in_proj_bias[:D]
        q    = F.linear(x_t, attn.in_proj_weight[:D], bq)
        q    = self._split_heads(q, attn.num_heads)

        o = F.scaled_dot_product_attention(q, self.cross_k[i], self.cross_v[i])
        return attn.out_proj(self._merge_heads(o))

    @staticmethod
    def _ff(layer, x):
        act = layer.activation
        if isinstance(act, str):
            act = F.relu if act == "relu" else F.gelu
        return layer.linear2(layer.dropout(act(layer.linear1(x))))

    def step(self, x_t):
        """x_t: [B, 1, D] → [B, 1, D]"""
        for i, layer in enumerate(self.layers):
            if getattr(layer, "norm_first", False):
                x_t = x_t + layer.dropout1(self._self_attn_step(layer, i, layer.norm1(x_t)))
                x_t = x_t + layer.dropout2(self._cross_attn_step(layer, i, layer.norm2(x_t)))
                x_t = x_t + layer.dropout3(self._ff(layer, layer.norm3(x_t)))
            else:
                x_t = layer.norm1(x_t + layer.dropout1(self._self_attn_step(layer, i, x_t)))
                x_t = layer.norm2(x_t + layer.dropout2(self._cross_attn_step(layer, i, x_t)))
                x_t = layer.norm3(x_t + layer.dropout3(self._ff(layer, x_t)))
        return x_t if self.norm is None else self.norm(x_t)


@torch.no_grad()
def ar_decode_batch(model, images, max_seq_len=MAX_SEQ_LEN, device=None, use_cache=True):
    """
    Batch greedy AR decode. Encoder bir kez çalışır.
    use_cache=True → KV cache (L=372 için ~10-15x hızlı).
    Dönen: (tokens [B, L-1], scores [B, L-1])
    """
    model.eval()
    device = device or next(model.parameters()).device
    B      = images.size(0)

    memory = model.encode(images)
    cache  = KVCacheDecoder(model, memory) if use_cache else None

    seq      = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    scores   = torch.zeros(B, max_seq_len, device=device)
    pe       = model.seq_pos_encoding.pe

    for step in range(max_seq_len - 1):
        if use_cache:
            emb    = model.embedding(seq[:, -1:]) + pe[:, step:step + 1, :]
            out    = cache.step(model.emb_dropout(emb))
            logits = model.fc_out(out[:, -1, :]).float()
        else:
            tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(seq)))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
            out      = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
            logits   = model.fc_out(out[:, -1, :]).float()

        probs = torch.softmax(logits, -1)
        nxt   = probs.argmax(-1)
        scores[:, step] = probs.gather(1, nxt.unsqueeze(1)).squeeze(1)

        nxt = torch.where(finished, torch.full_like(nxt, PAD_TOKEN), nxt)
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)

        finished |= (nxt == EOS_TOKEN)
        if finished.all():
            break

    return seq[:, 1:], scores[:, :seq.size(1) - 1]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim {device} üzerinde başlıyor...")
    print(f"Token şeması: {NUM_POLY_PTS} poligon noktası → {TOKENS_PER_OBJ} token/nesne, "
          f"max_seq_len={MAX_SEQ_LEN}")

    USE_BF16    = True
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
        train_size  = int(0.9 * len(all_json_files))
        train_files = all_json_files[:train_size]
        val_files   = all_json_files[train_size:]
        with open(VAL_SPLIT_PATH, "w") as f:
            json.dump({"filenames": val_files}, f, indent=2)
        print(f"Val split kaydedildi ({len(val_files)} dosya)")

    train_dataset            = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform)
    train_dataset.json_files = train_files

    val_dataset            = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform)
    val_dataset.json_files = val_files

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=8, pin_memory=True, prefetch_factor=2)
    val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    max_seq_len = train_dataset.max_seq_len

    model     = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),          'lr': 3e-5},
        {'params': model.enc_proj.parameters(),         'lr': LEARNING_RATE},
        {'params': [model.pos_emb],                     'lr': LEARNING_RATE},
        {'params': model.embedding.parameters(),        'lr': LEARNING_RATE},
        {'params': model.seq_pos_encoding.parameters(), 'lr': LEARNING_RATE},
        {'params': model.decoder.parameters(),          'lr': LEARNING_RATE},
        {'params': model.fc_out.parameters(),           'lr': LEARNING_RATE},
    ], weight_decay=1e-4)

    total_steps = len(train_dataloader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[3e-5, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE,
                LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.05
    )

    # ── Augmentation sanity check: poligonları çiz ────────────────────────────
    os.makedirs("aug_check", exist_ok=True)
    ds   = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])

    for k in range(20):
        img_t, seq = ds[k % len(ds)]
        img = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d   = ImageDraw.Draw(img)

        toks = [t for t in seq.tolist() if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
        boxes, polys, labels, _ = seq_to_instances(toks)
        for box, poly in zip(boxes, polys):
            d.rectangle(box, outline="cyan", width=2)
            d.polygon([float(c) for p in poly for c in p], outline="lime")
            # ilk nokta kanonik başlangıç — tutarlı görünmeli
            d.ellipse([poly[0][0] - 4, poly[0][1] - 4, poly[0][0] + 4, poly[0][1] + 4],
                      fill="red")
        img.save(f"aug_check/aug_check_{k}.jpg")

    best_val_loss = float('inf')
    best_map      = 0.0

    from instance_seg_eval_utils_pix2seq import evaluate
    EVAL_EVERY = 4      # sekans 7x uzadı, eval maliyeti arttı → daha seyrek

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_dataloader, desc=f"  Train {epoch+1}/{EPOCHS}",
                         leave=False, unit="batch")

        for images, targets in train_bar:
            images  = images.to(device).float()
            targets = targets.to(device)

            decoder_input  = targets[:, :-1]
            decoder_target = targets[:, 1:]

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

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        val_bar = tqdm(val_dataloader, desc=f"  Val   {epoch+1}/{EPOCHS}",
                       leave=False, unit="batch")

        with torch.no_grad():
            for images, targets in val_bar:
                images  = images.to(device).float()
                targets = targets.to(device)

                decoder_input  = targets[:, :-1]
                decoder_target = targets[:, 1:]

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    logits = model(images, decoder_input)
                    loss   = criterion(logits.reshape(-1, VOCAB_SIZE), decoder_target.reshape(-1))

                val_loss += loss.item()
                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_dataloader)

        epoch_bar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            best=f"{best_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pix2seq_instseg_best.pth")
            tqdm.write(f"  ✓ Epoch {epoch+1:3d} — new best val loss: {avg_val_loss:.4f} "
                       f"→ saved pix2seq_instseg_best.pth")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch > EPOCHS - 20:
            res = evaluate(model, val_dataloader, max_seq_len, device)
            tqdm.write(f"  epoch {epoch+1:3d} | val_loss {avg_val_loss:.4f} "
                       f"| segm mAP {res['segm_map']:.4f} | segm mAP@50 {res['segm_map_50']:.4f} "
                       f"| bbox mAP {res['bbox_map']:.4f}")
            if res["segm_map"] > best_map:
                best_map = res["segm_map"]
                torch.save(model.state_dict(), "pix2seq_instseg_best_map.pth")

        torch.save(model.state_dict(), "pix2seq_instseg_last.pth")

    print(f"\nEğitim tamamlandı.")
    print(f"  Son model      : pix2seq_instseg_last.pth")
    print(f"  En iyi val loss: pix2seq_instseg_best.pth  ({best_val_loss:.4f})")
    print(f"  En iyi segm mAP: pix2seq_instseg_best_map.pth ({best_map:.4f})")
