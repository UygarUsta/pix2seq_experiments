import os
import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm



# --- CONFIGURATION ---
JSON_DIR       = "/mnt/d/Datasets/person.v2i.coco-segmentation/train"
IMG_DIR        = "/mnt/d/Datasets/person.v2i.coco-segmentation/train"
NUM_BINS       = 500
MAX_OBJECTS    = 10
BATCH_SIZE     = 16
EPOCHS         = 350
LEARNING_RATE  = 3e-4
IMG_SIZE       = (512, 512)
VAL_SPLIT_PATH = "val_split.json"

LABEL_TO_ID = {"person": 0}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)                   # real classes only

# ── Sequence augmentation ────────────────────────────────────────────────────
NOISE_CLASS_ID     = NUM_CLASSES                 # 1
NOISE_CLASS_TOKEN  = NUM_BINS + NOISE_CLASS_ID   # 501
NUM_CLASS_SLOTS    = NUM_CLASSES + 1             # real classes + noise
NOISE_FILL_TO_MAX  = True    # fill the tail out to MAX_OBJECTS with noise objects
NOISE_RATIO        = 0.4     # only used when NOISE_FILL_TO_MAX is False
NOISE_IOU_REJECT   = 0.5     # reject synthetic boxes landing on a real object
INFER_SCORE_THRESH = 0.05    # keep low: every kept instance costs a decoder-2 rollout

VOCAB_SIZE  = NUM_BINS + NUM_CLASS_SLOTS + 3     # 505
BOS_TOKEN   = VOCAB_SIZE - 3                     # 502
EOS_TOKEN   = VOCAB_SIZE - 2                     # 503 (decoder 2 only)
PAD_TOKEN   = VOCAB_SIZE - 1                     # 504

NUM_POLY_PTS        = 16
BOX_TOKENS_PER_OBJ  = 5
POLY_TOKENS_PER_OBJ = 2 * NUM_POLY_PTS     # 32
# No EOS in the box sequence any more: the tail is filled with noise objects, so
# box decoding is always exactly MAX_OBJECTS * BOX_TOKENS_PER_OBJ steps.
MAX_BOX_SEQ_LEN     = 1 + BOX_TOKENS_PER_OBJ * MAX_OBJECTS              # 51
MAX_MASK_SEQ_LEN    = 1 + BOX_TOKENS_PER_OBJ + POLY_TOKENS_PER_OBJ + 1  # 39

MIN_SIDE      = 2.0
MIN_AREA      = 16.0
MIN_AREA_KEEP = 0.15

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY & TOKEN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def clip_polygon(poly, w, h):
    def _clip_edge(pts, inside_fn, isect_fn):
        if not pts: return []
        out = []
        for i in range(len(pts)):
            cur, prev = pts[i], pts[i - 1]
            c_in, p_in = inside_fn(cur), inside_fn(prev)
            if c_in:
                if not p_in: out.append(isect_fn(prev, cur))
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
        if len(pts) < 3: return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)

def canonicalize_polygon(poly):
    poly = np.asarray(poly, dtype=np.float32)
    if polygon_area(poly) < 0:
        poly = poly[::-1].copy()
    start = int(np.lexsort((poly[:, 0], poly[:, 1]))[0])
    return np.roll(poly, -start, axis=0)

def resample_polygon(poly, K):
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) < 3: return None
    closed = np.vstack([poly, poly[:1]])
    seg    = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum    = np.concatenate([[0.0], np.cumsum(seg)])
    total  = float(cum[-1])
    if total < 1e-6: return None

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

def pad_to_square(image, fill_color=(128, 128, 128)):
    w, h = image.size
    max_dim = max(w, h)
    new_image = Image.new("RGB", (max_dim, max_dim), fill_color)
    new_image.paste(image, (0, 0))
    return new_image, max_dim

train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(std_range=(0.05, 0.1), p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE AUGMENTATION HELPERS
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


def generate_synthetic_boxes(real_boxes, img_w, img_h, count, max_iou=NOISE_IOU_REJECT):
    """
    Type A: jitter a real box (shift AND rescale) -> hard negatives.
    Type B: fully random background box.
    Candidates overlapping a real object above max_iou are rejected, otherwise
    the model would be taught that a true positive is noise.
    """
    synth = []
    attempts = 0
    while len(synth) < count and attempts < max(count, 1) * 20:
        attempts += 1
        if real_boxes and random.random() < 0.6:
            rx0, ry0, rx1, ry1 = random.choice(real_boxes)
            bw = max(rx1 - rx0, MIN_SIDE)
            bh = max(ry1 - ry0, MIN_SIDE)
            cx, cy = (rx0 + rx1) * 0.5, (ry0 + ry1) * 0.5

            sw = bw * np.random.uniform(0.5, 1.6)
            sh = bh * np.random.uniform(0.5, 1.6)
            cx += np.random.uniform(-0.7 * bw, 0.7 * bw)
            cy += np.random.uniform(-0.7 * bh, 0.7 * bh)

            x0, y0 = cx - sw * 0.5, cy - sh * 0.5
            x1, y1 = cx + sw * 0.5, cy + sh * 0.5
        else:
            w = np.random.uniform(20.0, img_w * 0.5)
            h = np.random.uniform(20.0, img_h * 0.5)
            x0 = np.random.uniform(0.0, img_w - w)
            y0 = np.random.uniform(0.0, img_h - h)
            x1, y1 = x0 + w, y0 + h

        x0 = float(np.clip(x0, 0.0, img_w - MIN_SIDE))
        y0 = float(np.clip(y0, 0.0, img_h - MIN_SIDE))
        x1 = float(np.clip(x1, x0 + MIN_SIDE, img_w))
        y1 = float(np.clip(y1, y0 + MIN_SIDE, img_h))

        cand = (x0, y0, x1, y1)
        if any(box_iou(cand, rb) > max_iou for rb in real_boxes):
            continue
        synth.append(cand)
    return synth


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DualPix2SeqDataset(Dataset):
    """
    Emits ALIGNED (input, target) pairs - the training loop must NOT shift them.

    Box sequence, with sequence augmentation:
        box_in  = [BOS, x0, y0, x1, y1, c, ...,  nx0, ny0, nx1, ny1, NOISE, ...]
        box_tgt = [     x0, y0, x1, y1, c, ...,  PAD, PAD, PAD, PAD, NOISE, ..., PAD]
    Noise coordinates are "n/a" in the target, so only the class position is
    supervised for them. Noise objects always sit at the TAIL, after the real
    objects, which is what lets the model be scored at every one of the
    MAX_OBJECTS class slots instead of relying on EOS to stop.
    """

    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS,
                 img_size=IMG_SIZE, transform=train_transform,
                 is_train=True, permute_prob=0.5,
                 noise_fill_to_max=NOISE_FILL_TO_MAX,
                 noise_ratio=NOISE_RATIO,
                 noise_iou_reject=NOISE_IOU_REJECT):
        self.json_dir    = json_dir
        self.img_dir     = img_dir
        self.img_size    = img_size
        self.max_objects = max_objects
        self.json_files  = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform   = transform
        # Noise/permutation are gated on is_train, NOT on `transform is not None`
        # (val_transform is also not None, which would leak noise into val GT).
        self.is_train          = is_train
        self.permute_prob      = permute_prob
        self.noise_fill_to_max = noise_fill_to_max
        self.noise_ratio       = noise_ratio
        self.noise_iou_reject  = noise_iou_reject

    def __len__(self):
        return len(self.json_files)

    def _load_records(self, item, new_points, pt_counts, valid_shapes, img_w, img_h):
        records = []
        cursor  = 0
        for shape, cnt in zip(valid_shapes, pt_counts):
            poly = np.asarray(new_points[cursor:cursor + cnt], dtype=np.float32)
            cursor += cnt
            if len(poly) < 3: continue

            area_before = abs(polygon_area(poly))
            poly = clip_polygon(poly, float(img_w), float(img_h))
            if len(poly) < 3: continue

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
            if poly is None: continue

            records.append({
                "class_id": LABEL_TO_ID[shape["label"]],
                "box": (x_min, y_min, x_max, y_max),
                "poly": poly,
                "is_noise": False,
            })
        return records

    def __getitem__(self, idx):
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
        except Exception:
            image_np = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

        shapes = item.get("shapes", [])
        valid_shapes = [s for s in shapes if s.get("label") in LABEL_TO_ID and len(s.get("points", [])) >= 3]

        all_points, pt_counts = [], []
        for shape in valid_shapes:
            pts = [(float(p[0]), float(p[1])) for p in shape["points"]]
            all_points.extend(pts)
            pt_counts.append(len(pts))

        if self.transform is not None:
            augmented    = self.transform(image=image_np, keypoints=all_points)
            image_tensor = augmented['image']
            new_points   = [(float(p[0]), float(p[1])) for p in augmented['keypoints']]
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            new_points   = all_points

        img_h, img_w = self.img_size[0], self.img_size[1]
        records = self._load_records(item, new_points, pt_counts, valid_shapes, img_w, img_h)

        # ── 1. Ordering (real objects only) ──────────────────────────────────
        if self.is_train and random.random() < self.permute_prob:
            random.shuffle(records)
        else:
            records.sort(key=lambda r: (r["box"][1], r["box"][0]))

        records = records[:self.max_objects]
        num_valid_objs = len(records)      # real objects only - GT parsing uses this

        # ── 2. Append noise objects at the TAIL ──────────────────────────────
        if self.is_train:
            real_boxes = [r["box"] for r in records]
            if self.noise_fill_to_max:
                num_noise = self.max_objects - len(records)
            else:
                num_noise = min(int(round(len(records) * self.noise_ratio)),
                                self.max_objects - len(records))
            if num_noise > 0:
                for sbox in generate_synthetic_boxes(real_boxes, img_w, img_h,
                                                     num_noise, self.noise_iou_reject):
                    records.append({
                        "class_id": NOISE_CLASS_ID,
                        "box": sbox,
                        "poly": None,
                        "is_noise": True,
                    })

        # ── 3. Build aligned input / target sequences ────────────────────────
        box_in_toks, box_tgt_toks = [], []
        mask_in_seqs, mask_tgt_seqs = [], []

        for rec in records:
            x_min, y_min, x_max, y_max = rec["box"]
            q_box = [
                quantize(x_min, img_w, NUM_BINS),
                quantize(y_min, img_h, NUM_BINS),
                quantize(x_max, img_w, NUM_BINS),
                quantize(y_max, img_h, NUM_BINS),
            ]

            if rec["is_noise"]:
                # Input carries the synthetic coordinates; the target marks them
                # "n/a" (PAD -> ignore_index) and supervises only the class.
                box_in_toks.extend(q_box + [NOISE_CLASS_TOKEN])
                box_tgt_toks.extend([PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN])
                mask_in_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
                mask_tgt_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
            else:
                class_tok = NUM_BINS + rec["class_id"]
                box_in_toks.extend(q_box + [class_tok])
                box_tgt_toks.extend(q_box + [class_tok])

                poly_tokens = []
                for px, py in rec["poly"]:
                    poly_tokens.append(quantize(float(px), img_w, NUM_BINS))
                    poly_tokens.append(quantize(float(py), img_h, NUM_BINS))

                m_in  = [BOS_TOKEN] + q_box + [class_tok] + poly_tokens
                m_tgt = q_box + [class_tok] + poly_tokens + [EOS_TOKEN]

                m_in  = (m_in  + [PAD_TOKEN] * MAX_MASK_SEQ_LEN)[:MAX_MASK_SEQ_LEN]
                m_tgt = (m_tgt + [PAD_TOKEN] * MAX_MASK_SEQ_LEN)[:MAX_MASK_SEQ_LEN]
                mask_in_seqs.append(m_in)
                mask_tgt_seqs.append(m_tgt)

        box_in_seq  = [BOS_TOKEN] + box_in_toks
        box_tgt_seq = box_tgt_toks + [PAD_TOKEN]   # last step predicts nothing

        box_in_seq  = (box_in_seq  + [PAD_TOKEN] * MAX_BOX_SEQ_LEN)[:MAX_BOX_SEQ_LEN]
        box_tgt_seq = (box_tgt_seq + [PAD_TOKEN] * MAX_BOX_SEQ_LEN)[:MAX_BOX_SEQ_LEN]

        while len(mask_in_seqs) < self.max_objects:
            mask_in_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)
            mask_tgt_seqs.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)

        return (
            image_tensor,
            torch.tensor(box_in_seq,   dtype=torch.long),
            torch.tensor(box_tgt_seq,  dtype=torch.long),
            torch.tensor(mask_in_seqs, dtype=torch.long),
            torch.tensor(mask_tgt_seqs, dtype=torch.long),
            torch.tensor(num_valid_objs, dtype=torch.long),
        )

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DualPix2SeqModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=256, nheads=8, num_layers=4):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder  = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)

        grid_h = IMG_SIZE[0] // 32
        grid_w = IMG_SIZE[1] // 32
        self.pos_emb = nn.Parameter(torch.randn(1, grid_h * grid_w, hidden_dim))

        self.embedding        = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max(MAX_BOX_SEQ_LEN, MAX_MASK_SEQ_LEN))
        self.emb_dropout      = nn.Dropout(0.1)

        box_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1)
        self.box_decoder = nn.TransformerDecoder(box_layer, num_layers=num_layers)

        mask_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1)
        self.mask_decoder = nn.TransformerDecoder(mask_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, images):
        features = self.enc_proj(self.encoder(images))
        memory   = features.flatten(2).permute(0, 2, 1)
        return memory + self.pos_emb

    def forward_boxes(self, memory, box_seq):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(box_seq.size(1)).to(memory.device)
        tgt_emb  = self.emb_dropout(self.seq_pos_encoding(self.embedding(box_seq)))
        out      = self.box_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

    def forward_masks(self, memory, mask_seqs):
        """
        Runs decoder 2 only on object slots that actually carry a sequence.
        Padded slots (empty slots and noise objects) are all-PAD in both input
        and target, so their logits are never used by the loss - computing them
        would waste most of the batch once the tail is filled with noise.
        """
        B, N_objs, S = mask_seqs.shape
        flat  = mask_seqs.reshape(B * N_objs, S)
        valid = (flat != PAD_TOKEN).any(dim=1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S).to(memory.device)

        if not bool(valid.any()):
            # Degenerate batch (no objects at all): return zeros, loss ignores them.
            zeros = memory.new_zeros((B * N_objs, S, self.fc_out.out_features))
            return zeros.view(B, N_objs, S, -1)

        sel      = valid.nonzero(as_tuple=True)[0]
        owner    = torch.div(sel, N_objs, rounding_mode='floor')
        sub_seqs = flat.index_select(0, sel)
        sub_mem  = memory.index_select(0, owner)

        tgt_emb = self.emb_dropout(self.seq_pos_encoding(self.embedding(sub_seqs)))
        out     = self.mask_decoder(tgt=tgt_emb, memory=sub_mem, tgt_mask=tgt_mask)
        sub_logits = self.fc_out(out)

        base   = sub_logits.new_zeros((B * N_objs, S, sub_logits.size(-1)))
        logits = base.index_copy(0, sel, sub_logits)
        return logits.view(B, N_objs, S, -1)

    def forward(self, images, box_targets, mask_targets):
        memory      = self.encode(images)
        box_logits  = self.forward_boxes(memory, box_targets)
        mask_logits = self.forward_masks(memory, mask_targets)
        return box_logits, mask_logits

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE DECODING
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dual_ar_decode(model, images, device, score_thresh=INFER_SCORE_THRESH):
    """
    Fixed-length constrained decode over MAX_OBJECTS slots:
      * coordinate steps are restricted to the bin range [0, NUM_BINS)
      * class steps are restricted to the class slots (real classes + noise)
      * no EOS, so chunking can never desync
      * score = p(best real class) / (p(best real class) + p(noise))

    Lowering score_thresh keeps more low-confidence boxes; mAP generally prefers
    that (they rank below the good ones), latency prefers the opposite, since
    every kept instance costs a decoder-2 rollout.
    """
    model.eval()
    B = images.size(0)
    memory = model.encode(images)

    seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    n_steps = BOX_TOKENS_PER_OBJ * MAX_OBJECTS
    slot_scores = torch.zeros(B, MAX_OBJECTS, device=device)

    for step in range(n_steps):
        tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(seq)))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
        out      = model.box_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        probs    = model.fc_out(out[:, -1, :]).float().softmax(-1)

        if step % BOX_TOKENS_PER_OBJ == BOX_TOKENS_PER_OBJ - 1:
            slot  = step // BOX_TOKENS_PER_OBJ
            cls_p = probs[:, NUM_BINS:NUM_BINS + NUM_CLASS_SLOTS]        # [B, C+1]
            real_p = cls_p[:, :NUM_CLASSES]
            p_real, best_real = real_p.max(-1)
            p_noise = cls_p[:, NOISE_CLASS_ID]

            slot_scores[:, slot] = p_real / (p_real + p_noise).clamp_min(1e-6)
            nxt = NUM_BINS + best_real       # feed the real class back as context
        else:
            nxt = probs[:, :NUM_BINS].argmax(-1)

        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)

    # ── Assemble surviving instances across the whole batch ──────────────────
    all_instances, flat_prefixes, flat_owner = [], [], []
    for b in range(B):
        toks = seq[b, 1:].tolist()
        inst = []
        for s in range(MAX_OBJECTS):
            chunk = toks[s * BOX_TOKENS_PER_OBJ:(s + 1) * BOX_TOKENS_PER_OBJ]
            score = float(slot_scores[b, s])
            if score < score_thresh:
                continue
            x0 = dequantize(chunk[0], IMG_SIZE[1], NUM_BINS)
            y0 = dequantize(chunk[1], IMG_SIZE[0], NUM_BINS)
            x1 = dequantize(chunk[2], IMG_SIZE[1], NUM_BINS)
            y1 = dequantize(chunk[3], IMG_SIZE[0], NUM_BINS)
            if x1 <= x0 or y1 <= y0:
                continue
            inst.append({"box": [x0, y0, x1, y1],
                         "cls": chunk[4] - NUM_BINS,
                         "score": score})
            flat_prefixes.append([BOS_TOKEN] + chunk)
            flat_owner.append(b)
        all_instances.append(inst)

    if not flat_prefixes:
        return [[] for _ in range(B)]

    # ── Decoder 2, batched across every kept instance in the batch ───────────
    prefixes    = torch.tensor(flat_prefixes, dtype=torch.long, device=device)
    owner       = torch.tensor(flat_owner, dtype=torch.long, device=device)
    inst_memory = memory.index_select(0, owner)

    cur      = prefixes
    finished = torch.zeros(cur.size(0), dtype=torch.bool, device=device)

    for _ in range(POLY_TOKENS_PER_OBJ + 1):
        tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(cur)))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur.size(1)).to(device)
        out      = model.mask_decoder(tgt=tgt_emb, memory=inst_memory, tgt_mask=tgt_mask)
        nxt      = model.fc_out(out[:, -1, :]).float().argmax(-1)
        nxt      = torch.where(finished, torch.full_like(nxt, PAD_TOKEN), nxt)
        cur      = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        finished |= (nxt == EOS_TOKEN)
        if finished.all():
            break

    all_image_preds = [[] for _ in range(B)]
    cursor = 0
    for b in range(B):
        for inst in all_instances[b]:
            poly_toks = cur[cursor, 1 + BOX_TOKENS_PER_OBJ:].tolist()
            cursor += 1
            if EOS_TOKEN in poly_toks:
                poly_toks = poly_toks[:poly_toks.index(EOS_TOKEN)]
            poly_toks = [t for t in poly_toks if 0 <= t < NUM_BINS]
            if len(poly_toks) < 2 * NUM_POLY_PTS:
                continue
            poly = np.array([
                [dequantize(poly_toks[k], IMG_SIZE[1], NUM_BINS),
                 dequantize(poly_toks[k + 1], IMG_SIZE[0], NUM_BINS)]
                for k in range(0, 2 * NUM_POLY_PTS, 2)
            ], dtype=np.float32)

            all_image_preds[b].append({
                "box": inst["box"],
                "polygon": poly,
                "label": inst["cls"],
                "score": inst["score"],
            })

    return all_image_preds

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dual-Decoder Pix2seq starting on {device}...")
    USE_BF16 = True
    amp_enabled = USE_BF16 and device.type == "cuda"
    amp_dtype   = torch.bfloat16 if amp_enabled else torch.float32  
    # Load / Build Dataset Splits
    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    if os.path.exists(VAL_SPLIT_PATH):
        print(f"Loading existing val split: {VAL_SPLIT_PATH}")
        with open(VAL_SPLIT_PATH) as f:
            saved = json.load(f)
        val_files_set = set(saved["filenames"])
        train_files   = [f for f in all_json_files if f not in val_files_set]
        val_files     = [f for f in all_json_files if f in val_files_set]
    else:
        print(f"Creating new val split -> {VAL_SPLIT_PATH}")
        random.shuffle(all_json_files)
        train_size  = int(0.9 * len(all_json_files))
        train_files = all_json_files[:train_size]
        val_files   = all_json_files[train_size:]
        with open(VAL_SPLIT_PATH, "w") as f:
            json.dump({"filenames": val_files}, f, indent=2)

    train_dataset            = DualPix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform,
                                                  is_train=True)
    train_dataset.json_files = train_files

    # is_train=False -> no noise objects and no permutation in the val targets
    val_dataset            = DualPix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform,
                                                is_train=False)
    val_dataset.json_files = val_files

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=8, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model     = DualPix2SeqModel(vocab_size=VOCAB_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),          'lr': 3e-5},
        {'params': model.enc_proj.parameters(),         'lr': LEARNING_RATE},
        {'params': [model.pos_emb],                     'lr': LEARNING_RATE},
        {'params': model.embedding.parameters(),        'lr': LEARNING_RATE},
        {'params': model.seq_pos_encoding.parameters(), 'lr': LEARNING_RATE},
        {'params': model.box_decoder.parameters(),      'lr': LEARNING_RATE},
        {'params': model.mask_decoder.parameters(),     'lr': LEARNING_RATE},
        {'params': model.fc_out.parameters(),           'lr': LEARNING_RATE},
    ], weight_decay=1e-4)

    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[3e-5, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE,
                LEARNING_RATE, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.05
    )
    from instance_seg_eval_utils_dual_pix2seq import evaluate_dual
    EVAL_EVERY    = 4
    best_val_loss = float("inf")
    best_segm_map = 0.0

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        # The dataset already emits aligned (input, target) pairs - do NOT shift.
        for images, box_in, box_out, mask_in, mask_out, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False):
            images   = images.to(device).float()
            box_in   = box_in.to(device)
            box_out  = box_out.to(device)
            mask_in  = mask_in.to(device)
            mask_out = mask_out.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                box_logits, mask_logits = model(images, box_in, mask_in)

                loss_box  = criterion(box_logits.reshape(-1, VOCAB_SIZE), box_out.reshape(-1))
                loss_mask = criterion(mask_logits.reshape(-1, VOCAB_SIZE), mask_out.reshape(-1))
                loss      = loss_box + 2.0 * loss_mask

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validation Loss ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, box_in, box_out, mask_in, mask_out, _ in val_loader:
                images   = images.to(device).float()
                box_in   = box_in.to(device)
                box_out  = box_out.to(device)
                mask_in  = mask_in.to(device)
                mask_out = mask_out.to(device)

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    box_logits, mask_logits = model(images, box_in, mask_in)
                    loss_box  = criterion(box_logits.reshape(-1, VOCAB_SIZE), box_out.reshape(-1))
                    loss_mask = criterion(mask_logits.reshape(-1, VOCAB_SIZE), mask_out.reshape(-1))
                    val_loss += (loss_box + 2.0 * loss_mask).item()

        avg_val_loss = val_loss / len(val_loader)

        epoch_bar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}", best_loss=f"{best_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "dual_pix2seq_best_loss.pth")

        # ── Evaluation (mAP) ──────────────────────────────────────────────────
        if (epoch + 1) % EVAL_EVERY == 0 or epoch >= (EPOCHS - 20):
            res = evaluate_dual(model, val_loader, device=device)
            tqdm.write(
                f"Epoch {epoch+1:3d} | Val Loss: {avg_val_loss:.4f} | "
                f"Segm mAP: {res['segm_map']:.4f} | Segm mAP@50: {res['segm_map_50']:.4f} | "
                f"BBox mAP: {res['bbox_map']:.4f}"
            )

            if res["segm_map"] > best_segm_map:
                best_segm_map = res["segm_map"]
                torch.save(model.state_dict(), "dual_pix2seq_best_map.pth")
                tqdm.write(f"  ✓ New best Segm mAP: {best_segm_map:.4f} -> saved dual_pix2seq_best_map.pth")

        torch.save(model.state_dict(), "dual_pix2seq_last.pth")
