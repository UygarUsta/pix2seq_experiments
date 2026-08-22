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

from instance_seg_eval_utils_dual_pix2seq import evaluate_dual

# --- CONFIGURATION ---
JSON_DIR       = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/"
IMG_DIR        = "/home/uygarusta/datasets/card_merged_datasets/merged_datasets/"
NUM_BINS       = 500
MAX_OBJECTS    = 10
BATCH_SIZE     = 16
EPOCHS         = 350
LEARNING_RATE  = 3e-4
IMG_SIZE       = (512, 512)
VAL_SPLIT_PATH = "val_split.json"

LABEL_TO_ID = {"card": 0}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)

VOCAB_SIZE  = NUM_BINS + NUM_CLASSES + 3
BOS_TOKEN   = VOCAB_SIZE - 3
EOS_TOKEN   = VOCAB_SIZE - 2
PAD_TOKEN   = VOCAB_SIZE - 1

NUM_POLY_PTS        = 16
BOX_TOKENS_PER_OBJ  = 5
POLY_TOKENS_PER_OBJ = 2 * NUM_POLY_PTS     # 32
MAX_BOX_SEQ_LEN     = 1 + BOX_TOKENS_PER_OBJ * MAX_OBJECTS + 1  # 52
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
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class DualPix2SeqDataset(Dataset):
    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS,
                 img_size=IMG_SIZE, transform=train_transform):
        self.json_dir    = json_dir
        self.img_dir     = img_dir
        self.img_size    = img_size
        self.max_objects = max_objects
        self.json_files  = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform   = transform

    def __len__(self):
        return len(self.json_files)

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
            })

        records.sort(key=lambda r: (r["box"][1], r["box"][0]))
        records = records[:self.max_objects]

        box_seq   = [BOS_TOKEN]
        mask_seqs = []

        for rec in records:
            x_min, y_min, x_max, y_max = rec["box"]
            q_box = [
                quantize(x_min, img_w, NUM_BINS),
                quantize(y_min, img_h, NUM_BINS),
                quantize(x_max, img_w, NUM_BINS),
                quantize(y_max, img_h, NUM_BINS),
                NUM_BINS + rec["class_id"],
            ]
            box_seq.extend(q_box)

            single_mask_seq = [BOS_TOKEN] + q_box
            for px, py in rec["poly"]:
                single_mask_seq.append(quantize(float(px), img_w, NUM_BINS))
                single_mask_seq.append(quantize(float(py), img_h, NUM_BINS))
            single_mask_seq.append(EOS_TOKEN)
            mask_seqs.append(single_mask_seq)

        box_seq.append(EOS_TOKEN)

        if len(box_seq) < MAX_BOX_SEQ_LEN:
            box_seq.extend([PAD_TOKEN] * (MAX_BOX_SEQ_LEN - len(box_seq)))
        else:
            box_seq = box_seq[:MAX_BOX_SEQ_LEN - 1] + [EOS_TOKEN]

        padded_masks = []
        for seq in mask_seqs:
            if len(seq) < MAX_MASK_SEQ_LEN:
                seq.extend([PAD_TOKEN] * (MAX_MASK_SEQ_LEN - len(seq)))
            else:
                seq = seq[:MAX_MASK_SEQ_LEN - 1] + [EOS_TOKEN]
            padded_masks.append(seq)

        num_valid_objs = len(padded_masks)
        while len(padded_masks) < self.max_objects:
            padded_masks.append([PAD_TOKEN] * MAX_MASK_SEQ_LEN)

        return (
            image_tensor,
            torch.tensor(box_seq, dtype=torch.long),
            torch.tensor(padded_masks, dtype=torch.long),
            torch.tensor(num_valid_objs, dtype=torch.long)
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
        B, N_objs, S = mask_seqs.shape
        D = memory.size(-1)
        flat_mask_seqs = mask_seqs.view(B * N_objs, S)
        flat_memory    = memory.unsqueeze(1).repeat(1, N_objs, 1, 1).view(B * N_objs, -1, D)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(S).to(memory.device)
        tgt_emb  = self.emb_dropout(self.seq_pos_encoding(self.embedding(flat_mask_seqs)))
        out      = self.mask_decoder(tgt=tgt_emb, memory=flat_memory, tgt_mask=tgt_mask)
        logits   = self.fc_out(out)
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
def dual_ar_decode(model, images, device):
    model.eval()
    B = images.size(0)
    memory = model.encode(images)

    box_seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(MAX_BOX_SEQ_LEN - 1):
        tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(box_seq)))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(box_seq.size(1)).to(device)
        out      = model.box_decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits   = model.fc_out(out[:, -1, :]).float()
        
        nxt = logits.argmax(-1)
        nxt = torch.where(finished, torch.full_like(nxt, PAD_TOKEN), nxt)
        box_seq = torch.cat([box_seq, nxt.unsqueeze(1)], dim=1)
        finished |= (nxt == EOS_TOKEN)
        if finished.all():
            break

    all_image_preds = []
    
    for b in range(B):
        b_toks = box_seq[b, 1:].tolist()
        if EOS_TOKEN in b_toks:
            b_toks = b_toks[:b_toks.index(EOS_TOKEN)]
        b_toks = [t for t in b_toks if t not in (PAD_TOKEN, BOS_TOKEN)]

        instances = []
        n_valid = len(b_toks) - (len(b_toks) % BOX_TOKENS_PER_OBJ)
        for i in range(0, n_valid, BOX_TOKENS_PER_OBJ):
            chunk = b_toks[i:i + BOX_TOKENS_PER_OBJ]
            if all(0 <= v < NUM_BINS for v in chunk[:4]) and (NUM_BINS <= chunk[4] < NUM_BINS + NUM_CLASSES):
                x0 = dequantize(chunk[0], IMG_SIZE[1], NUM_BINS)
                y0 = dequantize(chunk[1], IMG_SIZE[0], NUM_BINS)
                x1 = dequantize(chunk[2], IMG_SIZE[1], NUM_BINS)
                y1 = dequantize(chunk[3], IMG_SIZE[0], NUM_BINS)
                if x1 > x0 and y1 > y0:
                    instances.append({
                        "box": [x0, y0, x1, y1],
                        "cls": chunk[4] - NUM_BINS,
                        "raw_prefix": [BOS_TOKEN] + chunk
                    })

        if not instances:
            all_image_preds.append([])
            continue

        N_inst = len(instances)
        prefixes = torch.tensor([inst["raw_prefix"] for inst in instances], dtype=torch.long, device=device)
        inst_memory = memory[b:b+1].repeat(N_inst, 1, 1)

        cur_mask_seq = prefixes
        inst_finished = torch.zeros(N_inst, dtype=torch.bool, device=device)

        for _ in range(POLY_TOKENS_PER_OBJ + 1):
            tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(cur_mask_seq)))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(cur_mask_seq.size(1)).to(device)
            out      = model.mask_decoder(tgt=tgt_emb, memory=inst_memory, tgt_mask=tgt_mask)
            logits   = model.fc_out(out[:, -1, :]).float()
            
            nxt = logits.argmax(-1)
            nxt = torch.where(inst_finished, torch.full_like(nxt, PAD_TOKEN), nxt)
            cur_mask_seq = torch.cat([cur_mask_seq, nxt.unsqueeze(1)], dim=1)
            inst_finished |= (nxt == EOS_TOKEN)
            if inst_finished.all():
                break

        img_results = []
        for i_idx, inst in enumerate(instances):
            poly_toks = cur_mask_seq[i_idx, 6:].tolist()
            if EOS_TOKEN in poly_toks:
                poly_toks = poly_toks[:poly_toks.index(EOS_TOKEN)]
            poly_toks = [t for t in poly_toks if 0 <= t < NUM_BINS]
            
            if len(poly_toks) >= 2 * NUM_POLY_PTS:
                poly = np.array([
                    [dequantize(poly_toks[k], IMG_SIZE[1], NUM_BINS),
                     dequantize(poly_toks[k+1], IMG_SIZE[0], NUM_BINS)]
                    for k in range(0, 2 * NUM_POLY_PTS, 2)
                ], dtype=np.float32)
                
                img_results.append({
                    "box": inst["box"],
                    "polygon": poly,
                    "label": inst["cls"],
                    "score": 1.0
                })
        all_image_preds.append(img_results)

    return all_image_preds

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCRIPT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dual-Decoder Pix2seq starting on {device}...")

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

    train_dataset            = DualPix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform)
    train_dataset.json_files = train_files

    val_dataset            = DualPix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform)
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

    EVAL_EVERY    = 4
    best_val_loss = float("inf")
    best_segm_map = 0.0

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for images, box_targets, mask_targets, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}", leave=False):
            images       = images.to(device).float()
            box_targets  = box_targets.to(device)
            mask_targets = mask_targets.to(device)

            box_in, box_out   = box_targets[:, :-1], box_targets[:, 1:]
            mask_in, mask_out = mask_targets[:, :, :-1], mask_targets[:, :, 1:]

            optimizer.zero_grad()
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
            for images, box_targets, mask_targets, _ in val_loader:
                images       = images.to(device).float()
                box_targets  = box_targets.to(device)
                mask_targets = mask_targets.to(device)

                box_in, box_out   = box_targets[:, :-1], box_targets[:, 1:]
                mask_in, mask_out = mask_targets[:, :, :-1], mask_targets[:, :, 1:]

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
