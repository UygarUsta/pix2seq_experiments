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
from PIL import Image, ImageOps
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
JSON_DIR       = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
IMG_DIR        = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
NUM_BINS = 500
MAX_OBJECTS = 40
BATCH_SIZE = 16
EPOCHS = 350
LEARNING_RATE = 3e-4
IMG_SIZE = (512, 512)
VAL_SPLIT_PATH = "val_split.json"

LABEL_TO_ID = {
    "qr": 0, "menfaat": 1, "azami_yuk": 2, "kullanim_amaci": 3, 
    "net_agirlik": 4, "ruhsat": 5, "romork_azami_yuk": 6, 
    "plaka": 7, "tc": 8, "seri_no": 9
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_CLASSES = len(LABEL_TO_ID)
VOCAB_SIZE = NUM_BINS + NUM_CLASSES + 3
BOS_TOKEN = VOCAB_SIZE - 3
EOS_TOKEN = VOCAB_SIZE - 2
PAD_TOKEN = VOCAB_SIZE - 1




n_swap = n_diff = n_exif = 0
for jf in os.listdir(JSON_DIR):
    if not jf.endswith('.json'):
        continue
    with open(os.path.join(JSON_DIR, jf), encoding='utf-8') as f:
        item = json.load(f)
    jw, jh = item.get("imageWidth"), item.get("imageHeight")
    if not jw:
        continue
    p = os.path.join(IMG_DIR, item.get("imagePath", jf.replace('.json', '.jpg')))
    if not os.path.exists(p):
        continue

    im = Image.open(p)
    raw = im.size
    ori = (im.getexif() or {}).get(274, 1)      # 274 = Orientation
    if ori not in (1, 0, None):
        n_exif += 1
    if raw == (jh, jw) and jw != jh:
        n_swap += 1
        print(f"SWAP  {jf}: json=({jw},{jh}) raw={raw} exif_ori={ori}")
    elif raw != (jw, jh):
        n_diff += 1
        print(f"DIFF  {jf}: json=({jw},{jh}) raw={raw} exif_ori={ori}")

print(f"\nEXIF orientation != 1 : {n_exif}")
print(f"Boyut swap            : {n_swap}")
print(f"Diğer uyuşmazlık      : {n_diff}")


train_transform = A.Compose([
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.2),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, label_fields=['kp_ids']))


val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(
    format='xy',
    remove_invisible=False,
    label_fields=['kp_ids']
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
    if not (0.0 <= normalized <= 1.0) and os.environ.get("KP_DEBUG"):
        print(f"[KP] kadraj dışı: {coordinate:.1f} / {max_size}")
    normalized = max(0.0, min(1.0, normalized))
    return int(normalized * (num_bins - 1))

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


def dequantize(token, max_size, num_bins=NUM_BINS):
    return (token + 0.5) / num_bins * max_size

def decode_sequence(seq, img_w, img_h):
    """[BOS] (x1 y1 ... x4 y4 cls) x N [EOS] -> [(poly, label_str), ...]"""
    toks = seq.tolist() if torch.is_tensor(seq) else list(seq)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    out, i = [], 0
    while i + 9 <= len(toks):
        chunk = toks[i:i + 9]
        if any(t in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN) for t in chunk):
            break
        coords, cls_tok = chunk[:8], chunk[8]
        if any(c >= NUM_BINS for c in coords):          # hizalama bozulmuş
            break
        if not (NUM_BINS <= cls_tok < NUM_BINS + NUM_CLASSES):
            break
        poly = [(dequantize(coords[2 * j], img_w),
                 dequantize(coords[2 * j + 1], img_h)) for j in range(4)]
        out.append((poly, ID_TO_LABEL[cls_tok - NUM_BINS]))
        i += 9
    return out


PALETTE = ["#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231",
           "#48f90a", "#92cc17", "#3ddb86", "#1a9334", "#00c2ff"]

def visualize_augmentations(dataset, out_dir="aug_check", n=20):
    """Augment edilmiş görüntüyü ve sequence'tan geri çözülen poligonları çizer."""
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for k in range(n):
        img_t, seq = dataset[k % len(dataset)]

        img = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d   = ImageDraw.Draw(img)
        W, H = img.size

        instances = decode_sequence(seq, W, H)
        for poly, label in instances:
            color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]

            # HBB (kontrol amaçlı)
            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            d.rectangle([min(xs), min(ys), max(xs), max(ys)],
                        outline="#00e5ff", width=1)

            # poligon
            d.line([c for p in poly for c in p] + list(poly[0]),
                   fill=color, width=3)

            # 1. nokta kırmızı, 2. nokta sarı -> kanonik başlangıç + yön kontrolü
            x0, y0 = poly[0]; x1, y1 = poly[1]
            d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill="red")
            d.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill="yellow")

            d.text((x0 + 7, y0 - 13), label, fill=color)

        d.text((5, 5), f"{len(instances)} obj", fill="white")
        img.save(os.path.join(out_dir, f"aug_check_{k:02d}.jpg"), quality=92)
    print(f"{n} görsel kaydedildi -> {out_dir}/")



class Pix2SeqDataset(Dataset):
    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS, img_size=IMG_SIZE, transform=train_transform):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_seq_len = 1 + (9 * max_objects) + 1 
        self.transform = transform # Albumentations transformunu aldık

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
    
        # ── Mosaic (30% chance) ───────────────────────────────────────────────
        if self.transform is not None and random.random()  < 0.0 : #< 0.3:
            image_np, all_points, valid_shapes = self._mosaic(idx)
        else:
            # ── Normal loading ────────────────────────────────────────────────
            json_path = os.path.join(self.json_dir, self.json_files[idx])
            with open(json_path, 'r', encoding='utf-8') as f:
                item = json.load(f)

            img_name = item.get("imagePath")

            if img_name:
                img_path = os.path.join(self.img_dir, img_name)
            else:
                # imagePath missing — try common extensions
                base = self.json_files[idx].replace('.json', '')
                for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                    candidate = os.path.join(self.img_dir, base + ext)
                    if os.path.exists(candidate):
                        img_path = candidate
                        break
                else:
                    # nothing found
                    img_path = os.path.join(self.img_dir, base + '.jpg')  # will fail gracefully in the try/except below

            try:
                image = Image.open(img_path).convert("RGB")
                image = ImageOps.exif_transpose(image)
                image, _ = pad_to_square(image)
                image_np = np.array(image)
            except Exception as e:
                print(f"HATA: {img_path} okunamadı. Hata: {e}")
                image_np = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            shapes = item.get("shapes", [])
            # all_points  = []
            # valid_shapes = []

            # for shape in shapes:
            #     label_str = shape.get("label", "")
            #     points    = shape.get("points", [])
            #     if label_str in LABEL_TO_ID and len(points) == 4:
            #         valid_shapes.append(shape)
            #         for p in points:
            #             all_points.append(tuple(p))
            all_points = []
            kp_ids     = []          # her noktanın hangi shape'e ait olduğu
            valid_shapes = []

            for shape in shapes:
                label_str = shape.get("label", "")
                points    = shape.get("points", [])
                if label_str in LABEL_TO_ID and len(points) == 4:
                    si = len(valid_shapes)
                    valid_shapes.append(shape)
                    for p in points:
                        all_points.append((float(p[0]), float(p[1])))
                        kp_ids.append(si)

        # ── Augmentation (same path for both mosaic and normal) ───────────────
        # if self.transform is not None:
        #     augmented    = self.transform(image=image_np, keypoints=all_points)
        #     image_tensor = augmented['image']
        #     new_points   = list(augmented['keypoints'])
        # else:
        #     image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
        #     new_points   = all_points
        if self.transform is not None:
            augmented    = self.transform(image=image_np, keypoints=all_points, kp_ids=kp_ids)
            image_tensor = augmented['image']
            new_points   = [tuple(k[:2]) for k in augmented['keypoints']]
            new_ids      = list(augmented['kp_ids'])
            if len(new_points) != len(all_points) and os.environ.get("KP_DEBUG"):
                print(f"[KP] {self.json_files[idx]}: {len(all_points)} -> {len(new_points)}")

        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            new_points   = all_points
            new_ids      = kp_ids

        img_h, img_w = self.img_size

        # shape indeksine göre grupla
        groups = {}
        for (x, y), si in zip(new_points, new_ids):
            groups.setdefault(si, []).append((x, y))

        sequence = [BOS_TOKEN]
        for si, shape in enumerate(valid_shapes):
            quad = groups.get(si, [])
            if len(quad) != 4:                     # eksik/fazla -> bu objeyi at
                continue
            if not all(-8 <= x <= img_w + 8 and -8 <= y <= img_h + 8 for x, y in quad):
                continue                           # kadraj dışı -> clamp yerine at
            for x, y in quad:
                sequence.extend([quantize(x, img_w, NUM_BINS),
                                quantize(y, img_h, NUM_BINS)])
            sequence.append(NUM_BINS + LABEL_TO_ID[shape["label"]])
        sequence.append(EOS_TOKEN)

        if len(sequence) < self.max_seq_len:
            sequence.extend([PAD_TOKEN] * (self.max_seq_len - len(sequence)))
        else:
            sequence = sequence[:self.max_seq_len - 1] + [EOS_TOKEN]

        return image_tensor, torch.tensor(sequence, dtype=torch.long)


    def _mosaic(self, idx):
        """Combines 4 images into a 2x2 mosaic. Returns (image_np, all_points, valid_shapes)."""
        h, w    = self.img_size
        indices = [idx] + random.sample(range(len(self.json_files)), 3)

        mosaic_img    = np.full((h, w, 3), 114, dtype=np.uint8)
        all_points    = []
        valid_shapes  = []

        tiles = [
            (0,    0,    w//2, h//2),   # top-left
            (w//2, 0,    w//2, h//2),   # top-right
            (0,    h//2, w//2, h//2),   # bottom-left
            (w//2, h//2, w//2, h//2),   # bottom-right
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
            except:
                img = Image.fromarray(np.full((th, tw, 3), 114, dtype=np.uint8))
                max_dim = max(tw, th)

            mosaic_img[y_off:y_off+th, x_off:x_off+tw] = np.array(img)

            for shape in item.get("shapes", []):
                if shape.get("label") in LABEL_TO_ID and len(shape.get("points", [])) == 4:
                    valid_shapes.append(shape)
                    for px, py in shape["points"]:
                        # scale from original padded space → tile space → mosaic space
                        sx = (px / max_dim) * tw + x_off
                        sy = (py / max_dim) * th + y_off
                        all_points.append((sx, sy))

        return mosaic_img, all_points, valid_shapes

# --- MODEL ---
class Pix2SeqModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, nheads=8, num_layers=4, max_seq_len=200):
        super().__init__()
        # Özellik Çıkarıcı (Encoder)
        #resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2]) 
        #self.enc_proj = nn.Conv2d(512, hidden_dim, kernel_size=1) #resnet18
        self.enc_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        
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

    train_dataset            = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform)
    train_dataset.json_files = train_files

    val_dataset              = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform)
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
        max_lr=[1e-5, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE,
                LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.05
    )

    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)

    visualize_augmentations(train_dataset, n=20)
 
    best_val_loss = float('inf')
 
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    from quad_eval_utils_pix2seq import evaluate, MAX_SEQ_LEN
    EVAL_EVERY, best_map = 2, 0.0
 
    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
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
 
        # ── Validation ─────────────────────────────────────────────────────
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
 
        # ── Logging ────────────────────────────────────────────────────────
        epoch_bar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            best=f"{best_val_loss:.4f}"
        )
 
        # ── Checkpoint ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "pix2seq_best_quad.pth")
            tqdm.write(f"  ✓ Epoch {epoch+1:3d} — new best val loss: {avg_val_loss:.4f} → saved pix2seq_best_quad.pth")

        if (epoch + 1) % EVAL_EVERY == 0:
            res = evaluate(model, val_dataloader, MAX_SEQ_LEN, device)
            tqdm.write(f"  mAP {res['map']:.4f} | mAP50 {res['map_50']:.4f} "
                    f"| HBB50 {res['hbb_map_50']:.4f} | NME {res['nme']:.4f}")
            if res["map"] > best_map:
                best_map = res["map"]
                torch.save(model.state_dict(), "pix2seq_best_map_quad.pth")
 
    
    print(f"\nEğitim tamamlandı.")
    print(f"  Son model  : pix2seq_model_v2.pth")
    print(f"  En iyi model: pix2seq_best.pth  (val loss: {best_val_loss:.4f})")