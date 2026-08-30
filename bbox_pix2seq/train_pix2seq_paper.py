"""
train_pix2seq_paper.py
======================
Pix2Seq (Chen et al. 2021, arXiv:2109.10852) Appendix B'deki from-scratch
reçetesine hizalanmış eğitim scripti. training_pix2seq_bbox.py'yi DEĞİŞTİRMEZ —
annotasyon katmanı ve noise üretimi oradan import edilir, geri kalanı burada.

Paper'a göre değişenler (eski -> yeni)
-------------------------------------
  nesne sırası     deterministik (y,x)  ->  her gösterimde rastgele
                   Paper ablasyonu: rastgele sıralama hem AP hem AR@100'de en iyi.
                   Deterministik sırada model erken atladığı nesneyi telafi edemiyor.
  scale jitter     [0.5, 1.5]           ->  [0.1, 3.0] + random crop (LSJ)
  weight decay     1e-4                 ->  0.05
  stochastic depth yok                  ->  0.1  (encoder + decoder residual dalları)
  label smoothing  0.1                  ->  0.0  (paper düz CE kullanıyor)
  lr schedule      OneCycle             ->  warmup + lineer düşüş
  efektif batch    16                   ->  16 x GRAD_ACCUM
  çözünürlük       512                  ->  640  (paper 1333; bütçeye göre kıs)
  mosaic           0.4                  ->  0.0  (LSJ ile büyük ölçüde çakışıyor,
                                                  üstelik tile'ı 2x küçültüp AP_s'yi
                                                  zorlaştırıyor. Knob olarak duruyor.)

Paper'dan bilerek SAPILANLAR
----------------------------
  * batch 128 (+ batch repeat -> efektif 256) yerine 16 x GRAD_ACCUM. Bütçe.
  * 2000 bin yerine 500. Paper 640 görüntüde 500 binin yettiğini söylüyor
    (~1.3 piksel/bin); 640'ta bizde ~1.28 piksel/bin.
  * nucleus sampling yerine argmax decode. Paper p=0.4 ile biraz daha iyi AP
    alıyor ama basitlik tercih edildi.
  * ImageNet ön-eğitimli ResNet50 (paper tüm ağı sıfırdan eğitiyor) -> backbone
    için ayrı, düşük lr.

Kullanım
--------
    python train_pix2seq_paper.py
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
from PIL import Image, ImageDraw
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

# Saf yardımcılar — bunlar çözünürlükten bağımsız, tekrar yazmaya gerek yok
from training_pix2seq_bbox import (
    LABEL_TO_ID, ID_TO_LABEL, NUM_CLASSES,
    NUM_BINS, VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    NOISE_CLASS_ID, NOISE_CLASS_TOKEN, NUM_CLASS_SLOTS,
    TOKENS_PER_OBJ, MIN_SIDE, NOISE_IOU_REJECT,
    load_shapes, generate_noise_boxes, box_iou, pad_to_square, quantize,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
JSON_DIR       = "/home/uygarusta/dataset_toolkit/labelme_cikti/"
IMG_DIR        = "/home/uygarusta/Oriented-Centernet/centernet_ciou_iou_aware_pl/coco_dataset/train/"
VAL_SPLIT_PATH = "val_split.json"
CKPT_PREFIX    = "p2s_paper"          # eski checkpointlerin üzerine yazmaz

IMG_SIZE     = (640, 640)             # paper 1333; 512'de kaldıysan AP_s tavanı sert
MAX_OBJECTS  = 100                    # paper da 100 -> dizi uzunluğu 500
BATCH_SIZE   = 16
GRAD_ACCUM   = 4                      # efektif batch = 64 (paper 256)
EPOCHS       = 100                    # eski koşuda tepe epoch 64'teydi
BASE_LR      = 5e-4                   # aşağıdaki nota bak
BACKBONE_LR  = 5e-5
WEIGHT_DECAY = 0.05                   # paper
WARMUP_EPOCHS = 5                     # paper 300 epoch'ta 10 -> ~%3
GRAD_CLIP    = 0.1                    # DETR/Pix2Seq değeri
DROP_PATH    = 0.1                    # paper: stochastic depth %10
DROPOUT      = 0.1
LABEL_SMOOTH = 0.0                    # paper düz CE
EVAL_EVERY   = 2
MOSAIC_PROB  = 0.0
PERMUTE_PROB = 1.0                    # paper: her gösterimde rastgele sıra
SCALE_RANGE  = (0.1, 2.0)             # paper LSJ gücü
USE_BF16     = True

# lr notu: paper 0.003 @ efektif 256. Lineer ölçekleme -> 0.003 * 64/256 = 7.5e-4.
# Önceki koşun 3e-4'te epoch ~126'da çökmüştü, ama o koşuda wd 1e-4, clip 1.0,
# stochastic depth yok ve zayıf augmentasyon vardı. Bu reçete çok daha
# regularize; 5e-4 ara bir değer. İlk 10 epoch'ta loss zıplarsa 3e-4'e düşür.

IMG_H, IMG_W = IMG_SIZE
MAX_SEQ_LEN  = 1 + TOKENS_PER_OBJ * MAX_OBJECTS
NOISE_FILL_TO_MAX = True
MIN_VISIBLE_FRAC  = 0.2   # LSJ crop sonrası kadraj dışına taşan nesne eşiği

_SCALE = torch.tensor([IMG_W, IMG_H, IMG_W, IMG_H], dtype=torch.float32) / (NUM_BINS - 1)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASYON — Large Scale Jittering
# ─────────────────────────────────────────────────────────────────────────────
# Sıra önemli: önce IMG_SIZE'a indir (bellek sınırlı kalsın), sonra jitter,
# sonra pad/crop. RandomScale ve RandomCrop p=1.0 — LSJ her örnekte uygulanır,
# paper'da olasılıklı değil.
train_transform = A.Compose([
    A.Resize(IMG_H, IMG_W),
    A.HorizontalFlip(p=0.5),
    A.RandomScale(scale_limit=(SCALE_RANGE[0] - 1.0, SCALE_RANGE[1] - 1.0), p=1.0),
    A.PadIfNeeded(min_height=IMG_H, min_width=IMG_W,
                  border_mode=0, fill=(128, 128, 128)),
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
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class Pix2SeqDataset(Dataset):
    """
    seq_in  = [BOS, x0,y0,x1,y1,cls, ..., nx0,ny0,nx1,ny1,NOISE, ...]
    seq_tgt = [     x0,y0,x1,y1,cls, ..., PAD,PAD,PAD,PAD,NOISE, ...]

    Noise koordinatları hedefte PAD ("n/a", loss ağırlığı 0) — paper Eq.1'deki
    w_j = 1[y_j != "n/a"] tam olarak bu.
    """

    def __init__(self, json_dir, img_dir, transform, is_train,
                 max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 permute_prob=PERMUTE_PROB, mosaic_prob=MOSAIC_PROB):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform
        self.is_train = is_train
        self.max_objects = max_objects
        self.img_size = img_size
        self.max_seq_len = 1 + TOKENS_PER_OBJ * max_objects
        self.permute_prob = permute_prob if is_train else 0.0
        self.mosaic_prob = mosaic_prob if is_train else 0.0

    def __len__(self):
        return len(self.json_files)

    def _read_item(self, idx):
        with open(os.path.join(self.json_dir, self.json_files[idx]),
                  'r', encoding='utf-8') as f:
            item = json.load(f)
        img_name = item.get("imagePath") or item.get("file_name")
        if img_name:
            img_path = os.path.join(self.img_dir, os.path.basename(img_name))
        else:
            base = self.json_files[idx][:-5]
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                cand = os.path.join(self.img_dir, base + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            else:
                img_path = os.path.join(self.img_dir, base + '.jpg')
        return item, img_path

    def _mosaic(self, idx):
        h, w = self.img_size
        indices = [idx] + random.sample(range(len(self.json_files)), 3)
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        out_shapes = []
        tiles = [(0, 0, w // 2, h // 2), (w // 2, 0, w // 2, h // 2),
                 (0, h // 2, w // 2, h // 2), (w // 2, h // 2, w // 2, h // 2)]
        for i, (xo, yo, tw, th) in zip(indices, tiles):
            item, img_path = self._read_item(i)
            try:
                img = Image.open(img_path).convert("RGB")
                img, max_dim = pad_to_square(img)
                img = img.resize((tw, th), Image.BILINEAR)
            except Exception:
                img = Image.fromarray(np.full((th, tw, 3), 114, dtype=np.uint8))
                max_dim = max(tw, th)
            canvas[yo:yo + th, xo:xo + tw] = np.array(img)
            for sh in load_shapes(item, self.img_dir):
                pts = sh["points"].copy()
                pts[:, 0] = pts[:, 0] / max_dim * tw + xo
                pts[:, 1] = pts[:, 1] / max_dim * th + yo
                out_shapes.append({"label": sh["label"], "points": pts})
        return canvas, out_shapes

    def __getitem__(self, idx):
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            image_np, shapes = self._mosaic(idx)
        else:
            item, img_path = self._read_item(idx)
            try:
                image = Image.open(img_path).convert("RGB")
                image, _ = pad_to_square(image)
                image_np = np.array(image)
            except Exception as e:
                print(f"HATA: {img_path} okunamadı: {e}")
                image_np = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
            shapes = load_shapes(item, self.img_dir)

        # Poligon noktaları keypoint olarak taşınır; kutu augmentasyondan SONRA
        pt_counts, all_points = [], []
        for sh in shapes:
            pt_counts.append(len(sh["points"]))
            all_points.extend([(float(p[0]), float(p[1])) for p in sh["points"]])

        aug = self.transform(image=image_np, keypoints=all_points)
        image_tensor = aug['image']
        new_points = [(float(p[0]), float(p[1])) for p in aug['keypoints']]

        # ── Gerçek kutular ──────────────────────────────────────────────────
        records, cursor = [], 0
        for sh, cnt in zip(shapes, pt_counts):
            pts = np.asarray(new_points[cursor:cursor + cnt], dtype=np.float32)
            cursor += cnt
            if len(pts) < 2:
                continue

            rx0, ry0 = float(pts[:, 0].min()), float(pts[:, 1].min())
            rx1, ry1 = float(pts[:, 0].max()), float(pts[:, 1].max())
            cx0, cy0 = max(0.0, rx0), max(0.0, ry0)
            cx1, cy1 = min(float(IMG_W), rx1), min(float(IMG_H), ry1)
            if (cx1 - cx0) < MIN_SIDE or (cy1 - cy0) < MIN_SIDE:
                continue

            # LSJ'de scale 3'e kadar çıkıyor; kadrajın çoğu dışında kalan nesneyi
            # kırpıp tutmak modele "kenarda hep şu sınıf var" diye yanlış bir
            # sinyal verir. Görünürlük oranı düşükse nesneyi tamamen at.
            raw_area = max((rx1 - rx0) * (ry1 - ry0), 1e-6)
            if ((cx1 - cx0) * (cy1 - cy0)) / raw_area < MIN_VISIBLE_FRAC:
                continue

            records.append({"box": (cx0, cy0, cx1, cy1),
                            "class_id": LABEL_TO_ID[sh["label"]],
                            "is_noise": False})

        # Paper: nesne sırası her gösterimde rastgele
        if self.permute_prob > 0 and random.random() < self.permute_prob:
            random.shuffle(records)
        records = records[:self.max_objects]
        n_real = len(records)

        # ── Sentetik noise — kuyruğa ────────────────────────────────────────
        if self.is_train:
            n_noise = self.max_objects - n_real
            if n_noise > 0:
                for nb in generate_noise_boxes([r["box"] for r in records],
                                               IMG_W, IMG_H, n_noise,
                                               NOISE_IOU_REJECT):
                    records.append({"box": nb, "class_id": NOISE_CLASS_ID,
                                    "is_noise": True})

        # ── Hizalanmış diziler ──────────────────────────────────────────────
        toks_in, toks_tgt = [], []
        for rec in records:
            x0, y0, x1, y1 = rec["box"]
            q = [quantize(x0, IMG_W, NUM_BINS), quantize(y0, IMG_H, NUM_BINS),
                 quantize(x1, IMG_W, NUM_BINS), quantize(y1, IMG_H, NUM_BINS)]
            if rec["is_noise"]:
                toks_in.extend(q + [NOISE_CLASS_TOKEN])
                toks_tgt.extend([PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN])
            else:
                c = NUM_BINS + rec["class_id"]
                toks_in.extend(q + [c])
                toks_tgt.extend(q + [c])

        L = self.max_seq_len
        seq_in = ([BOS_TOKEN] + toks_in)[:L]
        seq_tgt = toks_tgt[:L]
        seq_in += [PAD_TOKEN] * (L - len(seq_in))
        seq_tgt += [PAD_TOKEN] * (L - len(seq_tgt))

        return (image_tensor,
                torch.tensor(seq_in, dtype=torch.long),
                torch.tensor(seq_tgt, dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC DEPTH
# ─────────────────────────────────────────────────────────────────────────────
class DropPath(nn.Module):
    """
    Residual dalını örnek başına tamamen düşürür (paper: stochastic depth %10).

    nn.TransformerEncoderLayer/DecoderLayer'da residual şu şekilde:
        x = norm1(x + self.dropout1(self_attn(...)))
    dropout1/2/3 tam olarak dalın çıkışında duruyor, dolayısıyla onları bununla
    değiştirmek dalın TAMAMINI atlamak demek — eleman bazlı dropout'tan farkı bu.
    Eval modunda kimlik fonksiyonu, o yüzden KV-cache decode etkilenmez.
    """

    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x):
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # batch_first=True -> dim 0 batch
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"


def apply_stochastic_depth(module_stack, p_max):
    """Katman derinliğiyle lineer artan drop oranı (timm/ViT konvansiyonu)."""
    layers = list(module_stack.layers)
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
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Pix2SeqModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 max_seq_len=MAX_SEQ_LEN, img_size=IMG_SIZE,
                 dropout=DROPOUT, drop_path=DROP_PATH, norm_first=False,
                 dilated_c5=False):
        super().__init__()
        # dilated_c5=True -> son stage'in stride'ı kalkar, çıktı stride 32 yerine 16
        # (paper'ın DC5 varyantı). AP_s için en büyük tek kaldıraç ama encoder
        # self-attention maliyeti 16x. Bütçe varsa aç.
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
            norm=nn.LayerNorm(hidden_dim) if norm_first else None)

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[PAD_TOKEN].zero_()

        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(
            dec_layer, num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim) if norm_first else None)

        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        if drop_path > 0:
            apply_stochastic_depth(self.encoder, drop_path)
            apply_stochastic_depth(self.decoder, drop_path)

    def encode(self, images):
        f = self.enc_proj(self.backbone(images))
        return self.encoder(f.flatten(2).permute(0, 2, 1) + self.img_pos_emb)

    def forward(self, images, tgt_seq, tgt_key_padding_mask=None):
        memory = self.encode(images)
        mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_seq.size(1), device=images.device)
        emb = self.emb_dropout(self.seq_pos_encoding(self.embedding(tgt_seq)))
        out = self.decoder(tgt=emb, memory=memory, tgt_mask=mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        return self.fc_out(out)


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE'Lİ DECODE + EVAL  (bu dosyanın IMG_SIZE'ıyla tutarlı)
# ─────────────────────────────────────────────────────────────────────────────
class KVCacheDecoder:
    def __init__(self, model, memory):
        self.layers = model.decoder.layers
        self.norm = model.decoder.norm
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

    def _self_attn(self, layer, i, x):
        a = layer.self_attn
        D = a.embed_dim
        q, k, v = F.linear(x, a.in_proj_weight, a.in_proj_bias).split(D, dim=-1)
        q, k, v = (self._split(t, a.num_heads) for t in (q, k, v))
        self.self_k[i] = k if self.self_k[i] is None else torch.cat([self.self_k[i], k], 2)
        self.self_v[i] = v if self.self_v[i] is None else torch.cat([self.self_v[i], v], 2)
        o = F.scaled_dot_product_attention(q, self.self_k[i], self.self_v[i])
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

    def step(self, x):
        # model.eval() -> DropPath ve Dropout kimlik; residual'lar sadeleşiyor
        for i, layer in enumerate(self.layers):
            if getattr(layer, "norm_first", False):
                x = x + layer.dropout1(self._self_attn(layer, i, layer.norm1(x)))
                x = x + layer.dropout2(self._cross_attn(layer, i, layer.norm2(x)))
                x = x + layer.dropout3(self._ff(layer, layer.norm3(x)))
            else:
                x = layer.norm1(x + layer.dropout1(self._self_attn(layer, i, x)))
                x = layer.norm2(x + layer.dropout2(self._cross_attn(layer, i, x)))
                x = layer.norm3(x + layer.dropout3(self._ff(layer, x)))
        return x if self.norm is None else self.norm(x)


@torch.no_grad()
def ar_decode_batch(model, images):
    """
    Sabit uzunlukta, kısıtlı argmax decode. EOS yok — model "dur" kararını her
    slotta noise sınıfını seçerek veriyor (paper: altered inference).
    Returns coord_bins [B,S,4], cls [B,S], noise_won [B,S], obj_conf [B,S].
    """
    model.eval()
    B, S = images.size(0), MAX_OBJECTS
    device = images.device
    cache = KVCacheDecoder(model, model.encode(images))
    pe = model.seq_pos_encoding.pe
    seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)

    coord = torch.zeros(B, S, 4, dtype=torch.long, device=device)
    cls_id = torch.zeros(B, S, dtype=torch.long, device=device)
    noise_won = torch.zeros(B, S, dtype=torch.bool, device=device)
    conf = torch.zeros(B, S, device=device)
    lo, hi = NUM_BINS, NUM_BINS + NUM_CLASS_SLOTS

    for step in range(TOKENS_PER_OBJ * S):
        emb = model.embedding(seq[:, -1:]) + pe[:, step:step + 1, :]
        probs = torch.softmax(model.fc_out(cache.step(emb)[:, -1, :]).float(), -1)
        obj, pos = divmod(step, TOKENS_PER_OBJ)
        if pos == TOKENS_PER_OBJ - 1:
            p_real, r = probs[:, NUM_BINS:NUM_BINS + NUM_CLASSES].max(-1)
            p_noise = probs[:, NOISE_CLASS_TOKEN]
            nxt = probs[:, lo:hi].argmax(-1) + lo      # diziye eğitimdeki gibi
            cls_id[:, obj] = r
            conf[:, obj] = p_real / (p_real + p_noise + 1e-6)
            noise_won[:, obj] = (nxt == NOISE_CLASS_TOKEN)
        else:
            nxt = probs[:, :NUM_BINS].argmax(-1)
            coord[:, obj, pos] = nxt
        seq = torch.cat([seq, nxt.unsqueeze(1)], 1)

    return coord.cpu(), cls_id.cpu(), noise_won.cpu(), conf.cpu()


def gt_from_target(seq_tgt):
    toks = seq_tgt.tolist()
    boxes, labels = [], []
    for i in range(0, (len(toks) // TOKENS_PER_OBJ) * TOKENS_PER_OBJ, TOKENS_PER_OBJ):
        t = toks[i:i + TOKENS_PER_OBJ]
        c = t[4]
        if c in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break
        if c == NOISE_CLASS_TOKEN:
            continue
        if not (NUM_BINS <= c < NUM_BINS + NUM_CLASSES):
            continue
        if not all(0 <= t[j] < NUM_BINS for j in range(4)):
            continue
        b = [float(t[j] * _SCALE[j]) for j in range(4)]
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        boxes.append(b)
        labels.append(int(c - NUM_BINS))
    return boxes, labels


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype=None):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()
    n_img = n_pred = 0
    for images, _si, targets in tqdm(loader, desc="    eval", leave=False):
        images = images.to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            coord, cls_id, noise_won, conf = ar_decode_batch(model, images)

        for b in range(images.size(0)):
            boxes = coord[b].float() * _SCALE
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            keep &= ~noise_won[b]
            gb, gl = gt_from_target(targets[b])
            metric.update(
                [{"boxes": boxes[keep].reshape(-1, 4),
                  "scores": conf[b][keep].reshape(-1),
                  "labels": cls_id[b][keep].reshape(-1).long()}],
                [{"boxes": torch.tensor(gb, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(gl, dtype=torch.long).reshape(-1)}],
            )
            n_img += 1
            n_pred += int(keep.sum())

    r = metric.compute()
    return {k: float(r[k]) for k in
            ("map", "map_50", "map_75", "map_small", "map_medium", "map_large")}, \
           n_pred / max(n_img, 1)


# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTASYON KONTROLÜ
# ─────────────────────────────────────────────────────────────────────────────
def visualize_augmentations(dataset, out_dir="aug_check_paper", n=20):
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    tot_gt = tot_noise = 0
    for k in range(n):
        img_t, seq_in, _ = dataset[k % len(dataset)]
        img = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        toks = seq_in.tolist()[1:]
        n_gt = n_noise = 0
        for i in range(0, (len(toks) // 5) * 5, 5):
            t = toks[i:i + 5]
            if t[4] == PAD_TOKEN:
                break
            if not all(0 <= v < NUM_BINS for v in t[:4]):
                continue
            box = [float(t[j] * _SCALE[j]) for j in range(4)]
            if t[4] == NOISE_CLASS_TOKEN:
                n_noise += 1
                d.rectangle(box, outline="#ff2d2d", width=1)
            elif NUM_BINS <= t[4] < NUM_BINS + NUM_CLASSES:
                n_gt += 1
                d.rectangle(box, outline="#48f90a", width=3)
                d.text((box[0] + 3, max(0, box[1] - 12)),
                       ID_TO_LABEL[t[4] - NUM_BINS], fill="#48f90a")
        d.rectangle([0, 0, img.size[0], 18], fill="#141414")
        d.text((5, 4), f"GT: {n_gt}   noise: {n_noise}", fill="#eeeeee")
        img.save(os.path.join(out_dir, f"aug_{k:02d}_gt{n_gt}.jpg"), quality=92)
        tot_gt += n_gt; tot_noise += n_noise
    print(f"{n} görsel -> {out_dir}/  | ort GT {tot_gt/n:.2f} | ort noise {tot_noise/n:.2f}")
    if tot_gt / n < 3.0:
        print("  [uyarı] LSJ sonrası ortalama GT çok düşük. SCALE_RANGE üst sınırını "
              "veya MIN_VISIBLE_FRAC'i gözden geçir.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = USE_BF16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"device={device} | img={IMG_SIZE} | batch={BATCH_SIZE}x{GRAD_ACCUM}"
          f"={BATCH_SIZE*GRAD_ACCUM} | epochs={EPOCHS}")

    all_json = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    with open(VAL_SPLIT_PATH) as f:
        val_set = set(json.load(f)["filenames"])
    train_files = [f for f in all_json if f not in val_set]
    val_files = [f for f in all_json if f in val_set]

    train_ds = Pix2SeqDataset(JSON_DIR, IMG_DIR, train_transform, is_train=True)
    train_ds.json_files = train_files
    val_ds = Pix2SeqDataset(JSON_DIR, IMG_DIR, val_transform, is_train=False)
    val_ds.json_files = val_files
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=8, pin_memory=True, prefetch_factor=2,
                              drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    visualize_augmentations(train_ds)

    model = Pix2SeqModel().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN,
                                    label_smoothing=LABEL_SMOOTH)

    # AdamW: norm/bias/embedding parametrelerine weight decay uygulanmaz.
    # wd 0.05 ile bu ayrım artık önemli — 1e-4'te fark edilmezdi.
    decay, no_decay, bb_decay, bb_no_decay = [], [], [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_bb = n_.startswith("backbone")
        skip = p.ndim <= 1 or "img_pos_emb" in n_ or "embedding" in n_
        (bb_no_decay if skip else bb_decay).append(p) if is_bb else \
            (no_decay if skip else decay).append(p)

    optimizer = torch.optim.AdamW([
        {"params": bb_decay,    "lr": BACKBONE_LR, "weight_decay": WEIGHT_DECAY},
        {"params": bb_no_decay, "lr": BACKBONE_LR, "weight_decay": 0.0},
        {"params": decay,       "lr": BASE_LR,     "weight_decay": WEIGHT_DECAY},
        {"params": no_decay,    "lr": BASE_LR,     "weight_decay": 0.0},
    ])
    assert sum(len(g["params"]) for g in optimizer.param_groups) == \
        sum(1 for p in model.parameters() if p.requires_grad), "parametre kaçağı"

    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    def lr_lambda(step):                      # paper: warmup + lineer düşüş
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        prog = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - prog)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_map = 0.0
    history = []
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="ep")

    for epoch in epoch_bar:
        model.train()
        run_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        bar = tqdm(train_loader, desc=f"  Train {epoch+1}/{EPOCHS}",
                   leave=False, unit="b")
        for i, (images, dec_in, dec_tgt) in enumerate(bar):
            images = images.to(device, non_blocking=True).float()
            dec_in = dec_in.to(device, non_blocking=True)
            dec_tgt = dec_tgt.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type,
                                dtype=amp_dtype or torch.float32,
                                enabled=amp_enabled):
                logits = model(images, dec_in,
                               tgt_key_padding_mask=(dec_in == PAD_TOKEN))
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), dec_tgt.reshape(-1))

            (loss / GRAD_ACCUM).backward()

            if (i + 1) % GRAD_ACCUM == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                bar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{gn:.2f}")

            run_loss += loss.item()
            n_batches += 1

        train_loss = run_loss / max(n_batches, 1)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for images, dec_in, dec_tgt in tqdm(val_loader, desc=f"  Val {epoch+1}",
                                                leave=False, unit="b"):
                images = images.to(device, non_blocking=True).float()
                dec_in, dec_tgt = dec_in.to(device), dec_tgt.to(device)
                with torch.autocast(device_type=device.type,
                                    dtype=amp_dtype or torch.float32,
                                    enabled=amp_enabled):
                    logits = model(images, dec_in,
                                   tgt_key_padding_mask=(dec_in == PAD_TOKEN))
                    val_loss += criterion(logits.reshape(-1, VOCAB_SIZE),
                                          dec_tgt.reshape(-1)).item()
                n_val += 1
        val_loss /= max(n_val, 1)

        lr_now = optimizer.param_groups[2]["lr"]
        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
                              lr=f"{lr_now:.2e}", best=f"{best_map:.4f}")

        torch.save(model.state_dict(), f"{CKPT_PREFIX}_last.pth")

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == EPOCHS - 1:
            m, ppi = evaluate(model, val_loader, device, amp_dtype)
            history.append({"epoch": epoch + 1, "train": train_loss,
                            "val": val_loss, "lr": lr_now, "preds_per_img": ppi, **m})
            tqdm.write(
                f"  ep {epoch+1:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| lr {lr_now:.2e} | mAP {m['map']:.4f} | @50 {m['map_50']:.4f} "
                f"| S {m['map_small']:.4f} M {m['map_medium']:.4f} L {m['map_large']:.4f} "
                f"| pred/img {ppi:.1f}")
            if m["map"] > best_map:
                best_map = m["map"]
                torch.save(model.state_dict(), f"{CKPT_PREFIX}_best_map.pth")
                tqdm.write(f"       ✓ yeni en iyi mAP {best_map:.4f}")
            with open(f"{CKPT_PREFIX}_history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"\nBitti. En iyi mAP: {best_map:.4f} -> {CKPT_PREFIX}_best_map.pth")
