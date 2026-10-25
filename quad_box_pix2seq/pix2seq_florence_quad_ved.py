import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image, ImageOps, ImageDraw
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from transformers import (
    VisionEncoderDecoderModel,
    SiglipVisionConfig, SiglipVisionModel,
    MBartConfig, MBartForCausalLM,
    get_cosine_schedule_with_warmup,
)

# =============================================================================
# CONFIG
# =============================================================================
JSON_DIR       = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
IMG_DIR        = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
MAX_OBJECTS    = 128          # dataset_object_stats() p99'una göre ayarla
BATCH_SIZE     = 8
EPOCHS         = 350
LEARNING_RATE  = 3e-4         # enc_to_dec_proj + decoder (sıfırdan)
VISION_LR      = 2e-5         # SigLIP (pretrained)
WARMUP_RATIO   = 0.03
WEIGHT_DECAY   = 0.05
VAL_SPLIT_PATH = "val_split.json"
MOSAIC_PROB    = 0.0
EVAL_EVERY     = 10
INFER_SCORE_THRESH = 0.05

# SigLIP: 512 çözünürlük, patch16 -> 32x32 = 1024 patch; decoder hepsine cross-attention yapar.
VISION_MODEL_NAME = "google/siglip-base-patch16-512"   # alternatif: "google/siglip2-base-patch16-512"
IMG_SIZE          = (512, 512)                         # (h, w) -> vision config image_size ile aynı olmalı
SIGLIP_MEAN       = (0.5, 0.5, 0.5)
SIGLIP_STD        = (0.5, 0.5, 0.5)

# Decoder (mBART, sıfırdan): pre-LN, self-attn + cross-attn + FFN
DEC_HIDDEN = 512
DEC_LAYERS = 6
DEC_HEADS  = 8
DEC_FFN    = 2048

SAVE_BEST_LOSS = "pix2seq_ved_best_loss"
SAVE_BEST_MAP  = "pix2seq_ved_best_map"

LABEL_TO_ID = {
    "qr": 0, "menfaat": 1, "azami_yuk": 2, "kullanim_amaci": 3,
    "net_agirlik": 4, "ruhsat": 5, "romork_azami_yuk": 6,
    "plaka": 7, "tc": 8, "seri_no": 9
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
NUM_CLASSES = len(LABEL_TO_ID)

# =============================================================================
# TOKENIZER (Florence mantığı) — DEĞİŞMEDİ
# -----------------------------------------------------------------------------
# Sözlük düzeni:  [<s>, </s>, <pad>, <QUADBOX>] [<loc_0> ... <loc_999>] [sınıflar]
#                   0    1     2       3          4 ...           1003   1004 ...
#
# Bir obje = 9 token:  SINIF  x1 y1 x2 y2 x3 y3 x4 y4
#
# labels          : [<s>,       cls, loc×8, cls, loc×8, ..., </s>]   (collate üretir)
# decoder girdisi : [<QUADBOX>, <s>, cls, loc×8, ...]                (model üretir)
# -> VisionEncoderDecoderModel labels'ı sağa kaydırıp başa decoder_start_token_id
#    (= <QUADBOX>) koyar. Kaydırma ve loss transformers'ta.
# =============================================================================
BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TASK_TOKEN = "<s>", "</s>", "<pad>", "<QUADBOX>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, TASK_TOKEN]
NUM_BINS = 1000

all_tokens = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
all_tokens |= {f"<loc_{i}>": len(SPECIAL_TOKENS) + i for i in range(NUM_BINS)}
all_tokens |= {lbl: len(SPECIAL_TOKENS) + NUM_BINS + cid for lbl, cid in LABEL_TO_ID.items()}
id_to_token = {v: k for k, v in all_tokens.items()}

BOS_ID, EOS_ID, PAD_ID, TASK_ID = (all_tokens[t] for t in SPECIAL_TOKENS)
LOC_START   = all_tokens["<loc_0>"]          # 4
CLASS_START = LOC_START + NUM_BINS           # 1004
VOCAB_SIZE  = len(all_tokens)                # 1014
BLOCK       = 9                              # sınıf + 8 loc
MAX_SEQ_LEN = 2 + BLOCK * MAX_OBJECTS        # [task, bos] + objeler


def quantize(pts, w, h):
    """Florence binning: piksel -> [0, 999] int.  bin = floor(x / W * 1000)"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x = np.clip(pts[:, 0] / w * NUM_BINS, 0, NUM_BINS - 1)
    y = np.clip(pts[:, 1] / h * NUM_BINS, 0, NUM_BINS - 1)
    return np.stack([x, y], axis=1).astype(int)


def dequantize(bins, w, h):
    """bin -> piksel. +0.5: bin'in MERKEZİ."""
    b = np.asarray(bins, dtype=np.float64).reshape(-1, 2)
    return (b + 0.5) / NUM_BINS * np.array([w, h], dtype=np.float64)


def order_quad(pts):
    """(4,2) -> sol-üst köşeden başlayıp saat yönünde sıralı (y aşağı)."""
    pts = np.asarray(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -start, axis=0)


def encode_objects(objs):
    """[(class_id, (4,2) bin)] -> token id listesi (sadece gövde)."""
    ids = []
    for cls, bins in objs:
        ids.append(CLASS_START + cls)
        ids.extend((LOC_START + np.asarray(bins).reshape(-1)).tolist())
    return ids


def decode_tokens(tokens, scores=None):
    """Token listesi (task/bos SONRASI) -> [(class_id, (4,2) bin, score), ...]"""
    out = []
    for k, i in enumerate(range(0, len(tokens) - BLOCK + 1, BLOCK)):
        cls_tok, locs = tokens[i], tokens[i + 1:i + BLOCK]
        if cls_tok in (EOS_ID, PAD_ID):
            break
        if not (CLASS_START <= cls_tok < CLASS_START + NUM_CLASSES):
            break
        if not all(LOC_START <= t < CLASS_START for t in locs):
            break
        bins = (np.asarray(locs) - LOC_START).reshape(4, 2)
        s = float(scores[k]) if scores is not None and k < len(scores) else 1.0
        out.append((cls_tok - CLASS_START, bins, s))
    return out


# =============================================================================
# VERİ KONTROLLERİ
# =============================================================================
def check_exif_and_sizes(json_dir=JSON_DIR, img_dir=IMG_DIR):
    n_swap = n_diff = n_exif = 0
    for jf in os.listdir(json_dir):
        if not jf.endswith('.json'):
            continue
        with open(os.path.join(json_dir, jf), encoding='utf-8') as f:
            item = json.load(f)
        jw, jh = item.get("imageWidth"), item.get("imageHeight")
        if not jw:
            continue
        p = os.path.join(img_dir, item.get("imagePath", jf.replace('.json', '.jpg')))
        if not os.path.exists(p):
            continue
        im = Image.open(p)
        raw = im.size
        ori = (im.getexif() or {}).get(274, 1)
        if ori not in (1, 0, None):
            n_exif += 1
        if raw == (jh, jw) and jw != jh:
            n_swap += 1
            print(f"SWAP  {jf}: json=({jw},{jh}) raw={raw} exif_ori={ori}")
        elif raw != (jw, jh):
            n_diff += 1
            print(f"DIFF  {jf}: json=({jw},{jh}) raw={raw} exif_ori={ori}")
    print(f"\nEXIF orientation != 1 : {n_exif}\nBoyut swap            : {n_swap}\n"
          f"Diğer uyuşmazlık      : {n_diff}")


def dataset_object_stats(dataset, n=None):
    n = n or len(dataset)
    counts = []
    for i in range(min(n, len(dataset))):
        with open(os.path.join(dataset.json_dir, dataset.json_files[i]), encoding='utf-8') as f:
            item = json.load(f)
        counts.append(sum(1 for sh in item.get("shapes", [])
                          if sh.get("label") in LABEL_TO_ID and len(sh.get("points", [])) == 4))
    c = np.asarray(counts)
    print(f"Nesne/görüntü — ort {c.mean():.2f} | medyan {np.median(c):.0f} | "
          f"p99 {np.percentile(c, 99):.0f} | max {c.max()} | MAX_OBJECTS={MAX_OBJECTS}")
    return c


# =============================================================================
# AUGMENTATION  (normalize: SigLIP mean/std = 0.5)
# =============================================================================
_kp = A.KeypointParams(format='xy', remove_invisible=False, label_fields=['kp_ids'])

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
    A.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ToTensorV2()
], keypoint_params=_kp)

val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=SIGLIP_MEAN, std=SIGLIP_STD),
    ToTensorV2()
], keypoint_params=_kp)


def pad_to_square(image, fill_color=(128, 128, 128)):
    """Görüntü SOL-ÜSTTE -> koordinatlar değişmez."""
    w, h = image.size
    max_dim = max(w, h)
    new_image = Image.new("RGB", (max_dim, max_dim), fill_color)
    new_image.paste(image, (0, 0))
    return new_image, max_dim


def load_image(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


# =============================================================================
# DATASET + COLLATE
# =============================================================================
class Pix2SeqDataset(Dataset):
    def __init__(self, json_dir, img_dir, img_size=IMG_SIZE, transform=train_transform,
                 max_objects=MAX_OBJECTS, mosaic_prob=0.0):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.transform = transform
        self.max_objects = max_objects
        self.mosaic_prob = mosaic_prob

    def __len__(self):
        return len(self.json_files)

    def _read(self, idx):
        with open(os.path.join(self.json_dir, self.json_files[idx]), encoding='utf-8') as f:
            item = json.load(f)
        img_name = item.get("imagePath")
        if img_name:
            img_path = os.path.join(self.img_dir, img_name)
        else:
            base = self.json_files[idx].replace('.json', '')
            img_path = next((os.path.join(self.img_dir, base + e)
                             for e in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                             if os.path.exists(os.path.join(self.img_dir, base + e))),
                            os.path.join(self.img_dir, base + '.jpg'))
        try:
            img, max_dim = pad_to_square(load_image(img_path))
        except Exception as e:
            print(f"HATA: {img_path} okunamadı. Hata: {e}")
            img, max_dim = Image.new("RGB", (self.img_size[1], self.img_size[0])), self.img_size[1]

        shapes = [(LABEL_TO_ID[s["label"]], s["points"]) for s in item.get("shapes", [])
                  if s.get("label") in LABEL_TO_ID and len(s.get("points", [])) == 4]
        return img, max_dim, shapes

    def _mosaic(self, idx):
        h, w = self.img_size
        mosaic = np.full((h, w, 3), 114, dtype=np.uint8)
        shapes = []
        tiles = [(0, 0), (w // 2, 0), (0, h // 2), (w // 2, h // 2)]
        tw, th = w // 2, h // 2
        for i, (xo, yo) in zip([idx] + random.sample(range(len(self)), 3), tiles):
            img, max_dim, shp = self._read(i)
            mosaic[yo:yo + th, xo:xo + tw] = np.array(img.resize((tw, th), Image.BILINEAR))
            for cls, pts in shp:
                shapes.append((cls, [(px / max_dim * tw + xo, py / max_dim * th + yo) for px, py in pts]))
        return mosaic, shapes

    def __getitem__(self, idx):
        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            image_np, shapes = self._mosaic(idx)
        else:
            img, _, shapes = self._read(idx)
            image_np = np.array(img)

        pts = [(float(x), float(y)) for _, s in shapes for x, y in s]
        kp_ids = [si for si, (_, s) in enumerate(shapes) for _ in s]
        aug = self.transform(image=image_np, keypoints=pts, kp_ids=kp_ids)
        image = aug["image"]

        groups = {}
        for (x, y), si in zip((tuple(k[:2]) for k in aug["keypoints"]), aug["kp_ids"]):
            groups.setdefault(si, []).append((x, y))

        H, W = self.img_size
        objs = []
        for si, (cls, _) in enumerate(shapes):
            q = groups.get(si, [])
            if len(q) != 4:
                continue
            q = np.asarray(q, dtype=np.float32)
            if q[:, 0].min() < -8 or q[:, 1].min() < -8 or q[:, 0].max() > W + 8 or q[:, 1].max() > H + 8:
                continue
            objs.append((cls, quantize(order_quad(q), W, H)))

        objs.sort(key=lambda o: (o[1][0, 1], o[1][0, 0]))  # raster sırası
        objs = objs[:self.max_objects]
        return {"image": image.float(), "token_ids": encode_objects(objs)}


def collate_fn(batch):
    """labels = [<s>] + gövde + [</s>]; padding -100 (loss'ta yok sayılır)."""
    labels = [torch.tensor([BOS_ID] + b["token_ids"] + [EOS_ID], dtype=torch.long) for b in batch]
    return {
        "pixel_values": torch.stack([b["image"] for b in batch]),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }


# =============================================================================
# MODEL — VisionEncoderDecoderModel
# -----------------------------------------------------------------------------
#   SigLIP (pretrained)  --enc_to_dec_proj (768->512, otomatik)-->  mBART decoder (sıfırdan)
#
# Cross-attention, KV cache ve generate() transformers'ta. Görüntü token'ları decoder
# dizisine girmez, cross-attention ile okunur. Bu yüzden 32x32 = 1024 patch'in hepsi
# kullanılabilir, pixel shuffle'a gerek yok.
#
# Decoder seçimi: VisionEncoderDecoder decoder'a cross-attention ekler. Llama/Qwen gibi
# decoder-only modellerde cross-attention yok. mBART decoder pre-LN (sıfırdan eğitimde
# stabil); Florence-2 de bu ailede bir decoder kullanıyor.
# Alternatifler: GPT2LMHeadModel(add_cross_attention=True), TrOCRForCausalLM.
# =============================================================================
def build_model(vision_name=VISION_MODEL_NAME):
    """Eğitim başlangıcı: SigLIP pretrained, projeksiyon + decoder sıfırdan."""
    vcfg = SiglipVisionConfig.from_pretrained(vision_name)
    vcfg.vision_use_head = False          # pooling head'e gerek yok (cross-attn tüm patch'leri okur)
    encoder = SiglipVisionModel.from_pretrained(vision_name, config=vcfg)
    assert encoder.config.image_size == IMG_SIZE[0] == IMG_SIZE[1], \
        f"IMG_SIZE {IMG_SIZE} != SigLIP image_size {encoder.config.image_size}"

    dec_cfg = MBartConfig(
        vocab_size=VOCAB_SIZE,
        d_model=DEC_HIDDEN,
        decoder_layers=DEC_LAYERS,
        decoder_attention_heads=DEC_HEADS,
        decoder_ffn_dim=DEC_FFN,
        max_position_embeddings=MAX_SEQ_LEN + 8,
        dropout=0.1, attention_dropout=0.0, activation_dropout=0.0,
        activation_function="gelu",
        scale_embedding=True,
        pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID,
        decoder_start_token_id=TASK_ID,
        forced_eos_token_id=None,      # mBART varsayılanı 2 (= bizim <pad>) -> kapat
        is_decoder=True, add_cross_attention=True, is_encoder_decoder=False,
        tie_word_embeddings=False,
    )
    model = VisionEncoderDecoderModel(encoder=encoder, decoder=MBartForCausalLM(dec_cfg))

    for cfg in (model.config, model.generation_config):
        cfg.decoder_start_token_id = TASK_ID
        cfg.pad_token_id = PAD_ID
        cfg.eos_token_id = EOS_ID
        cfg.bos_token_id = BOS_ID
        cfg.forced_eos_token_id = None   # sonlandırmayı gramer processor'ı yönetir
    model.config.vocab_size = VOCAB_SIZE
    return model


# =============================================================================
# INFERENCE
# =============================================================================
PROMPT = [TASK_ID, BOS_ID]
_CLASS_OR_EOS = list(range(CLASS_START, CLASS_START + NUM_CLASSES)) + [EOS_ID]
_LOCS = list(range(LOC_START, CLASS_START))


@torch.no_grad()
def generate(model, images, max_objects=MAX_OBJECTS, amp_dtype=torch.bfloat16):
    """
    Greedy, gramer kısıtlı model.generate(). Başlangıç [<QUADBOX>, <s>].
    Döner:
        tokens : (B, T) üretilen token'lar (prompt hariç; </s> sonrası <pad>)
        scores : (B, n_blok) her blok başındaki token olasılığı = obje skoru
    """
    max_new = max_objects * BLOCK + 1

    def allowed(batch_id, ids):
        t = ids.shape[-1] - len(PROMPT)
        if t % BLOCK:
            return _LOCS
        return [EOS_ID] if t == max_new - 1 else _CLASS_OR_EOS

    prompt = torch.tensor([PROMPT], device=images.device).repeat(images.size(0), 1)
    with torch.autocast(device_type=images.device.type, dtype=amp_dtype,
                        enabled=images.device.type == "cuda" and amp_dtype is not None):
        out = model.generate(pixel_values=images, decoder_input_ids=prompt,
                             max_new_tokens=max_new, do_sample=False, num_beams=1,
                             prefix_allowed_tokens_fn=allowed,
                             return_dict_in_generate=True, output_scores=True)

    tokens = out.sequences[:, len(PROMPT):]
    logp = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
    return tokens, logp[:, ::BLOCK].float().exp()


@torch.no_grad()
def predict(model, img_path, device="cuda", score_thresh=INFER_SCORE_THRESH):
    """Tek görüntü -> orijinal piksel koordinatlarında [(label, (4,2) pts, score), ...]"""
    img = load_image(img_path)
    padded, max_dim = pad_to_square(img)
    x = val_transform(image=np.array(padded), keypoints=[], kp_ids=[])["image"]
    tokens, scores = generate(model, x[None].float().to(device))
    preds = []
    for cls, bins, s in decode_tokens(tokens[0].tolist(), scores[0].tolist()):
        if s < score_thresh:
            continue
        preds.append((ID_TO_LABEL[cls], dequantize(bins, max_dim, max_dim), s))
    return img, preds


PALETTE = ["#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231",
           "#48f90a", "#92cc17", "#3ddb86", "#1a9334", "#00c2ff"]


def draw_predictions(img, preds, out_path=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    for label, pts, s in preds:
        poly = [tuple(p) for p in pts]
        color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]
        d.line(poly + [poly[0]], fill=color, width=3)
        d.ellipse([poly[0][0] - 5, poly[0][1] - 5, poly[0][0] + 5, poly[0][1] + 5], fill="red")
        d.text((poly[0][0] + 7, poly[0][1] - 13), f"{label} {s:.2f}", fill=color)
    if out_path:
        img.save(out_path)
    return img


# =============================================================================
# DEĞERLENDİRME (mAP, quad IoU) — DEĞİŞMEDİ
# =============================================================================
def _poly_signed_area(p):
    p = np.asarray(p, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _poly_area(p):
    return abs(_poly_signed_area(p))


def _convex_clip(subject, clip):
    def side(a, b, q):
        return (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])

    def isect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return (x2, y2)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    clip = np.asarray(clip, dtype=np.float64)
    if _poly_signed_area(clip) < 0:
        clip = clip[::-1]
    out = [(float(q[0]), float(q[1])) for q in subject]
    for i in range(len(clip)):
        if not out:
            return np.zeros((0, 2))
        a, b = clip[i], clip[(i + 1) % len(clip)]
        new = []
        for j in range(len(out)):
            cur, prev = out[j], out[j - 1]
            c_in, p_in = side(a, b, cur) >= -1e-12, side(a, b, prev) >= -1e-12
            if c_in:
                if not p_in:
                    new.append(isect(prev, cur, a, b))
                new.append(cur)
            elif p_in:
                new.append(isect(prev, cur, a, b))
        out = new
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def quad_iou(a, b):
    ua, ub = _poly_area(a), _poly_area(b)
    if ua < 1e-6 or ub < 1e-6:
        return 0.0
    inter = _convex_clip(a, b)
    if len(inter) < 3:
        return 0.0
    ai = _poly_area(inter)
    return float(ai / max(ua + ub - ai, 1e-6))


def _average_precision(tp, n_gt):
    if n_gt == 0:
        return None
    if len(tp) == 0:
        return 0.0
    tp = np.asarray(tp, dtype=np.float64)
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    rec = tpc / n_gt
    prec = tpc / np.maximum(tpc + fpc, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@torch.no_grad()
def evaluate(model, loader, device, iou_thrs=np.arange(0.5, 0.96, 0.05)):
    model.eval()
    preds = {c: [] for c in range(NUM_CLASSES)}
    gts = {}
    n_gt = np.zeros(NUM_CLASSES, dtype=int)
    key = 0
    for batch in tqdm(loader, desc="  mAP", leave=False):
        tokens, scores = generate(model, batch["pixel_values"].to(device).float())
        for i in range(tokens.size(0)):
            for cls, bins, s in decode_tokens(tokens[i].tolist(), scores[i].tolist()):
                preds[cls].append((s, key, bins.astype(np.float64)))
            for cls, bins, _ in decode_tokens(batch["labels"][i, 1:].tolist()):
                gts.setdefault((key, cls), []).append(bins.astype(np.float64))
                n_gt[cls] += 1
            key += 1

    aps = {thr: [] for thr in iou_thrs}
    corner_err = []
    for c in range(NUM_CLASSES):
        if n_gt[c] == 0:
            continue
        ranked = sorted(preds[c], key=lambda p: -p[0])
        for thr in iou_thrs:
            used = {k: [False] * len(v) for k, v in gts.items() if k[1] == c}
            tp = []
            for _, k, q in ranked:
                cand = gts.get((k, c), [])
                best, bj = 0.0, -1
                for j, g in enumerate(cand):
                    if used[(k, c)][j]:
                        continue
                    iou = quad_iou(q, g)
                    if iou > best:
                        best, bj = iou, j
                if best >= thr:
                    used[(k, c)][bj] = True
                    tp.append(1)
                    if abs(thr - 0.5) < 1e-9:
                        corner_err.append(np.linalg.norm(q - cand[bj], axis=1).mean() / NUM_BINS)
                else:
                    tp.append(0)
            aps[thr].append(_average_precision(tp, n_gt[c]))

    map_per_thr = [np.mean(v) for v in aps.values() if v]
    return {
        "map": float(np.mean(map_per_thr)) if map_per_thr else 0.0,
        "map_50": float(np.mean(aps[iou_thrs[0]])) if aps[iou_thrs[0]] else 0.0,
        "nme": float(np.mean(corner_err)) if corner_err else float("nan"),
    }


# =============================================================================
# GÖRSEL KONTROL
# =============================================================================
def visualize_augmentations(dataset, out_dir="aug_check", n=20):
    os.makedirs(out_dir, exist_ok=True)
    mean, std = np.array(SIGLIP_MEAN), np.array(SIGLIP_STD)
    H, W = dataset.img_size
    total = 0
    for k in range(n):
        idx = k % len(dataset)
        b = collate_fn([dataset[idx]])
        img = (b["pixel_values"][0].permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d = ImageDraw.Draw(img)

        objs = decode_tokens(b["labels"][0, 1:].tolist())
        for order, (cls, bins, _) in enumerate(objs):
            poly = [tuple(p) for p in dequantize(bins, W, H)]
            color = PALETTE[cls % len(PALETTE)]
            d.line(poly + [poly[0]], fill=color, width=3)
            d.ellipse([poly[0][0] - 5, poly[0][1] - 5, poly[0][0] + 5, poly[0][1] + 5], fill="red")
            d.ellipse([poly[1][0] - 4, poly[1][1] - 4, poly[1][0] + 4, poly[1][1] + 4], fill="yellow")
            d.text((poly[0][0] + 7, poly[0][1] - 13), f"{order}:{ID_TO_LABEL[cls]}", fill=color)

        stem = os.path.splitext(dataset.json_files[idx])[0][:40]
        img.save(os.path.join(out_dir, f"aug_check_{k:02d}_{stem}_n{len(objs)}.jpg"), quality=92)
        total += len(objs)
    print(f"{n} görsel -> {out_dir}/ | ort. quad: {total / max(n, 1):.2f}")


# =============================================================================
# EĞİTİM
# =============================================================================
def param_groups(model):
    """SigLIP düşük LR; norm/bias/embedding'lerde weight decay yok."""
    groups = {"v_decay": [], "v_nodecay": [], "d_decay": [], "d_nodecay": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nodecay = p.ndim < 2 or "norm" in n or "embed" in n
        pre = "v" if n.startswith("encoder.") else "d"
        groups[f"{pre}_{'nodecay' if nodecay else 'decay'}"].append(p)
    return [
        {"params": groups["v_decay"],   "lr": VISION_LR,     "weight_decay": WEIGHT_DECAY},
        {"params": groups["v_nodecay"], "lr": VISION_LR,     "weight_decay": 0.0},
        {"params": groups["d_decay"],   "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        {"params": groups["d_nodecay"], "lr": LEARNING_RATE, "weight_decay": 0.0},
    ]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_BF16 = True
    amp_enabled = USE_BF16 and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    if os.path.exists(VAL_SPLIT_PATH):
        print(f"Mevcut val split yükleniyor: {VAL_SPLIT_PATH}")
        with open(VAL_SPLIT_PATH) as f:
            val_set = set(json.load(f)["filenames"])
        train_files = [f for f in all_json_files if f not in val_set]
        val_files = [f for f in all_json_files if f in val_set]
    else:
        print(f"Val split bulunamadı, yeni oluşturuluyor → {VAL_SPLIT_PATH}")
        random.shuffle(all_json_files)
        cut = int(0.9 * len(all_json_files))
        train_files, val_files = all_json_files[:cut], all_json_files[cut:]
        with open(VAL_SPLIT_PATH, "w") as f:
            json.dump({"filenames": val_files}, f, indent=2)

    train_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform, mosaic_prob=MOSAIC_PROB)
    train_dataset.json_files = train_files
    val_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform)
    val_dataset.json_files = val_files
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                  pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                pin_memory=True, prefetch_factor=2, collate_fn=collate_fn)

    model = build_model().to(device)
    # Bellek darsa: model.gradient_checkpointing_enable()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Eğitim {device} | vocab {VOCAB_SIZE} | max_seq {MAX_SEQ_LEN} | {n_params:.1f}M param")

    optimizer = torch.optim.AdamW(param_groups(model), betas=(0.9, 0.98), eps=1e-6)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    dataset_object_stats(train_dataset)
    visualize_augmentations(train_dataset, out_dir="aug_check", n=20)

    best_val_loss, best_map = float('inf'), 0.0
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"  Train {epoch+1}/{EPOCHS}", leave=False, unit="batch")
        for batch in train_bar:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = model(pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                             labels=batch["labels"].to(device, non_blocking=True)).loss
            optimizer.zero_grad(set_to_none=True)
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
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"  Val   {epoch+1}/{EPOCHS}", leave=False, unit="batch"):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    val_loss += model(pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                                      labels=batch["labels"].to(device, non_blocking=True)).loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        epoch_bar.set_postfix(train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}",
                              best=f"{best_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(SAVE_BEST_LOSS)
            tqdm.write(f"  ✓ Epoch {epoch+1:3d} — val loss {avg_val_loss:.4f} → {SAVE_BEST_LOSS}/")

        if (epoch + 1) % EVAL_EVERY == 0:
            res = evaluate(model, val_dataloader, device)
            tqdm.write(f"  mAP {res['map']:.4f} | mAP50 {res['map_50']:.4f} | NME {res['nme']:.4f}")
            if res["map"] > best_map:
                best_map = res["map"]
                model.save_pretrained(SAVE_BEST_MAP)
                tqdm.write(f"  ✓ yeni en iyi mAP {best_map:.4f} → {SAVE_BEST_MAP}/")

    print("\nEğitim tamamlandı.")
    print(f"  En iyi loss modeli: {SAVE_BEST_LOSS}/ (val loss {best_val_loss:.4f})")
    print(f"  En iyi mAP modeli : {SAVE_BEST_MAP}/  (mAP {best_map:.4f})")