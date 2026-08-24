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
MAX_OBJECTS = 30
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

NUM_CLASSES = len(LABEL_TO_ID)          # SADECE gerçek sınıflar

# ── Sequence augmentation ────────────────────────────────────────────────────
# DİKKAT: "noise" LABEL_TO_ID'ye EKLENMEZ. Eklenirse (a) NUM_CLASSES kayar,
# (b) eval per-class AP'ye sahte bir sınıf girer, (c) labelme'de yanlışlıkla
# "noise" etiketli bir shape gerçek GT sayılır. Ayrı bir token olarak durur.
NOISE_CLASS_ID     = NUM_CLASSES                   # 10
NOISE_CLASS_TOKEN  = NUM_BINS + NOISE_CLASS_ID     # 510
NUM_CLASS_SLOTS    = NUM_CLASSES + 1               # gerçek + noise

# Kuyruk MAX_OBJECTS'e kadar noise ile DOLDURULUR: görüntü başına noise sayısı
# serbest bir parametre değil, (MAX_OBJECTS - gerçek nesne) olarak belirlenir.
# Bu yüzden MAX_OBJECTS'i gerçek dağılıma göre ayarlamak ZORUNLU:
# dataset_object_stats() p99'u söyler, MAX_OBJECTS ~ p99 + biraz pay olmalı.
# 40 iken 8 nesneli bir ruhsatta 32 noise üretilir - oran 4:1, dizi 361 token.
# 16 iken 8 noise üretilir - oran 1:1, dizi 145 token, decode 2.5x hızlı.
NOISE_FILL_TO_MAX  = True
NOISE_MAX_PER_IMG  = None   # normalde None. Sadece MAX_OBJECTS'i küçültemiyorsan sert sınır koy.
NOISE_RATIO        = 0.5    # NOISE_FILL_TO_MAX=False iken gerçek nesne başına oran
NOISE_IOU_REJECT   = 0.5    # gerçek bir quad ile bu IoU'yu aşan sentetik quad elenir
NOISE_JITTER_FRAC  = 0.75   # sentetik quadların ne kadarı gerçek quad bozularak üretilir
INFER_SCORE_THRESH = 0.05

VOCAB_SIZE = NUM_BINS + NUM_CLASS_SLOTS + 3
BOS_TOKEN = VOCAB_SIZE - 3
EOS_TOKEN = VOCAB_SIZE - 2
PAD_TOKEN = VOCAB_SIZE - 1




def check_exif_and_sizes(json_dir=JSON_DIR, img_dir=IMG_DIR):
    """JSON boyutları ile gerçek görüntü boyutlarını karşılaştırır.
    ÖNEMLİ: eskiden modül seviyesindeydi, yani bu dosyayı import eden HER script
    (eval, inference) tüm dataseti tarıyordu. Artık isteyerek çağrılıyor."""
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


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE AUGMENTATION - QUAD GEOMETRİSİ
# ─────────────────────────────────────────────────────────────────────────────
# Quad'a özgü kritik nokta: sentetik nesneler GERÇEKÇİ quad olmak zorunda.
# Kendi kendini kesen ya da absürt oranlı bir dörtgen üretirsek model "garip
# şekil = noise" kısayolunu öğrenir ve bu hiçbir işe yaramaz. O yüzden sentetik
# quadlar da döndürülmüş dikdörtgen + hafif perspektif olarak üretilir ve
# köşe sırası (winding + başlangıç köşesi) gerçek annotasyonla eşleştirilir.

def _poly_signed_area(p):
    p = np.asarray(p, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _poly_area(p):
    return abs(_poly_signed_area(p))


def _convex_clip(subject, clip):
    """Sutherland-Hodgman; clip konveks olmalı."""
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


def quad_iou_fast(a, b):
    """Konveks quadlar için kesin IoU; konveks değilse 0 döner (elenmez)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    ua = _poly_area(a); ub = _poly_area(b)
    if ua < 1e-6 or ub < 1e-6:
        return 0.0
    inter = _convex_clip(a, b)
    if len(inter) < 3:
        return 0.0
    ai = _poly_area(inter)
    return float(ai / max(ua + ub - ai, 1e-6))


def _match_winding_and_start(quad, ref):
    """Sentetik quadın köşe sırasını referans (gerçek) quadın konvansiyonuna uydurur."""
    quad = np.asarray(quad, dtype=np.float64)
    if ref is None:
        return quad
    ref = np.asarray(ref, dtype=np.float64)
    if np.sign(_poly_signed_area(quad)) != np.sign(_poly_signed_area(ref)):
        quad = quad[::-1].copy()

    # referansın 1. köşesi merkeze göre hangi yönde ise, sentetiği de oradan başlat
    rc = ref.mean(0); qc = quad.mean(0)
    rv = ref[0] - rc
    rv = rv / max(np.linalg.norm(rv), 1e-6)
    best, best_dot = 0, -2.0
    for k in range(4):
        v = quad[k] - qc
        v = v / max(np.linalg.norm(v), 1e-6)
        d = float(np.dot(v, rv))
        if d > best_dot:
            best, best_dot = k, d
    return np.roll(quad, -best, axis=0)


def _rect_quad(cx, cy, w, h, angle_deg, persp=0.0):
    """Döndürülmüş dikdörtgen + köşe başına hafif perspektif sapması."""
    hw, hh = w * 0.5, h * 0.5
    base = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    if persp > 0.0:
        base += np.random.uniform(-persp, persp, size=(4, 2)) * np.array([w, h])
    t = math.radians(angle_deg)
    R = np.array([[math.cos(t), -math.sin(t)],
                  [math.sin(t),  math.cos(t)]], dtype=np.float64)
    return base @ R.T + np.array([cx, cy], dtype=np.float64)


def generate_noise_quads(real_quads, img_w, img_h, count,
                         max_iou=NOISE_IOU_REJECT, jitter_frac=NOISE_JITTER_FRAC):
    """
    Tip A (jitter_frac): gerçek bir quadı ölçekle + döndür + kaydır + köşe titret.
                         Köşe sırası otomatik olarak gerçek konvansiyonu miras alır.
                         Asıl öğretici olan zor negatifler bunlar.
    Tip B: rastgele döndürülmüş dikdörtgen. Boyut/oran, varsa gerçek quadlardan
           örneklenir - "ince uzun alan kutusu" görünümünü korumak için.

    Eleme kuralları:
      * gerçek bir quad ile IoU > max_iou  -> modele "doğru tespit = noise"
        öğretmemek için elenir
      * kadraj dışına taşan aday elenir (köşeleri tek tek clip ETMEK yerine).
        Clip, dörtgeni kenara yapıştırıp dejenere/yassı şekiller üretiyordu;
        model bunu "yamuk şekil = noise" kısayoluna çevirir.
    """
    out = []
    attempts = 0
    ref = real_quads[0] if real_quads else None
    m = 2.0   # kadraj toleransı (px)

    def in_frame(q):
        return (q[:, 0].min() >= -m and q[:, 1].min() >= -m and
                q[:, 0].max() <= img_w + m and q[:, 1].max() <= img_h + m)

    # gerçek quadların boyut dağılımı (Tip B için)
    if real_quads:
        sizes = np.array([[q[:, 0].max() - q[:, 0].min(),
                           q[:, 1].max() - q[:, 1].min()] for q in real_quads])
    else:
        sizes = None

    while len(out) < count and attempts < max(count, 1) * 40:
        attempts += 1

        if real_quads and random.random() < jitter_frac:
            q = np.asarray(random.choice(real_quads), dtype=np.float64)
            c = q.mean(0)
            diag = float(np.linalg.norm(q.max(0) - q.min(0))) or 10.0

            scale = np.random.uniform(0.6, 1.5)
            ang   = math.radians(np.random.uniform(-20, 20))
            R = np.array([[math.cos(ang), -math.sin(ang)],
                          [math.sin(ang),  math.cos(ang)]], dtype=np.float64)
            cand = (q - c) * scale @ R.T + c
            cand += np.random.uniform(-0.7, 0.7, size=2) * diag       # kaydırma
            cand += np.random.normal(0, 0.04 * diag, size=(4, 2))     # köşe titremesi
        else:
            if sizes is not None:
                w, h = sizes[np.random.randint(len(sizes))] * np.random.uniform(0.7, 1.4, size=2)
            else:
                w = np.random.uniform(0.10, 0.40) * img_w
                h = np.random.uniform(0.03, 0.10) * img_h
            cand = _rect_quad(np.random.uniform(0.1, 0.9) * img_w,
                              np.random.uniform(0.1, 0.9) * img_h,
                              float(w), float(h), np.random.uniform(-20, 20), persp=0.03)
            cand = _match_winding_and_start(cand, ref)

        if not in_frame(cand):
            continue
        if _poly_area(cand) < 64.0:
            continue
        if any(quad_iou_fast(cand, rq) > max_iou for rq in real_quads):
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


def dequantize(token, max_size, num_bins=NUM_BINS):
    # quantize -> int(norm * (num_bins-1)) olduğu için tersi de (num_bins-1)'e bölmeli.
    # Eski (token+0.5)/num_bins hâli yarım bin kayma bırakıyordu.
    return (token / (num_bins - 1)) * max_size

def decode_sequence(seq, img_w, img_h, keep_noise=False):
    """
    [BOS] (x1 y1 ... x4 y4 cls) x N [EOS] -> [(poly, label_str, is_noise), ...]

    Noise objeleri de 9 tokenlık tam bir chunk kaplar, o yüzden hizalamayı
    bozmadan atlanabilirler (break değil, continue). keep_noise=True ile
    görselleştirme için geri döndürülürler.
    """
    toks = seq.tolist() if torch.is_tensor(seq) else list(seq)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    out, i = [], 0
    while i + 9 <= len(toks):
        chunk = toks[i:i + 9]
        coords, cls_tok = chunk[:8], chunk[8]
        i += 9

        if cls_tok in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break                                        # dizi bitti
        is_noise = (cls_tok == NOISE_CLASS_TOKEN)
        if not is_noise and not (NUM_BINS <= cls_tok < NUM_BINS + NUM_CLASSES):
            break                                        # hizalama bozulmuş
        if any(not (0 <= c < NUM_BINS) for c in coords):
            continue                                     # hedef dizisinde noise: koordinatlar PAD
        if is_noise and not keep_noise:
            continue

        poly = [(dequantize(coords[2 * j], img_w),
                 dequantize(coords[2 * j + 1], img_h)) for j in range(4)]
        label = "noise" if is_noise else ID_TO_LABEL[cls_tok - NUM_BINS]
        out.append((poly, label, is_noise))
    return out


PALETTE = ["#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231",
           "#48f90a", "#92cc17", "#3ddb86", "#1a9334", "#00c2ff"]

NOISE_COLOR = "#ff2d2d"


def _dashed_polygon(d, pts, fill, width=2, dash=9):
    """Noise quadları kesikli çizilir - gerçek GT ile karışmasın."""
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


def visualize_augmentations(dataset, out_dir="aug_check", n=20, verify=True):
    """
    Augment edilmiş görüntüyü ve INPUT dizisinden geri çözülen quadları çizer.

    Çizim `records` ara yapısından değil, dataset'in döndürdüğü token
    dizisinden yapılır - yani modelin gerçekten gördüğü şey görselleşir
    (quantize kaybı, sıralama, noise'un kuyrukta durması dahil).

        yeşil/palet + dolu çizgi : gerçek GT quad (1. köşe kırmızı, 2. köşe sarı)
        kırmızı + kesikli çizgi  : sentetik noise quad
    """
    os.makedirs(out_dir, exist_ok=True)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    tot_gt = tot_noise = 0
    for k in range(n):
        idx = k % len(dataset)
        img_t, seq_in, seq_tgt = dataset[idx]

        img = (img_t.permute(1, 2, 0).numpy() * std + mean).clip(0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8))
        d   = ImageDraw.Draw(img)
        W, H = img.size

        instances = decode_sequence(seq_in, W, H, keep_noise=True)
        n_gt = n_noise = 0

        for poly, label, is_noise in instances:
            if is_noise:
                n_noise += 1
                _dashed_polygon(d, poly, NOISE_COLOR, width=2)
                d.text((poly[0][0] + 5, poly[0][1] - 12), "noise", fill=NOISE_COLOR)
                continue

            n_gt += 1
            color = PALETTE[LABEL_TO_ID[label] % len(PALETTE)]

            xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
            d.rectangle([min(xs), min(ys), max(xs), max(ys)], outline="#00e5ff", width=1)
            d.line([c for p in poly for c in p] + list(poly[0]), fill=color, width=3)

            # 1. nokta kırmızı, 2. nokta sarı -> kanonik başlangıç + yön kontrolü
            x0, y0 = poly[0]; x1, y1 = poly[1]
            d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill="red")
            d.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill="yellow")
            d.text((x0 + 7, y0 - 13), label, fill=color)

        if verify:
            verify_sequence(seq_in, seq_tgt, expected_gt=n_gt)

        d.rectangle([0, 0, W, 18], fill="#141414")
        d.text((5, 4), f"GT: {n_gt}", fill="#48f90a")
        d.text((70, 4), f"noise: {n_noise}", fill=NOISE_COLOR)
        d.text((165, 4), f"slot: {n_gt + n_noise}/{dataset.max_objects}", fill="#bbbbbb")

        stem = os.path.splitext(dataset.json_files[idx])[0][:40]
        img.save(os.path.join(out_dir, f"aug_check_{k:02d}_{stem}_gt{n_gt}_noise{n_noise}.jpg"),
                 quality=92)
        tot_gt += n_gt; tot_noise += n_noise

    print(f"{n} görsel kaydedildi -> {out_dir}/")
    print(f"  gerçek quad: {tot_gt} (ort. {tot_gt / max(n,1):.2f}) | "
          f"noise quad: {tot_noise} (ort. {tot_noise / max(n,1):.2f})")
    if verify:
        print("  dizi kontrolü: OK (noise kuyrukta, hedefte koordinatlar PAD, hiza doğru)")


def verify_sequence(seq_in, seq_tgt, expected_gt=None):
    """Görselin yanında sessiz sağlık kontrolü; bozuk örnekte AssertionError atar."""
    ti = seq_in.tolist() if torch.is_tensor(seq_in) else list(seq_in)
    tt = seq_tgt.tolist() if torch.is_tensor(seq_tgt) else list(seq_tgt)
    assert ti[0] == BOS_TOKEN, "seq_in BOS ile başlamalı"
    assert len(ti) == len(tt), "input/target uzunlukları farklı"

    body_in, body_tgt = ti[1:], tt[:-1]
    n_gt = n_noise = 0
    seen_noise = False
    for i in range(0, (len(body_in) // 9) * 9, 9):
        cin, ctgt = body_in[i:i + 9], body_tgt[i:i + 9]
        if cin[8] == PAD_TOKEN:
            # kuyruk: sadece EOS ve PAD kalabilir (fill_to_max kapalıyken EOS burada olur)
            assert all(t in (PAD_TOKEN, EOS_TOKEN) for t in ctgt), "PAD slotunda hedef kirli"
            continue
        if cin[8] == NOISE_CLASS_TOKEN:
            n_noise += 1; seen_noise = True
            assert ctgt == [PAD_TOKEN] * 8 + [NOISE_CLASS_TOKEN], f"noise hedefi bozuk: {ctgt}"
            assert all(0 <= c < NUM_BINS for c in cin[:8]), "noise girdisi gerçek bin olmalı"
        else:
            n_gt += 1
            assert not seen_noise, "noise gerçek nesnelerin arasına girmiş (kuyrukta olmalı)"
            assert cin == ctgt, "gerçek nesnede input/target hizası bozuk"
    if expected_gt is not None:
        assert n_gt == expected_gt, f"GT sayısı tutmuyor: {n_gt} != {expected_gt}"
    return n_gt, n_noise


def dataset_object_stats(dataset, n=None):
    """MAX_OBJECTS'i doğru boyutlamak için gerçek nesne sayısı dağılımı."""
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


class Pix2SeqDataset(Dataset):
    """
    HİZALANMIŞ (input, target) çifti döndürür - eğitim döngüsü ARTIK kaydırma yapmaz.

        seq_in  = [BOS, x1..y4, cls, ...,  nx1..ny4, NOISE, ...]
        seq_tgt = [     x1..y4, cls, ...,  PAD x8,   NOISE, ..., EOS]

    Noise objesinin koordinatları hedefte PAD ("n/a"), yani sadece SINIF tokenı
    öğrenilir. Tek tensör + shift ile bu mümkün değildi: aynı pozisyon girdide
    gerçek koordinat, hedefte PAD olmak zorunda.
    """

    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 transform=train_transform, is_train=True, permute_prob=0.0,
                 noise_fill_to_max=NOISE_FILL_TO_MAX, noise_ratio=NOISE_RATIO,
                 noise_iou_reject=NOISE_IOU_REJECT):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        self.json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_objects = max_objects
        self.max_seq_len = 1 + (9 * max_objects) + 1
        self.transform = transform # Albumentations transformunu aldık
        # noise/permutation `transform is not None` ile değil is_train ile kontrol edilir
        # (val_transform da None değil - öyle olsaydı val hedeflerine noise sızardı).
        self.is_train          = is_train
        self.permute_prob      = permute_prob
        self.noise_fill_to_max = noise_fill_to_max
        self.noise_ratio       = noise_ratio
        self.noise_iou_reject  = noise_iou_reject

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

        # ── 1. Gerçek quadlar ────────────────────────────────────────────────
        records = []
        for si, shape in enumerate(valid_shapes):
            quad = groups.get(si, [])
            if len(quad) != 4:                     # eksik/fazla -> bu objeyi at
                continue
            if not all(-8 <= x <= img_w + 8 and -8 <= y <= img_h + 8 for x, y in quad):
                continue                           # kadraj dışı -> clamp yerine at
            records.append({"quad": np.asarray(quad, dtype=np.float64),
                            "class_id": LABEL_TO_ID[shape["label"]],
                            "is_noise": False})

        if self.is_train and self.permute_prob > 0 and random.random() < self.permute_prob:
            random.shuffle(records)
        records = records[:self.max_objects]
        n_real = len(records)

        # ── 2. Sentetik noise objeleri - KUYRUĞA eklenir ─────────────────────
        if self.is_train:
            real_quads = [r["quad"] for r in records]
            if self.noise_fill_to_max:
                n_noise = self.max_objects - n_real
            else:
                n_noise = int(round(n_real * self.noise_ratio))
            n_noise = min(n_noise, self.max_objects - n_real)
            if NOISE_MAX_PER_IMG is not None:
                n_noise = min(n_noise, NOISE_MAX_PER_IMG)
            if n_noise > 0:
                for nq in generate_noise_quads(real_quads, img_w, img_h, n_noise,
                                               self.noise_iou_reject):
                    records.append({"quad": nq, "class_id": NOISE_CLASS_ID, "is_noise": True})

        # ── 3. Hizalanmış input / target dizileri ────────────────────────────
        toks_in, toks_tgt = [], []
        for rec in records:
            q = []
            for x, y in rec["quad"]:
                q.extend([quantize(float(x), img_w, NUM_BINS),
                          quantize(float(y), img_h, NUM_BINS)])
            if rec["is_noise"]:
                # koordinatlar girdide gerçek, hedefte "n/a" -> sadece sınıf öğrenilir
                toks_in.extend(q + [NOISE_CLASS_TOKEN])
                toks_tgt.extend([PAD_TOKEN] * 8 + [NOISE_CLASS_TOKEN])
            else:
                cls_tok = NUM_BINS + rec["class_id"]
                toks_in.extend(q + [cls_tok])
                toks_tgt.extend(q + [cls_tok])

        # NOISE_FILL_TO_MAX iken dizi HER ZAMAN tam MAX_OBJECTS slot uzunluğundadır,
        # dolayısıyla EOS'a gerek yoktur ve konulmaz: model "dur" kararını EOS ile
        # değil, her slotta noise sınıfını seçerek verir. Bu, erken EOS yüzünden
        # nesne kaçırma problemini kökten kaldırır ve her slota bir skor kazandırır.
        L = self.max_seq_len - 1
        tail = [] if self.noise_fill_to_max else [EOS_TOKEN]
        seq_in  = ([BOS_TOKEN] + toks_in)[:L]
        seq_tgt = (toks_tgt + tail)[:L]
        seq_in  = seq_in  + [PAD_TOKEN] * (L - len(seq_in))
        seq_tgt = seq_tgt + [PAD_TOKEN] * (L - len(seq_tgt))

        return (image_tensor,
                torch.tensor(seq_in,  dtype=torch.long),
                torch.tensor(seq_tgt, dtype=torch.long))


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
                img = ImageOps.exif_transpose(img)
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
        max_lr=[1e-5, LEARNING_RATE, LEARNING_RATE, LEARNING_RATE,
                LEARNING_RATE, LEARNING_RATE, LEARNING_RATE],
        total_steps=total_steps,
        pct_start=0.05
    )

    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model)

    dataset_object_stats(train_dataset)
    visualize_augmentations(train_dataset, out_dir="aug_check", n=20)
 
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
 
        # dataset zaten hizalı (input, target) veriyor - KAYDIRMA YOK
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