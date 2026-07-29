import os
import json
import math
import copy
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random

# --- YAPILANDIRMA (CONFIG) ---
# Kendi yollarına göre burayı güncelleyebilirsin
JSON_DIR = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01/" #"/home/uygarusta/Oriented-Centernet/2stage_corner_regression/ruhsat_refined/"
IMG_DIR = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01/" #"/home/uygarusta/Oriented-Centernet/2stage_corner_regression/ruhsat_refined/"

# Bin sayısı görüntü çözünürlüğünden anlamlı ölçüde büyük olmamalı: koordinatlar
# IMG_SIZE'a resize EDİLDİKTEN SONRA kuantize ediliyor, yani 512px'lik bir grid'de
# 2000 bin = 0.26 piksel/bin. Backbone stride 32 ile memory grid'i 16x16 olan bir
# model bu çözünürlüğü ayıramaz — fazladan her bin ikiye katlaması loss'a ~ln(2)
# nat indirgenemez gürültü ekler ve softmax'ı gereksiz yere zorlaştırır.
NUM_BINS = 1000
# Bu veri setinde her görüntüde her sınıftan tam olarak 1 obje var (7 gerçek obje).
# MAX_OBJECTS yalnızca üst sınır; gereğinden büyük tutmak sequence'i uzatıp
# eğitimi yavaşlatır (ve noise-obje sayısını şişirir).
MAX_OBJECTS = 10
# FEATURE_LEVELS=(-2,-1) ile memory token sayısı 256'dan 1024'e çıktığı için
# BATCH_SIZE düşürülüp GRAD_ACCUM_STEPS aynı oranda artırıldı — efektif batch
# 16 sabit kalıyor (bkz. GRAD_ACCUM_STEPS). VRAM bolsa 8/2'ye geri alınabilir.
BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 2e-4
# Pretrained backbone çok daha düşük LR ister; ayrı param grubu olarak verilir.
ENCODER_LR = 2e-5
IMG_SIZE = (512, 512)

# --- Loss ayarları ---
# DİKKAT: label smoothing burada 2014'lük bir vocab üzerinde çalışıyor ve
# PyTorch'un formülü `(1-ε)·NLL(hedef) + ε·mean_i(-log p_i)` — yani ε>0, modeli
# hedeften 1000 bin uzaktaki token'lara bile olasılık ayırmaya zorluyor. Bu hem
# koordinat dağılımının keskinleşmesini engelliyor hem de loss'a ~1.09 nat'lık
# indirgenemez bir taban ekliyor (bu yüzden loss "takılmış" gibi görünüyor).
# Koordinat-as-classification için 0.0 doğru varsayılan.
LABEL_SMOOTHING = 0.0

# Koordinat token'ları için gauss "soft target" (bin cinsinden std; 0 => kapalı).
# Cross-entropy bin'leri SIRASIZ sınıflar gibi ele alır: hedef 501 iken 500
# tahmin etmekle 900 tahmin etmek aynı cezayı alır. Model bin'lerin ordinal
# olduğunu ancak dolaylı olarak öğrenebilir. Hedefi tek bir bin yerine gerçek
# konumun etrafına yerleştirilmiş dar bir gauss yapmak bu yapıyı doğrudan
# öğretir ve tepe noktasını keskinleştirir — kutuların sıkılığını artıran şey de
# tam olarak budur (bkz. pix2seq_decode.SOFT_ARGMAX_WINDOW, çıkarım tarafındaki
# karşılığı).
#
# NOT: soft target loss'a indirgenemez bir taban ekler (gauss etiketin kendi
# entropisi, σ=1 için ≈ ln(1·√(2πe)) = 1.42 nat). Eğitim başında bu değer
# yazdırılıyor; loss'u geçmiş koşularla karşılaştırırken hesaba katın.
COORD_LABEL_SIGMA = 1.0

# --- Augmentasyon ayarları ---
# Mosaic, COCO tarzı çok-objeli sahneler için tasarlandı. Burada 4 belge x 7 alan
# = ~28 obje üretiyor, MAX_OBJECTS'e kırpılıyor ve modele "görüntüde duran
# objeleri hedefe yazma" diye yanlış bir sinyal veriyor. Sabit düzenli belgelerde
# kapalı olmalı.
MOSAIC_PROB = 0.0

# Rotasyon augmentasyonunun üst sınırı (derece).
#
# OUTPUT_MODE="hbb" iken bu ayar göründüğünden çok daha önemli: w×h bir kutu θ
# kadar döndüğünde eksen-hizalı kutusunun yüksekliği w·sinθ + h·cosθ olur, yani
# İNCE alanlarda hedef dramatik biçimde şişer —
#     110×22 metin alanı:  5° -> 1.43x,  10° -> 1.85x,  15° -> 2.26x yükseklik
#     400×300 ruhsat     :  5° -> 1.11x,  10° -> 1.22x,  15° -> 1.31x
# Hedef tutarlı olduğu için model bunu öğrenebilir (görüntü de dönüyor), ama
# koordinat dağılımı çok genişler ve aynı kapasiteyle daha kaba bir çözüm
# öğrenilir — tam olarak "kutular gevşek" şikâyetinin beslendiği yer.
#
# Üretimdeki fotoğraflar büyük ölçüde dikse 5-7 dereceye çekmek kenar
# hassasiyetini artırır. Gerçekten eğik çekimler geliyorsa 15'te bırakın —
# aksi halde dayanıklılığı kaybedersiniz. "corner" modunda bu etki yoktur
# (dönen quad'ın köşeleri yine 4 nokta).
ROTATE_LIMIT = 15

# Objelerin sequence içindeki sırası:
#   "class"  -> sınıf ID'sine (eşitlikte y,x'e) göre deterministik sıralama
#   "random" -> Pix2Seq makalesindeki rastgele sıra
# Bu veri setinde her sınıftan tam olarak bir obje olduğu için "class" sıralaması
# sequence yapısını tamamen deterministik yapar (1. obje hep qr, 2. hep menfaat...).
# "random" ise modele her slotta "sıradaki 7 alandan hangisi?" tahmini yaptırır —
# ln(7!) ≈ 8.5 nat tamamen boşa giden entropi. Aynı sınıftan birden fazla obje
# çıkabilen veri setlerinde "random" tercih edilmeli.
OBJECT_ORDER = "class"  # "class" | "random"

# Her objenin 4 köşesinin KENDİ İÇİNDEKİ sırası. Model bu 8 koordinatı sabit bir
# sırayla üretir, dolayısıyla sıra ancak etiketlerde tutarlı bir kural varsa
# öğrenilebilir. Tutarsızsa model iki köşeyi yer değiştirir, ortaya kendisiyle
# kesişen ("papyon") bir dörtgen çıkar ve polygon IoU bunu 0 sayar — 4 nokta
# geometrik olarak kusursuz olsa bile. Bu, ince metin alanlarında (iki komşu
# köşe ~20 px arayken) büyük kutulara göre çok daha sık olur.
#
# ÖNCE `python check_corner_order.py` çalıştırın. Tutarlılık %98'in altındaysa
# burayı True yapın: köşeler ağırlık merkezi etrafında açıya göre sıralanır
# (papyon imkânsız hale gelir) ve daima sol-üste en yakın köşeden başlatılır.
#
# UYARI: True yapmak köşe sırasını GEOMETRİK bir kurala bağlar. Etiketlerdeki
# sıra belgenin okuma yönünü kodluyorsa (ör. crop'u düzeltirken hangi köşenin
# metnin sol-üstü olduğu önemliyse) bu bilgi kaybolur — 180° dönmüş belgeler
# ters crop üretir. Sadece hbb/kutu konumu önemliyse sakınca yok.
CANONICAL_CORNER_ORDER = False

# --- Convergence / hız ayarları ---
# Efektif batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16. VRAM yetmiyorsa BATCH_SIZE'ı
# düşür, GRAD_ACCUM_STEPS'i aynı oranda artır (efektif batch sabit kalır).
GRAD_ACCUM_STEPS = 4
# bf16 autocast: Ampere+ GPU (RTX 30xx/40xx, A100...) gerektirir. fp16'nın
# aksine bf16, fp32 ile aynı exponent aralığına sahip olduğu için GradScaler'a
# ihtiyaç yok (overflow/underflow riski yok) — sadece forward+loss autocast
# altında çalıştırılır.
USE_BF16 = True
# İlk ENCODER_FREEZE_EPOCHS boyunca ResNet50 backbone donuk kalır (yalnızca
# enc_proj/img_encoder/decoder ısınır); sıfırdan eğitimin ilk adımlarındaki
# büyük/gürültülü gradyanların ImageNet-pretrained backbone'u bozmasını önler.
ENCODER_FREEZE_EPOCHS = 5
# EMA (exponential moving average) ağırlıkları val/checkpoint için kullanılır;
# mosaic/jitter/perspective gibi gürültülü augmentasyonlar altında ham
# ağırlıklardan genelde daha stabil ve iyi genelleşir. 0 => EMA kapalı.
EMA_DECAY = 0.999
# Augmente edilmemiş train alt kümesinde kaç epoch'ta bir ölçüm yapılsın
# (bkz. trainclean_dataset). Her epoch yapmak val süresini iki katına çıkarır.
TRAINCLEAN_EVERY = 10

# --- Backbone (timm) ---
# timm.list_models(pretrained=True) ile yüzlerce alternatif görülebilir, örn:
#   "resnet50", "resnet50d", "convnext_tiny", "convnext_small",
#   "efficientnet_b0" .. "efficientnet_b3", "regnety_016", "seresnet50" ...
# NOT: backbone değiştirmek encoder'ın state_dict yapısını değiştirir — farklı
# bir BACKBONE_NAME ile eğitilmiş bir checkpoint, başka bir backbone ile
# YÜKLENEMEZ (state_dict key mismatch). Her checkpoint'in .meta.json'ı hangi
# backbone ile eğitildiğini kaydeder (bkz. save_checkpoint), inference tarafı
# (pix2seq_decode.load_model) bunu okuyup doğru backbone'u otomatik kurar.
BACKBONE_NAME = "resnet50"

# Hangi backbone stage'lerinin feature map'i decoder'a memory olarak verilecek.
# Birden fazla verilirse FPN tarzı birleştirilir: kaba olanlar en ince olanın
# çözünürlüğüne upsample edilip toplanır (bkz. Pix2SeqModel.encode).
#
# KUTULARIN "GEVŞEK" ÇIKMASININ KÖK NEDENİ BURASI. Sadece son stage (stride 32)
# kullanıldığında 512px girdide grid 16x16 olur, yani bir hücre 32 piksele denk
# gelir. 110x22'lik ince bir metin alanı bu grid'de 3.4 x 0.7 hücre kaplar —
# alanın üst ve alt kenarı AYNI hücrenin içinde kalır ve decoder'ın onları
# ayırabileceği bir çözünürlük yoktur. NUM_BINS=1000 zaten 0.51 px/bin veriyor,
# yani darboğaz kuantizasyon değil, feature çözünürlüğü.
#
#   (-1,)     -> stride 32, grid 16x16 =  256 token  (eski davranış)
#   (-2, -1)  -> stride 16, grid 32x32 = 1024 token  (varsayılan: 2x hassasiyet,
#                son stage'in semantiği upsample edilip korunur)
#   (-3,-2,-1)-> stride  8, grid 64x64 = 4096 token  (4x hassasiyet, pahalı)
#
# state_dict'i değiştirdiği için .meta.json'a kaydediliyor.
FEATURE_LEVELS = (-2, -1)

# Transformer encoder'ın (img_encoder) self-attention'ı hangi çözünürlükte
# çalışsın?
#   False -> birleştirilmiş İNCE grid üzerinde (basit, ama maliyet token²)
#   True  -> yalnızca EN KABA seviyede; sonuç upsample edilip ince grid'e
#            eklenir, decoder memory'si yine ince grid olur
#
# Bu ayrım stride 8'i mümkün kılan şey. Self-attention token sayısının KARESİYLE
# büyür: 768px girdide stride 8, 96x96 = 9216 token demek ve 9216² çift
# hesaplanamaz. Oysa global bağlam kaba seviyede zaten mevcut (DETR/FPN
# mantığı), ince seviyenin katkısı lokalizasyon detayı — ve decoder'ın
# cross-attention'ı token sayısında LİNEER olduğu için 9216 memory token'a
# rahatça bakabilir. Yani: self-attention kabada, detay incede.
ENCODER_ON_COARSE = True

# Model hangi geometriyi öğrensin:
#   "corner" -> yalnızca 4 köşe (oriented quad)
#   "hbb"    -> yalnızca eksen-hizalı (horizontal) bounding box
#   "both"   -> ikisi birden, aynı obje içinde peş peşe
# Bu, sequence formatını belirlediği için train/inference arasında MUTLAKA aynı
# kalmalı. pix2seq_decode.py bu değeri doğrudan buradan okur; ayrıca her
# checkpoint yanına bir .meta.json kaydedilir (bkz. save_checkpoint) ki farklı
# bir OUTPUT_MODE ile eğitilmiş bir checkpoint yanlışlıkla yüklenirse fark edilsin.
OUTPUT_MODE = "both"  # "corner" | "hbb" | "both"

_TOKENS_PER_OBJECT_BY_MODE = {"corner": 9, "hbb": 5, "both": 13}
if OUTPUT_MODE not in _TOKENS_PER_OBJECT_BY_MODE:
    raise ValueError(f"Geçersiz OUTPUT_MODE: {OUTPUT_MODE!r}. 'corner', 'hbb' veya 'both' olmalı.")

# Obje başına token sayısı, seçilen OUTPUT_MODE'a göre değişir:
#   corner -> 4 köşe (8 koordinat) + 1 sınıf                      = 9
#   hbb    -> eksen-hizalı bbox (4 koordinat) + 1 sınıf            = 5
#   both   -> 4 köşe (8) + köşelerden türetilen hbb (4) + 1 sınıf  = 13
TOKENS_PER_OBJECT = _TOKENS_PER_OBJECT_BY_MODE[OUTPUT_MODE]

LABEL_TO_ID = {
    "qr": 0, "menfaat": 1, "azami_yuk": 2, "kullanim_amaci": 3,
    "net_agirlik": 4, "ruhsat": 5, "romork_azami_yuk": 6,
    "plaka": 7, "tc": 8, "seri_no": 9
}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

NUM_CLASSES = len(LABEL_TO_ID)  # gerçek sınıf sayısı (noise hariç)

# --- Noise-object sequence augmentation (Pix2Seq makalesi, Böl. 3.3) ---
# Eğitimde gerçek objelerin yanına, MAX_OBJECTS'e tamamlayacak kadar sahte
# ("noise") obje ekliyoruz; bunlar özel bir NOISE_CLASS_ID sınıf token'ı alıyor.
# Model böylece "bu kutu gerçek bir obje mi yoksa gürültü mü" ayrımını öğreniyor
# ve inference'ta üretilen sınıf token'ının softmax olasılığı gerçek, kalibre
# edilmiş bir confidence score'a dönüşüyor (bkz. pix2seq_decode.py — detokenize
# zaten NOISE_CLASS_ID'yi kabul edilen sınıf aralığının dışında bıraktığı için
# noise tahmin edilen "objeler" otomatik elenir, ekstra kod gerekmiyor).
NOISE_CLASS_ID = NUM_CLASSES               # "gürültü / obje yok" sözde-sınıfı
NOISE_AUGMENTATION = True                  # train'de açık, val'de kapalı (bkz. __main__)
NOISE_JITTERED_RATIO = 0.5                 # eklenen noise objelerin ne kadarı gerçek bir objenin jitter'lanmış hali (kalanı tamamen rastgele kutu)
NOISE_JITTER_STD = 0.05                    # jitter için gauss std, görüntü boyutunun oranı olarak
# Makale (Böl. 3.3) noise objelerin KOORDİNAT token'larına loss uygulamaz —
# sadece sınıf token'ına ("bu bir noise objesi") uygular. Bu kritik: noise
# koordinatları tanım gereği rastgele üretiliyor, dolayısıyla öğrenilebilir bir
# sinyal içermiyor. Loss'a dahil edilirlerse (7 gerçek + 8 noise objede) hedef
# token'ların ~%49'u saf gürültü olur ve gradyanın yarısı modele "rastgele sayı
# tahmin et" der — koordinat dağılımı keskinleşemez, loss plato yapar.
# True yapmak eski (hatalı) davranışa döner; yalnızca karşılaştırma için.
NOISE_COORD_LOSS = False

# +1: noise sınıfı için ekstra vocab slotu, +3: BOS/EOS/PAD
VOCAB_SIZE = NUM_BINS + NUM_CLASSES + 1 + 3
NOISE_TOKEN = NUM_BINS + NOISE_CLASS_ID
BOS_TOKEN = VOCAB_SIZE - 3
EOS_TOKEN = VOCAB_SIZE - 2
PAD_TOKEN = VOCAB_SIZE - 1


train_transform = A.Compose([
    # Resize EN BAŞTA: aşağıdaki pixel-level augmentasyonlar orijinal (genelde
    # telefon kamerası çözünürlüğünde, 512'den çok daha büyük) görüntü yerine
    # doğrudan IMG_SIZE üzerinde çalışsın diye — CPU maliyetleri IMG_SIZE'a
    # bağlı hale gelir ve DataLoader darboğazı büyük ölçüde azalır.
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=ROTATE_LIMIT, p=0.4),
    A.Perspective(scale=(0.05, 0.1), p=0.4),
    A.RandomBrightnessContrast(p=0.4),
    # DİKKAT (albumentations >= 2.0): `var_limit` / `quality_lower` / `quality_upper`
    # artık geçerli argüman değil ve SESSİZCE yok sayılıyorlar (sadece bir
    # UserWarning basılıyor) — transform varsayılanlarıyla çalışıyordu.
    # GaussNoise'un varsayılanı std_range=(0.2, 0.44), yani normalize [0,1]
    # ölçeğinde σ = 51..112/255. Hedeflenen var_limit=(10,50) ise σ ≈ 3.2..7.1/255,
    # yani ~10-16 KAT daha zayıf. Eğitim görüntülerinin %30'u okunamayacak kadar
    # bozuluyordu. Yeni API ile eşdeğeri açıkça yazılıyor:
    A.GaussNoise(std_range=(0.012, 0.028), p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
    A.ImageCompression(quality_range=(60, 100), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.2),
    A.RandomShadow(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Validation için: mosaic yok, geometrik/fotometrik jitter yok — sadece resize +
# normalize. Val loss'un gerçek genelleme performansını yansıtması ve checkpoint
# seçiminin augmentasyon gürültüsünden etkilenmemesi için train_transform'dan
# kasıtlı olarak ayrı tutuluyor.
val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


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
    # round, int() (floor) değil: dequantize `token/(num_bins-1)*max_size` ile
    # ters çeviriyor, dolayısıyla floor sistematik olarak yarım bin'lik bir
    # sola-kayma (bias) üretiyordu. round ile iki fonksiyon birbirinin tam tersi.
    return int(round(normalized * (num_bins - 1)))


def canonicalize_corners(points):
    """4 köşeyi ağırlık merkezi etrafındaki açıya göre sıralar, sonra diziyi
    sol-üste (0,0) en yakın köşeden başlayacak şekilde döndürür.

    Açıya göre sıralama, sonucun daima BASİT (kendisiyle kesişmeyen) bir
    dörtgen olmasını garanti eder; başlangıç köşesinin sabitlenmesi de hedefi
    deterministik yapar. bkz. CANONICAL_CORNER_ORDER."""
    cx = sum(p[0] for p in points) / 4.0
    cy = sum(p[1] for p in points) / 4.0
    ordered = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    start = min(range(4), key=lambda i: ordered[i][0] + ordered[i][1])
    return ordered[start:] + ordered[:start]


def random_box_points(img_w, img_h):
    """Tamamen rastgele, eksen-hizalı küçük bir kutunun 4 köşesini döndürür
    (noise-object augmentation için 'tamamen uydurma' obje)."""
    bw = random.uniform(0.05, 0.3) * img_w
    bh = random.uniform(0.05, 0.3) * img_h
    cx = random.uniform(0, img_w)
    cy = random.uniform(0, img_h)
    x1 = max(0.0, min(float(img_w), cx - bw / 2))
    x2 = max(0.0, min(float(img_w), cx + bw / 2))
    y1 = max(0.0, min(float(img_h), cy - bh / 2))
    y2 = max(0.0, min(float(img_h), cy + bh / 2))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def jitter_box_points(points, img_w, img_h, std_frac=0.05):
    """Gerçek bir objenin köşelerine gauss gürültüsü ekler (noise-object
    augmentation için 'neredeyse doğru ama değil' obje — modelin yakın
    kaçırmaları da reddetmeyi öğrenmesi için)."""
    jittered = []
    for x, y in points:
        nx = x + random.gauss(0, std_frac * img_w)
        ny = y + random.gauss(0, std_frac * img_h)
        nx = max(0.0, min(float(img_w), nx))
        ny = max(0.0, min(float(img_h), ny))
        jittered.append((nx, ny))
    return jittered

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



# --- DATASET ---
class Pix2SeqDataset(Dataset):
    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 transform=train_transform, json_files=None, mosaic_prob=MOSAIC_PROB,
                 order_mode=OBJECT_ORDER, noise_augmentation=NOISE_AUGMENTATION):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        # json_files verilirse (train/val split'i) o listeye sabitlenir; verilmezse
        # dizindeki tüm .json dosyaları kullanılır (tek dataset kullanım senaryosu).
        self.json_files = json_files if json_files is not None else \
            [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_objects = max_objects
        self.max_seq_len = 1 + (TOKENS_PER_OBJECT * max_objects) + 1
        self.transform = transform  # Albumentations transformunu aldık
        self.mosaic_prob = mosaic_prob
        # Objelerin sequence İÇİNDEKİ SIRASI (bkz. OBJECT_ORDER). Bu, YALNIZCA
        # hangi objenin kaçıncı sırada yazılacağını etkiler. Her objenin kendi 4
        # köşesinin sırası (canonical etiketleme) buna dokunulmadan aynen korunur.
        self.order_mode = order_mode
        # Gerçek objelerin yanına MAX_OBJECTS'e tamamlayacak kadar "noise" obje
        # eklenir (bkz. modül üstündeki NOISE_AUGMENTATION açıklaması). Val'de
        # kapalı tutulmalı — val loss/AP sadece gerçek objeler üzerinden ölçülsün.
        self.noise_augmentation = noise_augmentation

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):

        # ── Mosaic (MOSAIC_PROB, sadece bu dataset'in kendi split'i içinden) ───
        if self.mosaic_prob > 0 and self.transform is not None and len(self.json_files) >= 4 \
                and random.random() < self.mosaic_prob:
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
                image, _ = pad_to_square(image)
                image_np = np.array(image)
            except Exception as e:
                print(f"HATA: {img_path} okunamadı. Hata: {e}")
                image_np = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)

            shapes = item.get("shapes", [])
            all_points  = []
            valid_shapes = []

            for shape in shapes:
                label_str = shape.get("label", "")
                points    = shape.get("points", [])
                if label_str in LABEL_TO_ID and len(points) == 4:
                    valid_shapes.append(shape)
                    for p in points:
                        all_points.append(tuple(p))

        # ── Augmentation (same path for both mosaic and normal) ───────────────
        if self.transform is not None:
            augmented    = self.transform(image=image_np, keypoints=all_points)
            image_tensor = augmented['image']
            new_points   = list(augmented['keypoints'])
        else:
            image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0
            new_points   = all_points

        img_h, img_w = self.img_size[0], self.img_size[1]

        # Önce her objeyi (sınıf, köşe noktaları) olarak topla — köşelerin KENDİ
        # içindeki sıra burada da hiç değişmiyor, aşağıdaki `for x, y in
        # shape_points` döngüsü canonical etiketleme sırasını aynen kullanıyor.
        objects   = []
        point_idx = 0
        for shape in valid_shapes:
            class_id     = LABEL_TO_ID[shape["label"]]
            shape_points = new_points[point_idx:point_idx + 4]
            point_idx   += 4

            if len(shape_points) < 4:
                continue

            # Köşe sırası: augmentasyon SONRASI kanonikleştirilir — döndürme /
            # perspektif köşelerin geometrik konumunu değiştirdiği için sıranın
            # da o yeni konuma göre belirlenmesi gerekir.
            if CANONICAL_CORNER_ORDER:
                shape_points = canonicalize_corners(shape_points)

            objects.append((class_id, shape_points))

        # Gerçek objelerin sequence İÇİNDEKİ SIRASI (bkz. OBJECT_ORDER) — hangi
        # objenin köşeleri olduğu ya da o köşelerin sırası değişmiyor, sadece
        # "önce hangi obje yazılsın" belirleniyor. Val'de daima deterministik.
        if self.order_mode == "random":
            random.shuffle(objects)
        else:  # "class": sınıf ID'si, eşitlikte sol-üst köşe konumu
            objects.sort(key=lambda o: (o[0],
                                        min(p[1] for p in o[1]),
                                        min(p[0] for p in o[1])))

        # Noise-object augmentation: gerçek objelerin ARDINA MAX_OBJECTS'e
        # tamamlayacak kadar sahte obje ekle (yarısı gerçek bir objenin jitter'lı
        # hali, yarısı tamamen rastgele kutu). Jitter için HER ZAMAN orijinal
        # gerçek obje listesinden (`real_objects`) örnekleniyor — sonradan eklenen
        # noise objeler üzerinden jitter yapılmıyor ki gürültü gürültüyü büyütmesin.
        # Makaledeki gibi noise objeler SONA ekleniyor (shuffle'a dahil değil):
        # araya karıştırılırlarsa gerçek objelerin decoder context'i rastgele
        # koordinatlarla kirlenir.
        n_real = len(objects)
        if self.noise_augmentation:
            real_objects = list(objects)
            n_missing = self.max_objects - len(objects)
            for _ in range(max(0, n_missing)):
                if real_objects and random.random() < NOISE_JITTERED_RATIO:
                    _, base_points = random.choice(real_objects)
                    noise_points = jitter_box_points(base_points, img_w, img_h, NOISE_JITTER_STD)
                else:
                    noise_points = random_box_points(img_w, img_h)
                objects.append((NOISE_CLASS_ID, noise_points))

        sequence  = [BOS_TOKEN]
        # loss_mask[i] == 1  =>  sequence[i] bir hedef olarak supervise edilir.
        # BOS asla hedef değil (train loop targets[:, 1:] alıyor), değeri önemsiz.
        loss_mask = [0]

        for obj_i, (class_id, shape_points) in enumerate(objects):
            is_noise = (obj_i >= n_real)
            # Noise objelerin koordinatları rastgele üretildiği için öğrenilebilir
            # sinyal içermez; makale bunlara loss uygulamaz (bkz. NOISE_COORD_LOSS).
            coord_mask = 1 if (not is_noise or NOISE_COORD_LOSS) else 0

            if OUTPUT_MODE in ("corner", "both"):
                # 4 köşe noktası (quad / oriented polygon)
                for x, y in shape_points:
                    token_x = quantize(x, img_w, NUM_BINS)
                    token_y = quantize(y, img_h, NUM_BINS)
                    sequence.extend([token_x, token_y])
                    loss_mask.extend([coord_mask, coord_mask])

            if OUTPUT_MODE in ("hbb", "both"):
                # Köşelerden türetilen eksen-hizalı (horizontal) bounding box
                xs = [p[0] for p in shape_points]
                ys = [p[1] for p in shape_points]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                sequence.append(quantize(xmin, img_w, NUM_BINS))
                sequence.append(quantize(ymin, img_h, NUM_BINS))
                sequence.append(quantize(xmax, img_w, NUM_BINS))
                sequence.append(quantize(ymax, img_h, NUM_BINS))
                loss_mask.extend([coord_mask] * 4)

            # Sınıf token'ı her modda son sırada — decode tarafı bunu varsayar.
            # Noise objelerde de DAİMA supervise edilir: modelin "bu obje gerçek
            # değil" diyebilmesi (ve skorunun kalibre olması) buna bağlı.
            sequence.append(NUM_BINS + class_id)
            loss_mask.append(1)

        sequence.append(EOS_TOKEN)
        loss_mask.append(1)

        if len(sequence) > self.max_seq_len:
            # Obje sınırında kes: TOKENS_PER_OBJECT'in ortasından kesip modele
            # bozuk/yarım bir hedef öğretmeyi önler (özellikle mosaic ile obje
            # sayısı MAX_OBJECTS'i kolayca aşabiliyor).
            n_objects = (self.max_seq_len - 2) // TOKENS_PER_OBJECT
            usable    = 1 + n_objects * TOKENS_PER_OBJECT
            sequence  = sequence[:usable]  + [EOS_TOKEN]
            loss_mask = loss_mask[:usable] + [1]

        if len(sequence) < self.max_seq_len:
            n_pad = self.max_seq_len - len(sequence)
            sequence.extend([PAD_TOKEN] * n_pad)
            loss_mask.extend([0] * n_pad)

        return (image_tensor,
                torch.tensor(sequence,  dtype=torch.long),
                torch.tensor(loss_mask, dtype=torch.bool))


    def _mosaic(self, idx):
        """Combines 4 images into a 2x2 mosaic. Returns (image_np, all_points, valid_shapes).
        Diğer 3 parça yalnızca bu dataset instance'ının kendi json_files listesinden
        (yani kendi split'inden) seçilir — train/val arası sızıntı olmaz."""
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
    def __init__(self, vocab_size, hidden_dim=256, nheads=8, num_layers=4, max_seq_len=200,
                 num_encoder_layers=2, backbone_name=BACKBONE_NAME,
                 feature_levels=FEATURE_LEVELS):
        super().__init__()
        # Özellik Çıkarıcı (Encoder) — timm ile herhangi bir backbone'a
        # değiştirilebilir (bkz. BACKBONE_NAME). features_only=True +
        # out_indices=FEATURE_LEVELS, seçilen stage'lerin feature map'lerini
        # kabadan inceye sıralı bir liste olarak döner (bkz. encode()).
        self.feature_levels = tuple(feature_levels)
        self.encoder = timm.create_model(backbone_name, pretrained=True,
                                          features_only=True,
                                          out_indices=self.feature_levels)
        channels = self.encoder.feature_info.channels()
        strides  = self.encoder.feature_info.reduction()

        # FPN tarzı lateral projeksiyonlar: her stage kendi 1x1 conv'u ile
        # hidden_dim'e indirgenir, sonra hepsi EN İNCE stride'ın çözünürlüğüne
        # upsample edilip toplanır. Böylece hem ince grid (lokalizasyon) hem son
        # stage'in derin semantiği (sınıf/bağlam) aynı memory'de bulunur —
        # yalnızca stride-16 kullanmak semantiği, yalnızca stride-32 kullanmak
        # çözünürlüğü kaybettirirdi.
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1) for c in channels])
        # Upsample sonrası aliasing'i azaltan standart FPN smoothing conv'u.
        self.fpn_smooth = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # timm, feature'ları out_indices'te verildikleri sırayla döndürür — yani
        # (-2, -1) için [stride16, stride32]: son eleman EN KABA olan. Hedef
        # çözünürlük en küçük stride'lı olandır, listedeki yeri sabit değil.
        self.finest_idx = min(range(len(strides)), key=lambda i: strides[i])
        self.out_stride = strides[self.finest_idx]
        self.coarsest_idx = max(range(len(strides)), key=lambda i: strides[i])
        # Tek seviye seçilmişse "kabada self-attention" ayrımı anlamsız.
        self.encoder_on_coarse = ENCODER_ON_COARSE and len(strides) > 1

        # Resim özellikleri için konum kodlaması — grid boyutu birleştirilmiş
        # feature map'in stride'ından hesaplanır.
        # std=1 (torch.randn) init, projeksiyon çıktısıyla aynı mertebede gürültü
        # ekleyip ilk adımları bozuyordu; 0.02 transformer pos-embedding standardı.
        grid_h = IMG_SIZE[0] // self.out_stride
        grid_w = IMG_SIZE[1] // self.out_stride
        self.pos_emb = nn.Parameter(torch.randn(1, grid_h * grid_w, hidden_dim) * 0.02)

        # Self-attention kaba seviyede çalışacaksa oraya ayrı bir konum kodlaması
        # gerekir (grid boyutu farklı).
        if self.encoder_on_coarse:
            c_stride = strides[self.coarsest_idx]
            c_h, c_w = IMG_SIZE[0] // c_stride, IMG_SIZE[1] // c_stride
            self.pos_emb_coarse = nn.Parameter(torch.randn(1, c_h * c_w, hidden_dim) * 0.02)

        # CNN özelliklerini decoder'a memory olarak vermeden önce self-attention
        # ile bağlamsallaştıran hafif bir transformer encoder (DETR/Pix2Seq'teki
        # gibi). Backbone zaten derin olduğu ve dataset küçük/tek-domain olduğu
        # için 2 katman ile başlıyoruz — istenirse num_encoder_layers ile artırılabilir.
        #
        # norm_first=True (pre-LN): PyTorch varsayılanı post-LN'dir ve post-LN
        # transformerlar sıfırdan eğitilirken belirgin biçimde daha yavaş/kırılgan
        # yakınsar, yüksek LR'yi kaldıramaz. Pre-LN + sondaki LayerNorm, gradyan
        # akışını düzeltip LEARNING_RATE'i güvenle yukarı çekmeyi mümkün kılar.
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nheads,
                                                   batch_first=True, dropout=0.1,
                                                   norm_first=True)
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                 norm=nn.LayerNorm(hidden_dim))

        # Hedef Dizi (Sequence) Modellemesi
        #self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(0.1)

        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nheads,
                                                   batch_first=True, dropout=0.1,
                                                   norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, images):
        """Görüntüyü bir kez encode eder. Autoregressive decode döngüsünde her
        adımda tekrar çağrılmamalı — memory sabittir, sadece tgt_seq değişir."""
        feats = self.encoder(images)  # out_indices sırasında
        lat = [layer(f) for layer, f in zip(self.lateral, feats)]

        if self.encoder_on_coarse:
            # Global bağlam yalnızca en kaba seviyede kuruluyor — self-attention
            # maliyeti token² olduğu için ince grid'de yapılamaz (bkz.
            # ENCODER_ON_COARSE). Sonuç FPN toplamına geri konuyor.
            c = lat[self.coarsest_idx]
            b, ch, h, w = c.shape
            t = c.flatten(2).permute(0, 2, 1) + self.pos_emb_coarse
            t = self.img_encoder(t)
            lat[self.coarsest_idx] = t.permute(0, 2, 1).reshape(b, ch, h, w)

        # FPN birleştirme: hepsi en ince seviyenin çözünürlüğüne taşınıp toplanır.
        target_hw = feats[self.finest_idx].shape[-2:]
        fused = None
        for p in lat:
            if p.shape[-2:] != target_hw:
                p = nn.functional.interpolate(p, size=target_hw, mode="nearest")
            fused = p if fused is None else fused + p
        features = self.fpn_smooth(fused)

        memory = features.flatten(2).permute(0, 2, 1)
        memory = memory + self.pos_emb
        if not self.encoder_on_coarse:
            memory = self.img_encoder(memory)
        return memory

    def decode(self, memory, tgt_seq):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq.size(1)).to(tgt_seq.device)

        tgt_emb = self.embedding(tgt_seq)
        tgt_emb = self.seq_pos_encoding(tgt_emb)
        tgt_emb = self.emb_dropout(tgt_emb)

        out = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

    def forward(self, images, tgt_seq):
        memory = self.encode(images)
        return self.decode(memory, tgt_seq)


def set_encoder_trainable(model, trainable):
    """ENCODER_FREEZE_EPOCHS penceresi için ResNet50 backbone'u dondurur/çözer."""
    for p in model.encoder.parameters():
        p.requires_grad_(trainable)


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """EMA ağırlıklarını her optimizer.step() sonrası günceller. Buffer'lar
    (BatchNorm running mean/var, num_batches_tracked) EMA'lanmaz, doğrudan
    kopyalanır — aksi halde BN istatistikleri EMA gecikmesiyle yanlışlaşır."""
    ema_params = dict(ema_model.named_parameters())
    for name, p in model.named_parameters():
        ema_params[name].mul_(decay).add_(p.detach(), alpha=1 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    for name, b in model.named_buffers():
        ema_buffers[name].copy_(b)


def save_checkpoint(model, path):
    """state_dict'in yanına, hangi OUTPUT_MODE ile eğitildiğini kaydeden bir
    <path>.meta.json bırakır. Sequence formatı OUTPUT_MODE'a göre değiştiği için
    bu, yanlış moddaki bir checkpoint'in inference'ta sessizce anlamsız sonuç
    üretmesini önlemek amaçlı — bkz. pix2seq_decode.load_model."""
    torch.save(model.state_dict(), path)
    meta_path = os.path.splitext(path)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "output_mode": OUTPUT_MODE,
            "tokens_per_object": TOKENS_PER_OBJECT,
            "max_objects": MAX_OBJECTS,
            "num_bins": NUM_BINS,
            "noise_augmentation": NOISE_AUGMENTATION,
            "backbone_name": BACKBONE_NAME,
            "feature_levels": list(FEATURE_LEVELS),
            "encoder_on_coarse": ENCODER_ON_COARSE,
            "img_size": list(IMG_SIZE),
            "object_order": OBJECT_ORDER,
            "canonical_corner_order": CANONICAL_CORNER_ORDER,
        }, f, indent=2)

# --- EĞİTİM DÖNGÜSÜ ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim {device} üzerinde başlıyor...")

    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    random.shuffle(all_json_files)
    train_size  = int(0.9 * len(all_json_files))
    train_files = all_json_files[:train_size]
    val_files   = all_json_files[train_size:]

    # Train ve val için AYRI dataset instance'ları: val'de mosaic yok, jitter yok
    # (val_transform: sadece resize+normalize), obje sırası karıştırılmıyor ve
    # noise-object augmentation kapalı — val loss/AP sadece gerçek objeler
    # üzerinden, deterministik şekilde ölçülsün. _mosaic de sadece kendi
    # split'inin dosya listesinden örnekler, dolayısıyla train/val arasında
    # sızıntı olmaz.
    train_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform,
                                    json_files=train_files, mosaic_prob=MOSAIC_PROB,
                                    order_mode=OBJECT_ORDER, noise_augmentation=NOISE_AUGMENTATION)
    val_dataset   = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform,
                                    json_files=val_files, mosaic_prob=0.0,
                                    order_mode="class", noise_augmentation=False)

    # Train verisinin AUGMENTE EDİLMEMİŞ bir alt kümesi, val ile birebir aynı
    # şartlarda ölçülür. Bu, "model neden 5 px hata yapıyor" sorusunu ikiye böler:
    #   temiz-train ≈ val  -> model veriyi ZATEN fit edemiyor: kapasite/çözünürlük
    #                         sınırı. Daha ince feature grid / daha büyük girdi gerek.
    #   temiz-train << val  -> model fit edebiliyor ama genelleyemiyor: daha çok
    #                         veri / daha güçlü regularizasyon gerek; çözünürlük
    #                         artırmak burada işe YARAMAZ.
    # İkisi tamamen farklı işler olduğu için tahmin etmek yerine ölçüyoruz.
    trainclean_files = train_files[:len(val_files)]  # val ile aynı büyüklükte
    trainclean_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform,
                                         json_files=trainclean_files, mosaic_prob=0.0,
                                         order_mode="class", noise_augmentation=False)

    with open("val_split.json", "w") as f:
        json.dump({"filenames": val_files}, f, indent=2)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dataloader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    trainclean_dataloader = DataLoader(trainclean_dataset, batch_size=BATCH_SIZE,
                                        shuffle=False, num_workers=2)

    max_seq_len = train_dataset.max_seq_len

    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)

    # Lokalizasyon hassasiyetini belirleyen sayıları baştan görelim — kutuların
    # ne kadar sıkı olabileceğinin üst sınırı feature grid'i, NUM_BINS değil.
    _g = (IMG_SIZE[0] // model.out_stride, IMG_SIZE[1] // model.out_stride)
    _n_mem = _g[0] * _g[1]
    print(f"  feature levels : {FEATURE_LEVELS}  -> stride {model.out_stride}, "
          f"grid {_g[0]}x{_g[1]} = {_n_mem} memory token "
          f"(1 hücre = {model.out_stride} px)")
    if model.encoder_on_coarse:
        _cs = model.encoder.feature_info.reduction()[model.coarsest_idx]
        _cn = (IMG_SIZE[0] // _cs) * (IMG_SIZE[1] // _cs)
        print(f"  self-attention : kaba seviyede (stride {_cs}, {_cn} token, "
              f"{_cn**2/1e6:.1f}M çift) — ince grid olsaydı {_n_mem**2/1e6:.1f}M çift")
    else:
        print(f"  self-attention : ince grid üzerinde ({_n_mem} token, "
              f"{_n_mem**2/1e6:.1f}M çift)")
    print(f"  koordinat bin  : {NUM_BINS} bin -> {IMG_SIZE[0]/(NUM_BINS-1):.3f} px/bin")
    if COORD_LABEL_SIGMA > 0:
        _floor = math.log(COORD_LABEL_SIGMA * math.sqrt(2 * math.pi * math.e))
        print(f"  soft target    : sigma={COORD_LABEL_SIGMA} bin -> loss tabanı ≈ {_floor:.3f} nat "
              f"(koordinat token'ları bu değerin altına inemez)")

    # reduction='none': loss'u token bazında alıp dataset'ten gelen loss_mask ile
    # çarpıyoruz (noise objelerin koordinat token'ları maskeleniyor, bkz.
    # NOISE_COORD_LOSS). Ortalama, yalnızca supervise edilen token sayısına
    # bölünür — aksi halde maskelenen token sayısı değiştikçe loss ölçeği kayar
    # ve epoch'lar arası karşılaştırılamaz hale gelirdi.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN,
                                    label_smoothing=LABEL_SMOOTHING,
                                    reduction='none')

    # Gauss soft-target için sabit bin ekseni (her adımda yeniden kurulmasın).
    _bin_axis = torch.arange(NUM_BINS, device=device, dtype=torch.float32)

    def masked_loss(logits, target, mask):
        flat_logits = logits.reshape(-1, VOCAB_SIZE)
        flat_target = target.reshape(-1)
        flat_mask   = mask.reshape(-1)

        per_token = criterion(flat_logits, flat_target)

        if COORD_LABEL_SIGMA > 0:
            # Koordinat token'larında sert hedef yerine hedef bin'in etrafına
            # yerleştirilmiş gauss dağılım kullanılır (bkz. COORD_LABEL_SIGMA).
            # Sınıf/EOS token'ları sert hedefte kalır — onlar ordinal değil.
            is_coord = flat_mask & (flat_target < NUM_BINS)
            idx = is_coord.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                logp = torch.log_softmax(flat_logits[idx].float(), dim=-1)
                d = _bin_axis.unsqueeze(0) - flat_target[idx].unsqueeze(1).float()
                w = torch.exp(-0.5 * (d / COORD_LABEL_SIGMA) ** 2)
                w = w / w.sum(dim=-1, keepdim=True)
                soft_ce = -(w * logp[:, :NUM_BINS]).sum(dim=-1)
                per_token = per_token.clone()
                per_token[idx] = soft_ce.to(per_token.dtype)

        m = flat_mask.to(per_token.dtype)
        return (per_token * m).sum() / m.sum().clamp(min=1.0)

    PX_PER_BIN = IMG_SIZE[0] / (NUM_BINS - 1)

    @torch.no_grad()
    def token_metrics(logits, target, mask):
        """Loss skaleri tek başına yorumlanamaz (koordinat entropisi + sınıf
        doğruluğu + EOS aynı sayıda toplanıyor). Bunları ayırıyoruz.

        Koordinat hatası için ORTALAMA yeterli değil: 5 px ortalama, "her kenar
        tutarlı 5 px kaba" da olabilir, "çoğu kenar 1 px ama %5'i 50 px" de —
        ve bu ikisi tamamen farklı düzeltmeler gerektirir (birincisi çözünürlük/
        kapasite, ikincisi belirli bir bozulma modu). O yüzden ham hata vektörü
        döndürülüp epoch sonunda percentile'ları alınıyor.

        Returns: (coord_err_px [tensor], class_hit, n_class)"""
        pred = logits.argmax(-1)
        is_coord = mask & (target < NUM_BINS)
        is_class = mask & (target >= NUM_BINS) & (target < NUM_BINS + NUM_CLASSES + 1)

        coord_err = ((pred[is_coord] - target[is_coord]).abs().float() * PX_PER_BIN).cpu()
        n_class = int(is_class.sum())
        class_hit = float((pred[is_class] == target[is_class]).float().sum()) if n_class else 0.0
        return coord_err, class_hit, n_class

    # EMA modeli: her zaman eval modunda, kendi gradyanı yok — sadece
    # update_ema() ile ham `model`in ağırlıklarının hareketli ortalamasını tutar.
    ema_model = None
    ema_step  = 0
    if EMA_DECAY > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_model.eval()

    if ENCODER_FREEZE_EPOCHS > 0:
        set_encoder_trainable(model, False)

    def amp_ctx():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                               enabled=(USE_BF16 and device.type == "cuda"))

    # Parametre grupları modül modül elle sayılmıyor: yalnızca iki LR rejimi var
    # (pretrained backbone ve geri kalan her şey), o yüzden named_parameters()
    # üzerinden ayrılıyorlar. Elle liste tutmak kırılgandı — mimariye yeni bir
    # modül eklenince ya AttributeError veriyor ya da (daha kötüsü) o parametreler
    # hiçbir gruba girmediği için SESSİZCE hiç eğitilmiyordu.
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        (backbone_params if name.startswith("encoder.") else head_params).append(p)

    n_covered = len(backbone_params) + len(head_params)
    n_total   = len(list(model.parameters()))
    assert n_covered == n_total, f"optimizer {n_covered}/{n_total} parametreyi kapsıyor"
    print(f"  optimizer      : backbone {len(backbone_params)} tensör @ lr={ENCODER_LR}, "
          f"geri kalan {len(head_params)} tensör @ lr={LEARNING_RATE}")

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': ENCODER_LR},
        {'params': head_params,     'lr': LEARNING_RATE},
    ], weight_decay=1e-4)

    # optimizer.step() artık her batch'te değil, GRAD_ACCUM_STEPS'te bir
    # çağrılıyor — scheduler'ın total_steps'i buna göre hesaplanmalı.
    steps_per_epoch = math.ceil(len(train_dataloader) / GRAD_ACCUM_STEPS)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[ENCODER_LR, LEARNING_RATE],  # optimizer'daki grup sırasıyla aynı
        total_steps=total_steps,
        pct_start=0.05
    )

    best_val_loss = float('inf')

    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")

    n_train_batches = len(train_dataloader)

    for epoch in epoch_bar:
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        if epoch < ENCODER_FREEZE_EPOCHS:
            model.encoder.eval()  # requires_grad zaten False — BN running stats de donuk kalsın
        elif epoch == ENCODER_FREEZE_EPOCHS and ENCODER_FREEZE_EPOCHS > 0:
            set_encoder_trainable(model, True)
            tqdm.write(f"  → Epoch {epoch+1}: encoder çözüldü (unfreeze).")

        train_loss = 0.0
        optimizer.zero_grad()

        train_bar = tqdm(train_dataloader, desc=f"  Train {epoch+1}/{EPOCHS}",
                         leave=False, unit="batch")

        for step, (images, targets, masks) in enumerate(train_bar):
            images  = images.to(device).float()
            targets = targets.to(device)
            masks   = masks.to(device)

            decoder_input  = targets[:, :-1]
            decoder_target = targets[:, 1:]
            target_mask    = masks[:, 1:]

            with amp_ctx():
                logits = model(images, decoder_input)
                loss   = masked_loss(logits, decoder_target, target_mask)

            (loss / GRAD_ACCUM_STEPS).backward()

            is_accum_boundary = ((step + 1) % GRAD_ACCUM_STEPS == 0) or (step + 1 == n_train_batches)
            if is_accum_boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if ema_model is not None:
                    decay = min(EMA_DECAY, (ema_step + 1) / (ema_step + 10))
                    update_ema(ema_model, model, decay)
                    ema_step += 1

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / n_train_batches

        # ── Validation ─────────────────────────────────────────────────────
        # Hem ham model hem EMA ayrı ayrı ölçülüyor. EMA_DECAY=0.999 ve
        # ~124 step/epoch ile EMA ufku ~8 epoch — yani EMA val loss'u ham modelin
        # 8 epoch gerisinden takip eder ve eğriyi yapay olarak pürüzsüz/yavaş
        # gösterir. İkisini birlikte loglamak "model mi yavaş öğreniyor, yoksa
        # EMA mı geride kalıyor" sorusunu tek bakışta ayırır.
        @torch.no_grad()
        def evaluate(m, tag, loader=None):
            m.eval()
            total, n_batches = 0.0, 0
            errs, class_hit, class_n = [], 0.0, 0

            bar = tqdm(loader if loader is not None else val_dataloader,
                       desc=f"  Val[{tag}] {epoch+1}/{EPOCHS}",
                       leave=False, unit="batch")
            for images, targets, masks in bar:
                images  = images.to(device).float()
                targets = targets.to(device)
                masks   = masks.to(device)

                decoder_input  = targets[:, :-1]
                decoder_target = targets[:, 1:]
                target_mask    = masks[:, 1:]

                with amp_ctx():
                    logits = m(images, decoder_input)
                    loss   = masked_loss(logits, decoder_target, target_mask)

                ce, ch, kn = token_metrics(logits.float(), decoder_target, target_mask)
                errs.append(ce); class_hit += ch; class_n += kn

                total += loss.item(); n_batches += 1
                bar.set_postfix(loss=f"{loss.item():.4f}")

            # torch.cat boş tensörler üretebilir (bir batch'te hiç koordinat
            # token'ı olmaması); quantile/max boş girdide patlar.
            all_err = torch.cat(errs) if errs else torch.zeros(0)
            if all_err.numel() == 0:
                all_err = torch.zeros(1)
            q = torch.quantile(all_err, torch.tensor([0.5, 0.9, 0.99]))
            stats = {
                "mean": float(all_err.mean()),
                "p50":  float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                "max":  float(all_err.max()),
            }
            return total / max(n_batches, 1), stats, class_hit / max(class_n, 1)

        raw_val_loss, raw_px, raw_class_acc = evaluate(model, "raw")
        model.train()

        if ema_model is not None:
            ema_val_loss, ema_px, ema_class_acc = evaluate(ema_model, "ema")
        else:
            ema_val_loss, ema_px, ema_class_acc = raw_val_loss, raw_px, raw_class_acc

        # Checkpoint hangi ağırlıklarla seçilecek: hangisi o an daha iyiyse.
        # (Eğitimin başında ham model, sonlarında genelde EMA kazanır.)
        if ema_model is not None and ema_val_loss <= raw_val_loss:
            eval_model, avg_val_loss = ema_model, ema_val_loss
            best_tag, px, class_acc = "ema", ema_px, ema_class_acc
        else:
            eval_model, avg_val_loss = model, raw_val_loss
            best_tag, px, class_acc = "raw", raw_px, raw_class_acc

        # ── Logging ────────────────────────────────────────────────────────
        epoch_bar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            raw=f"{raw_val_loss:.4f}",
            ema=f"{ema_val_loss:.4f}",
            p50=f"{px['p50']:.1f}",
            p90=f"{px['p90']:.1f}",
            acc=f"{class_acc:.3f}",
            best=f"{best_val_loss:.4f}"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(eval_model, "pix2seq_best.pth")
            # p50 >> p90 ise hata birkaç kötü kenarda yoğunlaşmış demektir
            # (belirli bir bozulma modu); p50 ≈ p90 ise hata tabana yayılmış
            # demektir (çözünürlük/kapasite sınırı). İkisi farklı iş gerektirir.
            tqdm.write(
                f"  ✓ Epoch {epoch+1:3d} — val loss {avg_val_loss:.4f} ({best_tag}) "
                f"| kenar hatası px: ort {px['mean']:.2f} / p50 {px['p50']:.2f} / "
                f"p90 {px['p90']:.2f} / p99 {px['p99']:.2f} / maks {px['max']:.1f} "
                f"| sınıf {class_acc:.3f} → pix2seq_best.pth"
            )

        # Temiz (augmente edilmemiş) train alt kümesi — pahalı olduğu için her
        # epoch değil. val ile arasındaki fark, çözünürlük artırmanın işe yarayıp
        # yaramayacağını söyler (bkz. trainclean_dataset açıklaması).
        if (epoch + 1) % TRAINCLEAN_EVERY == 0 or epoch + 1 == EPOCHS:
            tc_loss, tc_px, tc_acc = evaluate(eval_model, "train-clean",
                                              trainclean_dataloader)
            model.train()
            gap = px["p50"] - tc_px["p50"]
            verdict = ("kapasite/çözünürlük sınırı — daha ince grid işe yarar"
                       if abs(gap) < 0.25 * max(px["p50"], 1e-6)
                       else "genelleme açığı — daha çok veri/regularizasyon gerek, "
                            "çözünürlük artırmak işe yaramaz")
            tqdm.write(
                f"    train-clean: loss {tc_loss:.4f} | kenar p50 {tc_px['p50']:.2f} px "
                f"vs val p50 {px['p50']:.2f} px  ->  {verdict}"
            )

    save_checkpoint(model, "pix2seq_model_v2.pth")
    print(f"\nEğitim tamamlandı.")
    print(f"  Son model  : pix2seq_model_v2.pth")
    print(f"  En iyi model: pix2seq_best.pth  (val loss: {best_val_loss:.4f})")
