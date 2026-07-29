import os
import json
import math
import copy
import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

from augmentations import (
    ImageAugmentor,
    LabelCorruptionStrategy,
    QuadAugmentation,
    corrupt_class_labels,
    letterbox_params,
)
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

# --- Augmentasyon ayarları (Chris-hughes10/pix2seq pipeline'ı) ---
# Görüntü / kutu / sequence olmak üzere üç katman da augmentations.py'de.
# Bu bölüm o modülün parametrelerini toplar.
#
# NOT: Bu pipeline COCO için yazıldı ve önceki (rotate/perspective/blur/gürültü/
# gölge/JPEG ağırlıklı) pipeline'ın YERİNE geçti. Mosaic de kaldırıldı —
# repo'da yok. Önceki sürüme dönmek gerekirse "training - Kopya.py" duruyor.

# Large scale jitter: görüntü letterbox'landıktan sonra 0.3x .. 2.0x arası
# rastgele ölçeklenip 512'ye geri kırpılır. Repo'daki en güçlü geometrik
# augmentasyon bu; eski `ShiftScaleRotate(scale_limit=0.2)`nin çok ötesinde bir
# ölçek aralığı gördürür ve modelin obje büyüklüğüne aşırı uyum sağlamasını
# engeller.
JITTER_SCALE = (0.3, 2.0)

# ColorJitter'ın brightness/contrast/saturation genliği (hue bunun 0.2 katı).
COLOR_JITTER_STRENGTH = 0.4

# Yatay çevirme olasılığı (repo: 0.5).
#
# UYARI — bu veri seti için ciddi bir yan etkisi var: ruhsat bir METİN belgesi,
# hflip plaka/tc/seri_no alanlarındaki yazıyı aynalar ve modelin göreceği
# görüntülerin yarısı üretimde asla karşılaşmayacağı bir dağılımdan gelir.
# Ayrıca köşe sırasını da bozar (sol-üst köşe sağ-üste geçer) — bu yüzden
# CANONICAL_CORNER_ORDER aşağıda True'ya çekildi, aksi halde model tutarsız
# hedef görüp papyon quad üretir. Metin yönü önemliyse 0.0 yapın.
HFLIP_PROB = 0.5

# ColorJitter, LongestMaxSize'dan ÖNCE mi çalışsın (repo sırası) yoksa SONRA mı?
# Sonra çalıştırmak sonucu pratikte değiştirmez ama telefon çözünürlüğündeki
# görüntülerde CPU maliyetini ~30x düşürür (bkz. augmentations.ImageAugmentor).
COLOR_FIRST = False

# --- Kutu (quad) augmentasyonu ---
# Gerçek objelere uygulanan jitter genliği; gürültü kutunun KENDİ boyutuyla
# ölçeklenir (repo `jitter_bbox`). Etiket gerçek kalır — model "kutu tam
# oturmasa da sınıfı doğru bul" sinyalini buradan alır.
MAX_JITTER = 0.05

# Noise objelerin gerçeklerin ARASINA karıştırılma olasılığı (repo `mix_rate`).
# 0.0 -> hep sona eklenir (bu projenin eski davranışı), 1.0 -> hep karıştırılır.
MIX_RATE = 0.5

# Sequence augmentation: decoder GİRDİ dizisindeki sınıf token'larının bozulma
# stratejisi (repo config'i: "rand_n_fake_cls").
#   NONE            -> bozma yok (eski davranış)
#   RANDOM          -> %50 orijinal, %50 rastgele geçerli sınıf
#   RANDOM_AND_FAKE -> %50 orijinal, %25 rastgele sınıf, %25 noise token
# Hedef dizi HER ZAMAN temiz kalır; bkz. augmentations.corrupt_class_labels.
CLASS_LABEL_CORRUPTION = LabelCorruptionStrategy.RANDOM_AND_FAKE

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
#
# HFLIP_PROB > 0 İKEN ZORUNLU: yatay çevirme köşelerin geometrik konumunu
# değiştirir (sol-üst köşe sağ-üste geçer) ama etiketteki sırayı değiştirmez,
# yani model aynı slotta bazen sol-üst bazen sağ-üst köşe görür. Kanonikleştirme
# sırayı augmentasyon SONRASI geometriden yeniden türeterek bunu ortadan kaldırır.
CANONICAL_CORNER_ORDER = True

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

# --- Mimari ayarları (repo model config'i) ---
# Stochastic depth (repo `drop_path: 0.1`). Küçük veri setlerinde transformer
# katmanlarının ezberlemesini geciktiren, dropout'tan daha etkili bir
# regularizasyon. Uygulaması için bkz. DropPathTransformer*Layer.
DROP_PATH = 0.1

# Girdi embedding'i ile çıkış projeksiyonunun ağırlığını paylaştır
# (repo `shared_decoder_embedding: true`).
TIE_WORD_EMBEDDINGS = True

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
#
# BBOX (RECTANGLE) İLE ETİKETLENMİŞ DATASET KULLANIYORSANIZ "hbb" SEÇİN.
# Veri okuma tarafı her iki etiket formatını da kabul eder (bkz. shape_to_quad:
# labelme'nin 2-noktalı `rectangle` shape'i otomatik 4 köşeye çevrilir), yani
# "both"/"corner" da teknik olarak çalışır — ama eksen-hizalı bir kutunun 4
# köşesi zaten min/max'ından tamamen belirlidir, dolayısıyla o modlarda model
# obje başına 8 token'lık BİLGİ TAŞIMAYAN bir hedef üretmeyi öğrenir:
#     "hbb"  ->  5 token/obje, MAX_OBJECTS=10 ile dizi 52 token
#     "both" -> 13 token/obje, MAX_OBJECTS=10 ile dizi 132 token
# Yani 2.5x daha uzun dizi, 2.5x daha yavaş autoregressive decode ve daha çok
# hata birikme fırsatı — karşılığında sıfır ek bilgi.
# __main__ içindeki dataset taraması bu durumu tespit edip uyarır.
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

# --- Token ağırlıkları (repo `TokenProcessor`) ---
# Loss artık ikili bir maske değil, token başına FLOAT ağırlık kullanıyor.
# Repo'nun ana katkısı burada: karmaşıklık loss fonksiyonundan veri
# temsiline taşınmış durumda, tek ayar noktası bu üç sayı.
#
# Noise objelerin KOORDİNAT token'larının ağırlığı. Makale (Böl. 3.3) ve repo'nun
# kendi docstring'i 0 diyor: noise koordinatları tanım gereği rastgele, yani
# öğrenilebilir sinyal içermiyor. Loss'a dahil edilirlerse (7 gerçek + 3 noise
# objede) hedef token'ların önemli kısmı saf gürültü olur ve gradyan modele
# "rastgele sayı tahmin et" der. (Repo'nun train.yaml'ı 1.0 veriyor — kodun
# kendi yorumuyla çelişiyor; 0.0 doğru olan.)
NOISE_BBOX_WEIGHT = 0.0

# BOS/EOS token'larının ağırlığı (repo config: 0.1).
# EOS, dizinin NE ZAMAN biteceğini belirleyen tek token. Tam ağırlıkta
# tutulduğunda tüm koordinat/sınıf token'larıyla eşit yarışır ve model "erken
# bitir" çözümüne kayar — sonuç obje düşürmedir (recall kaybı), üstelik dizi
# kesildiği için sonraki objeler hiç üretilmez. 0.1 bu baskıyı azaltır.
EOS_TOKEN_WEIGHT = 0.1

# +1: noise sınıfı için ekstra vocab slotu, +3: BOS/EOS/PAD
VOCAB_SIZE = NUM_BINS + NUM_CLASSES + 1 + 3
NOISE_TOKEN = NUM_BINS + NOISE_CLASS_ID
BOS_TOKEN = VOCAB_SIZE - 3
EOS_TOKEN = VOCAB_SIZE - 2
PAD_TOKEN = VOCAB_SIZE - 1


assert IMG_SIZE[0] == IMG_SIZE[1], "Repo pipeline'ı (LongestMaxSize + kare pad) kare girdi varsayar."

# Görüntü augmentasyonu artık augmentations.ImageAugmentor'da. Eğitim ve
# değerlendirme için ayrı iki instance: değerlendirme yolunda yalnızca
# LongestMaxSize + PadIfNeeded (deterministik letterbox) çalışır, böylece val
# loss/AP augmentasyon gürültüsünden etkilenmez ve inference ile birebir aynı
# ön-işlemeyi görür (bkz. pix2seq_decode.encode_image).
train_augmentor = ImageAugmentor(
    image_size=IMG_SIZE[0], jitter_scale=JITTER_SCALE,
    color_jitter_strength=COLOR_JITTER_STRENGTH, hflip_prob=HFLIP_PROB,
    training=True, color_first=COLOR_FIRST,
)
val_augmentor = ImageAugmentor(image_size=IMG_SIZE[0], training=False)


def quantize(normalized, num_bins):
    """[0,1] koordinatını bin indeksine çevirir.

    Koordinatlar artık ImageAugmentor'dan normalize geliyor (repo da normalize
    uzayda kuantize ediyor), o yüzden ayrı bir `max_size` argümanı yok.

    round, int() (floor) değil: dequantize `token/(num_bins-1)` ile ters
    çeviriyor, dolayısıyla floor sistematik olarak yarım bin'lik bir sola-kayma
    (bias) üretiyordu. round ile iki fonksiyon birbirinin tam tersi.
    """
    normalized = max(0.0, min(1.0, float(normalized)))
    return int(round(normalized * (num_bins - 1)))


def _axis_aligned_quad(points):
    """Nokta kümesinin eksen-hizalı kutusunu 4 köşe olarak döndürür.

    Sıra (sol-üst, sağ-üst, sağ-alt, sol-alt) zaten kanoniktir — ağırlık merkezi
    etrafında açı sırasında ve sol-üstten başlar — yani canonicalize_corners bu
    diziyi değiştirmez (bkz. CANONICAL_CORNER_ORDER).
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def shape_to_quad(shape, bbox_fallback=False):
    """labelme shape'ini 4 köşeye normalize eder; desteklenmiyorsa None döner.

    Bu proje objeleri her zaman 4 köşe olarak taşır (OUTPUT_MODE ne olursa
    olsun), ama etiketleme aracı iki farklı format üretir:

      shape_type="rectangle" -> İKİ nokta: [[x1,y1],[x2,y2]]  (karşıt köşeler)
      shape_type="polygon"   -> N nokta;   bu projede beklenen N=4

    Rectangle'ın iki noktası, kullanıcı kutuyu hangi yönde sürüklediyse ona göre
    ters sırada gelebilir (x1 > x2 olabilir), o yüzden min/max ile normalize
    ediliyor. Dönüşüm KAYIPSIZ: eksen-hizalı bir kutunun 4 köşesi iki karşıt
    köşesinden tam olarak türetilir.

    `bbox_fallback=True` iken 3+ noktalı serbest poligonlar da eksen-hizalı
    kutularına indirgenir. Yalnızca OUTPUT_MODE="hbb" için anlamlı: o modda
    zaten sadece min/max kullanılıyor, dolayısıyla bilgi kaybı yok. "corner"/
    "both" modunda poligonun şekli hedefin parçası olduğu için bu indirgeme
    yapılmaz — shape atlanır.
    """
    pts = [(float(p[0]), float(p[1])) for p in shape.get("points", []) if len(p) >= 2]
    stype = shape.get("shape_type", "")

    if stype == "rectangle" or len(pts) == 2:
        return _axis_aligned_quad(pts) if len(pts) >= 2 else None
    if len(pts) == 4:
        return list(pts)
    if bbox_fallback and len(pts) >= 3:
        return _axis_aligned_quad(pts)
    return None


def scan_dataset_shapes(json_dir, json_files=None):
    """Etiket dosyalarını tarayıp shape formatlarını sayar.

    Bunun varlık sebebi: uyumsuz bir shape sessizce atlanıyor ve eğitim "hiç
    gerçek obje yok" haliyle sorunsuz koşuyor (tüm slotlar noise objeyle
    doluyor, loss düşüyor, hiçbir hata çıkmıyor). Sayıları eğitim başında
    yazdırmak bu senaryoyu görünür kılar.
    """
    files = json_files if json_files is not None else \
        [f for f in os.listdir(json_dir) if f.endswith('.json')]
    counts = {"rectangle": 0, "quad": 0, "bbox_fallback": 0,
              "skipped": 0, "unknown_label": 0, "files": 0}
    skipped_kinds = {}
    fallback_kinds = {}
    bbox_fallback = (OUTPUT_MODE == "hbb")

    for fn in files:
        try:
            with open(os.path.join(json_dir, fn), encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue
        counts["files"] += 1
        for shape in item.get("shapes", []):
            if shape.get("label", "") not in LABEL_TO_ID:
                counts["unknown_label"] += 1
                continue
            n_pts = len(shape.get("points", []))
            stype = shape.get("shape_type", "")
            key = f"{stype or 'belirtilmemiş'}/{n_pts} nokta"
            if shape_to_quad(shape, bbox_fallback) is None:
                counts["skipped"] += 1
                skipped_kinds[key] = skipped_kinds.get(key, 0) + 1
            elif stype == "rectangle" or n_pts == 2:
                counts["rectangle"] += 1
            elif n_pts == 4:
                counts["quad"] += 1
            else:
                # Yalnızca bbox_fallback sayesinde geçti: şekli kaybedildi,
                # eksen-hizalı kutusuna indirgendi. "quad" diye saymak yanıltıcı
                # olurdu — ayrı raporlanıyor.
                counts["bbox_fallback"] += 1
                fallback_kinds[key] = fallback_kinds.get(key, 0) + 1
    counts["skipped_kinds"] = skipped_kinds
    counts["fallback_kinds"] = fallback_kinds
    return counts


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


# NOT: `random_box_points` / `jitter_box_points` kaldırıldı — yerlerini
# augmentations.QuadAugmentation'ın `random_quad` / `jitter_quad` / `shift_quad`
# üçlüsü aldı. Fark sadece isim değil: eski jitter gürültüyü GÖRÜNTÜ boyutuyla
# ölçeklediği için ince metin alanlarında objeyi tamamen ıskalıyordu; yenisi
# kutunun kendi boyutuyla ölçekliyor (bkz. QuadAugmentation.jitter_quad).

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
    """Repo'nun üç katmanlı augmentasyon şemasını uygular:

        1. ImageAugmentor      -> görüntü + köşe noktaları (piksel -> normalize)
        2. QuadAugmentation    -> gerçeklere jitter, eksik slotlara noise obje
        3. corrupt_class_labels-> decoder GİRDİ dizisindeki sınıf token'larını bozar

    Döndürdüğü 4'lü: (image, input_seq, target_seq, token_weights).

    input_seq ve target_seq artık AYRI: eskiden tek bir `sequence` üretilip
    eğitim döngüsünde `[:, :-1]` / `[:, 1:]` diye kaydırılıyordu, yani girdi ile
    hedef aynı token'lardan oluşuyordu. Sequence augmentation'ın çalışabilmesi
    için ikisinin FARKLI olması gerekiyor (girdi bozuk, hedef temiz), o yüzden
    kaydırma buraya, dataset'e taşındı.
    """

    def __init__(self, json_dir, img_dir, max_objects=MAX_OBJECTS, img_size=IMG_SIZE,
                 augmentor=None, json_files=None, order_mode=OBJECT_ORDER,
                 noise_augmentation=NOISE_AUGMENTATION,
                 label_corruption=CLASS_LABEL_CORRUPTION):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.img_size = img_size
        # json_files verilirse (train/val split'i) o listeye sabitlenir; verilmezse
        # dizindeki tüm .json dosyaları kullanılır (tek dataset kullanım senaryosu).
        self.json_files = json_files if json_files is not None else \
            [f for f in os.listdir(json_dir) if f.endswith('.json')]
        self.max_objects = max_objects
        self.max_seq_len = 1 + (TOKENS_PER_OBJECT * max_objects) + 1
        self.augmentor = augmentor if augmentor is not None else train_augmentor
        # Objelerin sequence İÇİNDEKİ SIRASI (bkz. OBJECT_ORDER). Bu, YALNIZCA
        # hangi objenin kaçıncı sırada yazılacağını etkiler. Her objenin kendi 4
        # köşesinin sırası (canonical etiketleme) buna dokunulmadan aynen korunur.
        self.order_mode = order_mode
        # Gerçek objelerin yanına MAX_OBJECTS'e tamamlayacak kadar "noise" obje
        # eklenir (bkz. modül üstündeki NOISE_AUGMENTATION açıklaması). Val'de
        # kapalı tutulmalı — val loss/AP sadece gerçek objeler üzerinden ölçülsün.
        self.noise_augmentation = noise_augmentation
        # Val'de daima NONE olmalı: val loss'un checkpoint seçimi için
        # karşılaştırılabilir kalması, bozulmuş girdilerin gürültüsünden
        # etkilenmemesine bağlı.
        self.label_corruption = label_corruption
        self.quad_augmentor = QuadAugmentation(NOISE_CLASS_ID)

    def __len__(self):
        return len(self.json_files)

    def _load(self, idx):
        """Ham görüntüyü ve objeleri PİKSEL uzayında döndürür.

        Letterbox/resize artık burada yapılmıyor — ImageAugmentor'ın
        LongestMaxSize + PadIfNeeded adımları hallediyor (bkz. augmentations.py).
        """
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
                img_path = os.path.join(self.img_dir, base + '.jpg')

        try:
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
        except Exception as e:
            # Görüntü okunamadıysa OBJELERİ DE atıyoruz: koordinatlar orijinal
            # görüntünün uzayında, boş tuval ise başka bir boyutta — ikisini
            # eşleştirmek modele tamamen yanlış bir hedef öğretirdi.
            print(f"HATA: {img_path} okunamadı. Hata: {e}")
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8), []

        # shape_to_quad hem 4-noktalı poligonu hem 2-noktalı `rectangle`ı kabul
        # eder — yani hbb ile etiketlenmiş bir dataset ek iş gerektirmeden
        # çalışır (bkz. shape_to_quad).
        bbox_fallback = (OUTPUT_MODE == "hbb")
        objects = []
        for shape in item.get("shapes", []):
            label_str = shape.get("label", "")
            if label_str not in LABEL_TO_ID:
                continue
            quad = shape_to_quad(shape, bbox_fallback)
            if quad is not None:
                objects.append((LABEL_TO_ID[label_str], quad))
        return image_np, objects

    def __getitem__(self, idx):
        image_np, objects = self._load(idx)

        # ── 1. Görüntü augmentasyonu ──────────────────────────────────────────
        # Çıkışta koordinatlar [0,1]'e normalize; kadraj dışında kalan objeler
        # atılmış olur (bkz. augmentations.MIN_VISIBILITY).
        image_tensor, objects = self.augmentor(image_np, objects)

        # Köşe sırası: augmentasyon SONRASI kanonikleştirilir — yatay çevirme /
        # ölçekleme köşelerin geometrik konumunu değiştirdiği için sıranın da o
        # yeni konuma göre belirlenmesi gerekir (bkz. CANONICAL_CORNER_ORDER).
        if CANONICAL_CORNER_ORDER:
            objects = [
                (cid, np.asarray(canonicalize_corners([tuple(p) for p in quad]),
                                 dtype=np.float32))
                for cid, quad in objects
            ]

        # Gerçek objelerin sequence İÇİNDEKİ SIRASI (bkz. OBJECT_ORDER) — hangi
        # objenin köşeleri olduğu ya da o köşelerin sırası değişmiyor, sadece
        # "önce hangi obje yazılsın" belirleniyor. Val'de daima deterministik.
        if self.order_mode == "random":
            random.shuffle(objects)
        else:  # "class": sınıf ID'si, eşitlikte sol-üst köşe konumu
            objects.sort(key=lambda o: (o[0],
                                        float(o[1][:, 1].min()),
                                        float(o[1][:, 0].min())))

        # MAX_OBJECTS'ten fazla gerçek obje varsa: eğitimde rastgele örnekle
        # (repo davranışı — hep aynı objeleri kesmemek için), val'de ilk N.
        if len(objects) > self.max_objects:
            if self.noise_augmentation:
                idxs = random.sample(range(len(objects)), self.max_objects)
                objects = [objects[i] for i in sorted(idxs)]
            else:
                objects = objects[:self.max_objects]

        # ── 2. Kutu (quad) augmentasyonu ──────────────────────────────────────
        # Gerçeklere jitter + boş slotlara noise obje. Val'de kapalı.
        if self.noise_augmentation:
            objects = self.quad_augmentor.augment(
                objects,
                n_noise=self.max_objects - len(objects),
                max_jitter=MAX_JITTER,
                mix_rate=MIX_RATE,
                # order_mode="class" iken gerçeklerin deterministik sırası
                # korunur, noise objeler yalnızca ARALARINA serpiştirilir.
                keep_real_order=(self.order_mode != "random"),
            )

        return (image_tensor,) + self._build_sequences(objects)

    def _build_sequences(self, objects):
        """objects -> (input_seq, target_seq, token_weights).

        Ağırlık şeması (repo `TokenProcessor.build_sequences`):
          gerçek obje koordinatları -> 1.0
          noise obje koordinatları  -> NOISE_BBOX_WEIGHT (0.0)
          sınıf token'ı (gerçek VE noise) -> 1.0   ["fake" demeyi öğrenmeli]
          BOS / EOS                 -> EOS_TOKEN_WEIGHT (0.1)
          PAD                       -> 0.0
        """
        clean = [BOS_TOKEN]
        corrupt = [BOS_TOKEN]
        weights = [EOS_TOKEN_WEIGHT]

        class_ids = [cid for cid, _ in objects]
        corrupt_ids = corrupt_class_labels(
            class_ids, NUM_CLASSES, NOISE_CLASS_ID, self.label_corruption)

        for (class_id, quad), corrupt_id in zip(objects, corrupt_ids):
            coord_w = NOISE_BBOX_WEIGHT if class_id == NOISE_CLASS_ID else 1.0
            coords = []

            if OUTPUT_MODE in ("corner", "both"):
                # 4 köşe noktası (quad / oriented polygon)
                for x, y in quad:
                    coords.append(quantize(x, NUM_BINS))
                    coords.append(quantize(y, NUM_BINS))

            if OUTPUT_MODE in ("hbb", "both"):
                # Köşelerden türetilen eksen-hizalı (horizontal) bounding box
                coords.append(quantize(quad[:, 0].min(), NUM_BINS))
                coords.append(quantize(quad[:, 1].min(), NUM_BINS))
                coords.append(quantize(quad[:, 0].max(), NUM_BINS))
                coords.append(quantize(quad[:, 1].max(), NUM_BINS))

            clean.extend(coords)
            corrupt.extend(coords)
            weights.extend([coord_w] * len(coords))

            # Sınıf token'ı her modda son sırada — decode tarafı bunu varsayar.
            # HEDEF daima temiz, GİRDİ bozulmuş olabilir: sequence augmentation
            # tam olarak bu ikisinin ayrışmasından ibaret.
            clean.append(NUM_BINS + class_id)
            corrupt.append(NUM_BINS + corrupt_id)
            weights.append(1.0)

        clean.append(EOS_TOKEN)
        corrupt.append(EOS_TOKEN)
        weights.append(EOS_TOKEN_WEIGHT)

        if len(clean) > self.max_seq_len:
            # Obje sınırında kes: TOKENS_PER_OBJECT'in ortasından kesip modele
            # bozuk/yarım bir hedef öğretmeyi önler.
            n_objects = (self.max_seq_len - 2) // TOKENS_PER_OBJECT
            usable = 1 + n_objects * TOKENS_PER_OBJECT
            clean = clean[:usable] + [EOS_TOKEN]
            corrupt = corrupt[:usable] + [EOS_TOKEN]
            weights = weights[:usable] + [EOS_TOKEN_WEIGHT]

        if len(clean) < self.max_seq_len:
            n_pad = self.max_seq_len - len(clean)
            clean.extend([PAD_TOKEN] * n_pad)
            corrupt.extend([PAD_TOKEN] * n_pad)
            weights.extend([0.0] * n_pad)

        # Kaydırma: adım i'de girdi corrupt[i], hedef clean[i+1].
        return (torch.tensor(corrupt[:-1], dtype=torch.long),
                torch.tensor(clean[1:], dtype=torch.long),
                torch.tensor(weights[1:], dtype=torch.float32))


class _DropPathMixin:
    """Katman ÇIKTISININ residual katkısına örnek-bazlı stochastic depth uygular.

    Repo `drop_path: 0.1` kullanıyor ama kendi transformer bloklarını yazdığı
    için her alt-blok (attention / FFN) ayrı ayrı drop edilebiliyor.
    `nn.TransformerEncoderLayer`/`DecoderLayer` alt-bloklara erişim vermiyor;
    bunun yerine katmanın TAMAMINI atlıyoruz.

    Bu ancak norm_first=True (pre-LN) iken doğru: o durumda katman tam olarak
    `out = x + sa(norm(x)) + ff(norm(x))` biçiminde olduğu için `out - x` gerçek
    residual katkıdır ve onu ölçeklemek/sıfırlamak standart stochastic depth ile
    aynı şeydir. Post-LN'de `out - x` residual değildir, o yüzden burada
    norm_first zorunlu tutuluyor.

    state_dict anahtarları değişmez (yeni parametre eklenmiyor), yani mevcut
    checkpoint'ler bu değişiklikten etkilenmez.
    """

    def _apply_drop_path(self, x, out):
        p = getattr(self, "drop_path", 0.0)
        if not self.training or p <= 0.0:
            return out
        keep = 1.0 - p
        # Örnek bazlı maske: batch'in bazı elemanları katmanı atlar.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=out.dtype) < keep
        return x + (out - x) * mask / keep


class DropPathTransformerEncoderLayer(_DropPathMixin, nn.TransformerEncoderLayer):
    def __init__(self, *args, drop_path=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.norm_first, "drop_path yalnızca pre-LN (norm_first=True) ile doğru"
        self.drop_path = drop_path

    def forward(self, src, *args, **kwargs):
        return self._apply_drop_path(src, super().forward(src, *args, **kwargs))


class DropPathTransformerDecoderLayer(_DropPathMixin, nn.TransformerDecoderLayer):
    def __init__(self, *args, drop_path=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.norm_first, "drop_path yalnızca pre-LN (norm_first=True) ile doğru"
        self.drop_path = drop_path

    def forward(self, tgt, *args, **kwargs):
        return self._apply_drop_path(tgt, super().forward(tgt, *args, **kwargs))


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
        encoder_layer = DropPathTransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1,
            norm_first=True, drop_path=DROP_PATH)
        # enable_nested_tensor=False: girdi maskesiz olduğu için nested-tensor
        # hızlı yolu zaten devreye girmiyor, ama özel katman sınıfıyla birlikte
        # sessiz bir uyarı basıyor — açıkça kapatıyoruz.
        self.img_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers,
                                                 norm=nn.LayerNorm(hidden_dim),
                                                 enable_nested_tensor=False)

        # Hedef Dizi (Sequence) Modellemesi
        #self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.seq_pos_encoding = PositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.emb_dropout = nn.Dropout(0.1)

        decoder_layer = DropPathTransformerDecoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True, dropout=0.1,
            norm_first=True, drop_path=DROP_PATH)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(hidden_dim))
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Weight tying (repo `shared_decoder_embedding: true`, `decoder_output_bias: true`).
        # Girdi embedding'i ile çıkış projeksiyonu aynı matrisi paylaşır: vocab
        # 1014 x hidden 256 = ~260K parametre tasarrufu ve — asıl önemlisi —
        # nadir görülen koordinat bin'lerinin embedding'i çıkış tarafından da
        # gradyan alır. 1000 bin'in çoğu bir epoch'ta yalnızca birkaç kez hedef
        # olduğu için bu, koordinat token'larının temsilini belirgin biçimde
        # hızlandırır. bias ayrı kalır (paylaşılmaz).
        if TIE_WORD_EMBEDDINGS:
            # nn.Embedding varsayılan init'i N(0,1); nn.Linear'ınki ±1/sqrt(256).
            # Paylaştırınca embedding'in N(0,1) ağırlıkları ÇIKIŞ projeksiyonu
            # olarak da kullanılır ve logit'ler ~16x büyür — ilk adımlarda loss
            # patlar. Transformer standardı olan 0.02'ye çekiyoruz.
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embedding.weight[PAD_TOKEN].zero_()
            self.fc_out.weight = self.embedding.weight

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
            # Mimari state_dict'i etkileyen ayarlar — inference tarafı bunları
            # okuyup modeli aynı şekilde kurabilsin.
            "tie_word_embeddings": TIE_WORD_EMBEDDINGS,
            "drop_path": DROP_PATH,
            # Ön-işleme şeması: "letterbox" = LongestMaxSize + ORTALANMIŞ pad.
            # Eski checkpoint'ler sağ-alta dolgu yapan şemayla eğitildi ve bu
            # alan onlarda yok — pix2seq_decode bunu görüp uyarır.
            "preprocess": "letterbox_centered",
        }, f, indent=2)

# --- EĞİTİM DÖNGÜSÜ ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim {device} üzerinde başlıyor...")

    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]

    # ── Etiket formatı taraması ────────────────────────────────────────────
    # Uyumsuz bir shape sessizce atlanıyor ve eğitim "hiç gerçek obje yok"
    # haliyle sorunsuz koşuyor: tüm slotlar noise objeyle doluyor, loss düşüyor,
    # hiçbir hata çıkmıyor. Sayıları başta yazdırmak bunu görünür kılar.
    _scan = scan_dataset_shapes(JSON_DIR, all_json_files)
    _n_obj = _scan["rectangle"] + _scan["quad"] + _scan["bbox_fallback"]
    print(f"  etiket formatı : {_scan['files']} dosya, {_n_obj} obje "
          f"({_scan['rectangle']} rectangle/bbox, {_scan['quad']} 4-noktalı quad)")
    if _scan["bbox_fallback"]:
        print(f"    NOT: {_scan['bbox_fallback']} shape eksen-hizalı kutusuna indirgendi "
              f"(OUTPUT_MODE='hbb' olduğu için kayıpsız): {_scan['fallback_kinds']}")
    if _scan["skipped"]:
        print(f"    UYARI: {_scan['skipped']} shape ATLANDI (desteklenmeyen format): "
              f"{_scan['skipped_kinds']}")
    if _scan["unknown_label"]:
        print(f"    NOT: {_scan['unknown_label']} shape LABEL_TO_ID'de olmayan bir etikete sahip.")
    if _n_obj == 0:
        raise RuntimeError(
            "Hiç kullanılabilir obje bulunamadı. LABEL_TO_ID etiketleriyle JSON'daki "
            "'label' değerleri eşleşiyor mu ve shape'ler rectangle/4-noktalı poligon mu?"
        )
    if _scan["rectangle"] and not _scan["quad"] and OUTPUT_MODE != "hbb":
        print(f"    UYARI: dataset tamamen bbox (rectangle) etiketli ama OUTPUT_MODE="
              f"'{OUTPUT_MODE}'. Köşe token'ları min/max'tan türetildiği için ek bilgi "
              f"taşımıyor; OUTPUT_MODE='hbb' dizi uzunluğunu {TOKENS_PER_OBJECT} -> 5 "
              f"token/obje'ye indirir ve decode'u ~{TOKENS_PER_OBJECT/5:.1f}x hızlandırır.")

    random.shuffle(all_json_files)
    train_size  = int(0.9 * len(all_json_files))
    train_files = all_json_files[:train_size]
    val_files   = all_json_files[train_size:]

    # Train ve val için AYRI dataset instance'ları: val'de geometrik/fotometrik
    # jitter yok (val_augmentor: sadece deterministik letterbox), obje sırası
    # karıştırılmıyor, noise-object augmentation kapalı ve sequence augmentation
    # (label corruption) kapalı — val loss/AP sadece gerçek objeler üzerinden,
    # deterministik ve karşılaştırılabilir şekilde ölçülsün.
    train_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, augmentor=train_augmentor,
                                    json_files=train_files,
                                    order_mode=OBJECT_ORDER,
                                    noise_augmentation=NOISE_AUGMENTATION,
                                    label_corruption=CLASS_LABEL_CORRUPTION)
    val_dataset   = Pix2SeqDataset(JSON_DIR, IMG_DIR, augmentor=val_augmentor,
                                    json_files=val_files,
                                    order_mode="class", noise_augmentation=False,
                                    label_corruption=LabelCorruptionStrategy.NONE)

    # Train verisinin AUGMENTE EDİLMEMİŞ bir alt kümesi, val ile birebir aynı
    # şartlarda ölçülür. Bu, "model neden 5 px hata yapıyor" sorusunu ikiye böler:
    #   temiz-train ≈ val  -> model veriyi ZATEN fit edemiyor: kapasite/çözünürlük
    #                         sınırı. Daha ince feature grid / daha büyük girdi gerek.
    #   temiz-train << val  -> model fit edebiliyor ama genelleyemiyor: daha çok
    #                         veri / daha güçlü regularizasyon gerek; çözünürlük
    #                         artırmak burada işe YARAMAZ.
    # İkisi tamamen farklı işler olduğu için tahmin etmek yerine ölçüyoruz.
    trainclean_files = train_files[:len(val_files)]  # val ile aynı büyüklükte
    trainclean_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, augmentor=val_augmentor,
                                         json_files=trainclean_files,
                                         order_mode="class", noise_augmentation=False,
                                         label_corruption=LabelCorruptionStrategy.NONE)

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

    # reduction='none': loss'u token bazında alıp dataset'ten gelen FLOAT token
    # ağırlıklarıyla çarpıyoruz (bkz. NOISE_BBOX_WEIGHT / EOS_TOKEN_WEIGHT).
    # Repo'nun yaklaşımı bu: "karmaşıklık loss fonksiyonundan veri temsiline
    # taşındı" — focal loss, IoU loss, Hungarian matching yok, sadece ağırlıklı
    # cross-entropy.
    #
    # Ortalama, ağırlıkların TOPLAMINA bölünür (repo token sayısına bölüyor).
    # Ağırlık toplamına bölmek loss'u "supervise edilen token başına ortalama
    # nat" olarak yorumlanabilir tutar; token sayısına bölmek ise EOS ağırlığı
    # 0.1'e çekilince loss'u yapay olarak düşürürdü ve geçmiş koşularla
    # karşılaştırılamaz hale getirirdi.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN,
                                    label_smoothing=LABEL_SMOOTHING,
                                    reduction='none')

    # Gauss soft-target için sabit bin ekseni (her adımda yeniden kurulmasın).
    _bin_axis = torch.arange(NUM_BINS, device=device, dtype=torch.float32)

    def weighted_loss(logits, target, weights):
        flat_logits  = logits.reshape(-1, VOCAB_SIZE)
        flat_target  = target.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_mask    = flat_weights > 0

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

        m = flat_weights.to(per_token.dtype)
        return (per_token * m).sum() / m.sum().clamp(min=1e-6)

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
        # `mask` artık float ağırlık tensörü — metrikte yalnızca "supervise
        # ediliyor mu" bilgisi lazım (ağırlığın büyüklüğü değil).
        mask = mask > 0
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

        # Kaydırma artık dataset'te (input_seq zaten hedefe göre bir kaydırılmış
        # ve sequence augmentation ile bozulmuş halde geliyor) — burada `[:, :-1]`
        # / `[:, 1:]` dilimlemesi YAPILMAMALI, aksi halde ikinci kez kaydırılır.
        for step, (images, input_seq, target_seq, weights) in enumerate(train_bar):
            images     = images.to(device).float()
            input_seq  = input_seq.to(device)
            target_seq = target_seq.to(device)
            weights    = weights.to(device)

            with amp_ctx():
                logits = model(images, input_seq)
                loss   = weighted_loss(logits, target_seq, weights)

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
            for images, input_seq, target_seq, weights in bar:
                images     = images.to(device).float()
                input_seq  = input_seq.to(device)
                target_seq = target_seq.to(device)
                weights    = weights.to(device)

                with amp_ctx():
                    logits = m(images, input_seq)
                    loss   = weighted_loss(logits, target_seq, weights)

                ce, ch, kn = token_metrics(logits.float(), target_seq, weights)
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
