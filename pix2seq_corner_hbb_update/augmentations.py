"""
Chris-hughes10/pix2seq (blog: "Rethinking Object Detection as Language
Modelling") augmentasyon pipeline'ının bu projeye portu.

Repo üç KATMANDA augmentasyon yapıyor ve bu modül üçünü de içerir:

  1. GÖRÜNTÜ  (ImageAugmentor)      — repo/data/augmentations.py::ImageAugmentor
  2. KUTU     (QuadAugmentation)    — repo/data/augmentations.py::BBoxAugmentation
  3. SEQUENCE (corrupt_class_labels)— repo/data/tokenizer.py::TokenProcessor

Repo'dan tek yapısal sapma: repo eksen-hizalı kutularla (albumentations
`BboxParams`) çalışıyor, bu proje her obje için 4 KÖŞE (oriented quad) üretiyor.
Dolayısıyla:
  - albumentations tarafında `bbox_params` yerine `keypoint_params` kullanılıyor,
  - `BboxParams(min_visibility=0.1)`in karşılığı elle hesaplanıyor (albumentations
    keypoint'ler için görünürlük filtresi sunmuyor — bkz. MIN_VISIBILITY),
  - jitter/shift/random kutu üreticileri quad üzerinde çalışacak şekilde
    genelleştirildi (x koordinatları kutunun genişliğiyle, y'ler yüksekliğiyle
    ölçekleniyor — repo'daki `sizes = [h, w, h, w]` mantığının aynısı).

Koordinat uzayı: QuadAugmentation ve corrupt_class_labels repo ile aynı şekilde
NORMALİZE [0,1] uzayda çalışır. Görüntü augmentasyonu piksel uzayında çalışıp
çıktıyı normalize eder.
"""

import random
from enum import Enum

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

# Repo'nun letterbox dolgu rengi: int(0.3 * 255).
BACKGROUND_VALUE = int(0.3 * 255)

# `BboxParams(min_visibility=0.1)` karşılığı: augmentasyon sonrası kadrajın
# içinde kalan alanı, orijinal alanının bu oranından az olan objeler atılır.
# Repo bunu albumentations'a bırakıyor; keypoint API'sinde böyle bir filtre
# olmadığı için burada elle uyguluyoruz.
MIN_VISIBILITY = 0.1


# ---------------------------------------------------------------------------
# Letterbox geometrisi
# ---------------------------------------------------------------------------
def letterbox_params(width, height, image_size):
    """`LongestMaxSize` + ortalanmış `PadIfNeeded` ikilisinin ürettiği dönüşüm.

    Returns (scale, pad_left, pad_top, new_w, new_h) — bir orijinal piksel
    (x, y), letterbox'lanmış tuvalde (x * scale + pad_left, y * scale + pad_top)
    konumuna düşer.

    Bu fonksiyonun varlık sebebi inference: eğitimde koordinatlar zaten
    albumentations tarafından taşınıyor, ama çıkarımda modelin ürettiği tuval
    koordinatlarını orijinal görüntüye GERİ çevirmek gerekiyor ve bu ancak
    dolgunun nereye konduğu bilinirse yapılabilir. Eski `pad_to_square` dolguyu
    sağ-alta koyuyordu (offset hep 0), `PadIfNeeded` ise ORTALIYOR — yani bu iki
    şema birbirinin yerine kullanılamaz.
    """
    scale = image_size / max(width, height)
    # albumentations LongestMaxSize round() kullanır (floor değil).
    new_w = int(round(width * scale))
    new_h = int(round(height * scale))
    pad_left = (image_size - new_w) // 2
    pad_top = (image_size - new_h) // 2
    return scale, pad_left, pad_top, new_w, new_h


def _quad_visibility(quad, width, height):
    """Quad'ın eksen-hizalı kutusunun kadraj içinde kalan oranı ([0,1])."""
    x1, y1 = float(quad[:, 0].min()), float(quad[:, 1].min())
    x2, y2 = float(quad[:, 0].max()), float(quad[:, 1].max())
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if area <= 0.0:
        return 0.0
    ix = max(min(x2, width) - max(x1, 0.0), 0.0)
    iy = max(min(y2, height) - max(y1, 0.0), 0.0)
    return (ix * iy) / area


# ---------------------------------------------------------------------------
# 1. Görüntü augmentasyonu
# ---------------------------------------------------------------------------
class ImageAugmentor:
    """Repo'daki `ImageAugmentor`ın quad (keypoint) uyarlaması.

    Eğitim pipeline'ı (repo sırası):
        OneOf([ColorJitter(p=0.8), ToGray(p=0.2)], p=0.8)
        LongestMaxSize(image_size)
        PadIfNeeded(image_size, image_size)     # ortalanmış, dolgu = 76
        RandomScale(scale_limit=jitter_scale-1) # 0.3x .. 2.0x  (large scale jitter)
        RandomSizedCrop(min_max_height=(0.8*S, S), size=(S, S))
        HorizontalFlip(p=0.5)

    İKİ BİLİNÇLİ SAPMA (ikisi de bayrakla geri alınabilir):

    a) `color_first=False` (varsayılan) ile ColorJitter, LongestMaxSize'ın
       ARDINA alınır. Repo COCO'nun ~640px görüntüleriyle çalışıyor; buradaki
       veri telefon kamerası çözünürlüğünde (12MP+) olduğu için ColorJitter'ı
       ham görüntüde çalıştırmak DataLoader'ı CPU'ya bağlar. ColorJitter
       piksel-bazlı ve geometriden bağımsız olduğu için küçültme sonrası
       uygulamak sonucu pratikte değiştirmez (yalnızca uint8 kırpma
       sınırlarında ihmal edilebilir fark oluşur). `color_first=True` ile
       repo sırası birebir geri gelir.

    b) RandomScale'den SONRA ikinci bir PadIfNeeded var. RandomScale görüntüyü
       0.3x'e kadar küçültebiliyor (512 -> 153 px), RandomSizedCrop ise en az
       0.8*512 = 409 px yükseklikte bir kırpma istiyor — yani kırpma penceresi
       görüntüden büyük kalıyor ve albumentations hata veriyor. Repo'da bu
       koruma yok (COCO'da da aynı sorun oluşur, muhtemelen eski bir
       albumentations sürümünde sessizce kırpılıyordu). Dolgu, kırpmanın her
       zaman geçerli olmasını garanti eder ve 0.3x ölçeğin amacı olan
       "obje küçülsün" etkisini bozmaz.
    """

    def __init__(
        self,
        image_size=512,
        jitter_scale=(0.3, 2.0),
        color_jitter_strength=0.4,
        hflip_prob=0.5,
        training=True,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        min_visibility=MIN_VISIBILITY,
        color_first=False,
    ):
        self.image_size = int(image_size)
        self.training = training
        self.min_visibility = min_visibility

        # Repo COCO'da ViT'i sıfırdan eğittiği için sadece /255 yapıyor. Burada
        # backbone ImageNet-pretrained bir ResNet olduğundan mean/std normalize
        # ŞART — kaldırılırsa pretrained ağırlıklar yanlış girdi dağılımı görür.
        tail = [A.Normalize(mean=normalize_mean, std=normalize_std), ToTensorV2()]

        color = A.OneOf(
            [
                A.ColorJitter(
                    brightness=color_jitter_strength,
                    contrast=color_jitter_strength,
                    saturation=color_jitter_strength,
                    hue=0.2 * color_jitter_strength,
                    p=0.8,
                ),
                A.ToGray(p=0.2),
            ],
            p=0.8,
        )
        letterbox = [
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=BACKGROUND_VALUE,
            ),
        ]

        if training:
            scale_limit = (jitter_scale[0] - 1.0, jitter_scale[1] - 1.0)
            geometric = [
                A.RandomScale(scale_limit=scale_limit, p=1.0),
                # (b) — RandomSizedCrop penceresinin görüntüden büyük kalmasını önler.
                A.PadIfNeeded(
                    min_height=self.image_size,
                    min_width=self.image_size,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=BACKGROUND_VALUE,
                ),
                A.RandomSizedCrop(
                    min_max_height=(int(self.image_size * 0.8), self.image_size),
                    size=(self.image_size, self.image_size),
                    w2h_ratio=1.0,
                    p=1.0,
                ),
                A.HorizontalFlip(p=hflip_prob),
            ]
            steps = ([color] + letterbox if color_first else letterbox + [color]) + geometric
        else:
            steps = letterbox

        self.transform = A.Compose(
            steps + tail,
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )

    def __call__(self, image, objects):
        """image: HxWx3 uint8. objects: [(class_id, [(x, y) x4]), ...] piksel uzayında.

        Returns (image_tensor, objects_norm) — objects_norm, [0,1]'e normalize
        edilmiş (4,2) float32 quad'lar. Kadrajın büyük kısmı dışında kalan
        objeler (bkz. MIN_VISIBILITY) atılır.
        """
        keypoints = np.array(
            [pt for _, quad in objects for pt in quad], dtype=np.float32
        ).reshape(-1, 2)

        out = self.transform(image=image, keypoints=keypoints)
        image_tensor = out["image"]
        kps = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 4, 2)

        s = float(self.image_size)
        kept = []
        for (class_id, _), quad in zip(objects, kps):
            if _quad_visibility(quad, s, s) < self.min_visibility:
                continue
            quad = np.clip(quad, 0.0, s) / s
            kept.append((class_id, quad.astype(np.float32)))
        return image_tensor, kept


# ---------------------------------------------------------------------------
# 2. Kutu augmentasyonu
# ---------------------------------------------------------------------------
class QuadAugmentation:
    """Repo'daki `BBoxAugmentation`ın quad uyarlaması.

    Üç tip obje üretir:
      1. Pozitif  — gerçek objeye küçük jitter, GERÇEK etiketi korunur.
                    (Sende bu yoktu: pix2seq'in "kutu tam oturmasa da doğru
                    sınıfı bul" sinyali buradan geliyor.)
      2. Hard negatif (`shift`) — objenin BOYUTU korunur, merkezi rastgele bir
                    yere taşınır. Makul görünen ama yanlış konumdaki kutu.
      3. Rastgele negatif (`random`) — tamamen uydurma kutu.

    2 ve 3 `noise_class_id` etiketini alır. `mix_rate` olasılıkla noise objeler
    gerçeklerin ARASINA karıştırılır (repo/TF davranışı); aksi halde sona eklenir.
    """

    def __init__(self, noise_class_id, rng=None):
        self.noise_class_id = noise_class_id
        self.rng = rng or random

    # -- üreticiler ---------------------------------------------------------
    def jitter_quad(self, quad, max_range=0.05, truncation=True):
        """Repo `jitter_bbox`: gürültü, kutunun KENDİ boyutuyla ölçeklenir.

        Bu ölçekleme kritik: gürültüyü görüntü boyutuyla ölçeklemek (eski
        NOISE_JITTER_STD davranışı) 110x22'lik bir metin alanını 25 px oynatır,
        yani objeyi tamamen ıskalar; kutu boyutuyla ölçeklemek ise ~1-5 px
        oynatır ve gerçekten "az kaçırılmış" bir örnek üretir.
        """
        w = float(quad[:, 0].max() - quad[:, 0].min())
        h = float(quad[:, 1].max() - quad[:, 1].min())
        noise = np.array(
            [[self.rng.gauss(0.0, max_range / 2.0) for _ in range(2)] for _ in range(4)],
            dtype=np.float32,
        )
        noise = np.clip(noise, -max_range, max_range)
        out = quad + noise * np.array([w, h], dtype=np.float32)
        return np.clip(out, 0.0, 1.0) if truncation else out

    def shift_quad(self, quad, truncation=True):
        """Repo `shift_bbox`: şekil/boyut aynı, merkez rastgele."""
        cx = 0.5 * float(quad[:, 0].min() + quad[:, 0].max())
        cy = 0.5 * float(quad[:, 1].min() + quad[:, 1].max())
        out = quad + np.array(
            [self.rng.random() - cx, self.rng.random() - cy], dtype=np.float32
        )
        return np.clip(out, 0.0, 1.0) if truncation else out

    def random_quad(self, max_size=1.0, truncation=True):
        """Repo `random_bbox`: merkez U(0,1), kenarlar |N(0, max_size/2)|.

        Normal dağılım (uniform değil) bilinçli: küçük kutular baskın olur,
        gerçek obje boyut dağılımına daha yakın negatifler üretir.
        """
        cx, cy = self.rng.random(), self.rng.random()
        w = abs(self.rng.gauss(0.0, max_size / 2.0))
        h = abs(self.rng.gauss(0.0, max_size / 2.0))
        x1, x2 = cx - w / 2.0, cx + w / 2.0
        y1, y2 = cy - h / 2.0, cy + h / 2.0
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        # Eksen-hizalı dikdörtgenin bu sırası zaten kanonik (ağırlık merkezi
        # etrafında açı sırasında ve sol-üstten başlıyor) — bkz.
        # training.canonicalize_corners.
        return np.clip(quad, 0.0, 1.0) if truncation else quad

    # -- ana giriş ----------------------------------------------------------
    def augment(self, objects, n_noise, max_jitter=0.05, mix_rate=0.5,
                keep_real_order=False):
        """objects: [(class_id, quad(4,2) normalize)]. Returns aynı formatta liste.

        Repo `augment_bbox` ile birebir akış: önce gerçekler jitter'lanır, sonra
        n_noise adet sahte obje `dup` (shift) ve `bad` (shift/random karışımı)
        olarak bölüştürülür.

        `keep_real_order=True` repo'dan tek sapma: mix_rate tetiklendiğinde repo
        TÜM listeyi karıştırıyor, bu da gerçek objelerin deterministik sırasını
        yok ediyor. OBJECT_ORDER="class" ile çalışan bir veri setinde bu sıra
        bilinçli bir tercih (model her slotta "sıradaki alan hangisi?" tahmini
        yapmak zorunda kalmasın diye), o yüzden bu modda gerçeklerin göreli
        sırası korunur ve noise objeler yalnızca ARALARINA serpiştirilir.
        mix_rate'in asıl amacı — modelin "sondakiler noise" kısayolunu
        öğrenememesi — bu şekilde de sağlanır.
        """
        n = len(objects)
        n_noise = max(0, n_noise)

        if n > 0:
            objects = [
                (cid, self.jitter_quad(quad, max_range=max_jitter))
                for cid, quad in objects
            ]
            dup_size = self.rng.randint(0, n_noise) if n_noise > 0 else 0
        else:
            dup_size = n_noise
        bad_size = n_noise - dup_size

        noise = []
        for _ in range(bad_size):
            if n > 0 and self.rng.random() < 0.5:
                quad = self.shift_quad(objects[self.rng.randrange(n)][1])
            else:
                quad = self.random_quad()
            noise.append((self.noise_class_id, quad))

        if dup_size > 0 and n > 0:
            # Repo `torch.randperm(n)[:dup_bbox_size]`: dup_size > n ise
            # tekrarsız olduğu için n taneyle sınırlı kalır.
            idxs = list(range(n))
            self.rng.shuffle(idxs)
            for i in idxs[:dup_size]:
                noise.append((self.noise_class_id, self.shift_quad(objects[i][1])))

        if n > 0 and noise and self.rng.random() < mix_rate:
            if keep_real_order:
                out = list(objects)
                for item in noise:
                    out.insert(self.rng.randint(0, len(out)), item)
            else:
                out = list(objects) + noise
                self.rng.shuffle(out)
        else:
            out = list(objects) + noise
        return out


# ---------------------------------------------------------------------------
# 3. Sequence augmentasyonu (class label corruption)
# ---------------------------------------------------------------------------
class LabelCorruptionStrategy(str, Enum):
    NONE = "none"                    # etiketler bozulmaz
    RANDOM = "rand_cls"              # %50 orijinal, %50 rastgele geçerli sınıf
    RANDOM_AND_FAKE = "rand_n_fake_cls"  # %50 orijinal, %25 rastgele, %25 fake


def corrupt_class_labels(class_ids, num_classes, noise_class_id, strategy, rng=None):
    """Repo `TokenProcessor.corrupt_class_labels`.

    YALNIZCA decoder'ın GİRDİ dizisine uygulanır; hedef dizi temiz kalır. Amacı
    exposure bias: teacher forcing'de model her adımda kusursuz bir geçmiş
    görüyor, çıkarımda ise kendi (bazen hatalı) token'larını görüyor. Girdiyi
    bozup hedefi doğru bırakmak, modele "geçmişte yanlış sınıf yazılmış olsa
    bile doğrusunu üret" demeyi öğretir.

    Not: bozulan etiketler sınıf loss'unda tam ağırlıkla kalır — model bozuk
    objeyi görmezden gelmeyi değil, AKTİF olarak reddetmeyi öğrenir.
    """
    rng = rng or random
    if strategy == LabelCorruptionStrategy.NONE:
        return list(class_ids)

    out = []
    for cid in class_ids:
        if rng.random() < 0.5:            # %50: dokunma
            out.append(cid)
            continue
        if strategy == LabelCorruptionStrategy.RANDOM:
            out.append(rng.randrange(num_classes))
        else:                              # RANDOM_AND_FAKE
            if rng.random() < 0.5:
                out.append(rng.randrange(num_classes))
            else:
                out.append(noise_class_id)
    return out
