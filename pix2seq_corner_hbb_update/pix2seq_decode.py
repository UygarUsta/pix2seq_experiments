"""
Paylaşılan inference yardımcıları: encode-once + KV-cache'li autoregressive
decode, yapısal kısıtlı (constrained) token üretimi, confidence score'lu
detokenization ve tahmin çizimi.

inference.py, api.py ve eval_iou.py bu modülü kullanır — böylece üç yerde ayrı
ayrı (ve tutarsız) decode/format mantığı tekrarlanmaz.

Üç katmanlı hızlandırma/doğruluk şeması:

  1. Encode-once — encoder tek görüntü için ~131 kez değil, BİR kez çalışır.
  2. KV cache    — decoder'ın self-attention K/V'si ve cross-attention'ın memory
                   K/V'si adımlar arasında saklanır. Naif döngü her adımda tüm
                   diziyi baştan işlediği için O(S²), cache ile O(S).
  3. Constrained decoding — her pozisyonda YAPISAL olarak geçerli olmayan
                   token'lar softmax'tan önce -inf'e çekilir (bkz.
                   `allowed_token_mask`). Repo'nun (Chris-hughes10/pix2seq)
                   `inference.py`sindeki şemanın karşılığı.

OUTPUT_MODE ("corner" | "hbb" | "both"), training.py'de tanımlı tek kaynaktır;
bu modül sequence'i nasıl gruplayıp parse edeceğini oradan okur. Hangi mod
seçilirse seçilsin, her tahmin sözlüğünde hem "points" (4 köşe) hem "bbox"
(eksen-hizalı) alanı bulunur — model sadece birini ürettiyse diğeri ondan
türetilir, böylece eval_iou.py / çizim kodu mod farkı gözetmeden çalışabilir.

ÖN-İŞLEME UYARISI: koordinatlar artık LongestMaxSize + ORTALANMIŞ pad
("letterbox") uzayında öğreniliyor (bkz. augmentations.ImageAugmentor). Eski
şema görüntüyü sol-üste yapıştırıp sağ-alta gri dolgu koyuyordu. Bu iki şema
birbirinin yerine kullanılamaz — eski bir checkpoint yeni kodla çalıştırılırsa
kutular kayar. load_model bunu .meta.json üzerinden tespit edip uyarır.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from augmentations import letterbox_params
from training import (
    BACKBONE_NAME,
    BOS_TOKEN,
    EOS_TOKEN,
    EOS_TOKEN_WEIGHT,
    FEATURE_LEVELS,
    ID_TO_LABEL,
    IMG_SIZE,
    LABEL_TO_ID,
    MAX_OBJECTS,
    NOISE_CLASS_ID,
    NUM_BINS,
    NUM_CLASSES,
    OUTPUT_MODE,
    TIE_WORD_EMBEDDINGS,
    TOKENS_PER_OBJECT,
    VOCAB_SIZE,
    Pix2SeqModel,
    val_augmentor,
)

MAX_SEQ_LEN = 1 + (TOKENS_PER_OBJECT * MAX_OBJECTS) + 1

# --- Decode ayarları ---
# Yapısal kısıt maskesi. Kapatmak sadece hata ayıklama için anlamlı: kapalıyken
# model bir koordinat pozisyonunda sınıf token'ı üretebilir, bu da tüm dizinin
# gruplamasını kaydırır ve o görüntüdeki KALAN objelerin hepsi çöpe gider.
CONSTRAINED_DECODING = True

# KV cache. Doğruluğu değiştirmez (bkz. `verify_kv_cache`), yalnızca hızlandırır.
USE_KV_CACHE = True

# Nucleus (top-p) örnekleme. Repo 0.4 kullanıyor çünkü COCO'da aynı sahnenin
# birden çok geçerli obje sıralaması var ve örnekleme çeşitliliği recall'u
# artırıyor. Burada her görüntüde her sınıftan tam olarak bir obje var, yani
# doğru cevap tek — örnekleme sadece gürültü ekler. 0 => greedy (argmax).
TOP_P = 0.0

# --- Soft-argmax koordinat çözümü ---
# Model her koordinat için NUM_BINS üzerinde bir dağılım üretir. argmax, bu
# dağılımı tek bir bin'e yuvarlar ve komşu bin'lerdeki bilgiyi çöpe atar:
# dağılım 500'de 0.11, 501'de 0.10, 502'de 0.09 ise gerçek beklenen değer
# ~500.9'dur ama argmax 500 der. Dağılım tepe noktası etrafında asimetrik
# olduğunda (kutu kenarları için tipik) bu sistematik bir kayma üretir.
#
# Burada argmax'ın komşuluğunda olasılık-ağırlıklı ortalama alıyoruz.
# Sequence'e beslenen token DEĞİŞMİYOR (autoregressive tutarlılık bozulmasın
# diye hâlâ argmax) — sadece piksele çevrilen değer rafine ediliyor. Yeniden
# eğitim gerektirmez, mevcut checkpoint'lerde çalışır.
#
# PENCERE GENİŞLİĞİ KRİTİK. Sabit ±3 bin, dağılımın dar olduğunu varsayıyordu;
# ölçülen σ ≈ 12 bin çıkınca bu pencere olasılık kütlesinin yalnızca %19'unu
# kapsıyor, yani soft-argmax neredeyse hiç iş yapmıyordu. Sabit bir sayı yerine
# pencereyi kütleye göre uyarlıyoruz: argmax'tan başlayıp SOFT_ARGMAX_MASS
# oranında olasılık toplanana kadar simetrik genişliyoruz. Böylece model emin
# olduğunda pencere daralıyor, belirsiz olduğunda genişliyor.
SOFT_ARGMAX_MASS = 0.8      # 0 => sabit SOFT_ARGMAX_WINDOW kullan
SOFT_ARGMAX_WINDOW = 16     # SOFT_ARGMAX_MASS = 0 iken kullanılan sabit yarıçap
SOFT_ARGMAX_MAX_WINDOW = 64  # uyarlamalı pencere için üst sınır (ikinci bir tepeyi
                             # içeri almasın diye)

# Sınıf token'ı olarak kabul edilen aralık: gerçek sınıflar + noise sınıfı.
# Noise token'ı BİLEREK serbest bırakılıyor — model "burada obje yok" diyebilmeli.
# detokenize bunu zaten eliyor (bkz. aşağıda).
CLASS_TOKEN_LO = NUM_BINS
CLASS_TOKEN_HI = NUM_BINS + NUM_CLASSES + 1  # NOISE_CLASS_ID dahil

# hbb dörtlüsünün obje içindeki başlangıç indeksi (xmin, ymin, xmax, ymax).
# "corner" modunda hbb yok, sıralama kısıtı da yok.
_HBB_OFFSET = {"both": 8, "hbb": 0, "corner": None}[OUTPUT_MODE]


def load_model(model_path, device, max_seq_len=MAX_SEQ_LEN):
    # Checkpoint'in .meta.json'ı hangi backbone/OUTPUT_MODE ile eğitildiğini
    # kaydeder (bkz. training.save_checkpoint). backbone_name'i buradan okuyup
    # modeli onunla kuruyoruz ki training.py'deki BACKBONE_NAME o an başka bir
    # şeye ayarlanmış olsa bile (farklı backbone'lar test edilirken) her
    # checkpoint kendi doğru mimarisiyle yüklenebilsin.
    meta_path = os.path.splitext(model_path)[0] + ".meta.json"
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    # Weight tying SESSİZ bir tuzak: tying açıkken state_dict'te hem
    # `embedding.weight` hem `fc_out.weight` aynı tensöre işaret eder, dolayısıyla
    # tying OLMADAN eğitilmiş bir checkpoint hatasız yüklenir ama iki farklı
    # matristen hangisi sonra işlendiyse o ikisinin YERİNE geçer — model
    # bozulur, hiçbir uyarı çıkmaz. O yüzden burada açıkça reddediyoruz.
    ckpt_tied = bool(meta.get("tie_word_embeddings", False)) if meta else False
    if TIE_WORD_EMBEDDINGS and not ckpt_tied:
        raise RuntimeError(
            f"{model_path} weight tying OLMADAN eğitilmiş (meta: tie_word_embeddings="
            f"{ckpt_tied}), ancak training.py'de TIE_WORD_EMBEDDINGS=True. Bu checkpoint "
            f"hatasız yüklenir ama embedding/çıkış matrisi birbirinin üzerine yazılacağı "
            f"için model bozulur. Modeli güncel training.py ile yeniden eğitin ya da "
            f"karşılaştırma yapıyorsanız TIE_WORD_EMBEDDINGS=False yapın."
        )

    backbone_name = meta.get("backbone_name", BACKBONE_NAME) if meta else BACKBONE_NAME
    feature_levels = tuple(meta.get("feature_levels", FEATURE_LEVELS)) if meta else FEATURE_LEVELS
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len,
                         backbone_name=backbone_name,
                         feature_levels=feature_levels).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        # Pre-LN geçişiyle birlikte img_encoder/decoder'a birer son LayerNorm
        # eklendi (training.Pix2SeqModel). Bu tarihten ÖNCE eğitilmiş
        # checkpoint'lerde bu ağırlıklar yok; state_dict key'leri eşleşse bile
        # post-LN ağırlıklarını pre-LN mimarisine yüklemek anlamsız sonuç üretir.
        # Sessizce yanlış çalışmak yerine ne yapılması gerektiğini söylüyoruz.
        raise RuntimeError(
            f"{model_path} bu mimariye yüklenemedi. Muhtemelen post-LN döneminde "
            f"(img_encoder.norm / decoder.norm eklenmeden önce) eğitilmiş eski bir "
            f"checkpoint. Modeli güncel training.py ile yeniden eğitin.\n\nOrijinal hata: {e}"
        ) from e
    model.eval()

    # Checkpoint hangi OUTPUT_MODE ile eğitilmiş — training.py'nin şu anki
    # OUTPUT_MODE'u ile uyuşmuyorsa sequence'i yanlış grupluyor olabiliriz.
    # Sessizce yanlış (ama çökmeyen) sonuç üretmek yerine burada uyarıyoruz.
    if meta is not None:
        if meta.get("output_mode") != OUTPUT_MODE:
            print(
                f"UYARI: {model_path} '{meta.get('output_mode')}' modu ile eğitilmiş, "
                f"ancak training.py şu an OUTPUT_MODE='{OUTPUT_MODE}' olarak ayarlı. "
                f"Tahminler yanlış gruplanıp anlamsız çıkabilir — training.py'deki "
                f"OUTPUT_MODE'u '{meta.get('output_mode')}' yapın ya da modeli yeniden eğitin."
            )
        if meta.get("preprocess") != "letterbox_centered":
            print(
                f"UYARI: {model_path} eski ön-işleme şemasıyla (kareye tamamlama SAĞ-ALTA "
                f"dolgu) eğitilmiş; bu kod ise ortalanmış letterbox kullanıyor. Kutular "
                f"sistematik olarak kayacaktır — modeli güncel training.py ile yeniden eğitin."
            )
    else:
        print(f"NOT: {meta_path} bulunamadı — checkpoint'in OUTPUT_MODE/backbone/ön-işleme uyumu doğrulanamadı.")

    return model


def dequantize(token, num_bins):
    """Bin indeksini [0,1] normalize koordinata çevirir (training.quantize'ın tersi)."""
    return token / (num_bins - 1)


@torch.no_grad()
def encode_image(model, image: Image.Image, device):
    """Letterbox'lar, normalize eder ve encoder'ı BİR KEZ çalıştırır.

    Ön-işleme için doğrudan training.val_augmentor kullanılıyor — eğitimdeki
    değerlendirme yolunun ta kendisi. Ayrı bir torchvision pipeline'ı yazmak
    (eski hali) iki tarafın sessizce ayrışmasına açık kapı bırakıyordu.

    Returns (memory, geom); geom = (scale, pad_left, pad_top), tahmin edilen
    koordinatları orijinal görüntü uzayına geri çevirmek için gerekir.
    """
    image_np = np.array(image.convert("RGB"))
    img_tensor, _ = val_augmentor(image_np, [])
    memory = model.encode(img_tensor.unsqueeze(0).to(device))
    w, h = image.size
    scale, pad_left, pad_top, _, _ = letterbox_params(w, h, IMG_SIZE[0])
    return memory, (scale, pad_left, pad_top)


# ---------------------------------------------------------------------------
# Yapısal kısıt (constrained decoding)
# ---------------------------------------------------------------------------
def allowed_token_mask(step, generated, device):
    """step. adımda üretilecek token için izinli-token boolean maskesi.

    `generated`, BOS hariç şimdiye kadar üretilmiş token listesi (len == step).

    Kurallar:
      - obje başı  : koordinat VEYA EOS (dizi burada bitebilir)
      - obje sonu  : yalnızca sınıf token'ı (noise sınıfı dahil)
      - arası      : yalnızca koordinat
      - hbb'de xmax > xmin ve ymax > ymin zorunlu
      - MAX_OBJECTS dolduysa yalnızca EOS
    """
    mask = torch.zeros(VOCAB_SIZE, dtype=torch.bool, device=device)
    pos = step % TOKENS_PER_OBJECT
    n_done = step // TOKENS_PER_OBJECT

    if pos == 0 and n_done >= MAX_OBJECTS:
        mask[EOS_TOKEN] = True
        return mask

    if pos == TOKENS_PER_OBJECT - 1:
        mask[CLASS_TOKEN_LO:CLASS_TOKEN_HI] = True
        return mask

    lo = 0
    if _HBB_OFFSET is not None:
        obj_start = n_done * TOKENS_PER_OBJECT
        if pos == _HBB_OFFSET + 2:      # xmax > xmin
            lo = generated[obj_start + _HBB_OFFSET] + 1
        elif pos == _HBB_OFFSET + 3:    # ymax > ymin
            lo = generated[obj_start + _HBB_OFFSET + 1] + 1
        # min değeri son bin ise "kesinlikle büyük" imkânsız — eşitliğe izin ver
        # ki maske tamamen boş kalmasın (boş maske softmax'ı NaN yapar).
        lo = min(lo, NUM_BINS - 1)

    mask[lo:NUM_BINS] = True
    if pos == 0:
        mask[EOS_TOKEN] = True
    return mask


def _sample(logits, top_p):
    """top_p > 0 ise nucleus örnekleme, değilse argmax."""
    if top_p <= 0.0:
        return int(torch.argmax(logits).item())
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    keep = cum - sorted_probs < top_p       # ilk token daima kalır
    sorted_probs = sorted_probs * keep
    sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
    pick = int(torch.multinomial(sorted_probs, 1).item())
    return int(sorted_idx[pick].item())


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------
def _split_heads(x, num_heads):
    b, l, e = x.shape
    return x.view(b, l, num_heads, e // num_heads).transpose(1, 2)   # [B,H,L,D]


def _merge_heads(x):
    b, h, l, d = x.shape
    return x.transpose(1, 2).reshape(b, l, h * d)


def _in_proj(mha, x, part):
    """nn.MultiheadAttention'ın birleşik in_proj'undan q/k/v dilimini hesaplar.

    torch, q/k/v aynı embed_dim'e sahipse üç projeksiyonu tek bir [3E, E]
    matriste tutar (`in_proj_weight`); alt modüllere erişim olmadığı için
    dilimleri elle çıkarıyoruz.
    """
    e = mha.embed_dim
    sl = {"q": slice(0, e), "k": slice(e, 2 * e), "v": slice(2 * e, 3 * e)}[part]
    b = mha.in_proj_bias[sl] if mha.in_proj_bias is not None else None
    return F.linear(x, mha.in_proj_weight[sl], b)


class IncrementalDecoder:
    """model.decoder'ı adım adım, K/V cache'leyerek çalıştırır.

    `model.decode(memory, seq)` ile SAYISAL OLARAK AYNI logit'i üretir (bkz.
    `verify_kv_cache`), tek fark her adımda O(1) yeni token işlemesi. Cache'in
    doğruluğu norm_first=True (pre-LN) katman yapısına dayanır:
        x = x + sa(norm1(x));  x = x + mha(norm2(x), memory);  x = x + ff(norm3(x))
    """

    def __init__(self, model, memory):
        assert not model.training, "IncrementalDecoder yalnızca eval modunda doğrudur (dropout/drop_path)"
        self.model = model
        self.layers = list(model.decoder.layers)
        self.final_norm = model.decoder.norm
        self.nheads = self.layers[0].self_attn.num_heads

        # Cross-attention K/V: memory sabit, bir kez hesaplanır. Asıl kazanç
        # burada — memory 1024 token (bkz. FEATURE_LEVELS) ve naif döngü bunu
        # her adımda yeniden projeksiyondan geçiriyordu.
        self.cross_kv = []
        for layer in self.layers:
            mha = layer.multihead_attn
            k = _split_heads(_in_proj(mha, memory, "k"), self.nheads)
            v = _split_heads(_in_proj(mha, memory, "v"), self.nheads)
            self.cross_kv.append((k, v))

        self.self_kv = [None] * len(self.layers)
        self.pos = 0

    @torch.no_grad()
    def step(self, token_id):
        """Tek token besler, bir sonraki token için logit'leri döndürür [V]."""
        m = self.model
        device = m.embedding.weight.device
        x = m.embedding(torch.tensor([[token_id]], device=device))          # [1,1,E]
        x = x + m.seq_pos_encoding.pe[:, self.pos:self.pos + 1, :]

        for i, layer in enumerate(self.layers):
            # --- masked self-attention (cache'li) ---
            h = layer.norm1(x)
            q = _split_heads(_in_proj(layer.self_attn, h, "q"), self.nheads)
            k = _split_heads(_in_proj(layer.self_attn, h, "k"), self.nheads)
            v = _split_heads(_in_proj(layer.self_attn, h, "v"), self.nheads)
            if self.self_kv[i] is not None:
                pk, pv = self.self_kv[i]
                k = torch.cat([pk, k], dim=2)
                v = torch.cat([pv, v], dim=2)
            self.self_kv[i] = (k, v)
            # Causal maske gerekmez: cache yalnızca geçmişi + bu adımı içeriyor.
            a = F.scaled_dot_product_attention(q, k, v)
            x = x + layer.self_attn.out_proj(_merge_heads(a))

            # --- cross-attention (memory K/V cache'li) ---
            h = layer.norm2(x)
            q = _split_heads(_in_proj(layer.multihead_attn, h, "q"), self.nheads)
            ck, cv = self.cross_kv[i]
            a = F.scaled_dot_product_attention(q, ck, cv)
            x = x + layer.multihead_attn.out_proj(_merge_heads(a))

            # --- feed-forward ---
            h = layer.norm3(x)
            x = x + layer.linear2(layer.activation(layer.linear1(h)))

        self.pos += 1
        out = self.final_norm(x) if self.final_norm is not None else x
        return m.fc_out(out)[0, -1, :]


@torch.no_grad()
def verify_kv_cache(model, memory, seq, atol=1e-3):
    """Cache'li ve naif decode yollarının aynı logit'i ürettiğini doğrular.

    Cache mantığı katman iç yapısına (pre-LN sırası, in_proj düzeni) bağlı
    olduğu için PyTorch sürüm değişikliklerinde sessizce bozulabilir; bu yüzden
    ucuz ve çalıştırılabilir bir kontrol bırakıyoruz.
    """
    naive = model.decode(memory, torch.tensor([seq], device=memory.device))[0]
    dec = IncrementalDecoder(model, memory)
    cached = torch.stack([dec.step(t) for t in seq])
    err = float((naive - cached).abs().max())
    return err <= atol, err


# ---------------------------------------------------------------------------
# Decode döngüsü
# ---------------------------------------------------------------------------
def _soft_coordinate(probs, token, bin_idx):
    """Koordinat token'ı için argmax yerine yerel beklenen değer (kesirli bin).

    Pencere SOFT_ARGMAX_MASS'e göre uyarlanır: argmax'tan simetrik genişleyip
    hedeflenen olasılık kütlesi toplanınca durur (ikili arama). Koordinat
    olmayan token'lar ve kapalı ayar için token'ın kendisi döner."""
    if token >= NUM_BINS:
        return float(token)

    coord_probs = probs[:NUM_BINS]

    if SOFT_ARGMAX_MASS > 0:
        # cumsum üzerinden yarıçap araması: mass(r) = c[hi] - c[lo-1]
        c = torch.cumsum(coord_probs, dim=0)
        total_mass = float(c[-1].item())
        target = SOFT_ARGMAX_MASS * total_mass

        def mass(r):
            lo, hi = max(0, token - r), min(NUM_BINS - 1, token + r)
            return float(c[hi].item()) - (float(c[lo - 1].item()) if lo > 0 else 0.0)

        lo_r, hi_r = 0, SOFT_ARGMAX_MAX_WINDOW
        if mass(hi_r) < target:
            radius = hi_r
        else:
            while lo_r < hi_r:                 # en küçük yeterli yarıçap
                mid = (lo_r + hi_r) // 2
                if mass(mid) >= target:
                    hi_r = mid
                else:
                    lo_r = mid + 1
            radius = lo_r
    else:
        radius = SOFT_ARGMAX_WINDOW

    if radius <= 0:
        return float(token)

    lo = max(0, token - radius)
    hi = min(NUM_BINS, token + radius + 1)
    w = coord_probs[lo:hi]
    denom = float(w.sum().item())
    if denom <= 0:
        return float(token)
    return float((w * bin_idx[lo:hi]).sum().item() / denom)


@torch.no_grad()
def autoregressive_decode(model, memory, device, max_seq_len=MAX_SEQ_LEN):
    """Kısıtlı + KV-cache'li decode. `memory` sabittir, yeniden hesaplanmaz.

    Returns (sequence, token_probs, soft_bins):
      - sequence   : üretilen token'lar (BOS dahil)
      - token_probs: token_probs[i], sequence[i+1] için modelin olasılığı
      - soft_bins  : soft_bins[i], sequence[i+1] koordinat token'ıysa argmax
                     komşuluğunun olasılık-ağırlıklı ortalaması (kesirli bin),
                     değilse token'ın kendisi. bkz. SOFT_ARGMAX_WINDOW.

    NOT: olasılıklar KISITLANMIŞ dağılımdan alınır. Bu bilinçli — yapısal olarak
    imkânsız token'lara ayrılmış olasılık kütlesi confidence'ı sistematik olarak
    düşürüyordu; maskeledikten sonra score gerçekten "modelin bu obje için
    güveni" anlamına geliyor.
    """
    sequence = [BOS_TOKEN]
    generated = []          # BOS hariç
    token_probs = []
    soft_bins = []

    bin_idx = torch.arange(NUM_BINS, device=device, dtype=torch.float32)
    dec = IncrementalDecoder(model, memory) if USE_KV_CACHE else None

    for step in range(max_seq_len):
        if dec is not None:
            next_token_logits = dec.step(sequence[-1])
        else:
            tgt_seq = torch.tensor([sequence], dtype=torch.long, device=device)
            next_token_logits = model.decode(memory, tgt_seq)[0, -1, :]

        if CONSTRAINED_DECODING:
            mask = allowed_token_mask(step, generated, device)
            next_token_logits = next_token_logits.masked_fill(~mask, float("-inf"))

        probs = F.softmax(next_token_logits.float(), dim=-1)
        next_token = _sample(next_token_logits.float(), TOP_P)

        soft = _soft_coordinate(probs, next_token, bin_idx)

        sequence.append(next_token)
        generated.append(next_token)
        token_probs.append(float(probs[next_token].item()))
        soft_bins.append(soft)

        if next_token == EOS_TOKEN:
            break

    return sequence, token_probs, soft_bins


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _points_from_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def detokenize(sequence, token_probs, geom, soft_bins=None):
    """Üretilen tokenları obje bazında gruplar (grup boyutu OUTPUT_MODE'a göre
    9/5/13). Her tahmin şunları içerir:
      - label
      - points: 4 köşeli quad (oriented polygon) — corner tahmin edilmediyse
                bbox'tan (eksen-hizalı dikdörtgen köşeleri olarak) türetilir
      - bbox:   [xmin, ymin, xmax, ymax] — hbb tahmin edilmediyse points'in
                min/max'ından türetilir
      - score:  [0,1] arası confidence — objenin tüm token'larının
                olasılıklarının geometrik ortalaması (herhangi bir token
                belirsizse — kötü lokalizasyon YA DA kötü sınıflandırma —
                skoru aşağı çeker)

    `geom` = (scale, pad_left, pad_top): bin -> [0,1] -> letterbox tuvali (px)
    -> orijinal görüntü (px) zincirinin son halkası.
    """
    scale, pad_left, pad_top = geom
    canvas = float(IMG_SIZE[0])

    def to_x(bin_value):
        return (dequantize(bin_value, NUM_BINS) * canvas - pad_left) / scale

    def to_y(bin_value):
        return (dequantize(bin_value, NUM_BINS) * canvas - pad_top) / scale

    tokens = sequence[1:]
    probs = token_probs
    # soft_bins verilmezse (eski çağrı biçimi) token'ların kendisi kullanılır —
    # davranış saf argmax'a döner.
    softs = soft_bins if soft_bins is not None else [float(t) for t in tokens]
    if tokens and tokens[-1] == EOS_TOKEN:
        tokens = tokens[:-1]
        probs = probs[:-1]
        softs = softs[:-1]

    predictions = []
    n_complete = len(tokens) - (len(tokens) % TOKENS_PER_OBJECT)

    for i in range(0, n_complete, TOKENS_PER_OBJECT):
        obj_tokens = tokens[i:i + TOKENS_PER_OBJECT]
        obj_probs  = probs[i:i + TOKENS_PER_OBJECT]
        # Geçerlilik kontrolleri tamsayı token üzerinden, koordinata çevirme ise
        # kesirli soft_bin üzerinden yapılır.
        obj_soft   = softs[i:i + TOKENS_PER_OBJECT]

        # Sınıf token'ı, moddan bağımsız olarak her zaman son sıradadır.
        class_tok    = obj_tokens[-1]
        coord_tokens = obj_tokens[:-1]

        # CONSTRAINED_DECODING açıkken bu iki kontrol asla tetiklenmez; kısıt
        # kapatılırsa (hata ayıklama) son savunma hattı olarak duruyorlar.
        if not all(0 <= t < NUM_BINS for t in coord_tokens):
            continue
        # [NUM_BINS, NUM_BINS + len(LABEL_TO_ID)) sadece GERÇEK sınıfları kapsar.
        # training.py'nin NOISE_CLASS_ID'si tam olarak bu aralığın bir üstünde
        # (NUM_BINS + len(LABEL_TO_ID)) oturur, yani model "noise" tahmin ettiğinde
        # buradaki kontrol onu otomatik eler — ekstra bir if gerekmiyor. Bu da
        # noise-object augmentation ile eğitilmiş bir modelin ürettiği kalibre
        # confidence score'un (bkz. `score` altında) neden anlamlı olduğunun
        # sebebidir: düşük ihtimalli/gürültü tahminleri zaten burada düşer.
        if not (NUM_BINS <= class_tok < NUM_BINS + len(LABEL_TO_ID)):
            continue

        idx = 0
        points = None
        bbox = None

        if OUTPUT_MODE in ("corner", "both"):
            points = []
            for j in range(0, 8, 2):
                points.append((to_x(obj_soft[idx + j]), to_y(obj_soft[idx + j + 1])))
            idx += 8

        if OUTPUT_MODE in ("hbb", "both"):
            # Kısıtlı decode zaten xmax > xmin / ymax > ymin garantiliyor, ama
            # soft-argmax kesirli değerleri komşu bin'lere kaydırabildiği için
            # min/max ile normalize ediyoruz (PIL'in draw.rectangle'ı
            # "x1 must be >= x0" diye patlamasın).
            x_a, y_a = to_x(obj_soft[idx]),     to_y(obj_soft[idx + 1])
            x_b, y_b = to_x(obj_soft[idx + 2]), to_y(obj_soft[idx + 3])
            bbox = [min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)]
            idx += 4

        if points is None:
            points = _points_from_bbox(bbox)
        if bbox is None:
            bbox = _bbox_from_points(points)

        class_id = class_tok - NUM_BINS
        label = ID_TO_LABEL.get(class_id, f"Bilinmeyen_{class_id}")

        score = math.exp(sum(math.log(max(p, 1e-8)) for p in obj_probs) / len(obj_probs))

        predictions.append({
            "label": label,
            "points": points,
            "bbox": bbox,
            "score": score,
        })

    return predictions


@torch.no_grad()
def predict(model, image: Image.Image, device, score_threshold=0.0):
    """Uçtan uca: bir kez encode et, decode et, detokenize et, orijinal
    görüntü sınırlarına kırp. Returns (predictions, sequence)."""
    orig_w, orig_h = image.size
    memory, geom = encode_image(model, image, device)
    sequence, token_probs, soft_bins = autoregressive_decode(model, memory, device)
    predictions = detokenize(sequence, token_probs, geom, soft_bins)

    for p in predictions:
        p["points"] = [
            (min(max(x, 0.0), float(orig_w)), min(max(y, 0.0), float(orig_h)))
            for x, y in p["points"]
        ]
        xmin, ymin, xmax, ymax = p["bbox"]
        p["bbox"] = [
            min(max(xmin, 0.0), float(orig_w)), min(max(ymin, 0.0), float(orig_h)),
            min(max(xmax, 0.0), float(orig_w)), min(max(ymax, 0.0), float(orig_h)),
        ]

    if score_threshold > 0.0:
        predictions = [p for p in predictions if p["score"] >= score_threshold]

    return predictions, sequence


def draw_predictions(image: Image.Image, predictions, score_threshold=0.0):
    """OUTPUT_MODE'a göre çizer: corner varsa quad kırmızı, hbb varsa dikdörtgen
    yeşil. "both" dışındaki modlarda points/bbox'tan biri türetilmiş olduğu için
    (aynı dikdörtgeni iki kez çizmemek adına) sadece gerçekten tahmin edilen
    geometri çizilir."""
    draw = ImageDraw.Draw(image)
    for pred in predictions:
        if pred["score"] < score_threshold:
            continue

        pts = pred["points"]

        if OUTPUT_MODE in ("corner", "both"):
            flat_points = [coord for pt in pts for coord in pt]
            draw.polygon(flat_points, outline="red", width=3)

        if OUTPUT_MODE in ("hbb", "both"):
            draw.rectangle(pred["bbox"], outline="lime", width=2)

        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        caption = f"{pred['label']} {pred['score']:.2f}"
        text_w = 7 * len(caption)
        draw.rectangle([min_x, max(0, min_y - 15), min_x + text_w, min_y], fill="red")
        draw.text((min_x + 2, max(0, min_y - 15)), caption, fill="white")

    return image
