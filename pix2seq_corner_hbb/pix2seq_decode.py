"""
Paylaşılan inference yardımcıları: encode-once autoregressive decode,
confidence score'lu detokenization ve tahmin çizimi.

inference.py, api.py ve eval_iou.py bu modülü kullanır — böylece üç yerde ayrı
ayrı (ve tutarsız) decode/format mantığı tekrarlanmaz.

Encode/decode ayrımı: eskiden her decode adımında `model(images, tgt_seq)`
çağrılıyordu, bu da ResNet50 encoder'ı tek bir görüntü için ~max_seq_len kez
(yaklaşık 190 kez) gereksiz yere yeniden çalıştırıyordu. Burada encoder sadece
bir kez çağrılır (`model.encode`), sonrasında decode döngüsü yalnızca
`model.decode(memory, tgt_seq)` çağırır.

OUTPUT_MODE ("corner" | "hbb" | "both"), training.py'de tanımlı tek kaynaktır;
bu modül sequence'i nasıl gruplayıp parse edeceğini oradan okur. Hangi mod
seçilirse seçilsin, her tahmin sözlüğünde hem "points" (4 köşe) hem "bbox"
(eksen-hizalı) alanı bulunur — model sadece birini ürettiyse diğeri ondan
türetilir, böylece eval_iou.py / çizim kodu mod farkı gözetmeden çalışabilir.
"""

import json
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from training import (
    BOS_TOKEN,
    EOS_TOKEN,
    ID_TO_LABEL,
    IMG_SIZE,
    LABEL_TO_ID,
    MAX_OBJECTS,
    NUM_BINS,
    OUTPUT_MODE,
    TOKENS_PER_OBJECT,
    VOCAB_SIZE,
    Pix2SeqModel,
    pad_to_square,
)

MAX_SEQ_LEN = 1 + (TOKENS_PER_OBJECT * MAX_OBJECTS) + 1

_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_path, device, max_seq_len=MAX_SEQ_LEN):
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Checkpoint hangi OUTPUT_MODE ile eğitilmiş — training.py'nin şu anki
    # OUTPUT_MODE'u ile uyuşmuyorsa sequence'i yanlış grupluyor olabiliriz.
    # Sessizce yanlış (ama çökmeyen) sonuç üretmek yerine burada uyarıyoruz.
    meta_path = os.path.splitext(model_path)[0] + ".meta.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("output_mode") != OUTPUT_MODE:
            print(
                f"UYARI: {model_path} '{meta.get('output_mode')}' modu ile eğitilmiş, "
                f"ancak training.py şu an OUTPUT_MODE='{OUTPUT_MODE}' olarak ayarlı. "
                f"Tahminler yanlış gruplanıp anlamsız çıkabilir — training.py'deki "
                f"OUTPUT_MODE'u '{meta.get('output_mode')}' yapın ya da modeli yeniden eğitin."
            )
    else:
        print(f"NOT: {meta_path} bulunamadı — checkpoint'in OUTPUT_MODE uyumu doğrulanamadı.")

    return model


def dequantize(token, max_size, num_bins):
    return (token / (num_bins - 1)) * max_size


@torch.no_grad()
def encode_image(model, image: Image.Image, device):
    """Kareye tamamlar, IMG_SIZE'a resize eder, normalize eder ve encoder'ı BİR
    KEZ çalıştırır. (memory, max_dim) döner — max_dim, tahmin edilen
    koordinatları piksel uzayına geri çevirmek için gerekir."""
    padded_img, max_dim = pad_to_square(image)
    resized = padded_img.resize(IMG_SIZE, Image.BILINEAR)
    img_tensor = _NORMALIZE(resized).unsqueeze(0).to(device)
    memory = model.encode(img_tensor)
    return memory, max_dim


@torch.no_grad()
def autoregressive_decode(model, memory, device, max_seq_len=MAX_SEQ_LEN):
    """Greedy decode. `memory` sabittir ve döngü boyunca yeniden hesaplanmaz.
    Returns (sequence, token_probs) — token_probs[i], sequence[i+1] için
    modelin softmax olasılığıdır (BOS hariç)."""
    sequence = [BOS_TOKEN]
    token_probs = []

    for _ in range(max_seq_len):
        tgt_seq = torch.tensor([sequence], dtype=torch.long, device=device)
        logits = model.decode(memory, tgt_seq)

        next_token_logits = logits[0, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = int(torch.argmax(next_token_logits).item())

        sequence.append(next_token)
        token_probs.append(float(probs[next_token].item()))

        if next_token == EOS_TOKEN:
            break

    return sequence, token_probs


def _bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _points_from_bbox(bbox):
    xmin, ymin, xmax, ymax = bbox
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def detokenize(sequence, token_probs, max_dim):
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
    """
    tokens = sequence[1:]
    probs = token_probs
    if tokens and tokens[-1] == EOS_TOKEN:
        tokens = tokens[:-1]
        probs = probs[:-1]

    predictions = []
    n_complete = len(tokens) - (len(tokens) % TOKENS_PER_OBJECT)

    for i in range(0, n_complete, TOKENS_PER_OBJECT):
        obj_tokens = tokens[i:i + TOKENS_PER_OBJECT]
        obj_probs  = probs[i:i + TOKENS_PER_OBJECT]

        # Sınıf token'ı, moddan bağımsız olarak her zaman son sıradadır.
        class_tok    = obj_tokens[-1]
        coord_tokens = obj_tokens[:-1]

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
                x = dequantize(obj_tokens[idx + j],     max_dim, NUM_BINS)
                y = dequantize(obj_tokens[idx + j + 1], max_dim, NUM_BINS)
                points.append((x, y))
            idx += 8

        if OUTPUT_MODE in ("hbb", "both"):
            # Model bu 4 koordinatı birbirinden bağımsız token olarak üretiyor —
            # mimaride "xmin <= xmax" gibi bir kısıt yok, dolayısıyla (özellikle
            # eğitimin erken aşamalarında) ters sıralı gelebilir. min/max ile
            # normalize ediyoruz ki PIL'in draw.rectangle'ı ("x1 must be >= x0")
            # asla çökmesin ve bbox her zaman geçerli olsun.
            x_a = dequantize(obj_tokens[idx],     max_dim, NUM_BINS)
            y_a = dequantize(obj_tokens[idx + 1], max_dim, NUM_BINS)
            x_b = dequantize(obj_tokens[idx + 2], max_dim, NUM_BINS)
            y_b = dequantize(obj_tokens[idx + 3], max_dim, NUM_BINS)
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
    (padding'siz) görüntü boyutuna kırp/ölçekle. Returns (predictions, sequence)."""
    orig_w, orig_h = image.size
    memory, max_dim = encode_image(model, image, device)
    sequence, token_probs = autoregressive_decode(model, memory, device)
    predictions = detokenize(sequence, token_probs, max_dim)

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
