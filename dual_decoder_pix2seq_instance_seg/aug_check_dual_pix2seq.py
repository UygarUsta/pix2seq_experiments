"""
Sequence-augmentation görsel kontrolü.

Önemli nokta: bu script kutuları `records` ara yapısından DEĞİL, dataset'in
döndürdüğü token dizilerinden (box_in / mask_in) geri çözerek çizer. Yani
modelin gerçekten gördüğü şeyi görselleştirir - quantize/dequantize kaybı,
sıralama, noise'un kuyruğa eklenmesi ve poligon eşleşmesi hepsi burada
doğrulanmış olur.

Renkler:
    yeşil   - gerçek (GT) kutu
    kırmızı - sentetik noise kutusu (hedefte koordinatlar PAD, sadece sınıf öğrenilir)
    cyan    - GT poligonu (decoder 2 hedefi)

Kullanım:
    python aug_check_dual_pix2seq.py                 # 24 örnek -> aug_check/
    python aug_check_dual_pix2seq.py --n 60 --out aug_check_v2
    python aug_check_dual_pix2seq.py --val           # noise olmamalı (kontrol)
"""

import os
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from dual_decoder_pix2seq import (
    DualPix2SeqDataset, dequantize,
    JSON_DIR, IMG_DIR, IMG_SIZE, ID_TO_LABEL,
    NUM_BINS, NUM_CLASSES, MAX_OBJECTS,
    NOISE_CLASS_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    BOX_TOKENS_PER_OBJ, NUM_POLY_PTS,
    train_transform, val_transform,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

C_GT    = (60, 220, 90)
C_NOISE = (255, 70, 70)
C_POLY  = (70, 210, 255)
C_BG    = (20, 20, 20)


def tensor_to_pil(img_tensor):
    """Albumentations Normalize + ToTensorV2 çıktısını geri çevirir."""
    arr = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * STD + MEAN
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def decode_box_sequence(box_in):
    """box_in -> [{box, cls_token, is_noise, slot}] (modelin gördüğü hâliyle)."""
    toks = box_in.tolist()[1:]          # BOS'u at
    objs = []
    for s in range(MAX_OBJECTS):
        chunk = toks[s * BOX_TOKENS_PER_OBJ:(s + 1) * BOX_TOKENS_PER_OBJ]
        if len(chunk) < BOX_TOKENS_PER_OBJ or PAD_TOKEN in chunk:
            continue                     # boş slot
        cls_tok = chunk[4]
        objs.append({
            "slot": s,
            "cls_token": cls_tok,
            "is_noise": cls_tok == NOISE_CLASS_TOKEN,
            "box": [
                dequantize(chunk[0], IMG_SIZE[1], NUM_BINS),
                dequantize(chunk[1], IMG_SIZE[0], NUM_BINS),
                dequantize(chunk[2], IMG_SIZE[1], NUM_BINS),
                dequantize(chunk[3], IMG_SIZE[0], NUM_BINS),
            ],
        })
    return objs


def decode_mask_row(row):
    """mask_in satırı -> poligon [K,2] ya da None (noise / boş slot)."""
    toks = row.tolist()
    if all(t == PAD_TOKEN for t in toks):
        return None
    poly_toks = [t for t in toks[1 + BOX_TOKENS_PER_OBJ:] if 0 <= t < NUM_BINS]
    if len(poly_toks) < 2 * NUM_POLY_PTS:
        return None
    return np.array([
        [dequantize(poly_toks[k],     IMG_SIZE[1], NUM_BINS),
         dequantize(poly_toks[k + 1], IMG_SIZE[0], NUM_BINS)]
        for k in range(0, 2 * NUM_POLY_PTS, 2)
    ], dtype=np.float32)


def render_sample(img_tensor, box_in, box_tgt, mask_in, n_real, poly_alpha=0.28):
    base = tensor_to_pil(img_tensor).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    objs = decode_box_sequence(box_in)

    # Poligonlar önce (dolgu), kutular üstüne
    for o in objs:
        if o["is_noise"]:
            continue
        poly = decode_mask_row(mask_in[o["slot"]])
        if poly is not None:
            od.polygon([tuple(p) for p in poly],
                       fill=C_POLY + (int(255 * poly_alpha),), outline=C_POLY + (200,))

    img = Image.alpha_composite(base, overlay).convert("RGB")
    d = ImageDraw.Draw(img)

    n_noise = 0
    for o in objs:
        x0, y0, x1, y1 = o["box"]
        if o["is_noise"]:
            n_noise += 1
            color, tag = C_NOISE, "noise"
            # noise kutusu: kesikli kenar -> GT ile karışmasın
            d.rectangle([x0, y0, x1, y1], outline=color, width=2)
            step = 10
            for x in range(int(x0), int(x1), step * 2):
                d.line([x, y0, min(x + step, x1), y0], fill=(255, 255, 255), width=1)
                d.line([x, y1, min(x + step, x1), y1], fill=(255, 255, 255), width=1)
        else:
            color = C_GT
            tag = ID_TO_LABEL.get(o["cls_token"] - NUM_BINS, "?")
            d.rectangle([x0, y0, x1, y1], outline=color, width=3)

        label = f"{o['slot']}:{tag}"
        tw = 7 * len(label) + 8
        ty = max(0, y0 - 15)
        d.rectangle([x0, ty, x0 + tw, ty + 15], fill=color)
        d.text((x0 + 4, ty + 2), label, fill=(0, 0, 0))

    # Üst bilgi şeridi
    strip_h = 26
    out = Image.new("RGB", (img.width, img.height + strip_h), C_BG)
    out.paste(img, (0, strip_h))
    hd = ImageDraw.Draw(out)
    hd.rectangle([6, 8, 18, 20], outline=C_GT, width=3)
    hd.text((24, 9), f"GT: {n_real}", fill=C_GT)
    hd.rectangle([100, 8, 112, 20], outline=C_NOISE, width=3)
    hd.text((118, 9), f"noise: {n_noise}", fill=C_NOISE)
    hd.text((210, 9), f"slots: {len(objs)}/{MAX_OBJECTS}", fill=(200, 200, 200))
    return out, n_noise


def verify_sample(box_in, box_tgt, mask_in, n_real):
    """Görselin yanında sessiz sağlık kontrolü - bozuk örnekte hata verir."""
    objs = decode_box_sequence(box_in)
    real = [o for o in objs if not o["is_noise"]]
    noise = [o for o in objs if o["is_noise"]]

    assert len(real) == int(n_real), f"gerçek nesne sayısı tutmuyor: {len(real)} != {n_real}"
    if noise:
        first_noise = min(o["slot"] for o in noise)
        last_real = max((o["slot"] for o in real), default=-1)
        assert first_noise > last_real, "noise gerçek nesnelerin arasına girmiş (kuyrukta olmalı)"
    for o in noise:
        tgt = box_tgt[o["slot"] * BOX_TOKENS_PER_OBJ:(o["slot"] + 1) * BOX_TOKENS_PER_OBJ].tolist()
        assert tgt == [PAD_TOKEN] * 4 + [NOISE_CLASS_TOKEN], f"noise hedefi bozuk: {tgt}"
        assert all(t == PAD_TOKEN for t in mask_in[o["slot"]].tolist()), "noise'a maske dizisi verilmiş"
    for o in real:
        tgt = box_tgt[o["slot"] * BOX_TOKENS_PER_OBJ:(o["slot"] + 1) * BOX_TOKENS_PER_OBJ].tolist()
        src = box_in[1 + o["slot"] * BOX_TOKENS_PER_OBJ:
                     1 + (o["slot"] + 1) * BOX_TOKENS_PER_OBJ].tolist()
        assert tgt == src, "gerçek nesnede input/target hizası bozuk"
    return len(real), len(noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default=JSON_DIR)
    ap.add_argument("--img-dir",  default=IMG_DIR)
    ap.add_argument("--out",      default="aug_check")
    ap.add_argument("--n",        type=int, default=24, help="kaç örnek kaydedilecek")
    ap.add_argument("--seed",     type=int, default=0)
    ap.add_argument("--val",      action="store_true",
                    help="is_train=False ile çalıştır - hiç noise çıkmamalı")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)

    ds = DualPix2SeqDataset(
        args.json_dir, args.img_dir,
        transform=val_transform if args.val else train_transform,
        is_train=not args.val,
    )
    ds.json_files = sorted(ds.json_files)
    print(f"Dataset: {len(ds)} örnek | is_train={not args.val} -> {args.out}/")

    n = min(args.n, len(ds))
    idxs = random.sample(range(len(ds)), n)

    tot_real = tot_noise = 0
    empty_imgs = 0
    for i, idx in enumerate(idxs):
        img_t, box_in, box_tgt, mask_in, mask_tgt, n_real = ds[idx]

        if not args.no_verify:
            verify_sample(box_in, box_tgt, mask_in, n_real)

        canvas, n_noise = render_sample(img_t, box_in, box_tgt, mask_in, int(n_real))
        stem = os.path.splitext(ds.json_files[idx])[0]
        canvas.save(os.path.join(args.out, f"{i:03d}_{stem}_gt{int(n_real)}_noise{n_noise}.jpg"),
                    quality=92)

        tot_real += int(n_real)
        tot_noise += n_noise
        empty_imgs += int(n_real == 0)

    print(f"Kaydedildi: {n} görüntü")
    print(f"  gerçek nesne : {tot_real} (ort. {tot_real / max(n,1):.2f}/görüntü)")
    print(f"  noise kutusu : {tot_noise} (ort. {tot_noise / max(n,1):.2f}/görüntü)")
    print(f"  boş görüntü  : {empty_imgs} (bunlar tamamen noise hedefi üretir)")
    if not args.no_verify:
        print("  dizi kontrolü: OK (noise kuyrukta, hedef koordinatları PAD, hiza doğru)")


if __name__ == "__main__":
    main()
