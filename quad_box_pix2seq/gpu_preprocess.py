"""
GPU ön-işleme: nvJPEG decode + EXIF döndürme + kare pad + resize + normalize, hepsi GPU'da.

CPU yolu (eğitimde/val'de kullanılan):
    PIL decode -> ImageOps.exif_transpose -> pad_to_square(128) -> A.Resize(cv2 INTER_LINEAR)
    -> A.Normalize(ImageNet)
Bu dosya aynı adımları aynı sırayla GPU'da yapar ve sonucu CPU yoluna olabildiğince yakın tutar:
  * bilinear, align_corners=False, antialias=False  -> cv2 INTER_LINEAR ile aynı örnekleme
  * resize sonrası uint8'e yuvarlama                  -> cv2'nin uint8 çıktısını taklit eder
  * EXIF yönü PIL ile aynı şekilde okunur, aynı 8 dönüşüm uygulanır
JPEG olmayan dosyalar veya nvJPEG'in açamadığı JPEG'ler (CMYK vb.) için decode CPU'da yapılır,
kalan adımlar yine GPU'da.

Kullanım:
    pre = GpuPreprocessor("cuda")
    x, max_dim, (w, h) = pre(path)          # x: (1,3,512,512) float, GPU'da
    preds, (w, h) = predict_hier_gpu(model, path, pre, accept_tol=2)
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.io import decode_jpeg, read_file, ImageReadMode

import pix2seq_florence_quad_hier as H
from pix2seq_florence_quad import (IMG_SIZE, INFER_SCORE_THRESH, ID_TO_LABEL, load_image,
                                   dequantize, decode_tokens)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD_FILL = 128                     # pad_to_square ile aynı gri
EXIF_ORIENTATION = 274


def exif_orientation(path):
    """PIL.ImageOps.exif_transpose'un okuduğu etiket. Sadece başlık okunur (görüntü decode edilmez)."""
    try:
        with Image.open(path) as im:
            return int(im.getexif().get(EXIF_ORIENTATION, 1) or 1)
    except Exception:
        return 1


def apply_orientation(img, o):
    """(C,H,W) tensöre PIL exif_transpose'un uyguladığı dönüşümün aynısı."""
    if o == 2:   # FLIP_LEFT_RIGHT
        return img.flip(-1)
    if o == 3:   # ROTATE_180
        return img.flip(-1, -2)
    if o == 4:   # FLIP_TOP_BOTTOM
        return img.flip(-2)
    if o == 5:   # TRANSPOSE
        return img.transpose(-1, -2)
    if o == 6:   # ROTATE_270 (saat yönünde 90)
        return torch.rot90(img, k=-1, dims=(-2, -1))
    if o == 7:   # TRANSVERSE
        return img.transpose(-1, -2).flip(-1, -2)
    if o == 8:   # ROTATE_90 (saat yönü tersine 90)
        return torch.rot90(img, k=1, dims=(-2, -1))
    return img


class GpuPreprocessor:
    def __init__(self, device="cuda", img_size=IMG_SIZE):
        self.device = torch.device(device)
        self.size = tuple(img_size)
        self.mean = torch.tensor(MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.device).view(1, 3, 1, 1)
        self.stats = {"gpu_decode": 0, "cpu_decode": 0}

    def decode(self, path):
        """-> uint8 (3,h,w) tensör, cihazda, EXIF uygulanmış."""
        if path.lower().endswith((".jpg", ".jpeg")):
            try:
                data = read_file(path)
                img = decode_jpeg(data, mode=ImageReadMode.RGB, device=self.device)
                self.stats["gpu_decode"] += 1
                return apply_orientation(img, exif_orientation(path))
            except Exception:
                pass                       # CMYK / bozuk JPEG -> CPU yolu
        self.stats["cpu_decode"] += 1
        pil = load_image(path)             # exif_transpose + RGB (CPU yoluyla birebir)
        return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).to(self.device)

    @torch.no_grad()
    def __call__(self, path):
        """-> x (1,3,H,W) float normalize, max_dim (kare tuval kenarı), (w, h) orijinal boyut"""
        img = self.decode(path)
        _, h, w = img.shape
        s = max(h, w)
        canvas = torch.full((3, s, s), PAD_FILL, dtype=torch.uint8, device=self.device)
        canvas[:, :h, :w] = img            # sol-üste yapıştır (pad_to_square ile aynı)
        x = F.interpolate(canvas[None].float(), size=self.size, mode="bilinear",
                          align_corners=False, antialias=False)
        x = x.round_().clamp_(0, 255)      # cv2 uint8 çıktısı
        x = (x / 255.0 - self.mean) / self.std
        return x, s, (w, h)


@torch.no_grad()
def predict_hier_gpu(model, img_path, pre, score_thresh=INFER_SCORE_THRESH, accept_tol=0,
                     timings=None):
    """predict_hier'in GPU ön-işlemeli hali. Döner: preds, (w, h). (PIL görüntü döndürmez.)"""
    cuda = pre.device.type == "cuda"
    t0 = time.perf_counter()
    x, max_dim, size = pre(img_path)
    if timings is not None and cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    tokens, scores = H.generate_hier_fast(model, x, accept_tol=accept_tol)
    if timings is not None:
        if cuda:
            torch.cuda.synchronize()
        timings.setdefault("pre", []).append(t1 - t0)
        timings.setdefault("model", []).append(time.perf_counter() - t1)
    preds = []
    for cls, bins, s in decode_tokens(tokens[0].tolist(), scores[0].tolist()):
        if s >= score_thresh:
            preds.append((ID_TO_LABEL[cls], dequantize(bins, max_dim, max_dim), s))
    return preds, size
