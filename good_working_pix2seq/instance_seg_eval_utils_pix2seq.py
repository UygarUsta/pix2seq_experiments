import numpy as np
import torch
import torch.nn as nn
import cv2
from torchmetrics.detection import MeanAveragePrecision

from training_pix2seq_instance_seg import (
    Pix2SeqModel, KVCacheDecoder, ar_decode_batch, seq_to_instances,
    VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, MAX_SEQ_LEN,
    TOKENS_PER_OBJ, NUM_POLY_PTS, NUM_CLASSES,
)

# Maskeler torchmetrics içinde epoch boyunca birikiyor. 512x512 bool @ 10 nesne
# ≈ 2.6 MB/görüntü → büyük val setlerinde RAM patlar. 256'da rasterize etmek
# mAP'i pratikte etkilemiyor, belleği 4x düşürüyor. OOM alırsanız 192/128 yapın.
MASK_EVAL_SIZE = 256


def polys_to_masks(polys, out_size=MASK_EVAL_SIZE):
    """
    Model uzayındaki (IMG_SIZE) poligonları [N, out_size, out_size] bool maskeye çevirir.
    """
    if len(polys) == 0:
        return torch.zeros((0, out_size, out_size), dtype=torch.bool)

    img_h, img_w = IMG_SIZE
    sx = out_size / float(img_w)
    sy = out_size / float(img_h)

    masks = np.zeros((len(polys), out_size, out_size), dtype=np.uint8)
    for i, poly in enumerate(polys):
        pts = np.asarray(poly, dtype=np.float32).copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        cv2.fillPoly(masks[i], [np.round(pts).astype(np.int32)], 1)
    return torch.from_numpy(masks).bool()


def _strip_specials(tokens):
    """EOS'ta kes, BOS/PAD'i at."""
    if EOS_TOKEN in tokens:
        tokens = tokens[:tokens.index(EOS_TOKEN)]
    return [t for t in tokens if t not in (BOS_TOKEN, PAD_TOKEN)]


@torch.no_grad()
def evaluate(model, loader, max_seq_len=MAX_SEQ_LEN, device=None, use_cache=True):
    """
    Hem segm hem bbox mAP döner:
        {"segm_map", "segm_map_50", "bbox_map", "bbox_map_50"}
    bbox metriğini bilerek koruyoruz — poligon tokenları eklendikten sonra kutu
    kalitesinin düşüp düşmediğini (sekans uzunluğu modeli boğuyor mu) doğrudan gösterir.
    """
    device = device or next(model.parameters()).device
    metric = MeanAveragePrecision(box_format="xyxy", iou_type=("bbox", "segm"))
    model.eval()

    for images, targets in loader:
        images = images.to(device).float()
        seqs, scs = ar_decode_batch(model, images, max_seq_len, device, use_cache=use_cache)

        preds_batch, gts_batch = [], []
        for b in range(images.size(0)):
            toks = _strip_specials(seqs[b].tolist())
            pb, pp, pl, pc = seq_to_instances(toks, scs[b].float().cpu().numpy())

            gt_toks = _strip_specials(targets[b].tolist())
            gb, gp, gl, _ = seq_to_instances(gt_toks, None)

            preds_batch.append({
                "boxes":  torch.tensor(pb, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(pc, dtype=torch.float32).reshape(-1),
                "labels": torch.tensor(pl, dtype=torch.long).reshape(-1),
                "masks":  polys_to_masks(pp),
            })
            gts_batch.append({
                "boxes":  torch.tensor(gb, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(gl, dtype=torch.long).reshape(-1),
                "masks":  polys_to_masks(gp),
            })

        metric.update(preds_batch, gts_batch)

    res = metric.compute()

    def _get(*keys):
        for k in keys:
            if k in res:
                return float(res[k])
        return float("nan")

    return {
        "segm_map":    _get("segm_map", "map"),
        "segm_map_50": _get("segm_map_50", "map_50"),
        "bbox_map":    _get("bbox_map", "map"),
        "bbox_map_50": _get("bbox_map_50", "map_50"),
    }
