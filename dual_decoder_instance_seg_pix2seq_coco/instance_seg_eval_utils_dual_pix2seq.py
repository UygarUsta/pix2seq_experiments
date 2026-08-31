"""
instance_seg_eval_utils_dual_pix2seq.py
=======================================
Dual-decoder Pix2Seq için segm + bbox mAP.

Bellek notu (COCO'da önemli): torchmetrics MeanAveragePrecision maskeleri
compute() çağrılana kadar RAM'de tutar. Görüntü başına ~30 tahmin + ~7 GT,
MASK_EVAL_SIZE=256 -> yaklaşık 2.4 MB/görüntü. 5000 görüntülük tam val2017 bu
yüzden 12 GB'a yaklaşır. Varsayılan olarak `max_images` ile ilk N görüntüye
bakılır (eğitim sırasında sıralamayı izlemek için fazlasıyla yeter); nihai
sayı için max_images=None ve MASK_EVAL_SIZE=192 gibi bir kombinasyon kullan
ya da doğrudan pycocotools/RLE ile ölç.
"""

import numpy as np
import torch
import cv2
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

from dual_decoder_pix2seq import (
    dual_ar_decode, dequantize,
    IMG_SIZE, NUM_BINS, NUM_CLASSES,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    BOX_TOKENS_PER_OBJ, NUM_POLY_PTS,
    INFER_SCORE_THRESH, MAX_DETS,
)

MASK_EVAL_SIZE = 256   # maske mAP'inin çözünürlüğü (bellek <-> yüksek IoU hassasiyeti)


def polys_to_masks(polys, out_size=MASK_EVAL_SIZE):
    """[K,2] poligon listesini [N, out_size, out_size] bool maskeye çevirir."""
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


def parse_ground_truth_targets(mask_targets, num_valid_objs):
    """
    [MAX_OBJECTS, S] maske HEDEF tensöründen GT kutu/poligon/etiket çıkarır.
    Satır formatı: [x0, y0, x1, y1, cls, poly..., EOS, PAD...]

    num_valid_objs sadece gerçek nesneleri sayar; noise nesneleri kuyrukta ve
    tamamen PAD olduğu için GT olarak okunmaları mümkün değil.
    """
    gt_boxes, gt_polys, gt_labels = [], [], []

    for obj_idx in range(int(num_valid_objs)):
        seq = mask_targets[obj_idx].tolist()
        if EOS_TOKEN in seq:
            seq = seq[:seq.index(EOS_TOKEN)]
        seq = [t for t in seq if t not in (BOS_TOKEN, PAD_TOKEN)]
        if len(seq) < BOX_TOKENS_PER_OBJ + 2 * NUM_POLY_PTS:
            continue

        x0 = dequantize(seq[0], IMG_SIZE[1], NUM_BINS)
        y0 = dequantize(seq[1], IMG_SIZE[0], NUM_BINS)
        x1 = dequantize(seq[2], IMG_SIZE[1], NUM_BINS)
        y1 = dequantize(seq[3], IMG_SIZE[0], NUM_BINS)
        label = seq[4] - NUM_BINS
        if not (0 <= label < NUM_CLASSES) or x1 <= x0 or y1 <= y0:
            continue

        pc = seq[5:5 + 2 * NUM_POLY_PTS]
        poly = np.array([
            [dequantize(pc[k], IMG_SIZE[1], NUM_BINS),
             dequantize(pc[k + 1], IMG_SIZE[0], NUM_BINS)]
            for k in range(0, 2 * NUM_POLY_PTS, 2)
        ], dtype=np.float32)

        gt_boxes.append([x0, y0, x1, y1])
        gt_polys.append(poly)
        gt_labels.append(int(label))

    return gt_boxes, gt_polys, gt_labels


@torch.no_grad()
def evaluate_dual(model, loader, device=None, score_thresh=INFER_SCORE_THRESH,
                  max_images=None, max_dets=MAX_DETS, amp_dtype=None,
                  mask_size=MASK_EVAL_SIZE, iou_types=("bbox", "segm")):
    """
    Loader üzerinde dual AR decode çalıştırır ve şunları döndürür:
        segm_map, segm_map_50, bbox_map, bbox_map_50, preds_per_img

    Tahminler gerçek bir skor taşır (p(sınıf) vs p(noise)), yani mAP noise
    sınıfının sağladığı sıralamayı yansıtır - herkese düz 1.0 vermez.
    """
    device = device or next(model.parameters()).device
    metric = MeanAveragePrecision(box_format="xyxy", iou_type=tuple(iou_types))
    model.eval()

    n_img = n_pred = 0
    for images, _bi, _bt, _mi, mask_tgt, num_objs in tqdm(loader, desc="    eval",
                                                          leave=False, unit="b"):
        images = images.to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type,
                            dtype=amp_dtype or torch.float32,
                            enabled=amp_dtype is not None):
            batch_preds = dual_ar_decode(model, images, device,
                                         score_thresh=score_thresh, max_dets=max_dets)

        preds_batch, gts_batch = [], []
        for b in range(images.size(0)):
            preds = batch_preds[b]
            g_boxes, g_polys, g_labels = parse_ground_truth_targets(
                mask_tgt[b], int(num_objs[b]))

            p = {"boxes":  torch.tensor([x["box"] for x in preds],
                                        dtype=torch.float32).reshape(-1, 4),
                 "scores": torch.tensor([x["score"] for x in preds],
                                        dtype=torch.float32).reshape(-1),
                 "labels": torch.tensor([x["label"] for x in preds],
                                        dtype=torch.long).reshape(-1)}
            g = {"boxes":  torch.tensor(g_boxes, dtype=torch.float32).reshape(-1, 4),
                 "labels": torch.tensor(g_labels, dtype=torch.long).reshape(-1)}
            if "segm" in iou_types:
                p["masks"] = polys_to_masks([x["polygon"] for x in preds], mask_size)
                g["masks"] = polys_to_masks(g_polys, mask_size)

            preds_batch.append(p)
            gts_batch.append(g)
            n_img += 1
            n_pred += len(preds)

        metric.update(preds_batch, gts_batch)

        if max_images is not None and n_img >= max_images:
            break

    res = metric.compute()

    def _get(*keys):
        for k in keys:
            if k in res:
                return float(res[k])
        return float("nan")

    nan = float("nan")
    out = {
        "segm_map":    _get("segm_map", "map") if "segm" in iou_types else nan,
        "segm_map_50": _get("segm_map_50", "map_50") if "segm" in iou_types else nan,
        "bbox_map":    _get("bbox_map", "map") if "bbox" in iou_types else nan,
        "bbox_map_50": _get("bbox_map_50", "map_50") if "bbox" in iou_types else nan,
        "preds_per_img": n_pred / max(n_img, 1),
    }
    metric.reset()
    return out
