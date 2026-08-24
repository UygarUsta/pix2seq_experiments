import numpy as np
import torch
import cv2
from torchmetrics.detection import MeanAveragePrecision

from dual_decoder_pix2seq import (
    dual_ar_decode, dequantize,
    IMG_SIZE, NUM_BINS, NUM_CLASSES,
    BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    BOX_TOKENS_PER_OBJ, NUM_POLY_PTS,
    INFER_SCORE_THRESH,
)

MASK_EVAL_SIZE = 256  # Downsampled mask resolution to keep validation memory footprint low


def polys_to_masks(polys, out_size=MASK_EVAL_SIZE):
    """Converts a list of [K, 2] polygon arrays into an [N, out_size, out_size] boolean mask tensor."""
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
    Extracts GT boxes, polygons, and labels from the [MAX_OBJECTS, SEQ_LEN] mask
    TARGET tensor, whose rows are [x0, y0, x1, y1, cls, poly..., EOS, PAD...].

    num_valid_objs counts real objects only; noise objects sit after them in the
    tail and are all-PAD, so they can never be read as ground truth.
    """
    gt_boxes, gt_polys, gt_labels = [], [], []

    for obj_idx in range(int(num_valid_objs.item())):
        seq = mask_targets[obj_idx].tolist()
        if EOS_TOKEN in seq:
            seq = seq[:seq.index(EOS_TOKEN)]
        seq = [t for t in seq if t not in (BOS_TOKEN, PAD_TOKEN)]

        if len(seq) < BOX_TOKENS_PER_OBJ + 2 * NUM_POLY_PTS:
            continue

        # Box & Class
        x0 = dequantize(seq[0], IMG_SIZE[1], NUM_BINS)
        y0 = dequantize(seq[1], IMG_SIZE[0], NUM_BINS)
        x1 = dequantize(seq[2], IMG_SIZE[1], NUM_BINS)
        y1 = dequantize(seq[3], IMG_SIZE[0], NUM_BINS)
        label = seq[4] - NUM_BINS
        if not (0 <= label < NUM_CLASSES):
            continue

        # Polygon
        poly_coords = seq[5:5 + 2 * NUM_POLY_PTS]
        poly = np.array([
            [dequantize(poly_coords[k], IMG_SIZE[1], NUM_BINS),
             dequantize(poly_coords[k + 1], IMG_SIZE[0], NUM_BINS)]
            for k in range(0, 2 * NUM_POLY_PTS, 2)
        ], dtype=np.float32)

        gt_boxes.append([x0, y0, x1, y1])
        gt_polys.append(poly)
        gt_labels.append(label)

    return gt_boxes, gt_polys, gt_labels


@torch.no_grad()
def evaluate_dual(model, loader, device=None, score_thresh=INFER_SCORE_THRESH):
    """
    Runs dual autoregressive decoding across the loader and returns:
      - segm_map, segm_map_50
      - bbox_map, bbox_map_50

    Predictions now carry a real score (p(class) vs p(noise)), so mAP reflects
    the ranking the noise class provides instead of a flat 1.0 for everything.
    """
    device = device or next(model.parameters()).device
    metric = MeanAveragePrecision(box_format="xyxy", iou_type=("bbox", "segm"))
    model.eval()

    for images, _box_in, _box_tgt, _mask_in, mask_tgt, num_objs in loader:
        images = images.to(device).float()
        batch_preds = dual_ar_decode(model, images, device, score_thresh=score_thresh)

        preds_batch, gts_batch = [], []
        for b in range(images.size(0)):
            preds = batch_preds[b]

            p_boxes  = [p["box"] for p in preds]
            p_polys  = [p["polygon"] for p in preds]
            p_labels = [p["label"] for p in preds]
            p_scores = [p["score"] for p in preds]

            g_boxes, g_polys, g_labels = parse_ground_truth_targets(mask_tgt[b], num_objs[b])

            preds_batch.append({
                "boxes":  torch.tensor(p_boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(p_scores, dtype=torch.float32).reshape(-1),
                "labels": torch.tensor(p_labels, dtype=torch.long).reshape(-1),
                "masks":  polys_to_masks(p_polys),
            })
            gts_batch.append({
                "boxes":  torch.tensor(g_boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(g_labels, dtype=torch.long).reshape(-1),
                "masks":  polys_to_masks(g_polys),
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
