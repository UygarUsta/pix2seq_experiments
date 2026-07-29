"""
Pix2Seq Evaluation — Oriented Quad IoU
---------------------------------------
Computes per-class and overall metrics on the validation split.

Metrics:
  - IoU per prediction (polygon-based, handles rotated quads)
  - Precision / Recall / F1 — tek eşikli, modelin ürettiği HER tahmin kabul
    edilmiş gibi (score'a bakılmaksızın) hesaplanır. "Şu an deploy edilmiş
    haliyle model ne veriyor" sorusuna cevap verir.
  - AP / mAP (IoU ≥ IOU_THRESHOLD) — PASCAL VOC tarzı, skora göre sıralanmış,
    all-point interpolation ile. Modelin `score` alanının (bkz.
    pix2seq_decode.py, noise-object augmentation ile kalibre edilmiş
    confidence) ne kadar güvenilir bir sıralama sinyali olduğunu ölçer;
    farklı score eşiklerinde precision/recall trade-off'unu tek sayıya indirger.
  - Per-class breakdown

Run:
    python eval_iou.py
"""

import json
import os
from collections import Counter

import numpy as np
import torch
from PIL import Image
from shapely.geometry import Polygon
from tqdm import tqdm

from training import JSON_DIR, LABEL_TO_ID, OUTPUT_MODE, Pix2SeqModel, shape_to_quad
from pix2seq_decode import load_model as _load_model
from pix2seq_decode import predict as _predict

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH    = "pix2seq_best.pth"
VAL_SPLIT     = "val_split.json"       # saved during training
JSON_DIR_PATH = JSON_DIR
IMG_DIR       = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01/"
IOU_THRESHOLD = 0.5
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model() -> Pix2SeqModel:
    model = _load_model(MODEL_PATH, DEVICE)
    print(f"Loaded model from {MODEL_PATH}")
    return model


def predict(model: Pix2SeqModel, img: Image.Image) -> list[dict]:
    """Returns list of {label, points, bbox, score} in original image coordinates.
    Encoder bir kez çalışır (pix2seq_decode.predict), autoregressive decode
    döngüsü sadece decode adımını tekrarlar."""
    predictions, _sequence = _predict(model, img, DEVICE)
    return predictions


# Geçersiz (kendisiyle kesişen) polygon sayaçları — bkz. INVALID_STATS raporu.
INVALID_STATS = {"pred": 0, "gt": 0, "pred_total": 0, "gt_total": 0}


def poly_iou(pts_a: list, pts_b: list) -> float:
    """Polygon IoU using Shapely — works for any convex quad.

    DİKKAT: bu fonksiyon köşe SIRASINA duyarlıdır. Model 8 koordinatı
    autoregressive üretiyor; iki köşeyi yer değiştirirse ortaya "papyon"
    (kendisiyle kesişen) bir dörtgen çıkar, Shapely bunu geçersiz sayar ve
    sonuç 0.0 olur — 4 nokta geometrik olarak kusursuz olsa bile tahmin tam
    ıska sayılır. Bu, ince metin alanlarında (iki komşu köşe ~22 px arayken)
    büyük kutulara (~300 px) göre çok daha sık olur. Ne kadar sık olduğunu
    ölçmek için quad_iou_unordered ile karşılaştırın."""
    try:
        pa = Polygon(pts_a)
        pb = Polygon(pts_b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def canon_quad(points: list) -> list:
    """4 köşeyi ağırlık merkezi etrafındaki açıya göre sıralar. Sonuç her zaman
    basit (kendisiyle kesişmeyen) bir dörtgendir, dolayısıyla köşe sırası
    hatalarını IoU ölçümünden tamamen çıkarır."""
    p = np.asarray(points, dtype=float)
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    return p[order].tolist()


def _as_bbox(points: list) -> list:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


# ── IoU varyantları ───────────────────────────────────────────────────────────
# Üçünü birlikte raporlamak hatayı bileşenlerine ayırır:
#   quad          : deploy edilen davranış (köşe sırası dahil her şey doğru olmalı)
#   quad-unordered: aynı 4 nokta, sıra normalize edilmiş  -> aradaki fark = SIRA hatası
#   hbb           : modelin ayrıca ürettiği eksen-hizalı kutu -> köşe sırasından bağımsız
# quad ile quad-unordered arası büyük bir fark "model nereye bakacağını biliyor
# ama köşeleri hangi sırayla yazacağını bilmiyor" demektir; bu veri/hedef tanımı
# sorunudur, çözünürlük sorunu değildir.

def quad_iou(pred: dict, gt: dict) -> float:
    INVALID_STATS["pred_total"] += 1
    INVALID_STATS["gt_total"]   += 1
    if not Polygon(pred["points"]).is_valid:
        INVALID_STATS["pred"] += 1
    if not Polygon(gt["points"]).is_valid:
        INVALID_STATS["gt"] += 1
    return poly_iou(pred["points"], gt["points"])


def quad_iou_unordered(pred: dict, gt: dict) -> float:
    return poly_iou(canon_quad(pred["points"]), canon_quad(gt["points"]))


def hbb_iou(pred: dict, gt: dict) -> float:
    # pred["bbox"]: "both" modunda modelin AYRI token'larla ürettiği hbb
    # (türetilmiş değil); gt tarafı köşelerin min/max'ı.
    ax1, ay1, ax2, ay2 = pred["bbox"]
    bx1, by1, bx2, by2 = _as_bbox(gt["points"])
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


IOU_VARIANTS = {
    "quad (sıraya duyarlı)":   quad_iou,
    "quad (sıra-bağımsız)":    quad_iou_unordered,
    "hbb (eksen-hizalı)":      hbb_iou,
}


def load_ground_truth(json_file: str) -> tuple[Image.Image, list[dict]]:
    """Returns (image, list of {label, points}) in original image coordinates."""
    json_path = os.path.join(JSON_DIR_PATH, json_file)
    with open(json_path, encoding="utf-8") as f:
        item = json.load(f)

    img_name = item.get("imagePath", json_file.replace(".json", ".jpg"))
    img_path = os.path.join(IMG_DIR, img_name)

    # Try common extensions if file not found
    if not os.path.exists(img_path):
        base = os.path.splitext(img_path)[0]
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            if os.path.exists(base + ext):
                img_path = base + ext
                break

    img = Image.open(img_path).convert("RGB")

    # shape_to_quad, labelme'nin 2-noktalı `rectangle` shape'ini de kabul eder —
    # bbox etiketli datasetlerde GT'nin sessizce boş kalmaması için (eskiden
    # `len(points) == 4` şartı bunları eliyordu, sonuç: her tahmin FP sayılıyor
    # ve mAP 0 çıkıyordu).
    bbox_fallback = (OUTPUT_MODE == "hbb")
    gts = []
    for shape in item.get("shapes", []):
        label = shape.get("label", "")
        if label not in LABEL_TO_ID:
            continue
        quad = shape_to_quad(shape, bbox_fallback)
        if quad is not None:
            gts.append({"label": label, "points": [[p[0], p[1]] for p in quad]})

    return img, gts


def match_predictions(preds: list[dict],
                      gts:   list[dict],
                      iou_thresh: float) -> tuple[list, list, list]:
    """
    Greedy matching: highest IoU pair gets matched first.
    Returns (tp_list, fp_list, fn_list) where each element is a dict with label info.
    """
    matched_gt  = set()
    matched_pred = set()
    tp, fp, fn  = [], [], []

    # Build IoU matrix only for same-label pairs
    iou_matrix = np.zeros((len(preds), len(gts)))
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if pred["label"] == gt["label"]:
                iou_matrix[pi, gi] = poly_iou(pred["points"], gt["points"])

    # Greedy match highest IoU first
    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < iou_thresh:
            break
        pi, gi = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        tp.append({
            "pred_label": preds[pi]["label"],
            "gt_label":   gts[gi]["label"],
            "iou":        max_iou,
        })
        matched_pred.add(pi)
        matched_gt.add(gi)
        iou_matrix[pi, :] = -1
        iou_matrix[:, gi] = -1

    for pi, pred in enumerate(preds):
        if pi not in matched_pred:
            fp.append({"pred_label": pred["label"], "iou": 0.0})

    for gi, gt in enumerate(gts):
        if gi not in matched_gt:
            fn.append({"gt_label": gt["label"]})

    return tp, fp, fn


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """PASCAL VOC 2012 tarzı all-point interpolation ile AP. Precision zarfını
    (sağdan sola monotonik non-increasing) çıkarıp recall ekseni boyunca
    integral alır — standart, dedupe/threshold seçimi gerektirmeyen yöntem."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _precision_recall_curve(scored: list, total_gt: int):
    """scored: [(score, is_tp), ...] — herhangi bir sırada. Skora göre azalan
    sıraya dizip kümülatif precision/recall eğrisini döndürür."""
    if total_gt == 0 or not scored:
        return np.array([]), np.array([])
    scored = sorted(scored, key=lambda x: -x[0])
    tp = np.array([1.0 if is_tp else 0.0 for _, is_tp in scored])
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall    = cum_tp / total_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)
    return recall, precision


def match_for_ap(preds: list[dict], gts: list[dict], iou_thresh: float,
                  scored_by_class: dict, total_gt_by_class: dict,
                  iou_fn=quad_iou, best_iou_by_class: dict | None = None) -> None:
    """AP bookkeeping — tek bir görüntünün tahmin/GT'lerini `scored_by_class`
    ve `total_gt_by_class`'a (yerinde) ekler. match_predictions'tan farkı:
    eşleştirme IoU'ya göre değil, HER SINIF İÇİNDE SKORA göre önceliklendirilir
    (yüksek skorlu tahmin GT'yi önce seçer) — standart AP algoritmasının
    gerektirdiği şey budur, çünkü nihai AP tüm dataset genelinde skora göre
    sıralanmış bir liste üzerinden hesaplanacak."""
    for g in gts:
        if g["label"] in total_gt_by_class:
            total_gt_by_class[g["label"]] += 1

    preds_by_label: dict = {}
    for p in preds:
        preds_by_label.setdefault(p["label"], []).append(p)

    for label, label_preds in preds_by_label.items():
        if label not in scored_by_class:
            continue
        label_gts = [g for g in gts if g["label"] == label]
        matched_gt = set()
        for p in sorted(label_preds, key=lambda x: -x["score"]):
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(label_gts):
                if gi in matched_gt:
                    continue
                iou = iou_fn(p, g)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            is_tp = best_iou >= iou_thresh
            if is_tp:
                matched_gt.add(best_gi)
            scored_by_class[label].append((p["score"], is_tp))
            if best_iou_by_class is not None:
                best_iou_by_class[label].append(best_iou)


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate():
    # Load val filenames
    if not os.path.exists(VAL_SPLIT):
        raise FileNotFoundError(
            f"{VAL_SPLIT} not found. Run training first to generate the val split."
        )
    with open(VAL_SPLIT) as f:
        val_filenames = json.load(f)["filenames"]

    print(f"Evaluating on {len(val_filenames)} val images  |  IoU threshold: {IOU_THRESHOLD}")

    model = load_model()

    # Accumulators
    all_tp, all_fp, all_fn = [], [], []

    # Per-class: {label: {tp, fp, fn, iou_sum, count}}
    class_stats = {label: {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
                   for label in LABEL_TO_ID}

    # AP bookkeeping: her IoU varyantı için ayrı (score, is_tp) listeleri.
    # total_gt_by_class varyanttan bağımsız olduğu için tek kopya yeterli, ancak
    # match_for_ap onu her çağrıda artırdığından varyant başına ayrı tutuluyor.
    scored_by_variant   = {v: {label: [] for label in LABEL_TO_ID} for v in IOU_VARIANTS}
    total_gt_by_variant = {v: {label: 0  for label in LABEL_TO_ID} for v in IOU_VARIANTS}
    best_iou_by_variant = {v: {label: [] for label in LABEL_TO_ID} for v in IOU_VARIANTS}

    n_preds_per_image = []

    for json_file in tqdm(val_filenames, desc="Evaluating", unit="img"):
        try:
            img, gts = load_ground_truth(json_file)
        except Exception as e:
            print(f"  Skipping {json_file}: {e}")
            continue

        preds = predict(model, img)
        n_preds_per_image.append(len(preds))

        # ── Tek eşikli hard P/R/F1 (score'dan bağımsız, "modelin verdiği her
        #    şey kabul edilirse" senaryosu — mevcut rapor formatıyla uyumlu) ──
        tp, fp, fn = match_predictions(preds, gts, IOU_THRESHOLD)

        all_tp.extend(tp)
        all_fp.extend(fp)
        all_fn.extend(fn)

        for t in tp:
            l = t["gt_label"]
            class_stats[l]["tp"]      += 1
            class_stats[l]["iou_sum"] += t["iou"]

        for f_ in fp:
            l = f_["pred_label"]
            if l in class_stats:
                class_stats[l]["fp"] += 1

        for f_ in fn:
            l = f_["gt_label"]
            if l in class_stats:
                class_stats[l]["fn"] += 1

        # ── AP için skor-sıralı bookkeeping (aynı preds/gts, ikinci bir
        #    inference gerekmiyor) — üç IoU varyantı için ayrı ayrı ──
        for vname, iou_fn in IOU_VARIANTS.items():
            match_for_ap(preds, gts, IOU_THRESHOLD,
                         scored_by_variant[vname], total_gt_by_variant[vname],
                         iou_fn=iou_fn, best_iou_by_class=best_iou_by_variant[vname])

    # ── AP / mAP (IoU ≥ IOU_THRESHOLD, skora göre sıralı) — varyant başına ───
    ap_by_variant, mAP_by_variant = {}, {}
    for vname in IOU_VARIANTS:
        scored, total_gt = scored_by_variant[vname], total_gt_by_variant[vname]
        aps = {}
        for label in LABEL_TO_ID:
            recall, precision = _precision_recall_curve(scored[label], total_gt[label])
            aps[label] = _voc_ap(recall, precision) if total_gt[label] > 0 else 0.0
        ap_by_variant[vname] = aps
        with_gt = [l for l in LABEL_TO_ID if total_gt[l] > 0]
        mAP_by_variant[vname] = (sum(aps[l] for l in with_gt) / len(with_gt)) if with_gt else 0.0

    # Mevcut rapor formatı "quad (sıraya duyarlı)" varyantını kullanmaya devam
    # ediyor — deploy edilen davranış bu.
    primary          = "quad (sıraya duyarlı)"
    ap_per_class     = ap_by_variant[primary]
    total_gt_by_class = total_gt_by_variant[primary]
    mAP              = mAP_by_variant[primary]
    classes_with_gt  = [l for l in LABEL_TO_ID if total_gt_by_class[l] > 0]

    # ── Overall metrics ───────────────────────────────────────────────────────
    n_tp = len(all_tp)
    n_fp = len(all_fp)
    n_fn = len(all_fn)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0.0
    recall    = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mean_iou  = (sum(t["iou"] for t in all_tp) / n_tp) if n_tp > 0 else 0.0

    print("\n" + "═" * 55)
    print(f"  Overall Results  (IoU ≥ {IOU_THRESHOLD})")
    print("═" * 55)
    print(f"  TP: {n_tp:4d}   FP: {n_fp:4d}   FN: {n_fn:4d}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Mean IoU  : {mean_iou:.4f}  (matched predictions only)")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  Per-Class Breakdown  (IoU ≥ {IOU_THRESHOLD})")
    print("═" * 55)
    print(f"  {'Class':<20} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6} {'AP':>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 60)

    for label, s in sorted(class_stats.items()):
        tp_ = s["tp"];  fp_ = s["fp"];  fn_ = s["fn"]
        p_  = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        r_  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f_  = (2 * p_ * r_ / (p_ + r_)) if (p_ + r_) > 0 else 0.0
        miou = s["iou_sum"] / tp_ if tp_ > 0 else 0.0
        ap_  = ap_per_class[label]
        print(f"  {label:<20} {p_:>6.3f} {r_:>6.3f} {f_:>6.3f} {miou:>6.3f} {ap_:>6.3f} {tp_:>4} {fp_:>4} {fn_:>4}")

    # ── AP / mAP raporu ────────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print(f"  Average Precision (AP@{IOU_THRESHOLD}, skor-sıralı, all-point interpolation)")
    print("═" * 55)
    for label in sorted(LABEL_TO_ID):
        n_gt = total_gt_by_class[label]
        note = "" if n_gt > 0 else "  (val setinde GT yok, mAP'e dahil edilmedi)"
        print(f"  {label:<20} AP={ap_per_class[label]:.4f}   #GT={n_gt}{note}")
    print(f"\n  mAP@{IOU_THRESHOLD} ({len(classes_with_gt)}/{len(LABEL_TO_ID)} sınıf üzerinden): {mAP:.4f}")

    # ── TEŞHİS: hatanın kaynağı neresi? ───────────────────────────────────────
    print("\n" + "═" * 78)
    print("  TEŞHİS — aynı tahminler, üç farklı IoU tanımı")
    print("═" * 78)
    header = f"  {'Sınıf':<20}" + "".join(f"{v:>24}" for v in IOU_VARIANTS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in sorted(classes_with_gt):
        row = f"  {label:<20}" + "".join(
            f"{ap_by_variant[v][label]:>24.4f}" for v in IOU_VARIANTS)
        print(row)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'mAP':<20}" + "".join(f"{mAP_by_variant[v]:>24.4f}" for v in IOU_VARIANTS))

    strict, unordered, hbb = (mAP_by_variant[v] for v in IOU_VARIANTS)
    print()
    print(f"  Köşe SIRASI hatasının maliyeti : {unordered - strict:+.4f} mAP")
    print(f"    (aynı 4 nokta, sadece sıra normalize edildi — bu fark tamamen")
    print(f"     'papyon' polygonların IoU=0 sayılmasından geliyor)")
    print(f"  Köşe LOKALİZASYON açığı        : {hbb - unordered:+.4f} mAP")
    print(f"    (hbb, modelin ayrı token'larla ürettiği eksen-hizalı kutu —")
    print(f"     köşe sırasından tamamen bağımsız)")

    print()
    print(f"  Geçersiz (kendisiyle kesişen) polygon:")
    pt, gt_t = INVALID_STATS["pred_total"], INVALID_STATS["gt_total"]
    print(f"    tahminlerde : {INVALID_STATS['pred']:5d} / {pt:5d}  "
          f"({100*INVALID_STATS['pred']/max(pt,1):.1f}%)")
    print(f"    GT'lerde    : {INVALID_STATS['gt']:5d} / {gt_t:5d}  "
          f"({100*INVALID_STATS['gt']/max(gt_t,1):.1f}%)   "
          f"<- sıfır değilse ETİKETLERİN köşe sırası tutarsız demektir")

    print()
    print("  IoU dağılımı (sıra-bağımsız quad, her tahminin en iyi eşleşmesi):")
    allious = [i for l in classes_with_gt for i in best_iou_by_variant["quad (sıra-bağımsız)"][l]]
    buckets = [(0.0, 0.01, "0.00  (tam ıska)"), (0.01, 0.3, "0.01-0.30"),
               (0.3, 0.5, "0.30-0.50 (kıl payı)"), (0.5, 0.75, "0.50-0.75"),
               (0.75, 0.9, "0.75-0.90"), (0.9, 1.01, "0.90-1.00")]
    for lo, hi, tag in buckets:
        n = sum(1 for i in allious if lo <= i < hi)
        bar = "█" * int(40 * n / max(len(allious), 1))
        print(f"    {tag:<22} {n:5d}  {bar}")

    print()
    print(f"  Görüntü başına tahmin sayısı: {Counter(n_preds_per_image).most_common()}")
    print(f"    (7 dışındaki değerler decode'un raydan çıktığını gösterir)")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "iou_threshold": IOU_THRESHOLD,
        "overall": {
            "tp": n_tp, "fp": n_fp, "fn": n_fn,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(f1,        4),
            "mean_iou":  round(mean_iou,  4),
        },
        "per_class": {
            label: {
                "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
                "precision": round(s["tp"] / (s["tp"] + s["fp"]), 4) if (s["tp"] + s["fp"]) > 0 else 0.0,
                "recall":    round(s["tp"] / (s["tp"] + s["fn"]), 4) if (s["tp"] + s["fn"]) > 0 else 0.0,
                "mean_iou":  round(s["iou_sum"] / s["tp"], 4) if s["tp"] > 0 else 0.0,
                "ap":        round(ap_per_class[label], 4),
                "num_gt":    total_gt_by_class[label],
            }
            for label, s in class_stats.items()
        },
        "mAP": round(mAP, 4),
        "mAP_num_classes": len(classes_with_gt),
        "diagnostics": {
            "mAP_by_iou_variant": {v: round(mAP_by_variant[v], 4) for v in IOU_VARIANTS},
            "ap_by_iou_variant": {
                v: {l: round(ap_by_variant[v][l], 4) for l in classes_with_gt}
                for v in IOU_VARIANTS
            },
            "invalid_polygons": dict(INVALID_STATS),
            "preds_per_image": dict(Counter(n_preds_per_image)),
        },
    }

    out_path = "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("═" * 55)


if __name__ == "__main__":
    evaluate()
