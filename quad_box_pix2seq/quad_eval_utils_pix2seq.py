"""
Quadrilateral (oriented / perspective 4-corner) evaluation for the 9-token pix2seq model.

Why not torchmetrics: MeanAveragePrecision only understands axis-aligned boxes or
dense masks. A perspective-distorted document field is neither — its axis-aligned
envelope can be 40% larger than the quad itself, so HBB mAP is nearly blind to
corner quality, which is exactly what matters when the crop gets warped for OCR.

What this module reports:
  polygon mAP@[.50:.95] / @.50 / @.75   — exact convex-polygon IoU, COCO protocol
  per-class AP@.50                      — your 10 field classes are very imbalanced
  corner NME / PCK                      — localization precision, order-invariant
  self-intersection rate                — diagnostic for corner-ordering confusion
  HBB mAP@.50                           — envelope cross-check

Drop-in: no changes needed to training_pix2seq.py.

from quad_eval_utils_pix2seq import evaluate, MAX_SEQ_LEN
EVAL_EVERY, best_map = 5, 0.0
...
if (epoch + 1) % EVAL_EVERY == 0:
    res = evaluate(model, val_dataloader, MAX_SEQ_LEN, device)
    tqdm.write(f"  mAP {res['map']:.4f} | mAP50 {res['map_50']:.4f} "
               f"| HBB50 {res['hbb_map_50']:.4f} | NME {res['nme']:.4f}")
    if res["map"] > best_map:
        best_map = res["map"]
        torch.save(model.state_dict(), "pix2seq_best_map.pth")
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training_pix2seq_quad import (
    Pix2SeqModel, VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, NUM_CLASSES,
    NOISE_CLASS_TOKEN, NOISE_CLASS_ID, NUM_CLASS_SLOTS, NOISE_FILL_TO_MAX,
)

TOKENS_PER_OBJ = 9
MAX_SEQ_LEN    = 1 + TOKENS_PER_OBJ * MAX_OBJECTS + 1
IOU_THRS       = np.arange(0.50, 0.96, 0.05)
REC_THRS       = np.linspace(0.0, 1.0, 101)      # COCO 101-point interpolation
MIN_QUAD_AREA  = 4.0

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


# ─────────────────────────────────────────────────────────────────────────────
# POLYGON GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def poly_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def poly_signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def is_convex(p, eps=1e-9):
    """A self-intersecting ('bowtie') quad fails this — useful as a diagnostic."""
    n = len(p)
    crosses = []
    for i in range(n):
        a, b, c = p[i], p[(i + 1) % n], p[(i + 2) % n]
        crosses.append(np.cross(b - a, c - b))
    crosses = np.asarray(crosses, dtype=np.float64)
    return bool(np.all(crosses >= -eps) or np.all(crosses <= eps))


def _side(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _line_isect(p1, p2, a, b):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = a;  x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return (x2, y2)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _convex_clip(subject, clip):
    """Sutherland-Hodgman. `clip` must be convex; both are reoriented CCW."""
    if poly_signed_area(clip) < 0:
        clip = clip[::-1]
    out = [(float(q[0]), float(q[1])) for q in subject]
    n = len(clip)
    for i in range(n):
        if not out:
            return np.zeros((0, 2))
        a, b = clip[i], clip[(i + 1) % n]
        new = []
        for j in range(len(out)):
            cur, prev = out[j], out[j - 1]
            c_in = _side(a, b, cur) >= -1e-12
            p_in = _side(a, b, prev) >= -1e-12
            if c_in:
                if not p_in:
                    new.append(_line_isect(prev, cur, a, b))
                new.append(cur)
            elif p_in:
                new.append(_line_isect(prev, cur, a, b))
        out = new
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _iou_raster(p, q, res=192):
    """Fallback for non-convex/self-intersecting quads: rasterize on a shared grid."""
    import cv2
    lo = np.minimum(p.min(0), q.min(0))
    hi = np.maximum(p.max(0), q.max(0))
    span = np.maximum(hi - lo, 1e-6)
    sc   = (res - 1) / span

    mp = np.zeros((res, res), np.uint8)
    mq = np.zeros((res, res), np.uint8)
    cv2.fillPoly(mp, [np.round((p - lo) * sc).astype(np.int32)], 1)
    cv2.fillPoly(mq, [np.round((q - lo) * sc).astype(np.int32)], 1)

    inter = float(np.count_nonzero(mp & mq))
    union = float(np.count_nonzero(mp | mq))
    return inter / union if union > 0 else 0.0


def quad_iou(p, q):
    """
    Exact IoU between two quadrilaterals.
    shapely if available (handles any polygon), else convex clipping,
    else rasterized fallback for degenerate shapes.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if _HAS_SHAPELY:
        pp, qq = _ShapelyPolygon(p), _ShapelyPolygon(q)
        if not pp.is_valid:
            pp = pp.buffer(0)
        if not qq.is_valid:
            qq = qq.buffer(0)
        if pp.is_empty or qq.is_empty:
            return 0.0
        union = pp.union(qq).area
        return float(pp.intersection(qq).area / union) if union > 0 else 0.0

    if not (is_convex(p) and is_convex(q)):
        try:
            return _iou_raster(p, q)
        except ImportError:
            return 0.0

    inter = _convex_clip(p, q)
    if len(inter) < 3:
        return 0.0
    ia = poly_area(inter)
    ua = poly_area(p) + poly_area(q) - ia
    return float(ia / ua) if ua > 0 else 0.0


def quad_iou_matrix(preds, gts):
    m = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            m[i, j] = quad_iou(p, g)
    return m


def quad_to_hbb(q):
    q = np.asarray(q)
    return [float(q[:, 0].min()), float(q[:, 1].min()),
            float(q[:, 0].max()), float(q[:, 1].max())]


def hbb_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih   = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter    = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def corner_error(pred_q, gt_q):
    """
    Mean corner distance normalized by the GT diagonal, invariant to the corner
    ordering convention (tries all 4 cyclic shifts x both winding directions).
    Your LabelMe files are not canonically ordered, so a naive index-wise
    comparison would report large errors on perfectly correct quads.
    """
    p = np.asarray(pred_q, dtype=np.float64)
    g = np.asarray(gt_q,  dtype=np.float64)
    diag = float(np.linalg.norm(g.max(0) - g.min(0)))
    if diag < 1e-6:
        return float("nan")

    best = float("inf")
    for cand in (p, p[::-1]):
        for s in range(4):
            d = np.linalg.norm(np.roll(cand, s, axis=0) - g, axis=1).mean()
            best = min(best, float(d))
    return best / diag


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN DECODING
# ─────────────────────────────────────────────────────────────────────────────

def gt_seq_to_quads(seq):
    """
    Hedef dizisinden GT quadları okur.

    Neden ayrı bir fonksiyon: GT eskiden
    `[t for t in target if t not in (BOS,EOS,PAD)]` ile düzleştiriliyordu.
    Noise objelerinin hedefi [PAD x8, NOISE] olduğundan bu filtre 8 tokenı silip
    tek başına NOISE tokenını bırakır. Noise hep KUYRUKTA olduğu için gerçek
    nesnelerin hizası şu an tesadüfen korunuyor - ama bu kırılgan bir varsayım
    (noise araya girse ya da yarım bir obje kalsa sessizce kayar). Sabit stride
    ile okumak bunu garantiye alır.
    """
    toks = seq.tolist() if torch.is_tensor(seq) else list(seq)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    img_h, img_w = IMG_SIZE
    quads, labels = [], []

    for i in range(0, (len(toks) // TOKENS_PER_OBJ) * TOKENS_PER_OBJ, TOKENS_PER_OBJ):
        t = toks[i:i + TOKENS_PER_OBJ]
        cls_tok = t[8]
        if cls_tok in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break
        if cls_tok == NOISE_CLASS_TOKEN:
            continue                                    # sentetik obje GT değildir
        if not (NUM_BINS <= cls_tok < NUM_BINS + NUM_CLASSES):
            continue
        if not all(0 <= t[j] < NUM_BINS for j in range(8)):
            continue
        pts = np.array([[(t[j]     / (NUM_BINS - 1)) * img_w,
                         (t[j + 1] / (NUM_BINS - 1)) * img_h] for j in range(0, 8, 2)],
                       dtype=np.float64)
        if poly_area(pts) < MIN_QUAD_AREA:
            continue
        quads.append(pts)
        labels.append(int(cls_tok - NUM_BINS))
    return quads, labels


def seq_to_quads(tokens, tok_scores=None, noise_scores=None):
    """
    Flat token list → (quads [N,4,2] in model space, labels, scores).

    Skor tercihi:
      noise_scores verilmişse  -> p(sınıf) / (p(sınıf) + p(noise))  [kalibre objectness]
      sadece tok_scores varsa  -> 9 tokenın olasılık çarpımı        [eski davranış]

    Noise chunk'ları 9 tokenlık tam bir blok kapladığı için `continue` ile
    atlanır; stride sabit olduğundan hizalama bozulmaz.
    """
    img_h, img_w = IMG_SIZE
    quads, labels, scores = [], [], []

    n = len(tokens) - (len(tokens) % TOKENS_PER_OBJ)
    for i in range(0, n, TOKENS_PER_OBJ):
        t = tokens[i:i + TOKENS_PER_OBJ]

        if not all(0 <= t[j] < NUM_BINS for j in range(8)):
            continue
        if t[8] == NOISE_CLASS_TOKEN:          # model "burada nesne yok" dedi
            continue
        if not (NUM_BINS <= t[8] < NUM_BINS + NUM_CLASSES):
            continue

        pts = np.array(
            [[(t[j]     / (NUM_BINS - 1)) * img_w,
              (t[j + 1] / (NUM_BINS - 1)) * img_h] for j in range(0, 8, 2)],
            dtype=np.float64,
        )
        if poly_area(pts) < MIN_QUAD_AREA:
            continue

        quads.append(pts)
        labels.append(int(t[8] - NUM_BINS))
        if tok_scores is None:
            scores.append(1.0)
        elif noise_scores is not None:
            p_cls = float(tok_scores[i + 8])
            p_noi = float(noise_scores[i + 8])
            scores.append(p_cls / max(p_cls + p_noi, 1e-6))
        else:
            scores.append(float(np.prod(tok_scores[i:i + TOKENS_PER_OBJ])))

    return quads, labels, scores


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHED AUTOREGRESSIVE DECODE
# ─────────────────────────────────────────────────────────────────────────────
# nn.TransformerDecoder reprocesses the whole prefix every step. At
# max_seq_len = 272 that dominates eval time. This reuses the trained weights
# exactly — no retraining, no state_dict change.

class KVCacheDecoder:
    def __init__(self, model, memory):
        self.layers = model.decoder.layers
        self.norm   = model.decoder.norm
        self.self_k = [None] * len(self.layers)
        self.self_v = [None] * len(self.layers)
        self.cross_k, self.cross_v = [], []

        for layer in self.layers:
            attn = layer.multihead_attn
            D    = attn.embed_dim
            W, b = attn.in_proj_weight, attn.in_proj_bias
            bk = None if b is None else b[D:2 * D]
            bv = None if b is None else b[2 * D:]
            self.cross_k.append(self._split(F.linear(memory, W[D:2 * D], bk), attn.num_heads))
            self.cross_v.append(self._split(F.linear(memory, W[2 * D:],  bv), attn.num_heads))

    @staticmethod
    def _split(x, nheads):
        B, L, D = x.shape
        return x.view(B, L, nheads, D // nheads).transpose(1, 2)

    @staticmethod
    def _merge(x):
        B, H, L, hd = x.shape
        return x.transpose(1, 2).reshape(B, L, H * hd)

    def _self_attn(self, layer, i, x):
        a = layer.self_attn
        D = a.embed_dim
        q, k, v = F.linear(x, a.in_proj_weight, a.in_proj_bias).split(D, dim=-1)
        q = self._split(q, a.num_heads)
        k = self._split(k, a.num_heads)
        v = self._split(v, a.num_heads)
        self.self_k[i] = k if self.self_k[i] is None else torch.cat([self.self_k[i], k], 2)
        self.self_v[i] = v if self.self_v[i] is None else torch.cat([self.self_v[i], v], 2)
        o = F.scaled_dot_product_attention(q, self.self_k[i], self.self_v[i])
        return a.out_proj(self._merge(o))

    def _cross_attn(self, layer, i, x):
        a  = layer.multihead_attn
        D  = a.embed_dim
        bq = None if a.in_proj_bias is None else a.in_proj_bias[:D]
        q  = self._split(F.linear(x, a.in_proj_weight[:D], bq), a.num_heads)
        o  = F.scaled_dot_product_attention(q, self.cross_k[i], self.cross_v[i])
        return a.out_proj(self._merge(o))

    @staticmethod
    def _ff(layer, x):
        act = layer.activation
        if isinstance(act, str):
            act = F.relu if act == "relu" else F.gelu
        return layer.linear2(layer.dropout(act(layer.linear1(x))))

    def step(self, x):
        for i, layer in enumerate(self.layers):
            if getattr(layer, "norm_first", False):
                x = x + layer.dropout1(self._self_attn(layer, i, layer.norm1(x)))
                x = x + layer.dropout2(self._cross_attn(layer, i, layer.norm2(x)))
                x = x + layer.dropout3(self._ff(layer, layer.norm3(x)))
            else:
                x = layer.norm1(x + layer.dropout1(self._self_attn(layer, i, x)))
                x = layer.norm2(x + layer.dropout2(self._cross_attn(layer, i, x)))
                x = layer.norm3(x + layer.dropout3(self._ff(layer, x)))
        return x if self.norm is None else self.norm(x)


@torch.no_grad()
def ar_decode_batch(model, images, max_seq_len=MAX_SEQ_LEN, device=None, use_cache=True,
                    fixed_length=NOISE_FILL_TO_MAX):
    """
    Batch greedy decode. Returns (tokens [B,L], per-token probs [B,L], p(noise) [B,L]).

    fixed_length=True (sequence augmentation ile eğitilmiş model):
      * tam 9 * MAX_OBJECTS adım çözülür, EOS beklenmez - model zaten EOS
        görmedi, "dur" kararını her slotta noise sınıfını seçerek veriyor
      * kısıtlı örnekleme: koordinat adımlarında sadece [0, NUM_BINS),
        sınıf adımlarında sadece sınıf tokenları seçilebilir. Böylece 9'luk
        hizalama hiçbir koşulda kaymaz.
      * erken EOS yüzünden nesne kaçırma problemi ortadan kalkar

    fixed_length=False: eski EOS'ta duran davranış (noise'suz checkpointler için).
    """
    model.eval()
    device = device or next(model.parameters()).device
    B      = images.size(0)

    memory = model.enc_proj(model.encoder(images)).flatten(2).permute(0, 2, 1) + model.pos_emb
    cache  = KVCacheDecoder(model, memory) if use_cache else None

    n_steps  = TOKENS_PER_OBJ * MAX_OBJECTS if fixed_length else max_seq_len - 1
    seq      = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    scores   = torch.zeros(B, n_steps + 1, device=device)
    noise_p  = torch.zeros(B, n_steps + 1, device=device)
    pe       = model.seq_pos_encoding.pe

    cls_lo, cls_hi = NUM_BINS, NUM_BINS + NUM_CLASS_SLOTS

    for step in range(n_steps):
        if use_cache:
            emb = model.embedding(seq[:, -1:]) + pe[:, step:step + 1, :]
            out = cache.step(model.emb_dropout(emb))
        else:
            emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(seq)))
            mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
            out  = model.decoder(tgt=emb, memory=memory, tgt_mask=mask)

        probs = torch.softmax(model.fc_out(out[:, -1, :]).float(), -1)

        if fixed_length:
            if step % TOKENS_PER_OBJ == TOKENS_PER_OBJ - 1:      # sınıf pozisyonu
                sub = probs[:, cls_lo:cls_hi]
                nxt = sub.argmax(-1) + cls_lo
            else:                                                # koordinat pozisyonu
                nxt = probs[:, :NUM_BINS].argmax(-1)
        else:
            nxt = probs.argmax(-1)

        scores[:, step]  = probs.gather(1, nxt.unsqueeze(1)).squeeze(1)
        noise_p[:, step] = probs[:, NOISE_CLASS_TOKEN]

        if not fixed_length:
            nxt = torch.where(finished, torch.full_like(nxt, PAD_TOKEN), nxt)
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)

        if not fixed_length:
            finished |= (nxt == EOS_TOKEN)
            if finished.all():
                break

    L = seq.size(1) - 1
    return seq[:, 1:], scores[:, :L], noise_p[:, :L]


def _ap_from_pr(tp, fp, n_gt):
    """101-point interpolated AP (COCO protocol)."""
    if n_gt == 0:
        return float("nan")
    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    rec  = tp_c / n_gt
    prec = tp_c / np.maximum(tp_c + fp_c, 1e-12)

    # make precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])

    idx = np.searchsorted(rec, REC_THRS, side="left")
    q   = np.zeros(len(REC_THRS))
    valid = idx < len(prec)
    q[valid] = prec[idx[valid]]
    return float(q.mean())


def compute_ap(records, iou_fn, iou_thrs=IOU_THRS):
    """
    records: list of per-image dicts
        {"pred": [geom], "pred_label": [int], "pred_score": [float],
         "gt": [geom],   "gt_label": [int]}
    Returns (ap_matrix [n_class, n_thr], per-class GT counts).
    """
    classes = sorted(set(
        [l for r in records for l in r["gt_label"]] +
        [l for r in records for l in r["pred_label"]]
    ))
    ap  = np.full((len(classes), len(iou_thrs)), np.nan)
    cnt = {}

    # cache IoU matrices per (image, class) so we don't recompute per threshold
    for ci, cls in enumerate(classes):
        entries, n_gt = [], 0
        for img_i, r in enumerate(records):
            pi = [i for i, l in enumerate(r["pred_label"]) if l == cls]
            gi = [i for i, l in enumerate(r["gt_label"])  if l == cls]
            n_gt += len(gi)
            if not pi:
                continue
            ious = np.zeros((len(pi), len(gi)))
            for a, i in enumerate(pi):
                for b, j in enumerate(gi):
                    ious[a, b] = iou_fn(r["pred"][i], r["gt"][j])
            for a, i in enumerate(pi):
                entries.append((r["pred_score"][i], img_i, a, ious[a], len(gi)))

        cnt[cls] = n_gt
        if n_gt == 0 or not entries:
            continue

        entries.sort(key=lambda e: -e[0])
        for ti, thr in enumerate(iou_thrs):
            matched = {}
            tp = np.zeros(len(entries)); fp = np.zeros(len(entries))
            for k, (_, img_i, _, ious_row, n_g) in enumerate(entries):
                used = matched.setdefault(img_i, set())
                best_j, best_iou = -1, thr
                for j in range(n_g):
                    if j in used or ious_row[j] < best_iou:
                        continue
                    best_j, best_iou = j, ious_row[j]
                if best_j >= 0:
                    used.add(best_j); tp[k] = 1
                else:
                    fp[k] = 1
            ap[ci, ti] = _ap_from_pr(tp, fp, n_gt)

    return classes, ap, cnt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, max_seq_len=MAX_SEQ_LEN, device=None,
             use_cache=True, match_iou=0.5, verbose=False):
    device = device or next(model.parameters()).device
    model.eval()

    records = []
    n_self_intersect = 0
    n_pred_total     = 0
    corner_errs      = []

    for images, _seq_in, targets in loader:
        images = images.to(device).float()
        seqs, scs, nps = ar_decode_batch(model, images, max_seq_len, device, use_cache=use_cache)

        for b in range(images.size(0)):
            toks = seqs[b].tolist()
            if EOS_TOKEN in toks:
                toks = toks[:toks.index(EOS_TOKEN)]
            toks = [t for t in toks if t not in (BOS_TOKEN, PAD_TOKEN)]
            pq, pl, ps = seq_to_quads(toks,
                                      scs[b].float().cpu().numpy(),
                                      nps[b].float().cpu().numpy())

            gq, gl = gt_seq_to_quads(targets[b])

            n_pred_total     += len(pq)
            n_self_intersect += sum(0 if is_convex(q) else 1 for q in pq)

            records.append({
                "pred": pq, "pred_label": pl, "pred_score": ps,
                "gt": gq, "gt_label": gl,
            })

            # corner metrics on greedy 1-1 matches (class-aware, IoU >= match_iou)
            if pq and gq:
                order = np.argsort(-np.asarray(ps))
                used  = set()
                for i in order:
                    best_j, best_iou = -1, match_iou
                    for j, g in enumerate(gq):
                        if j in used or gl[j] != pl[i]:
                            continue
                        v = quad_iou(pq[i], g)
                        if v >= best_iou:
                            best_j, best_iou = j, v
                    if best_j >= 0:
                        used.add(best_j)
                        e = corner_error(pq[i], gq[best_j])
                        if not np.isnan(e):
                            corner_errs.append(e)

    classes, ap_poly, cnt = compute_ap(records, quad_iou)

    hbb_records = [{
        "pred": [quad_to_hbb(q) for q in r["pred"]], "pred_label": r["pred_label"],
        "pred_score": r["pred_score"],
        "gt": [quad_to_hbb(q) for q in r["gt"]], "gt_label": r["gt_label"],
    } for r in records]
    _, ap_hbb, _ = compute_ap(hbb_records, hbb_iou, iou_thrs=np.array([0.5]))

    ce = np.asarray(corner_errs) if corner_errs else np.array([np.nan])
    res = {
        "map":       float(np.nanmean(ap_poly))            if ap_poly.size else float("nan"),
        "map_50":    float(np.nanmean(ap_poly[:, 0]))      if ap_poly.size else float("nan"),
        "map_75":    float(np.nanmean(ap_poly[:, 5]))      if ap_poly.size else float("nan"),
        "hbb_map_50": float(np.nanmean(ap_hbb))            if ap_hbb.size else float("nan"),
        "nme":       float(np.nanmean(ce)),
        "nme_median": float(np.nanmedian(ce)),
        "pck_005":   float(np.mean(ce <= 0.05)) if corner_errs else float("nan"),
        "pck_010":   float(np.mean(ce <= 0.10)) if corner_errs else float("nan"),
        "self_intersect_rate": (n_self_intersect / n_pred_total) if n_pred_total else 0.0,
        "n_pred": n_pred_total,
        "n_matched": len(corner_errs),
        "per_class_ap50": {ID_TO_LABEL.get(c, str(c)): float(ap_poly[i, 0])
                           for i, c in enumerate(classes)},
        "per_class_gt": {ID_TO_LABEL.get(c, str(c)): cnt.get(c, 0) for c in classes},
    }

    if verbose:
        print_report(res)
    return res


def print_report(res):
    print(f"\n{'':-<58}")
    print(f"  polygon mAP@[.50:.95] : {res['map']:.4f}")
    print(f"  polygon mAP@.50       : {res['map_50']:.4f}")
    print(f"  polygon mAP@.75       : {res['map_75']:.4f}")
    print(f"  HBB envelope mAP@.50  : {res['hbb_map_50']:.4f}   <- gap vs polygon = corner error")
    print(f"{'':-<58}")
    print(f"  corner NME (mean/med) : {res['nme']:.4f} / {res['nme_median']:.4f}  "
          f"(fraction of GT diagonal)")
    print(f"  PCK@0.05 / @0.10      : {res['pck_005']:.3f} / {res['pck_010']:.3f}")
    print(f"  self-intersecting     : {res['self_intersect_rate']:.3f} "
          f"({res['n_pred']} preds, {res['n_matched']} matched)")
    print(f"{'':-<58}")
    print("  per-class AP@.50:")
    for k, v in sorted(res["per_class_ap50"].items(), key=lambda kv: -kv[1]):
        n = res["per_class_gt"].get(k, 0)
        print(f"    {k:<20} {v:.4f}   (n_gt={n})")
    print(f"{'':-<58}\n")


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from training_pix2seq_quad import Pix2SeqDataset, JSON_DIR, IMG_DIR, val_transform, VAL_SPLIT_PATH

    MODEL_PATH = "pix2seq_best.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not _HAS_SHAPELY:
        print("[warn] shapely not installed — using built-in convex clipping "
              "(fine for quads, `pip install shapely` for full generality)")

    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    with open(VAL_SPLIT_PATH) as f:
        val_files = json.load(f)["filenames"]

    val_ds = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform, is_train=False)
    val_ds.json_files = val_files
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

    print(f"Evaluating {MODEL_PATH} on {len(val_ds)} images...")
    evaluate(model, val_dl, MAX_SEQ_LEN, device, verbose=True)
