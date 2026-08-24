import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.detection import MeanAveragePrecision

from training_pix2seq_bbox import (
    Pix2SeqModel, VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, TOKENS_PER_OBJ, NUM_CLASSES,
    NOISE_CLASS_TOKEN, NOISE_CLASS_ID, NUM_CLASS_SLOTS, NOISE_FILL_TO_MAX,
    INFER_SCORE_THRESH,
)

MAX_SEQ_LEN = 1 + TOKENS_PER_OBJ * MAX_OBJECTS


# ─────────────────────────────────────────────────────────────────────────────
# KV-CACHE'Lİ DECODER
# ─────────────────────────────────────────────────────────────────────────────
# nn.TransformerDecoder incremental decode desteklemiyor; cache olmadan her adım
# tüm diziyi baştan işliyor -> O(L^2). Sabit uzunlukta çözdüğümüz için (EOS'ta
# erken durmuyoruz) bu maliyet artık her görüntüde ödeniyor, cache şart.

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
def ar_decode_batch(model, images, max_seq_len=MAX_SEQ_LEN, device=None,
                    use_cache=True, fixed_length=NOISE_FILL_TO_MAX):
    """
    Returns (tokens [B,L], per-token probs [B,L], p(noise) [B,L]).

    fixed_length=True (sequence augmentation ile eğitilmiş model):
      * tam TOKENS_PER_OBJ * MAX_OBJECTS adım çözülür, EOS beklenmez — model
        zaten EOS görmedi, "dur" kararını her slotta noise seçerek veriyor
      * kısıtlı örnekleme: koordinat adımlarında sadece [0, NUM_BINS), sınıf
        adımlarında sadece sınıf tokenları. 5'li hizalama hiçbir koşulda kaymaz.
    fixed_length=False: eski EOS'ta duran davranış (noise'suz checkpointler için).
    """
    model.eval()
    device = device or next(model.parameters()).device
    B = images.size(0)

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
                nxt = probs[:, cls_lo:cls_hi].argmax(-1) + cls_lo
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


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN -> KUTU
# ─────────────────────────────────────────────────────────────────────────────

def seq_to_boxes(tokens, tok_scores=None, noise_scores=None, score_thresh=0.0):
    """
    Skor tercihi:
      noise_scores verilmişse -> p(sınıf) / (p(sınıf) + p(noise))  [kalibre objectness]
      sadece tok_scores varsa -> 5 tokenın olasılık çarpımı        [eski davranış]

    Noise chunk'ı da tam 5 token kapladığı için `continue` ile atlanır; stride
    sabit olduğundan hizalama bozulmaz.
    """
    boxes, labels, confs = [], [], []
    n = len(tokens) - (len(tokens) % TOKENS_PER_OBJ)

    for i in range(0, n, TOKENS_PER_OBJ):
        t = tokens[i:i + TOKENS_PER_OBJ]
        if t[4] == NOISE_CLASS_TOKEN:          # model "burada nesne yok" dedi
            continue
        if not all(0 <= t[j] < NUM_BINS for j in range(4)):
            continue
        if not (NUM_BINS <= t[4] < NUM_BINS + NUM_CLASSES):
            continue

        x0, y0, x1, y1 = [(t[j] / (NUM_BINS - 1)) * IMG_SIZE[0] for j in range(4)]
        if x1 <= x0 or y1 <= y0:
            continue

        if tok_scores is None:
            conf = 1.0
        elif noise_scores is not None:
            p_cls = float(tok_scores[i + 4]); p_noi = float(noise_scores[i + 4])
            conf = p_cls / max(p_cls + p_noi, 1e-6)
        else:
            conf = float(np.prod(tok_scores[i:i + TOKENS_PER_OBJ]))

        if conf < score_thresh:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(t[4] - NUM_BINS)
        confs.append(conf)

    return boxes, labels, confs


def gt_seq_to_boxes(seq):
    """
    Hedef dizisinden GT kutuları okur.

    Neden ayrı: GT eskiden `[t for t in target if t not in (BOS,EOS,PAD)]` ile
    düzleştiriliyordu. Noise hedefi [PAD x4, NOISE] olduğu için bu filtre 4 tokenı
    silip tek başına NOISE tokenını bırakır. Noise hep kuyrukta olduğundan gerçek
    nesnelerin hizası şu an tesadüfen korunuyor — ama kırılgan bir varsayım.
    Sabit stride ile okumak bunu garantiye alır.
    """
    toks = seq.tolist() if torch.is_tensor(seq) else list(seq)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    boxes, labels = [], []

    for i in range(0, (len(toks) // TOKENS_PER_OBJ) * TOKENS_PER_OBJ, TOKENS_PER_OBJ):
        t = toks[i:i + TOKENS_PER_OBJ]
        cls_tok = t[4]
        if cls_tok in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN):
            break
        if cls_tok == NOISE_CLASS_TOKEN:
            continue                                    # sentetik obje GT değildir
        if not (NUM_BINS <= cls_tok < NUM_BINS + NUM_CLASSES):
            continue
        if not all(0 <= t[j] < NUM_BINS for j in range(4)):
            continue
        x0, y0, x1, y1 = [(t[j] / (NUM_BINS - 1)) * IMG_SIZE[0] for j in range(4)]
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(int(cls_tok - NUM_BINS))

    return boxes, labels


@torch.no_grad()
def evaluate(model, loader, max_seq_len=MAX_SEQ_LEN, device=None, score_thresh=0.0):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()
    device = device or next(model.parameters()).device

    for images, _seq_in, targets in loader:
        images = images.to(device).float()
        seqs, scs, nps = ar_decode_batch(model, images, max_seq_len, device)

        for b in range(images.size(0)):
            toks = seqs[b].tolist()
            if EOS_TOKEN in toks:
                toks = toks[:toks.index(EOS_TOKEN)]
            pb, pl, pc = seq_to_boxes(toks,
                                      scs[b].float().cpu().numpy(),
                                      nps[b].float().cpu().numpy(),
                                      score_thresh=score_thresh)
            gb, gl = gt_seq_to_boxes(targets[b])

            metric.update(
                [{"boxes":  torch.tensor(pb).reshape(-1, 4),
                  "scores": torch.tensor(pc).reshape(-1),
                  "labels": torch.tensor(pl, dtype=torch.long).reshape(-1)}],
                [{"boxes":  torch.tensor(gb).reshape(-1, 4),
                  "labels": torch.tensor(gl, dtype=torch.long).reshape(-1)}],
            )

    res = metric.compute()
    return float(res["map"]), float(res["map_50"])
