import numpy as np
import torch 
import torch.nn as nn
from torchmetrics.detection import MeanAveragePrecision
from training_pix2seq_bbox  import (
    Pix2SeqModel, VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, 
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS, TOKENS_PER_OBJ, PAD_TOKEN, NUM_CLASSES
)

@torch.no_grad()
def ar_decode_batch(model, images, max_seq_len, device):
    """Greedy AR decode for a batch. Encoder runs once."""
    model.eval()
    B = images.size(0)

    feats  = model.enc_proj(model.encoder(images))
    memory = feats.flatten(2).permute(0, 2, 1) + model.pos_emb

    seq      = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    scores   = torch.zeros(B, max_seq_len, device=device)

    for step in range(max_seq_len - 1):
        tgt_emb  = model.emb_dropout(model.seq_pos_encoding(model.embedding(seq)))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
        out      = model.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits   = model.fc_out(out[:, -1, :]).float()

        probs   = torch.softmax(logits, -1)
        nxt     = probs.argmax(-1)
        scores[:, step] = probs.gather(1, nxt.unsqueeze(1)).squeeze(1)

        nxt = torch.where(finished, torch.full_like(nxt, PAD_TOKEN), nxt)
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)

        finished |= (nxt == EOS_TOKEN)
        if finished.all():
            break

    return seq[:, 1:], scores




def seq_to_boxes(tokens, tok_scores):
    boxes, labels, confs = [], [], []
    n = len(tokens) - (len(tokens) % 5)
    for i in range(0, n, 5):
        t = tokens[i:i+5]
        if not all(0 <= t[j] < NUM_BINS for j in range(4)):
            continue
        if not (NUM_BINS <= t[4] < NUM_BINS + NUM_CLASSES):
            continue
        x0, y0, x1, y1 = [(t[j] / (NUM_BINS - 1)) * IMG_SIZE[0] for j in range(4)]
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append([x0, y0, x1, y1])
        labels.append(t[4] - NUM_BINS)
        confs.append(float(np.prod(tok_scores[i:i+5])))
    return boxes, labels, confs



@torch.no_grad()
def evaluate(model, loader, max_seq_len, device):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    model.eval()

    for images, targets in loader:
        images = images.to(device).float()
        seqs, scs = ar_decode_batch(model, images, max_seq_len, device)

        for b in range(images.size(0)):
            toks = seqs[b].tolist()
            toks = toks[:toks.index(EOS_TOKEN)] if EOS_TOKEN in toks else toks
            pb, pl, pc = seq_to_boxes(toks, scs[b].cpu().numpy())

            gt = [t for t in targets[b].tolist()
                  if t not in (BOS_TOKEN, EOS_TOKEN, PAD_TOKEN)]
            gb, gl, _ = seq_to_boxes(gt, np.ones(len(gt)))

            metric.update(
                [{"boxes":  torch.tensor(pb).reshape(-1, 4),
                  "scores": torch.tensor(pc).reshape(-1),
                  "labels": torch.tensor(pl, dtype=torch.long).reshape(-1)}],
                [{"boxes":  torch.tensor(gb).reshape(-1, 4),
                  "labels": torch.tensor(gl, dtype=torch.long).reshape(-1)}],
            )

    res = metric.compute()
    return float(res["map"]), float(res["map_50"])