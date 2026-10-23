"""
Pix2Seq quad — HİYERARŞİK (obje seviyesi AR + lokal AR decoder)  [Varyant B]
==============================================================================
Vanilla (pix2seq_florence_quad.py):
    tek decoder, token seviyesinde causal, dizi uzunluğu 2 + 9N
Bu dosya:
    OUTER decoder : obje seviyesinde causal, dizi uzunluğu N+1
                    girdi[i] = (i-1). objenin 9 token'ının tek vektöre sıkıştırılmış hali
    LOCAL decoder : her obje için 9 token'ı (sınıf + 8 loc) kendi içinde causal üretir,
                    prefix = outer hidden h_i, görüntü memory'sine cross-attention yapar
    -> obje içi bağımlılık TAM korunur (köşeler birbirine koşullu), ağır decoder N kez çalışır.

Tokenizer, dataset, augmentation, quad IoU / AP fonksiyonları vanilla dosyadan import edilir;
vanilla dosyada hiçbir değişiklik gerekmez.

Hedef düzeni (obje başına satır, 9 token):
    satır i<N : [cls_tok, loc×8]          (vanilla ile aynı token id'leri)
    satır N   : [</s>, <pad>×8]           (EOS satırı: sadece slot-0 loss'u)
    <pad> satırları: loss dışı
Head'ler ayrık: slot 0 -> cls_head (NUM_CLASSES + EOS), slot 1..8 -> loc_head (NUM_BINS)
=> gramer maskesine gerek yok, label smoothing güvenli.
"""
import os
import json
import math
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import torchvision.models as models
from tqdm import tqdm

import pix2seq_florence_quad as base
from pix2seq_florence_quad import (
    JSON_DIR, IMG_DIR, MAX_OBJECTS, BATCH_SIZE, EPOCHS, LEARNING_RATE, IMG_SIZE,
    VAL_SPLIT_PATH, MOSAIC_PROB, EVAL_EVERY, INFER_SCORE_THRESH,
    NUM_CLASSES, ID_TO_LABEL, NUM_BINS, BLOCK,
    BOS_ID, EOS_ID, PAD_ID, TASK_ID, LOC_START, CLASS_START, VOCAB_SIZE,
    PositionalEncoding, Pix2SeqDataset, train_transform, val_transform,
    load_image, pad_to_square, dequantize, decode_tokens, draw_predictions,
    quad_iou, _average_precision, dataset_object_stats,
)

# =============================================================================
# CONFIG (hiyerarşik modele özel)
# =============================================================================
HIDDEN_DIM      = 256
NHEADS          = 8
OUTER_LAYERS    = 4          # vanilla decoder ile aynı derinlik -> warm start mümkün
LOCAL_LAYERS    = 2          # lokal decoder küçük tutulur
N_LOC           = BLOCK - 1  # 8 koordinat token'ı
N_FOURIER       = 8          # obje embedding'ine eklenen sürekli koordinat kodlaması
LOC_WEIGHT      = 1.0        # loss = L_cls + LOC_WEIGHT * L_loc
DRAFT_WEIGHT    = 1.0        # taslak head'i loss ağırlığı (0 -> eğitme). Ana modeli etkilemez.
LABEL_SMOOTHING = 0.1
MAX_OBJ_SEQ     = MAX_OBJECTS + 1            # +1: EOS satırı

# Vanilla checkpoint'ten başlatmak için (None -> ImageNet ResNet50'den başla)
WARM_START_CKPT = None       # örn. "pix2seq_florence_best_map.pth"

CKPT_BEST_LOSS = "pix2seq_hier_best_loss.pth"
CKPT_BEST_MAP  = "pix2seq_hier_best_map.pth"


# =============================================================================
# COLLATE (obje satırları)
# =============================================================================
def collate_fn_hier(batch):
    """
    Dataset çıktısı (vanilla ile aynı) -> obje satırları.
        pixel_values : (B,3,H,W)
        objects      : (B, Nmax+1, 9)  hedef satırları (son geçerli satır = EOS satırı)
        valid        : (B, Nmax+1)     bool, loss alınacak satırlar (objeler + EOS satırı)
    """
    pixel_values = torch.stack([b["image"] for b in batch])
    n_objs = [len(b["token_ids"]) // BLOCK for b in batch]
    n_max = max(n_objs)
    objects = torch.full((len(batch), n_max + 1, BLOCK), PAD_ID, dtype=torch.long)
    valid = torch.zeros((len(batch), n_max + 1), dtype=torch.bool)
    for i, (b, n) in enumerate(zip(batch, n_objs)):
        if n:
            objects[i, :n] = torch.tensor(b["token_ids"], dtype=torch.long).view(n, BLOCK)
        objects[i, n, 0] = EOS_ID
        valid[i, :n + 1] = True
    return {"pixel_values": pixel_values, "objects": objects, "valid": valid}


# =============================================================================
# MODEL
# =============================================================================
class FourierCoords(nn.Module):
    """8 normalize koordinat -> sin/cos -> d. Bin embedding'inin taşıyamadığı 'yakınlık' bilgisi."""

    def __init__(self, d, n_coord=N_LOC, n_freq=N_FOURIER):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freq)) * math.pi, persistent=False)
        self.proj = nn.Linear(n_coord * n_freq * 2, d)

    def forward(self, coords01):                       # (..., 8) in [0,1]
        x = coords01[..., None] * self.freqs           # (..., 8, F)
        return self.proj(torch.cat([x.sin(), x.cos()], -1).flatten(-2))


def drop_fixed_pe(model, state_dict):
    """
    Sinüzoidal PositionalEncoding.pe öğrenilmez; değeri yalnızca pozisyona bağlıdır, max_len'e
    değil. MAX_OBJECTS değişince boyutu değişir ama ortak kısmı birebir aynıdır. Boyutu tutmayan
    pe'leri checkpoint'ten çıkarıp modelin kendi (aynı değerli) tablosunu kullanırız.
    """
    own = model.state_dict()
    bad = [k for k, v in state_dict.items()
           if k.endswith("pos_encoding.pe") and k in own and own[k].shape != v.shape]
    if not bad:
        return state_dict
    sd = dict(state_dict)
    for k in bad:
        n = min(sd[k].shape[1], own[k].shape[1])
        assert torch.allclose(sd[k][:, :n].float(), own[k][:, :n].float(), atol=1e-5), \
            f"{k}: checkpoint'teki pe sinüzoidal değil, atlanamaz"
        sd[k] = own[k]
    return sd


def load_checkpoint(model, path):
    """Hem Pix2SeqModel hem Pix2SeqHierModel için: pe boyutu farkını tolere ederek yükler."""
    sd = drop_fixed_pe(model, torch.load(path, map_location="cpu"))
    model.load_state_dict(sd)
    return model


class DraftHead(nn.Module):
    """
    Hız için taslak üretici (Medusa / Varyant A tarzı): h + görüntü -> 9 token TEK forward'da.
    Doğruluğu etkilemez: generate_hier_fast taslağı local decoder ile doğrular, sonuç greedy
    AR ile aynı kalır. Taslak ne kadar iyiyse o kadar az Jacobi iterasyonu gerekir.
    """

    def __init__(self, d, nheads, n_layers=LOCAL_LAYERS):
        super().__init__()
        self.query = nn.Parameter(torch.randn(BLOCK, d) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nheads, batch_first=True, dropout=0.0), n_layers)
        self.cls_head = nn.Linear(d, NUM_CLASSES + 1)
        self.loc_head = nn.Linear(d, NUM_BINS)

    def forward(self, h, memory):
        out = self.decoder(tgt=h[:, None] + self.query, memory=memory)   # maske yok: paralel
        return self.cls_head(out[:, 0]), self.loc_head(out[:, 1:])


class Pix2SeqHierModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, nheads=NHEADS,
                 outer_layers=OUTER_LAYERS, local_layers=LOCAL_LAYERS, max_obj_seq=MAX_OBJ_SEQ):
        super().__init__()
        d = hidden_dim
        # ── Encoder (vanilla ile aynı isimler -> state_dict uyumlu) ──────────
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.enc_proj = nn.Conv2d(2048, d, kernel_size=1)
        grid_h, grid_w = IMG_SIZE[0] // 32, IMG_SIZE[1] // 32
        self.pos_emb = nn.Parameter(torch.randn(1, grid_h * grid_w, d))

        # ── Paylaşılan token embedding (vanilla ile aynı isim) ──────────────
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=PAD_ID)

        # ── Obje embedding: 9 token -> 1 vektör ─────────────────────────────
        self.obj_slot_emb = nn.Parameter(torch.randn(BLOCK, d) * 0.02)
        self.obj_proj = nn.Linear(BLOCK * d, d)
        self.obj_coord = FourierCoords(d)
        self.obj_norm = nn.LayerNorm(d)
        self.start_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)    # <QUADBOX><s> yerine

        # ── Outer decoder (obje seviyesi) ───────────────────────────────────
        self.obj_pos_encoding = PositionalEncoding(d, max_len=max_obj_seq + 1)
        self.emb_dropout = nn.Dropout(0.1)
        self.decoder = nn.TransformerDecoder(            # isim 'decoder': vanilla'dan warm start
            nn.TransformerDecoderLayer(d, nheads, batch_first=True, dropout=0.1), outer_layers)

        # ── Local decoder (obje içi, token seviyesi) ────────────────────────
        self.local_slot_emb = nn.Parameter(torch.randn(BLOCK, d) * 0.02)
        self.local_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, nheads, batch_first=True, dropout=0.1), local_layers)

        # ── Ayrık head'ler ──────────────────────────────────────────────────
        self.cls_head = nn.Linear(d, NUM_CLASSES + 1)    # son indeks = EOS
        self.loc_head = nn.Linear(d, NUM_BINS)

        # ── Taslak head'i (sadece hızlı inference; train_draft_head.py ile eğitilir) ─
        self.draft = DraftHead(d, nheads)
        self.register_buffer("draft_trained", torch.zeros((), dtype=torch.bool))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """
        - Taslak head'i olmayan eski checkpoint'ler de yüklenir (draft_trained=False kalır).
        - Farklı MAX_OBJECTS ile eğitilmiş checkpoint'ler de yüklenir (bkz. drop_fixed_pe).
        """
        state_dict = drop_fixed_pe(self, state_dict)
        own = self.state_dict()
        missing = [k for k in own if (k.startswith("draft.") or k == "draft_trained")
                   and k not in state_dict]
        if missing:
            state_dict = dict(state_dict)
            for k in missing:
                state_dict[k] = own[k]
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def init_draft_from_local(self):
        """Taslak decoder'ı local decoder'dan, head'leri ana head'lerden başlat."""
        self.draft.decoder.load_state_dict(self.local_decoder.state_dict())
        self.draft.cls_head.load_state_dict(self.cls_head.state_dict())
        self.draft.loc_head.load_state_dict(self.loc_head.state_dict())

    # ------------------------------------------------------------------ encode
    def encode(self, images):
        memory = self.enc_proj(self.encoder(images)).flatten(2).permute(0, 2, 1)
        return memory + self.pos_emb

    # ------------------------------------------------------------------ outer
    def embed_objects(self, obj_tokens):
        """(B,N,9) token id -> (B,N,d)"""
        e = self.embedding(obj_tokens) + self.obj_slot_emb             # (B,N,9,d)
        x = self.obj_proj(e.flatten(2))
        coords = (obj_tokens[..., 1:] - LOC_START).clamp(0, NUM_BINS - 1).float()
        coords = (coords + 0.5) / NUM_BINS
        is_obj = (obj_tokens[..., 0] >= CLASS_START).unsqueeze(-1).to(x.dtype)
        x = x + self.obj_coord(coords) * is_obj                        # pad/eos satırına koordinat yok
        return self.obj_norm(x)

    def outer(self, memory, prev_objects):
        """
        prev_objects: (B, n, 9) şimdiye kadarki objeler (n>=0)
        -> (B, n+1, d) ; h[:, i] = i. objeyi üretecek bağlam
        """
        B = memory.size(0)
        x = self.start_emb.expand(B, -1, -1)
        if prev_objects.size(1) > 0:
            x = torch.cat([x, self.embed_objects(prev_objects)], 1)
        x = self.emb_dropout(self.obj_pos_encoding(x))
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        # tgt_is_causal=True: PyTorch maskeyi CPU'da kontrol etmez (sync yok) ve SDPA'nın
        # causal kernel'ini kullanır. Matematiksel olarak aynı sonuç.
        return self.decoder(tgt=x, memory=memory, tgt_mask=mask, tgt_is_causal=True)

    # ------------------------------------------------------------------ local
    def local(self, h, tokens, memory):
        """
        h      : (M, d)      obje bağlamı (outer çıktısı)
        tokens : (M, k)      bu objenin şimdiye kadarki token'ları, k = 0..8
        memory : (M, S, d)   görüntü özellikleri
        -> (M, k+1, d) ; çıktı[j] = j. slot'un tahmini (0: sınıf, 1..8: loc)
        """
        x = h[:, None]
        if tokens.size(1) > 0:
            x = torch.cat([x, self.embedding(tokens)], 1)
        x = x + self.local_slot_emb[:x.size(1)]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.local_decoder(tgt=x, memory=memory, tgt_mask=mask, tgt_is_causal=True)

    def _local_step(self, h, draft, memory):
        """
        Jacobi adımı: 8 taslak token (sınıf + 7 loc) ile TEK forward'da 9 slot'un tümünü tahmin eder.
        slot j'nin tahmini yalnızca h ve draft[:j]'ye bağlıdır (causal) -> taslağın doğru
        önekinin bir sonrası her zaman doğru çıkar.
        Döner: tok (B,9) global token id, p (B,) sınıf olasılığı
        """
        out = self.local(h, draft, memory)                             # (B,9,d)
        logits = self.cls_head(out[:, 0]).float()
        p, c = logits.softmax(-1).max(-1)
        tok0 = torch.where(c == NUM_CLASSES, torch.full_like(c, EOS_ID), CLASS_START + c)
        locs = LOC_START + self.loc_head(out[:, 1:]).float().argmax(-1)
        return torch.cat([tok0[:, None], locs], 1), p

    def _draft_tokens(self, h, memory):
        """Taslak head'i -> (B,8) koşullama token'ları (sınıf + ilk 7 loc)."""
        dc, dl = self.draft(h, memory)
        c = dc.float().argmax(-1)
        tok0 = torch.where(c == NUM_CLASSES, torch.full_like(c, EOS_ID), CLASS_START + c)
        locs = LOC_START + dl[:, :N_LOC - 1].float().argmax(-1)
        return torch.cat([tok0[:, None], locs], 1)

    def _draft_verify_step(self, h, memory):
        """Taslak + ilk Jacobi doğrulaması tek çağrıda (tek CUDA graph replay)."""
        draft = self._draft_tokens(h, memory)
        tok, p = self._local_step(h, draft, memory)
        return tok, p, draft

    # ------------------------------------------------------------------ train
    def forward(self, images, objects, valid, with_draft=False):
        """
        Teacher forcing. objects: (B, Nmax+1, 9) hedef satırları, valid: (B, Nmax+1)
        Döner: cls_logits (M, NUM_CLASSES+1), loc_logits (M, 8, NUM_BINS), targets (M, 9)
               M = geçerli satır sayısı (tüm batch'te objeler + EOS satırları)
        with_draft=True: + (draft_cls, draft_loc). Taslak head'i detach edilmiş girdi alır,
        ana modelin gradyanlarını etkilemez.
        """
        memory = self.encode(images)
        h = self.outer(memory, objects[:, :-1])                       # (B, Nmax+1, d)
        b_idx, _ = valid.nonzero(as_tuple=True)
        h_v, tgt = h[valid], objects[valid]                            # (M,d), (M,9)
        mem_v = memory[b_idx]
        out = self.local(h_v, tgt[:, :-1], mem_v)                      # (M,9,d)
        res = (self.cls_head(out[:, 0]), self.loc_head(out[:, 1:]), tgt)
        if with_draft:
            res = res + self.draft(h_v.detach(), mem_v.detach())
        return res


def hier_loss(cls_logits, loc_logits, tgt, loc_weight=LOC_WEIGHT, smoothing=LABEL_SMOOTHING):
    """Sınıf: tüm satırlar (EOS dahil). Loc: sadece gerçek obje satırları."""
    is_eos = tgt[:, 0] == EOS_ID
    cls_t = torch.where(is_eos, torch.full_like(tgt[:, 0], NUM_CLASSES), tgt[:, 0] - CLASS_START)
    l_cls = F.cross_entropy(cls_logits.float(), cls_t, label_smoothing=smoothing)
    if (~is_eos).any():
        loc_t = tgt[~is_eos, 1:] - LOC_START
        l_loc = F.cross_entropy(loc_logits[~is_eos].float().reshape(-1, NUM_BINS),
                                loc_t.reshape(-1), label_smoothing=smoothing)
    else:
        l_loc = cls_logits.sum() * 0.0
    return l_cls + loc_weight * l_loc, l_cls.detach(), l_loc.detach()


# =============================================================================
# WARM START (vanilla checkpoint -> hiyerarşik model)
# =============================================================================
def load_vanilla_weights(model, ckpt_path, init_local_from_decoder=True):
    """
    Vanilla Pix2SeqModel ağırlıklarını aktarır:
        encoder, enc_proj, pos_emb, embedding, decoder (outer)  -> birebir isimle
        local_decoder  <- vanilla decoder'ın SON `LOCAL_LAYERS` katmanı (token seviyesi AR
                          + görüntüye cross-attn: vanilla decoder'ın yaptığı işin aynısı)
        loc_head / cls_head <- vanilla fc_out'un ilgili satırları
    Kalanlar (obje embedding, start_emb, slot embedding'ler) sıfırdan öğrenilir.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    own = model.state_dict()
    copied = []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape and not k.startswith("seq_pos_encoding"):
            own[k] = v
            copied.append(k)
    if init_local_from_decoder:
        n_outer = len(model.decoder.layers)
        n_local = len(model.local_decoder.layers)
        for li in range(n_local):
            src = n_outer - n_local + li
            for k, v in sd.items():
                pre = f"decoder.layers.{src}."
                if k.startswith(pre):
                    tk = f"local_decoder.layers.{li}." + k[len(pre):]
                    if tk in own and own[tk].shape == v.shape:
                        own[tk] = v
                        copied.append(tk)
    if "fc_out.weight" in sd:
        W, b = sd["fc_out.weight"], sd["fc_out.bias"]
        own["loc_head.weight"] = W[LOC_START:LOC_START + NUM_BINS].clone()
        own["loc_head.bias"] = b[LOC_START:LOC_START + NUM_BINS].clone()
        cls_rows = list(range(CLASS_START, CLASS_START + NUM_CLASSES)) + [EOS_ID]
        own["cls_head.weight"] = W[cls_rows].clone()
        own["cls_head.bias"] = b[cls_rows].clone()
        copied += ["loc_head.*", "cls_head.*"]
    model.load_state_dict(own)
    print(f"Warm start: {len(copied)} tensör aktarıldı ({ckpt_path})")
    return model


# =============================================================================
# INFERENCE
# =============================================================================
@torch.no_grad()
def generate_hier(model, images, max_objects=MAX_OBJECTS, amp_dtype=torch.bfloat16):
    """
    Greedy hiyerarşik üretim. Çıktı formatı vanilla generate ile AYNI:
        tokens : (B, 9*n [+...]) düz token dizisi, EOS/PAD ile biten
        scores : (B, n_blok) her objenin sınıf olasılığı
    -> vanilla decode_tokens / predict / evaluate mantığı aynen kullanılabilir.
    Outer decoder N+1 kez, local decoder obje başına 9 kez (küçük) çalışır.
    """
    model.eval()
    dev = images.device
    B = images.size(0)
    amp_on = dev.type == "cuda" and amp_dtype is not None
    ac = lambda: torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_on)

    with ac():
        memory = model.encode(images)

    objects = torch.empty((B, 0, BLOCK), dtype=torch.long, device=dev)
    rows, scores = [], []
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    for i in range(max_objects + 1):
        with ac():
            h = model.outer(memory, objects)[:, -1]                   # (B,d)
        toks = torch.empty((B, 0), dtype=torch.long, device=dev)
        for k in range(BLOCK):
            with ac():
                last = model.local(h, toks, memory)[:, -1]
            if k == 0:
                logits = model.cls_head(last).float()
                if i == max_objects:                                   # kapasite doldu -> EOS
                    logits[:, :NUM_CLASSES] = float("-inf")
                p, c = logits.softmax(-1).max(-1)
                tok = torch.where(c == NUM_CLASSES, torch.full_like(c, EOS_ID), CLASS_START + c)
                scores.append(p)
                if ((tok == EOS_ID) | done).all():
                    toks = torch.cat([tok[:, None], toks.new_full((B, N_LOC), PAD_ID)], 1)
                    break
            else:
                tok = LOC_START + model.loc_head(last).float().argmax(-1)
            toks = torch.cat([toks, tok[:, None]], 1)

        is_eos = toks[:, 0] == EOS_ID
        row = toks.clone()
        row[is_eos, 1:] = PAD_ID
        row[done] = PAD_ID
        rows.append(row)
        done |= is_eos
        if done.all():
            break
        objects = torch.cat([objects, toks[:, None]], 1)

    return torch.cat(rows, 1), torch.stack(scores, 1)


# =============================================================================
# HIZLI INFERENCE: Jacobi decoding + CUDA Graph
# -----------------------------------------------------------------------------
# Darboğaz hesaplama değil, sıralı çağrı sayısı (~100 çağrı × ~1.2 ms launch overhead).
#   1) Local decoder: 9 sıralı adım yerine Jacobi iterasyonu. Her iterasyon 9 slot'u
#      tek forward'da hesaplar; taslak = önceki iterasyonun tahmini. Sabit noktaya
#      ulaşınca (tahmin == taslak) sonuç greedy AR ile BİREBİR aynıdır. En kötü
#      durumda 9 iterasyon (her iterasyon en az 1 token'ı kesinleştirir).
#   2) Outer decoder: her adımda sabit uzunlukta (max_objects+1) tam dizi çalıştırılır,
#      i. pozisyon okunur. Causal maske sayesinde sonraki <pad> satırları sonucu etkilemez.
#   3) İki çağrının da şekli sabit -> CUDA Graph ile yakalanıp tek replay'de çalışır.
# =============================================================================
class _GraphRunner:
    """fn(*statik_girdiler) -> tuple. CUDA'da graph olarak yakalar; yakalanamazsa eager çalışır."""

    def __init__(self, fn, example_inputs, amp_dtype, use_graph):
        self.fn = fn
        self.static = [t.clone() for t in example_inputs]
        self.dev = self.static[0].device
        self.amp_on = self.dev.type == "cuda" and amp_dtype is not None
        self.amp_dtype = amp_dtype or torch.bfloat16
        self.graph, self.out = None, None
        if use_graph and self.dev.type == "cuda":
            try:
                torch.cuda.synchronize()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):                  # graph öncesi ısınma
                    for _ in range(3):
                        self._call()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                # thread_local: yakalama sırasında BAŞKA thread'lerin CUDA çağrıları serbest
                # (DataLoader pin_memory thread'i, servis worker thread'leri vb.). Varsayılan
                # "global" mod bunları yasaklar ve yakalamayı + o thread'i düşürür.
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    self.out = self._call()
                self.graph = g
            except Exception as e:                           # güvenli geri dönüş
                print(f"[generate_hier_fast] CUDA graph yakalanamadı, eager çalışılacak: {e}")
                self.graph, self.out = None, None

    def _call(self):
        with torch.autocast(device_type=self.dev.type, dtype=self.amp_dtype, enabled=self.amp_on):
            return self.fn(*self.static)

    def __call__(self, *inputs):
        """None verilen girdi kopyalanmaz (statik buffer'daki değer kullanılır)."""
        for s, x in zip(self.static, inputs):
            if x is not None:
                s.copy_(x)
        if self.graph is None:
            return self._call()
        self.graph.replay()
        return self.out                                     # statik çıktı: saklanacaksa clone


def _fast_runners(model, memory, max_objects, amp_dtype, use_cuda_graph):
    """Batch boyutu / ayar başına bir kez yakalanır, model üstünde önbelleğe alınır."""
    B = memory.size(0)
    cache = model.__dict__.setdefault("_fast_cache", {})
    key = (B, memory.size(1), max_objects, amp_dtype, use_cuda_graph, str(memory.device))
    if key not in cache:
        objs0 = torch.full((B, max_objects, BLOCK), PAD_ID, dtype=torch.long, device=memory.device)
        outer_r = _GraphRunner(lambda m, o: (model.outer(m, o),), [memory, objs0],
                               amp_dtype, use_cuda_graph)
        h0 = outer_r(memory, objs0)[0][:, 0].clone()
        draft0 = torch.full((B, N_LOC), PAD_ID, dtype=torch.long, device=memory.device)
        local_r = _GraphRunner(model._local_step, [h0, draft0, memory], amp_dtype, use_cuda_graph)
        draft_r = _GraphRunner(model._draft_verify_step, [h0, memory], amp_dtype, use_cuda_graph)
        cache[key] = (outer_r, local_r, draft_r)
    return cache[key]


def _draft_accepted(tok, draft, tol):
    """
    Satır bazında: doğrulama tahmini (tok[:, :8]) taslakla uyumlu mu?
    tol=0 -> birebir (sabit nokta -> greedy ile AYNI sonuç).
    tol>0 -> sınıf birebir, loc'lar ±tol bin (Medusa 'typical acceptance'; KAYIPLI:
             sonraki köşeler ±tol bin oynamış bir köşeye koşullanmış olabilir).
    """
    if tol <= 0:
        return (tok[:, :N_LOC] == draft).all(1)
    return (tok[:, 0] == draft[:, 0]) & ((tok[:, 1:N_LOC] - draft[:, 1:]).abs() <= tol).all(1)


@torch.no_grad()
def generate_hier_fast(model, images, max_objects=MAX_OBJECTS, amp_dtype=torch.bfloat16,
                       use_cuda_graph=True, use_draft=None, stats=None, accept_tol=0):
    """
    generate_hier ile aynı çıktı formatı; accept_tol=0 iken (greedy için) aynı sonuç,
    daha az sıralı çağrı.
    use_draft: None -> model.draft_trained ise taslak head'i kullanılır.
               Taslak yoksa Jacobi <pad> taslağıyla başlar (köşeler birbirine çok bağımlı
               olduğundan ~9 iterasyon sürebilir; kazanç o zaman sadece CUDA graph'tan gelir).
    accept_tol: taslak loc token'ı doğrulamadan en fazla bu kadar bin saparsa kabul edilir.
               0 = kayıpsız. >0 = daha az iterasyon, çıktı greedy'den hafifçe sapabilir
               (benchmark_fps.py --accept-tol K --map ile etkisini ölç).
    stats: dict verilirse doldurulur -> {"jacobi_iters": [obje başına çağrı sayısı, ...]}
    İlk çağrı (her batch boyutu için) graph yakalama yüzünden yavaştır; warmup yap.
    """
    if use_draft is None:
        use_draft = bool(model.draft_trained)
    model.eval()
    dev = images.device
    B = images.size(0)
    amp_on = dev.type == "cuda" and amp_dtype is not None
    with torch.autocast(device_type=dev.type, dtype=amp_dtype or torch.bfloat16, enabled=amp_on):
        memory = model.encode(images)

    outer_r, local_r, draft_r = _fast_runners(model, memory, max_objects, amp_dtype, use_cuda_graph)
    outer_r.static[0].copy_(memory)
    local_r.static[2].copy_(memory)
    draft_r.static[1].copy_(memory)
    objs = outer_r.static[1]
    objs.fill_(PAD_ID)

    rows, scores, iters = [], [], []
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    pad_draft = torch.full((B, N_LOC), PAD_ID, dtype=torch.long, device=dev)
    for i in range(max_objects + 1):
        h = outer_r(None, None)[0][:, i]                              # (B,d)
        draft = pad_draft
        for it in range(BLOCK):
            if it == 0 and use_draft:
                tok, p, draft = draft_r(h, None)                     # taslak + 1. doğrulama
            else:
                tok, p = local_r(h, draft, None)
            if it == 0:
                # slot 0 (sınıf/EOS) sadece h'ye bağlı -> ilk iterasyonda kesin, p de öyle.
                # EOS satırlarının loc'ları önemsiz -> yakınsama kontrolünden çıkar.
                fin, p = tok.clone(), p.clone()
                pending = ~done & (tok[:, 0] != EOS_ID)
            else:
                fin[pending] = tok[pending]                          # kabul edilen satırlar donuk
            pending &= ~_draft_accepted(fin, draft, accept_tol)
            if not pending.any():
                break
            draft = fin[:, :N_LOC].clone()
        tok = fin
        iters.append(it + 1)

        if i == max_objects:                                          # kapasite doldu -> EOS
            tok[:, 0] = EOS_ID
            p = torch.ones_like(p)
        scores.append(p)
        is_eos = tok[:, 0] == EOS_ID
        row = tok.clone()
        row[is_eos, 1:] = PAD_ID
        row[done] = PAD_ID
        rows.append(row)
        done |= is_eos
        if done.all():
            break
        objs[:, i] = tok

    if stats is not None:
        stats.setdefault("jacobi_iters", []).extend(iters[:-1] if len(iters) > 1 else iters)
    return torch.cat(rows, 1), torch.stack(scores, 1)


@torch.no_grad()
def predict_hier(model, img_path, device="cuda", score_thresh=INFER_SCORE_THRESH, fast=True,
                 accept_tol=0, timings=None):
    """
    Tek görüntü -> orijinal piksel koordinatlarında [(label, (4,2) pts, score), ...]
    accept_tol: generate_hier_fast taslak toleransı (0 = birebir greedy).
    timings: dict verilirse {"pre": [...], "model": [...]} saniye olarak eklenir.
    """
    t0 = time.perf_counter()
    img = load_image(img_path)
    padded, max_dim = pad_to_square(img)
    x = val_transform(image=np.array(padded), keypoints=[], kp_ids=[])["image"]
    x = x[None].float().to(device)
    if timings is not None and x.is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    if fast:
        tokens, scores = generate_hier_fast(model, x, accept_tol=accept_tol)
    else:
        tokens, scores = generate_hier(model, x)
    if timings is not None:
        if x.is_cuda:
            torch.cuda.synchronize()
        timings.setdefault("pre", []).append(t1 - t0)
        timings.setdefault("model", []).append(time.perf_counter() - t1)
    preds = []
    for cls, bins, s in decode_tokens(tokens[0].tolist(), scores[0].tolist()):
        if s >= score_thresh:
            preds.append((ID_TO_LABEL[cls], dequantize(bins, max_dim, max_dim), s))
    return img, preds


# =============================================================================
# DEĞERLENDİRME (vanilla evaluate ile aynı metrikler)
# =============================================================================
@torch.no_grad()
def evaluate_hier(model, loader, device, iou_thrs=np.arange(0.5, 0.96, 0.05), gen=None):
    """gen: üretici fonksiyon (varsayılan generate_hier); örn. partial(generate_hier_fast, accept_tol=2)"""
    gen = gen or generate_hier
    preds = {c: [] for c in range(NUM_CLASSES)}
    gts, n_gt, key = {}, np.zeros(NUM_CLASSES, dtype=int), 0
    for batch in tqdm(loader, desc="  mAP", leave=False):
        tokens, scores = gen(model, batch["pixel_values"].to(device).float())
        for i in range(tokens.size(0)):
            for cls, bins, s in decode_tokens(tokens[i].tolist(), scores[i].tolist()):
                preds[cls].append((s, key, bins.astype(np.float64)))
            gt_flat = batch["objects"][i][batch["valid"][i]].reshape(-1).tolist()
            for cls, bins, _ in decode_tokens(gt_flat):
                gts.setdefault((key, cls), []).append(bins.astype(np.float64))
                n_gt[cls] += 1
            key += 1

    aps, corner_err = {thr: [] for thr in iou_thrs}, []
    for c in range(NUM_CLASSES):
        if n_gt[c] == 0:
            continue
        ranked = sorted(preds[c], key=lambda p: -p[0])
        for thr in iou_thrs:
            used = {k: [False] * len(v) for k, v in gts.items() if k[1] == c}
            tp = []
            for _, k, q in ranked:
                cand = gts.get((k, c), [])
                best, bj = 0.0, -1
                for j, g in enumerate(cand):
                    if not used[(k, c)][j]:
                        iou = quad_iou(q, g)
                        if iou > best:
                            best, bj = iou, j
                if best >= thr:
                    used[(k, c)][bj] = True
                    tp.append(1)
                    if abs(thr - 0.5) < 1e-9:
                        corner_err.append(np.linalg.norm(q - cand[bj], axis=1).mean() / NUM_BINS)
                else:
                    tp.append(0)
            aps[thr].append(_average_precision(tp, n_gt[c]))

    map_per_thr = [np.mean(v) for v in aps.values() if v]
    return {
        "map": float(np.mean(map_per_thr)) if map_per_thr else 0.0,
        "map_50": float(np.mean(aps[iou_thrs[0]])) if aps[iou_thrs[0]] else 0.0,
        "nme": float(np.mean(corner_err)) if corner_err else float("nan"),
    }


# =============================================================================
# EĞİTİM
# =============================================================================
def run_epoch(model, loader, device, amp_dtype, amp_enabled, optimizer=None, scheduler=None, desc=""):
    train = optimizer is not None
    model.train(train)
    tot = tot_cls = tot_loc = 0.0
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for batch in bar:
        images = batch["pixel_values"].to(device, non_blocking=True)
        objects = batch["objects"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(images, objects, valid, with_draft=DRAFT_WEIGHT > 0)
            cls_logits, loc_logits, tgt = out[:3]
            loss, l_cls, l_loc = hier_loss(cls_logits, loc_logits, tgt)
            if DRAFT_WEIGHT > 0:                     # raporlanan loss'a dahil değil
                l_draft, _, _ = hier_loss(out[3], out[4], tgt, smoothing=0.0)
                total = loss + DRAFT_WEIGHT * l_draft
            else:
                total = loss
        if train:
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        tot += loss.item(); tot_cls += l_cls.item(); tot_loc += l_loc.item()
        bar.set_postfix(loss=f"{loss.item():.4f}", cls=f"{l_cls.item():.3f}", loc=f"{l_loc.item():.3f}")
    n = max(len(loader), 1)
    return tot / n, tot_cls / n, tot_loc / n


def make_loaders(batch_size=BATCH_SIZE, num_workers=8, pin_memory=True):
    """Vanilla ile aynı val_split.json -> metrikler karşılaştırılabilir."""
    all_json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    if os.path.exists(VAL_SPLIT_PATH):
        with open(VAL_SPLIT_PATH) as f:
            val_set = set(json.load(f)["filenames"])
        train_files = [f for f in all_json_files if f not in val_set]
        val_files = [f for f in all_json_files if f in val_set]
    else:
        random.shuffle(all_json_files)
        cut = int(0.9 * len(all_json_files))
        train_files, val_files = all_json_files[:cut], all_json_files[cut:]
        with open(VAL_SPLIT_PATH, "w") as f:
            json.dump({"filenames": val_files}, f, indent=2)

    train_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=train_transform, mosaic_prob=MOSAIC_PROB)
    train_dataset.json_files = train_files
    val_dataset = Pix2SeqDataset(JSON_DIR, IMG_DIR, transform=val_transform)
    val_dataset.json_files = val_files
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                     prefetch_factor=2, collate_fn=collate_fn_hier)
    return (train_dataset, val_dataset,
            DataLoader(train_dataset, shuffle=True, **loader_kw),
            DataLoader(val_dataset, shuffle=False, **loader_kw))


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    print(f"Hiyerarşik Pix2Seq | {device} | outer {OUTER_LAYERS}L, local {LOCAL_LAYERS}L | "
          f"max obje dizisi {MAX_OBJ_SEQ} (vanilla: {base.MAX_SEQ_LEN})")

    train_dataset, val_dataset, train_dataloader, val_dataloader = make_loaders()

    model = Pix2SeqHierModel().to(device)
    if WARM_START_CKPT and os.path.exists(WARM_START_CKPT):
        load_vanilla_weights(model, WARM_START_CKPT)
        model.to(device)
    if DRAFT_WEIGHT > 0:
        model.draft_trained.fill_(True)     # checkpoint'e yazılır -> generate_hier_fast kullanır

    enc_params = list(model.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": 3e-5},
                                   {"params": rest, "lr": LEARNING_RATE}], weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=[1e-5, LEARNING_RATE],
                           total_steps=len(train_dataloader) * EPOCHS, pct_start=0.05)

    dataset_object_stats(train_dataset)
    best_val_loss, best_map = float("inf"), 0.0
    epoch_bar = tqdm(range(EPOCHS), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        tr, tr_c, tr_l = run_epoch(model, train_dataloader, device, amp_dtype, amp_enabled,
                                   optimizer, scheduler, desc=f"  Train {epoch+1}/{EPOCHS}")
        va, va_c, va_l = run_epoch(model, val_dataloader, device, amp_dtype, amp_enabled,
                                   desc=f"  Val   {epoch+1}/{EPOCHS}")
        epoch_bar.set_postfix(train=f"{tr:.4f}", val=f"{va:.4f}", v_cls=f"{va_c:.3f}",
                              v_loc=f"{va_l:.3f}", best=f"{best_val_loss:.4f}")

        if va < best_val_loss:
            best_val_loss = va
            torch.save(model.state_dict(), CKPT_BEST_LOSS)
            tqdm.write(f"  ✓ Epoch {epoch+1:3d} — val loss {va:.4f} (cls {va_c:.3f} / loc {va_l:.3f})")

        if (epoch + 1) % EVAL_EVERY == 0:
            res = evaluate_hier(model, val_dataloader, device)
            tqdm.write(f"  mAP {res['map']:.4f} | mAP50 {res['map_50']:.4f} | NME {res['nme']:.4f}")
            if res["map"] > best_map:
                best_map = res["map"]
                torch.save(model.state_dict(), CKPT_BEST_MAP)
                tqdm.write(f"  ✓ yeni en iyi mAP {best_map:.4f} → {CKPT_BEST_MAP}")

    print(f"\nEğitim tamamlandı. best loss {best_val_loss:.4f} | best mAP {best_map:.4f}")