"""
pix2seq_fast_decode.py
======================
nn.TransformerDecoder tabanli Pix2Seq modeli icin KV-cache'li incremental decode.

Neden gerekli:
  - nn.TransformerDecoder'in kendi forward'i cache desteklemez. Her adimda
    tum prefix'i bastan hesaplar  ->  adim basi O(L^2), toplam O(L^3).
  - Bu modul her katmanin self-attention K/V'sini biriktirir (adim basi O(L)),
    cross-attention K/V'sini ise memory sabit oldugu icin BIR KEZ hesaplar.
  - Sonuc: toplam O(L^2), pratikte L=752 icin ~30-60x hizlanma.

Agirliklar aynen kullanilir, model yeniden egitilmez. Sadece forward yolu
elle yeniden yazilir (mathematically identical).

Kullanim:
    from pix2seq_fast_decode import Pix2SeqKVDecoder
    dec = Pix2SeqKVDecoder(model, max_seq_len, device)
    seqs, scores = dec.decode(images)          # images: (B,3,H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Positional encoding tablosunu cikarma
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_pe_table(model, max_seq_len, device):
    """
    seq_pos_encoding additive oldugu icin (x -> x + pe[:L]) sifir tensor
    gecirerek pe tablosunu birebir cikarabiliyoruz. Boylece incremental
    decode sirasinda tek pozisyonu (pe[:, t]) dogrudan ekleyebiliyoruz.
    """
    d_model = model.embedding.embedding_dim
    zeros = torch.zeros(1, max_seq_len, d_model, device=device)
    pe = model.seq_pos_encoding(zeros)
    assert pe.shape == (1, max_seq_len, d_model), (
        f"seq_pos_encoding cikti sekli beklenmedik: {tuple(pe.shape)}. "
        "Additive olmayan bir PE kullaniyorsan bu fonksiyonu elle uyarla."
    )
    if pe.abs().sum() == 0:
        raise RuntimeError(
            "PE tablosu tamamen sifir cikti. seq_pos_encoding muhtemelen "
            "additive degil; extract_pe_table'i modeline gore uyarla."
        )
    return pe.contiguous()


# --------------------------------------------------------------------------- #
#  Cache'li MultiheadAttention
# --------------------------------------------------------------------------- #
class _CachedMHA:
    """nn.MultiheadAttention agirliklarini alip incremental calistirir."""

    def __init__(self, mha: nn.MultiheadAttention):
        if not getattr(mha, "_qkv_same_embed_dim", True):
            raise NotImplementedError(
                "Ayri q/k/v projeksiyonlu MultiheadAttention destegi yok."
            )
        D = mha.embed_dim
        W = mha.in_proj_weight
        b = mha.in_proj_bias

        self.Wq, self.Wk, self.Wv = W[:D], W[D:2 * D], W[2 * D:3 * D]
        if b is not None:
            self.bq, self.bk, self.bv = b[:D], b[D:2 * D], b[2 * D:3 * D]
        else:
            self.bq = self.bk = self.bv = None

        self.out_proj = mha.out_proj
        self.h  = mha.num_heads
        self.D  = D
        self.dh = D // self.h

    def _heads(self, x, B, L):
        # (B, L, D) -> (B, h, L, dh)
        return x.view(B, L, self.h, self.dh).transpose(1, 2)

    # --- self-attention: tek token, gecmisi cache'ten oku -------------------
    def self_step(self, x, k_cache, v_cache, t):
        B = x.size(0)
        q = self._heads(F.linear(x, self.Wq, self.bq), B, 1)
        k = self._heads(F.linear(x, self.Wk, self.bk), B, 1)
        v = self._heads(F.linear(x, self.Wv, self.bv), B, 1)

        cd = k_cache.dtype
        k_cache[:, :, t:t + 1] = k.to(cd)
        v_cache[:, :, t:t + 1] = v.to(cd)

        # causal mask gerekmez: zaten sadece [0..t] gorunuyor
        o = F.scaled_dot_product_attention(
            q.to(cd), k_cache[:, :, :t + 1], v_cache[:, :, :t + 1]
        )
        o = o.transpose(1, 2).reshape(B, 1, self.D)
        return self.out_proj(o)

    # --- cross-attention: memory sabit, K/V bir kez ------------------------
    def build_cross_kv(self, memory, dtype):
        B, N, _ = memory.shape
        k = self._heads(F.linear(memory, self.Wk, self.bk), B, N).to(dtype)
        v = self._heads(F.linear(memory, self.Wv, self.bv), B, N).to(dtype)
        return k.contiguous(), v.contiguous()

    def cross_step(self, x, k, v):
        B = x.size(0)
        q = self._heads(F.linear(x, self.Wq, self.bq), B, 1).to(k.dtype)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, 1, self.D)
        return self.out_proj(o)


# --------------------------------------------------------------------------- #
#  Cache'li decoder katmani
# --------------------------------------------------------------------------- #
class _CachedDecoderLayer:
    def __init__(self, layer: nn.TransformerDecoderLayer):
        self.self_attn  = _CachedMHA(layer.self_attn)
        self.cross_attn = _CachedMHA(layer.multihead_attn)
        self.norm1, self.norm2, self.norm3 = layer.norm1, layer.norm2, layer.norm3
        self.linear1, self.linear2 = layer.linear1, layer.linear2
        self.activation = layer.activation
        self.norm_first = bool(getattr(layer, "norm_first", False))

    def _ff(self, x):
        return self.linear2(self.activation(self.linear1(x)))

    def step(self, x, sk, sv, ck, cv, t):
        if self.norm_first:
            x = x + self.self_attn.self_step(self.norm1(x), sk, sv, t)
            x = x + self.cross_attn.cross_step(self.norm2(x), ck, cv)
            x = x + self._ff(self.norm3(x))
        else:
            x = self.norm1(x + self.self_attn.self_step(x, sk, sv, t))
            x = self.norm2(x + self.cross_attn.cross_step(x, ck, cv))
            x = self.norm3(x + self._ff(x))
        return x


# --------------------------------------------------------------------------- #
#  Ana decoder
# --------------------------------------------------------------------------- #
class Pix2SeqKVDecoder:
    """
    Bir kez kur, tekrar tekrar kullan:

        dec = Pix2SeqKVDecoder(model, max_seq_len, device)
        seqs, scores = dec.decode(images)
    """

    def __init__(self, model, max_seq_len, device,
                 bos_token=None, eos_token=None, pad_token=None,
                 amp_dtype=torch.float16, cache_dtype=None):
        model.eval()
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len

        # token id'leri: verilmezse training modulunden cek
        if bos_token is None or eos_token is None or pad_token is None:
            from training_pix2seq_bbox import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN
            bos_token = BOS_TOKEN if bos_token is None else bos_token
            eos_token = EOS_TOKEN if eos_token is None else eos_token
            pad_token = PAD_TOKEN if pad_token is None else pad_token
        self.BOS, self.EOS, self.PAD = bos_token, eos_token, pad_token

        dec = model.decoder
        if not isinstance(dec, nn.TransformerDecoder):
            raise TypeError(f"model.decoder nn.TransformerDecoder degil: {type(dec)}")

        self.layers = [_CachedDecoderLayer(l) for l in dec.layers]
        self.final_norm = dec.norm  # None olabilir
        self.n_layers = len(self.layers)
        self.n_heads = self.layers[0].self_attn.h
        self.d_head  = self.layers[0].self_attn.dh
        self.d_model = self.layers[0].self_attn.D

        self.use_amp = (device.type == "cuda") and amp_dtype is not None
        self.amp_dtype = amp_dtype if self.use_amp else torch.float32
        self.cache_dtype = cache_dtype or (self.amp_dtype if self.use_amp else torch.float32)

        self.pe = extract_pe_table(model, max_seq_len, device)

        self._sk = self._sv = None   # self-attn cache buffer'lari
        self._cap_B = 0

    # -- cache buffer'lari yeniden kullan (her cagride malloc yapma) --------
    def _ensure_cache(self, B, L):
        if self._sk is not None and self._cap_B >= B and self._sk.size(3) >= L:
            return
        self._sk = torch.zeros(self.n_layers, B, self.n_heads, L, self.d_head,
                               device=self.device, dtype=self.cache_dtype)
        self._sv = torch.zeros_like(self._sk)
        self._cap_B = B

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def encode(self, images):
        """Encoder'i bir kez calistirip memory uretir."""
        m = self.model
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            feats = m.enc_proj(m.encoder(images))
        memory = feats.flatten(2).permute(0, 2, 1).float() + m.pos_emb
        return memory

    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def decode(self, images=None, memory=None, max_steps=None, return_logits=False):
        """
        Greedy AR decode. images ya da hazir memory ver.
        Donen: (seq  [B, T]  BOS'suz,  scores [B, T])
        Orijinal ar_decode_batch ile birebir ayni semantik.
        """
        m = self.model
        if memory is None:
            memory = self.encode(images)
        B = memory.size(0)

        T = self.max_seq_len - 1 if max_steps is None else min(max_steps, self.max_seq_len - 1)
        self._ensure_cache(B, T)
        sk, sv = self._sk[:, :B], self._sv[:, :B]

        # cross-attn K/V: memory sabit -> tek sefer
        cross = [lyr.cross_attn.build_cross_kv(memory, self.cache_dtype)
                 for lyr in self.layers]

        seq      = torch.full((B, T), self.PAD, dtype=torch.long, device=self.device)
        scores   = torch.zeros(B, T, device=self.device)
        finished = torch.zeros(B, dtype=torch.bool, device=self.device)
        cur      = torch.full((B,), self.BOS, dtype=torch.long, device=self.device)

        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
            for t in range(T):
                x = m.embedding(cur).unsqueeze(1) + self.pe[:, t:t + 1]
                x = m.emb_dropout(x)                      # eval'de no-op

                for li, lyr in enumerate(self.layers):
                    x = lyr.step(x, sk[li], sv[li], cross[li][0], cross[li][1], t)

                if self.final_norm is not None:
                    x = self.final_norm(x)

                logits = m.fc_out(x[:, 0]).float()
                probs  = torch.softmax(logits, -1)
                nxt    = probs.argmax(-1)
                scores[:, t] = probs.gather(1, nxt.unsqueeze(1)).squeeze(1)

                nxt = torch.where(finished, torch.full_like(nxt, self.PAD), nxt)
                seq[:, t] = nxt
                finished |= (nxt == self.EOS)
                if bool(finished.all()):
                    seq = seq[:, :t + 1]
                    scores = scores[:, :t + 1]
                    break
                cur = nxt

        return seq, scores


# --------------------------------------------------------------------------- #
#  Dogrulama: cache'li ve cache'siz decode ayni mi?
# --------------------------------------------------------------------------- #
@torch.no_grad()
def verify_against_naive(model, images, max_seq_len, device,
                         BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, amp_dtype=None):
    """
    Yeni implementasyonun eskisiyle ayni token dizisini urettigini dogrular.
    amp_dtype=None ver -> fp32'de birebir esitlik beklenir.
    """
    import time

    # --- naive (orijinal ar_decode_batch) ---
    t0 = time.time()
    B = images.size(0)
    feats  = model.enc_proj(model.encoder(images))
    memory = feats.flatten(2).permute(0, 2, 1) + model.pos_emb
    seq = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
    fin = torch.zeros(B, dtype=torch.bool, device=device)
    for step in range(max_seq_len - 1):
        tgt = model.emb_dropout(model.seq_pos_encoding(model.embedding(seq)))
        msk = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(device)
        out = model.decoder(tgt=tgt, memory=memory, tgt_mask=msk)
        nxt = model.fc_out(out[:, -1, :]).float().argmax(-1)
        nxt = torch.where(fin, torch.full_like(nxt, PAD_TOKEN), nxt)
        seq = torch.cat([seq, nxt.unsqueeze(1)], 1)
        fin |= (nxt == EOS_TOKEN)
        if fin.all():
            break
    naive = seq[:, 1:]
    t_naive = time.time() - t0

    # --- cache'li ---
    t0 = time.time()
    dec = Pix2SeqKVDecoder(model, max_seq_len, device,
                           BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, amp_dtype=amp_dtype)
    fast, _ = dec.decode(images)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fast = time.time() - t0

    L = min(naive.size(1), fast.size(1))
    match = int((naive[:, :L] == fast[:, :L]).all(1).sum())
    print(f"[verify] naive {t_naive:.2f}s | kv-cache {t_fast:.2f}s "
          f"| hizlanma {t_naive / max(t_fast, 1e-9):.1f}x")
    print(f"[verify] birebir eslesen ornek: {match}/{B}")
    if match != B:
        diff = (naive[:, :L] != fast[:, :L]).float().mean().item()
        print(f"[verify] token uyusmazlik orani: {diff:.4%} "
              "(fp16 kullaniyorsan kucuk sapma normaldir)")
    return naive, fast
