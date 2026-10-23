"""
Mevcut hiyerarşik checkpoint için TASLAK HEAD'İNİ eğitir (ana model tamamen donuk).

Neden: generate_hier_fast, local decoder'ı Jacobi ile çalıştırır. Taslak <pad> ile başlarsa
quad köşeleri birbirine çok bağımlı olduğu için ~9 iterasyon sürer (kazanç yok). Taslak head'i
9 token'ı tek forward'da tahmin eder; doğruysa obje başına 1 çağrı yeter.
Çıktılar DEĞİŞMEZ: taslak her zaman local decoder ile doğrulanır.

Hedef (--target):
  model : ana modelin kendi tahminleri (distillation, varsayılan). Kabul oranını doğrudan
          maksimize eder çünkü doğrulama modelin greedy zinciriyle karşılaştırır.
  gt    : etiketler.

Kullanım:
  python train_draft_head.py --ckpt pix2seq_hier_best_map.pth --out pix2seq_hier_best_map_draft.pth
Sonra benchmark / inference'ta --hier-ckpt / CKPT olarak yeni dosyayı ver.
"""
import argparse

import torch
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

import pix2seq_florence_quad_hier as H
from pix2seq_florence_quad_hier import (
    Pix2SeqHierModel, hier_loss, make_loaders,
    NUM_CLASSES, EOS_ID, CLASS_START, LOC_START, N_LOC,
)


def to_tokens(cls_logits, loc_logits):
    c = cls_logits.float().argmax(-1)
    tok0 = torch.where(c == NUM_CLASSES, torch.full_like(c, EOS_ID), CLASS_START + c)
    return torch.cat([tok0[:, None], LOC_START + loc_logits.float().argmax(-1)], 1)   # (M,9)


def step(model, batch, device, amp_on, target):
    images = batch["pixel_values"].to(device, non_blocking=True)
    objects = batch["objects"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_on):
        with torch.no_grad():                                         # ana model donuk
            memory = model.encode(images)
            h = model.outer(memory, objects[:, :-1])
            b_idx, _ = valid.nonzero(as_tuple=True)
            h_v, gt = h[valid], objects[valid]
            mem_v = memory[b_idx]
            if target == "model":
                out = model.local(h_v, gt[:, :-1], mem_v)
                tgt = to_tokens(model.cls_head(out[:, 0]), model.loc_head(out[:, 1:]))
            else:
                tgt = gt
        dc, dl = model.draft(h_v, mem_v)
    return dc, dl, tgt


TOLS = (0, 1, 2, 4)


@torch.no_grad()
def acceptance(dc, dl, tgt):
    """
    SADECE gerçek obje satırları (EOS satırları hariç; onlar sınıf doğruysa zaten 1 çağrı).
    Döner: {tol: tek-çağrıda-kabul sayısı}, slot bazında {tol: isabet sayısı (8,)}, obje sayısı
    """
    draft = to_tokens(dc, dl)[:, :N_LOC]
    obj = tgt[:, 0] != EOS_ID
    d, t = draft[obj], tgt[obj, :N_LOC]
    cls_ok = d[:, 0] == t[:, 0]
    diff = (d[:, 1:] - t[:, 1:]).abs()
    full, slot = {}, {}
    for tol in TOLS:
        loc_ok = diff <= tol
        full[tol] = int((cls_ok & loc_ok.all(1)).sum())
        slot[tol] = torch.cat([cls_ok[:, None], loc_ok], 1).float().sum(0)
    return full, slot, int(obj.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=H.CKPT_BEST_MAP)
    ap.add_argument("--out", default=H.CKPT_BEST_MAP.replace(".pth", "_draft.pth"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=H.BATCH_SIZE)
    ap.add_argument("--target", choices=["model", "gt"], default="model")
    ap.add_argument("--accept-tol", type=int, default=2, choices=TOLS,
                    help="en iyi checkpoint bu toleranstaki kabul oranına göre seçilir")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_on = device.type == "cuda"
    model = Pix2SeqHierModel()
    H.load_checkpoint(model, args.ckpt)
    if not bool(model.draft_trained):
        model.init_draft_from_local()          # iyi başlangıç: local decoder ile aynı iş
        print("Taslak head'i local decoder ağırlıklarından başlatıldı.")
    model.to(device).eval()                    # dropout kapalı; ana model donuk

    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.draft.parameters())
    for p in params:
        p.requires_grad_(True)
    print(f"Eğitilen parametre: {sum(p.numel() for p in params) / 1e6:.2f}M")

    _, _, train_loader, val_loader = make_loaders(batch_size=args.batch_size)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = OneCycleLR(opt, max_lr=args.lr, total_steps=len(train_loader) * args.epochs, pct_start=0.05)

    best = -1.0
    for ep in range(args.epochs):
        bar = tqdm(train_loader, desc=f"draft {ep + 1}/{args.epochs}", leave=False)
        for batch in bar:
            dc, dl, tgt = step(model, batch, device, amp_on, args.target)
            loss, _, _ = hier_loss(dc, dl, tgt, smoothing=0.0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        full = {t: 0 for t in TOLS}
        slot = {t: torch.zeros(N_LOC, device=device) for t in TOLS}
        n_obj = 0
        with torch.no_grad():
            for batch in val_loader:
                f, s, n = acceptance(*step(model, batch, device, amp_on, args.target))
                n_obj += n
                for t in TOLS:
                    full[t] += f[t]
                    slot[t] += s[t]
        n = max(n_obj, 1)
        print(f"epoch {ep + 1}: tek çağrıda kabul (obje) — "
              + " | ".join(f"±{t}: {100 * full[t] / n:.1f}%" for t in TOLS))
        for t in (0, 2):
            print(f"    slot isabeti ±{t} (cls,x1..y4): "
                  + " ".join(f"{100 * a:.0f}" for a in (slot[t] / n).tolist()))
        rate = full[args.accept_tol] / n
        if rate > best:
            best = rate
            model.draft_trained.fill_(True)
            torch.save(model.state_dict(), args.out)
            print(f"  ✓ kaydedildi (±{args.accept_tol}: {100 * rate:.1f}%) -> {args.out}")

    print(f"\nEn iyi ±{args.accept_tol} bin kabul oranı: {100 * best:.1f}% (sadece objeler). "
          f"Bu oran tek local çağrıda çözülen objelerin payıdır; benchmark'ta "
          f"--accept-tol {args.accept_tol} --map ile hız ve mAP etkisini ölç.")


if __name__ == "__main__":
    main()