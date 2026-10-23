"""
FPS benchmark: vanilla Pix2Seq vs hiyerarşik (Varyant B)

Ölçülenler (her model için):
  1) Model-only  (batch=1): görüntü önceden GPU'da, sadece encode + decode
       -> encoder ms, decoder ms, toplam ms, FPS, obje başına decoder ms
  2) End-to-end  (batch=1): dosyadan okuma + exif + pad + resize + normalize + model + decode_tokens
  3) Throughput  (batch=B): model-only, birden fazla görüntü aynı anda

Kullanım:
  python benchmark_fps.py                                   # iki model, val split'ten 100 görüntü
  python benchmark_fps.py --n 200 --batch-sizes 1 4 8 16
  python benchmark_fps.py --models hier                     # sadece hiyerarşik
  python benchmark_fps.py --no-amp                          # fp32
"""
import os
import json
import time
import argparse
from functools import partial

import numpy as np
import torch

import pix2seq_florence_quad as V
import pix2seq_florence_quad_hier as H

IMG_ROOT = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
EXTS = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]


# -----------------------------------------------------------------------------
def val_image_paths(n):
    with open(V.VAL_SPLIT_PATH) as f:
        names = json.load(f)["filenames"]
    paths = []
    for name in names:
        stem = os.path.join(IMG_ROOT, name).replace(".json", "")
        p = next((stem + e for e in EXTS if os.path.exists(stem + e)), None)
        if p:
            paths.append(p)
        if len(paths) >= n:
            break
    return paths


def preprocess(path):
    img = V.load_image(path)
    padded, max_dim = V.pad_to_square(img)
    x = V.val_transform(image=np.array(padded), keypoints=[], kp_ids=[])["image"]
    return x.float(), max_dim


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def build(kind, ckpt, device, accept_tol=0):
    if kind == "vanilla":
        model = V.Pix2SeqModel(vocab_size=V.VOCAB_SIZE, max_seq_len=V.MAX_SEQ_LEN)
        gen = V.generate
    else:
        model = H.Pix2SeqHierModel()
        gen = (partial(H.generate_hier_fast, accept_tol=accept_tol) if kind == "hier_fast"
               else H.generate_hier)
    H.load_checkpoint(model, ckpt)     # farklı MAX_OBJECTS ile eğitilmiş ckpt'leri de yükler
    return model.to(device).eval(), gen


@torch.no_grad()
def check_fast_equivalence(model, tensors, device, amp_dtype, accept_tol=0):
    """generate_hier vs generate_hier_fast: aynı görüntülerde çıktılar ne kadar aynı?"""
    seq_eq = obj_eq = obj_tot = 0
    max_coord_diff = 0
    ious = []
    stats = {}
    for t in tensors:
        x = t[None].to(device)
        a, sa = H.generate_hier(model, x, amp_dtype=amp_dtype)
        b, sb = H.generate_hier_fast(model, x, amp_dtype=amp_dtype, stats=stats,
                                     accept_tol=accept_tol)
        da = V.decode_tokens(a[0].tolist(), sa[0].tolist())
        db = V.decode_tokens(b[0].tolist(), sb[0].tolist())
        seq_eq += int(len(da) == len(db) and all(
            ca == cb and np.array_equal(ba, bb) for (ca, ba, _), (cb, bb, _) in zip(da, db)))
        for (ca, ba, _), (cb, bb, _) in zip(da, db):
            obj_eq += int(ca == cb and np.array_equal(ba, bb))
            if ca == cb:
                max_coord_diff = max(max_coord_diff, int(np.abs(ba - bb).max()))
                ious.append(1.0 if np.array_equal(ba, bb)
                            else V.quad_iou(bb.astype(np.float64), ba.astype(np.float64)))
        obj_tot += max(len(da), len(db))
    it = np.array(stats.get("jacobi_iters", [0]))
    ious = np.array(ious) if ious else np.array([1.0])
    return {"seq_eq": seq_eq / len(tensors), "obj_eq": obj_eq / max(obj_tot, 1),
            "max_bin_diff": max_coord_diff, "iters_mean": it.mean(),
            "iters_hist": np.bincount(it, minlength=10)[1:10],
            "iou_mean": ious.mean(), "iou_min": ious.min()}


# -----------------------------------------------------------------------------
@torch.no_grad()
def bench_model_only(model, gen, tensors, device, amp_dtype, warmup):
    """batch=1, görüntü önceden GPU'da. encoder ve toplam süre ayrı ölçülür."""
    amp_on = device.type == "cuda" and amp_dtype is not None
    xs = [t[None].to(device) for t in tensors]
    for x in xs[:warmup]:
        gen(model, x, amp_dtype=amp_dtype)
    sync(device)

    enc_t, tot_t, n_obj = [], [], []
    for x in xs:
        sync(device); t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
            model.encode(x)
        sync(device); enc_t.append(time.perf_counter() - t0)

        sync(device); t0 = time.perf_counter()
        tokens, scores = gen(model, x, amp_dtype=amp_dtype)
        sync(device); tot_t.append(time.perf_counter() - t0)
        n_obj.append(len(V.decode_tokens(tokens[0].tolist(), scores[0].tolist())))

    enc, tot, n_obj = np.array(enc_t) * 1e3, np.array(tot_t) * 1e3, np.array(n_obj)
    dec = tot - enc
    return {
        "enc_ms": enc.mean(), "dec_ms": dec.mean(), "tot_ms": tot.mean(),
        "p50_ms": np.percentile(tot, 50), "p95_ms": np.percentile(tot, 95),
        "fps": 1000.0 / tot.mean(), "obj": n_obj.mean(),
        "dec_ms_per_obj": (dec / np.maximum(n_obj + 1, 1)).mean(),   # +1: EOS adımı
    }


@torch.no_grad()
def bench_end_to_end(model, gen, paths, device, amp_dtype, warmup):
    """Dosyadan okuma dahil, batch=1."""
    for p in paths[:warmup]:
        x, _ = preprocess(p)
        gen(model, x[None].to(device), amp_dtype=amp_dtype)
    sync(device)
    times = []
    for p in paths:
        sync(device); t0 = time.perf_counter()
        x, max_dim = preprocess(p)
        tokens, scores = gen(model, x[None].to(device), amp_dtype=amp_dtype)
        for cls, bins, s in V.decode_tokens(tokens[0].tolist(), scores[0].tolist()):
            V.dequantize(bins, max_dim, max_dim)
        sync(device); times.append(time.perf_counter() - t0)
    t = np.array(times) * 1e3
    return {"tot_ms": t.mean(), "fps": 1000.0 / t.mean()}


@torch.no_grad()
def bench_throughput(model, gen, tensors, device, amp_dtype, bs, warmup_batches=2):
    """batch=bs, model-only. Batch'teki en uzun dizi süreyi belirler."""
    batches = [torch.stack(tensors[i:i + bs]).to(device)
               for i in range(0, len(tensors) - bs + 1, bs)]
    if not batches:
        return None
    for b in batches[:warmup_batches]:
        gen(model, b, amp_dtype=amp_dtype)
    sync(device); t0 = time.perf_counter()
    for b in batches:
        gen(model, b, amp_dtype=amp_dtype)
    sync(device)
    n_img = len(batches) * bs
    return {"fps": n_img / (time.perf_counter() - t0)}


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["vanilla", "hier", "hier_fast"],
                    choices=["vanilla", "hier", "hier_fast"])
    ap.add_argument("--no-check", action="store_true", help="hier vs hier_fast eşitlik kontrolünü atla")
    ap.add_argument("--vanilla-ckpt", default="pix2seq_florence_best_map_0.8674.pth")
    ap.add_argument("--hier-ckpt", default="pix2seq_hier_best_map.pth")
    ap.add_argument("--hier-fast-ckpt", default=None,
                    help="varsayılan: <hier-ckpt>_draft.pth varsa o (taslak head'li), yoksa hier-ckpt")
    ap.add_argument("--n", type=int, default=100, help="ölçülecek görüntü sayısı")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--accept-tol", type=int, default=0,
                    help="hier_fast taslak kabul toleransı (bin). 0 = kayıpsız")
    ap.add_argument("--map", action="store_true",
                    help="val setinin tamamında hier vs hier_fast mAP/mAP50/NME karşılaştır")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = None if args.no_amp else torch.bfloat16
    torch.backends.cudnn.benchmark = True
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} | AMP: {'bf16' if amp_dtype else 'fp32'}")

    paths = val_image_paths(args.n + args.warmup)
    print(f"{len(paths)} val görüntüsü ön-işleniyor...")
    tensors = [preprocess(p)[0] for p in paths]
    run_paths, run_tensors = paths[args.warmup:], tensors[args.warmup:]
    fast_ckpt = args.hier_fast_ckpt
    if fast_ckpt is None:
        cand = args.hier_ckpt.replace(".pth", "_draft.pth")
        fast_ckpt = cand if os.path.exists(cand) else args.hier_ckpt
    ckpts = {"vanilla": args.vanilla_ckpt, "hier": args.hier_ckpt, "hier_fast": fast_ckpt}

    check = None
    if not args.no_check and "hier_fast" in args.models and os.path.exists(fast_ckpt):
        model, _ = build("hier_fast", fast_ckpt, device)
        print(f"\n[kontrol] {fast_ckpt} | taslak head'i: "
              f"{'VAR' if bool(model.draft_trained) else 'YOK (train_draft_head.py ile eğit)'} | "
              f"accept_tol={args.accept_tol}")
        print("[kontrol] generate_hier vs generate_hier_fast çıktıları karşılaştırılıyor...")
        check = check_fast_equivalence(model, run_tensors, device, amp_dtype, args.accept_tol)
        check["draft"] = bool(model.draft_trained)

        if args.map:
            _, _, _, val_loader = H.make_loaders(batch_size=8, pin_memory=False)
            print("[mAP] val setinde hier (greedy) ölçülüyor...")
            check["map_ref"] = H.evaluate_hier(model, val_loader, device, gen=H.generate_hier)
            print(f"[mAP] hier_fast (accept_tol={args.accept_tol}) ölçülüyor...")
            check["map_fast"] = H.evaluate_hier(
                model, val_loader, device,
                gen=partial(H.generate_hier_fast, accept_tol=args.accept_tol))
        del model

    results = {}
    for kind in args.models:
        if not os.path.exists(ckpts[kind]):
            print(f"[{kind}] checkpoint yok: {ckpts[kind]} — atlanıyor")
            continue
        model, gen = build(kind, ckpts[kind], device, args.accept_tol)
        print(f"\n[{kind}] ölçülüyor...")
        r = {"model": bench_model_only(model, gen, tensors, device, amp_dtype, args.warmup)}
        r["e2e"] = bench_end_to_end(model, gen, run_paths, device, amp_dtype, args.warmup)
        r["tp"] = {bs: bench_throughput(model, gen, run_tensors, device, amp_dtype, bs)
                   for bs in args.batch_sizes}
        results[kind] = r
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Rapor ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'':10s}{'enc ms':>8s}{'dec ms':>8s}{'top ms':>8s}{'p95 ms':>8s}"
          f"{'FPS':>8s}{'e2e FPS':>9s}{'obje':>7s}{'ms/obje':>9s}")
    for kind, r in results.items():
        m = r["model"]
        print(f"{kind:10s}{m['enc_ms']:8.1f}{m['dec_ms']:8.1f}{m['tot_ms']:8.1f}{m['p95_ms']:8.1f}"
              f"{m['fps']:8.1f}{r['e2e']['fps']:9.1f}{m['obj']:7.1f}{m['dec_ms_per_obj']:9.2f}")
    print("\nThroughput (model-only, img/s):")
    for kind, r in results.items():
        print(f"  {kind:10s}" + "  ".join(f"bs={bs}: {v['fps']:6.1f}" if v else f"bs={bs}: -"
                                          for bs, v in r["tp"].items()))
    print()
    for a, b in [("vanilla", "hier"), ("hier", "hier_fast"), ("vanilla", "hier_fast")]:
        if {a, b} <= results.keys():
            ra, rb = results[a]["model"], results[b]["model"]
            print(f"Hızlanma {a} -> {b} (batch=1): toplam {ra['tot_ms'] / rb['tot_ms']:.2f}x | "
                  f"decoder {ra['dec_ms'] / max(rb['dec_ms'], 1e-6):.2f}x")
    if check:
        print(f"\nhier vs hier_fast çıktı eşitliği: görüntü {100 * check['seq_eq']:.1f}% | "
              f"obje {100 * check['obj_eq']:.1f}% | eşleşen sınıfta maks bin farkı {check['max_bin_diff']}")
        print(f"Aynı sınıflı obje çiftlerinde quad IoU (fast vs greedy): ort {check['iou_mean']:.4f} | "
              f"min {check['iou_min']:.4f}")
        print(f"Local çağrı/obje ({'taslaklı' if check['draft'] else 'taslaksız'}, "
              f"accept_tol={args.accept_tol}): ort {check['iters_mean']:.2f} (eski: 9) | "
              f"dağılım (1..9): {check['iters_hist'].tolist()}")
        if check["seq_eq"] < 1.0 and args.accept_tol == 0:
            print("  (bf16'da farklı kernel şekilleri argmax eşitliklerini nadiren çevirebilir; "
                  "--no-amp ile birebir olmalı)")
        if "map_ref" in check:
            r, f = check["map_ref"], check["map_fast"]
            print(f"\n{'':22s}{'mAP':>8s}{'mAP50':>8s}{'NME':>9s}")
            print(f"{'hier (greedy)':22s}{r['map']:8.4f}{r['map_50']:8.4f}{r['nme']:9.5f}")
            print(f"{f'hier_fast (tol={args.accept_tol})':22s}{f['map']:8.4f}{f['map_50']:8.4f}{f['nme']:9.5f}")
    print("=" * 78)


if __name__ == "__main__":
    main()