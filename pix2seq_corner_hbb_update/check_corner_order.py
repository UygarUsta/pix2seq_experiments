"""
Etiketlerdeki 4 köşenin sıralama tutarlılığını denetler.
--------------------------------------------------------
Pix2Seq köşeleri sabit bir sırayla, token token üretir. Model bu sırayı ancak
etiketlerde TUTARLI bir kural varsa öğrenebilir. Annotator'lar farklı köşeden
başladıysa (ya da kimi saat yönünde kimi tersine tıkladıysa) hedef sıra
öğrenilemez hale gelir: model iki köşeyi yer değiştirir, ortaya kendisiyle
kesişen bir "papyon" dörtgen çıkar ve eval_iou.py bunu IoU=0 sayar — 4 nokta
geometrik olarak kusursuz olsa bile.

Bu script hiçbir şeyi değiştirmez, sadece raporlar:
  - winding (saat yönü / ters) dağılımı
  - ilk köşenin, kutunun kendi ekseninde hangi köşeye denk geldiği
    (sol-üst / sağ-üst / sağ-alt / sol-alt)
  - etiketlerde kendisiyle kesişen (geçersiz) dörtgen var mı
  - sınıf bazında kırılım + kutu boyutları

Çalıştırma:
    python check_corner_order.py
"""

import json
import os
from collections import Counter, defaultdict

import numpy as np

from training import JSON_DIR, LABEL_TO_ID, shape_to_quad

CORNER_NAMES = ["sol-üst", "sağ-üst", "sağ-alt", "sol-alt"]


def signed_area(pts):
    """Shoelace. >0 => saat yönünün tersi (görüntü koordinatlarında saat yönü,
    çünkü y aşağı doğru artar)."""
    p = np.asarray(pts, dtype=float)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def is_simple(pts):
    """Dörtgen kendisiyle kesişiyor mu (papyon)?"""
    try:
        from shapely.geometry import Polygon
        return bool(Polygon(pts).is_valid)
    except ImportError:
        # shapely yoksa: köşegenlerin kesişip kesişmediğine bakan basit kontrol
        p = np.asarray(pts, dtype=float)
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        def seg_x(a, b, c, d):
            d1, d2 = cross(c, d, a), cross(c, d, b)
            d3, d4 = cross(a, b, c), cross(a, b, d)
            return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
        return not (seg_x(p[0], p[1], p[2], p[3]) or seg_x(p[1], p[2], p[3], p[0]))


def first_corner_quadrant(pts):
    """İlk köşe, kutunun ağırlık merkezine göre hangi çeyrekte?
    Bu, 'annotator hangi köşeden başlamış' sorusunun döndürülmüş kutularda da
    çalışan halidir."""
    p = np.asarray(pts, dtype=float)
    c = p.mean(axis=0)
    dx, dy = p[0] - c
    if dx <= 0 and dy <= 0: return 0   # sol-üst
    if dx >  0 and dy <= 0: return 1   # sağ-üst
    if dx >  0 and dy >  0: return 2   # sağ-alt
    return 3                            # sol-alt


def main():
    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"{len(files)} json taranıyor: {JSON_DIR}\n")

    winding      = Counter()
    first_corner = Counter()
    invalid      = 0
    total        = 0
    n_rect       = 0
    per_class    = defaultdict(lambda: {"first": Counter(), "wind": Counter(),
                                        "n": 0, "w": [], "h": []})

    for fn in files:
        try:
            with open(os.path.join(JSON_DIR, fn), encoding="utf-8") as f:
                item = json.load(f)
        except Exception as e:
            print(f"  atlandı {fn}: {e}")
            continue

        for shape in item.get("shapes", []):
            label = shape.get("label", "")
            if label not in LABEL_TO_ID:
                continue
            # rectangle (2 nokta) shape'leri de sayılıyor; shape_to_quad onları
            # daima aynı kanonik sırada (sol-üst -> sağ-üst -> sağ-alt -> sol-alt)
            # üretir, dolayısıyla bu araç bbox etiketli bir datasette "%100
            # tutarlı" raporlar — ki doğrusu da budur.
            pts = shape_to_quad(shape)
            if pts is None:
                continue
            if shape.get("shape_type") == "rectangle" or len(shape.get("points", [])) == 2:
                n_rect += 1

            total += 1
            w = winding_dir = "saat yönü" if signed_area(pts) > 0 else "ters yön"
            q = first_corner_quadrant(pts)

            winding[w] += 1
            first_corner[CORNER_NAMES[q]] += 1
            if not is_simple(pts):
                invalid += 1

            s = per_class[label]
            s["n"] += 1
            s["first"][CORNER_NAMES[q]] += 1
            s["wind"][w] += 1
            arr = np.asarray(pts, dtype=float)
            s["w"].append(arr[:, 0].max() - arr[:, 0].min())
            s["h"].append(arr[:, 1].max() - arr[:, 1].min())

    if total == 0:
        print("Hiç geçerli shape bulunamadı.")
        return

    print("═" * 66)
    print(f"  Toplam {total} obje" +
          (f" ({n_rect} tanesi bbox/rectangle -> köşe sırası tanım gereği tutarlı)"
           if n_rect else ""))
    print("═" * 66)
    if n_rect == total:
        print("  Dataset tamamen bbox etiketli: köşe sırası bu araçla ölçülecek bir\n"
              "  serbestlik derecesi içermiyor. OUTPUT_MODE='hbb' kullanın.\n")

    print("\n  Winding (dönüş yönü):")
    for k, v in winding.most_common():
        print(f"    {k:<12} {v:6d}  ({100*v/total:5.1f}%)")

    print("\n  İlk köşe hangi çeyrekte (annotator nereden başlamış):")
    for name in CORNER_NAMES:
        v = first_corner.get(name, 0)
        bar = "█" * int(40 * v / total)
        print(f"    {name:<10} {v:6d}  ({100*v/total:5.1f}%)  {bar}")

    print(f"\n  Kendisiyle kesişen (geçersiz) dörtgen: {invalid} / {total} "
          f"({100*invalid/total:.2f}%)")

    print("\n" + "═" * 66)
    print(f"  {'Sınıf':<20} {'n':>5} {'baskın ilk köşe':>18} {'tutarlılık':>11} "
          f"{'ort w×h':>12}")
    print("  " + "-" * 64)
    for label, s in sorted(per_class.items()):
        top, cnt = s["first"].most_common(1)[0]
        print(f"  {label:<20} {s['n']:>5} {top:>18} {100*cnt/s['n']:>10.1f}% "
              f"{np.mean(s['w']):>6.0f}×{np.mean(s['h']):<5.0f}")

    dom_name, dom_cnt = first_corner.most_common(1)[0]
    consistency = dom_cnt / total
    dom_wind_cnt = winding.most_common(1)[0][1]
    wind_consistency = dom_wind_cnt / total

    print("\n" + "═" * 66)
    print("  SONUÇ")
    print("═" * 66)
    print(f"  İlk köşe tutarlılığı : {100*consistency:.1f}%  (baskın: {dom_name})")
    print(f"  Winding tutarlılığı  : {100*wind_consistency:.1f}%")
    if consistency > 0.98 and wind_consistency > 0.98:
        print("\n  → Etiketler tutarlı. Köşe sırası öğrenilebilir bir hedef;")
        print("    kalan hata lokalizasyondan geliyor demektir.")
    else:
        print("\n  → Etiketler TUTARSIZ. Model köşe sırasını öğrenemez; ince")
        print("    kutularda iki köşeyi yer değiştirip 'papyon' dörtgen üretir")
        print("    ve eval_iou.py bunları IoU=0 sayar.")
        print("    Çözüm: training.py'de CANONICAL_CORNER_ORDER = True yapın —")
        print("    köşeler ağırlık merkezi etrafında açıya göre sıralanıp daima")
        print("    sol-üste en yakın köşeden başlatılır, hedef deterministik olur.")


if __name__ == "__main__":
    main()
