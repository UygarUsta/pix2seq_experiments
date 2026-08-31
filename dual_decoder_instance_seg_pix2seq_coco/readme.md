# Dual-Decoder Pix2Seq — Instance Segmentation (COCO)

Decoder 1 kutu + sınıf üretir, Decoder 2 her kutuyu koşul alıp maskeyi poligon
olarak üretir. Sequence augmentation (noise nesneleri) her iki tarafta da açık.

## Bu sürümde değişenler

| | eski | yeni |
|---|---|---|
| veri | tek sınıf, labelme json | COCO instance segmentation (80 sınıf) |
| backbone | ResNet18 | ResNet50 (ImageNet, ayrı düşük lr) |
| transformer encoder | yok | 6 katman |
| decoder | 4 katman x2 | 6 katman x2 |
| çözünürlük | 512 | 640 |
| MAX_OBJECTS | 10 | 100 (paper) |
| augmentasyon | ShiftScaleRotate/Perspective | LSJ (scale 0.1–2.0 + pad + crop) + HFlip + ColorJitter |
| nesne sırası | %50 permütasyon | her gösterimde rastgele |
| weight decay | 1e-4 | 0.05 (norm/bias/embedding hariç) |
| stochastic depth | yok | 0.1 (encoder + iki decoder) |
| label smoothing | 0.1 | 0.0 |
| lr schedule | OneCycle | warmup (%3) + lineer düşüş |
| efektif batch | 16 | 16 x 4 = 64 |
| grad clip | 1.0 | 0.1 |
| decode | naif AR | KV-cache'li AR |

Token uzayı: 500 bin + 80 sınıf + noise + BOS/EOS/PAD = 584. Kutu dizisi 501
token (BOS + 100 x 5), maske dizisi 39 token (BOS + kutu + sınıf + 32 poligon
koordinatı + EOS).

## Veri düzeni

```
COCO_ROOT/
    train2017/*.jpg
    val2017/*.jpg
    annotations/instances_train2017.json
    annotations/instances_val2017.json
```

`dual_decoder_pix2seq.py` içindeki `COCO_ROOT`'u ayarla. İlk çalıştırmada
annotation json'ı taranıp `cocoidx_*.npz` index'ine yazılır; sonraki koşular
saniyeler içinde açılır ve index düz numpy dizisi olduğu için DataLoader
worker'ları fork ettiğinde kopyalanmaz.

Annotation filtreleri: `iscrowd=1` atlanır, RLE segmentasyonlar atlanır (bu
model poligon üretiyor), çok parçalı segmentasyonlarda **en büyük parça**
alınır, alanı 16 pikselden küçük nesneler atılır.

## Çalıştırma

```bash
python aug_check_dual_pix2seq.py --n 24      # önce bunu çalıştır
python aug_check_dual_pix2seq.py --val      # noise çıkmamalı
python dual_decoder_pix2seq.py              # eğitim
python inference_dual_pix2seq.py            # çıkarım + görselleştirme
```

`aug_check` kutuları ara veri yapısından değil, dataset'in döndürdüğü token
dizilerinden geri çözerek çizer; yani modelin gerçekten gördüğü şeyi
gösterir ve dizi hizasını assert'lerle doğrular.

## Bilinmesi gerekenler

**Bellek.** 640 + ResNet50 + 6L encoder + 2x6L decoder, batch 16 bf16 ile
yaklaşık 18–22 GB. OOM alırsan `BATCH_SIZE=8, GRAD_ACCUM=8` yap; efektif batch
64'te kalır.

**Süre.** COCO train2017 118k görüntü, batch 16 -> epoch başına ~7.4k iterasyon.
Tek GPU'da 50 epoch ciddi bir koşu (tipik olarak 1–2 hafta). Önce 5 epoch'luk
bir koşuyla loss eğrisinin ve `preds_per_img` sayısının makul olduğunu doğrula.

**Eval.** torchmetrics maskeleri `compute()`'a kadar RAM'de tutuyor; tam
val2017 maske mAP'i 10 GB'ı aşabilir. Bu yüzden eğitim içindeki değerlendirme
`EVAL_MAX_IMAGES=500` ile sınırlı — sıralamayı izlemek için yeter, yayınlanacak
sayı için ayrı bir koşuda `max_images=None` + `MASK_EVAL_SIZE=192` kullan ya da
doğrudan pycocotools/RLE ile ölç.

**Poligon tavanı.** `NUM_POLY_PTS=16`, maske AP'sinin üst sınırını belirliyor;
ince/karmaşık nesnelerde 24 veya 32 belirgin fark yaratır (maske dizisi uzar,
decoder 2 maliyeti lineer artar).

**AP_small.** `DualPix2SeqModel(dilated_c5=True)` çıktı stride'ını 32'den 16'ya
indirir (paper'ın DC5 varyantı). Küçük nesneler için en büyük tek kaldıraç, ama
encoder self-attention maliyeti 16x (400 -> 1600 token).
