import os
import torch
from PIL import Image

from pix2seq_decode import load_model, predict, draw_predictions


def _resolve_image_path(image_path):
    if not os.path.exists(image_path) or image_path.endswith(".json"):
        base = os.path.splitext(image_path)[0]
        print("Base:", base)
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(base + ext):
                return base + ext
    return image_path


def _run_and_save(image_path, model, device, folder_mode, score_threshold):
    image_path = _resolve_image_path(image_path)
    original_img = Image.open(image_path).convert("RGB")

    predictions, sequence = predict(model, original_img, device, score_threshold=score_threshold)
    print(f"Üretilen Ham Tokenlar: {sequence}")

    draw_predictions(original_img, predictions)

    if not folder_mode:
        output_path = "inference_result.jpg"
    else:
        os.makedirs("test_output", exist_ok=True)
        output_path = f"test_output/{os.path.basename(image_path)}"

    original_img.save(output_path)
    print(f"\nİşlem tamam! Sonuç '{output_path}' dosyasına kaydedildi.")
    return predictions


def predict_and_draw(image_path, model_path, folder_mode=False, score_threshold=0.0):
    """Modeli diskten yükler ve tek bir görüntü üzerinde çalıştırır."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference {device} cihazında yapılıyor...")
    model = load_model(model_path, device)
    return _run_and_save(image_path, model, device, folder_mode, score_threshold)


def predict_and_draw_nomodel(image_path, model, folder_mode=False, score_threshold=0.0):
    """Zaten belleğe yüklenmiş bir modelle çalışır (toplu/klasör işlemlerinde
    modeli her görüntüde yeniden yüklememek için)."""
    device = next(model.parameters()).device
    return _run_and_save(image_path, model, device, folder_mode, score_threshold)


if __name__ == "__main__":
    # Test edilecek resmin yolu
    TEST_IMAGE_PATH = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01/1477f2c5-0606-45ea-8403-c8e80a437d5e.json"
    #TEST_IMAGE_PATH = "/home/uygarusta/Oriented-Centernet/pix2seq/test_images/4d8b2d02-47bb-49ca-a810-a4cb577e1ddf.jpeg"
    MODEL_PATH = "pix2seq_best.pth"

    predict_and_draw(TEST_IMAGE_PATH, MODEL_PATH)

    FOLDER_PATH = ""  # "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01"
    if FOLDER_PATH:
        print("FOLDER_PATH active!")
        from glob import glob
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(MODEL_PATH, device)
        FOLDER_LIST = glob(os.path.join(FOLDER_PATH, "*"))
        for image in tqdm(FOLDER_LIST):
            predict_and_draw_nomodel(image, model, folder_mode=True)

    VAL_JSON = "val_split.json"

    if VAL_JSON:
        print("VAL_JSON active!")
        import json
        from tqdm import tqdm

        if not os.path.exists(VAL_JSON):
            raise FileNotFoundError(
                f"{VAL_JSON} not found. Run training first to generate the val split."
            )
        with open(VAL_JSON) as f:
            val_filenames = json.load(f)["filenames"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Inference {device} cihazında yapılıyor...")
        model = load_model("pix2seq_best.pth", device)

        for fname in tqdm(val_filenames):
            fname = f"/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_2023_01_01__2023_03_01/{fname}"
            predict_and_draw_nomodel(fname, model, folder_mode=True)
