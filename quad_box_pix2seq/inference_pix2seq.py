import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from training_pix2seq_quad  import (
    Pix2SeqModel, VOCAB_SIZE, NUM_BINS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
    LABEL_TO_ID, ID_TO_LABEL, IMG_SIZE, MAX_OBJECTS,
    NOISE_CLASS_TOKEN, NUM_CLASSES, NUM_CLASS_SLOTS,
    NOISE_FILL_TO_MAX, INFER_SCORE_THRESH
)

def dequantize(token, max_size, num_bins):
    return (token / (num_bins - 1)) * max_size


def pad_to_square_infer(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (128, 128, 128))
    new_img.paste(img, (0, 0))
    return new_img, max_dim

def strip_specials(sequence):
    """BOS'u atar, EOS'ta keser. Kalan PAD'ler stride'ı bozmasın diye BIRAKILMAZ
    ama silinmez de: EOS zaten dizinin sonunu işaretler."""
    toks = list(sequence)
    if toks and toks[0] == BOS_TOKEN:
        toks = toks[1:]
    if EOS_TOKEN in toks:
        toks = toks[:toks.index(EOS_TOKEN)]
    return toks


def tokens_to_predictions(tokens, max_dim):
    """
    De-tokenize a flat token list into predictions.
    Coordinates are returned in padded-square space [0, max_dim].
    """
    img_h_model, img_w_model = IMG_SIZE  # 512, 512

    predictions = []
    for i in range(0, len(tokens) - (len(tokens) % 9), 9):
        obj_tokens = tokens[i:i+9]

        # Noise objesi de tam 9 token kaplar -> `continue` hizalamayı bozmaz.
        # NOT: "noise" LABEL_TO_ID'de DEĞİL, ayrı bir token; etiket string'i
        # üzerinden filtrelemeye gerek yok, sınıf aralığı kontrolü zaten eler.
        if obj_tokens[8] == NOISE_CLASS_TOKEN:
            continue

        coords_valid = all(0 <= obj_tokens[j] < NUM_BINS for j in range(8))
        class_valid  = NUM_BINS <= obj_tokens[8] < NUM_BINS + NUM_CLASSES
        if not coords_valid or not class_valid:
            continue

        class_id = obj_tokens[8] - NUM_BINS
        label    = ID_TO_LABEL.get(class_id, f"Bilinmeyen_{class_id}")

        points = []
        for j in range(0, 8, 2):
            x_tok = obj_tokens[j]
            y_tok = obj_tokens[j + 1]

            x_model = dequantize(x_tok, img_w_model, NUM_BINS)
            y_model = dequantize(y_tok, img_h_model, NUM_BINS)

            x_padded = x_model * (max_dim / img_w_model)
            y_padded = y_model * (max_dim / img_h_model)

            points.append((x_padded, y_padded))

        predictions.append({"label": label, "points": points})

    return predictions


def draw_predictions(padded_img, predictions, orig_w, orig_h):
    """
    Draw predictions on the padded image then crop back to original size.
    This avoids any coordinate remapping — everything stays in padded space.
    """
    # FIX: draw on the padded image, not original_img.
    #      Coordinates from the model are in padded space so this is exact.
    result = padded_img.copy()
    draw   = ImageDraw.Draw(result)

    for pred in predictions:
        pts   = pred["points"]
        label = pred["label"]

        flat_points = [coord for pt in pts for coord in pt]
        draw.polygon(flat_points, outline="red", width=3)

        min_x = min(p[0] for p in pts)
        min_y = min(p[1] for p in pts)

        draw.rectangle([min_x, max(0, min_y - 15), min_x + 80, min_y], fill="red")
        draw.text((min_x + 2, max(0, min_y - 15)), label, fill="white")

    # FIX: crop back to original image dimensions (removes gray padding)
    result = result.crop((0, 0, orig_w, orig_h))
    return result


def run_autoregressive(model, img_tensor, max_seq_len, device, fixed_length=NOISE_FILL_TO_MAX):
    """
    fixed_length=True: tam 9*MAX_OBJECTS adım, EOS beklenmez (model EOS görmedi),
    koordinat adımlarında sadece bin tokenları, sınıf adımlarında sadece sınıf
    tokenları seçilir -> 9'luk hizalama garanti.
    """
    sequence = [BOS_TOKEN]
    n_steps  = 9 * MAX_OBJECTS if fixed_length else max_seq_len
    cls_lo, cls_hi = NUM_BINS, NUM_BINS + NUM_CLASS_SLOTS

    with torch.no_grad():
        for step in range(n_steps):
            tgt_seq = torch.tensor([sequence], dtype=torch.long).to(device)
            logits  = model(img_tensor, tgt_seq)[0, -1, :]

            if fixed_length:
                if step % 9 == 8:
                    next_token = int(torch.argmax(logits[cls_lo:cls_hi]).item()) + cls_lo
                else:
                    next_token = int(torch.argmax(logits[:NUM_BINS]).item())
            else:
                next_token = int(torch.argmax(logits).item())

            sequence.append(next_token)
            if not fixed_length and next_token == EOS_TOKEN:
                break
    return sequence


def resolve_image_path(image_path):
    """Try common extensions if path doesn't exist or is a json."""
    if not os.path.exists(image_path) or image_path.endswith(".json"):
        base = os.path.splitext(image_path)[0]
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".JPG", ".JPEG", ".PNG"]:
            if os.path.exists(base + ext):
                return base + ext
    return image_path


def preprocess_image(image_path):
    """Load, pad, resize, normalize. Returns (img_tensor, padded_img, orig_w, orig_h, max_dim)."""
    original_img        = Image.open(image_path).convert("RGB")
    orig_w, orig_h      = original_img.size
    padded_img, max_dim = pad_to_square_infer(original_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_tensor = TF.resize(padded_img, IMG_SIZE)
    img_tensor = transform(img_tensor).unsqueeze(0)
    return img_tensor, padded_img, orig_w, orig_h, max_dim


# ── Public API ────────────────────────────────────────────────────────────────

def predict_and_draw(image_path, model_path, original_image_size=None, folder_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference {device} cihazında yapılıyor...")

    max_seq_len = 1 + (9 * MAX_OBJECTS) + 1
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
    # state_dict = torch.load(model_path, map_location=device)
    # clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    # model.load_state_dict(clean_state_dict)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    state_dict = torch.load(model_path, map_location=device)
    clean_state_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    image_path = resolve_image_path(image_path)
    img_tensor, padded_img, orig_w, orig_h, max_dim = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    sequence = run_autoregressive(model, img_tensor, max_seq_len, device)
    print(f"Üretilen Ham Tokenlar: {sequence}")

    tokens      = strip_specials(sequence)
    predictions = tokens_to_predictions(tokens, max_dim)

    # FIX: draw on padded image and crop — replaces drawing on original_img
    result_img  = draw_predictions(padded_img, predictions, orig_w, orig_h)

    if not folder_mode:
        output_path = "inference_result.jpg"
    else:
        os.makedirs("test_output", exist_ok=True)
        output_path = f"test_output/{os.path.basename(image_path)}"

    result_img.save(output_path)
    print(f"\nİşlem tamam! Sonuç '{output_path}' dosyasına kaydedildi.")


def predict_and_draw_nomodel(image_path, model, device, folder_mode=False):
    # FIX: device is now an explicit parameter — was undefined global in original
    max_seq_len = 1 + (9 * MAX_OBJECTS) + 1

    image_path = resolve_image_path(image_path)
    if not os.path.exists(image_path):
        print(f"  Atlandı (bulunamadı): {image_path}")
        return

    img_tensor, padded_img, orig_w, orig_h, max_dim = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)

    sequence = run_autoregressive(model, img_tensor, max_seq_len, device)
    print(f"Üretilen Ham Tokenlar: {sequence}")

    tokens      = strip_specials(sequence)
    predictions = tokens_to_predictions(tokens, max_dim)

    # FIX: same draw fix as predict_and_draw
    result_img  = draw_predictions(padded_img, predictions, orig_w, orig_h)

    if not folder_mode:
        output_path = "inference_result.jpg"
    else:
        os.makedirs("test_output", exist_ok=True)
        output_path = f"test_output/{os.path.basename(image_path)}"

    result_img.save(output_path)
    print(f"\nİşlem tamam! Sonuç '{output_path}' dosyasına kaydedildi.")


if __name__ == "__main__":
    TEST_IMAGE_PATH = "test_images/2ruhsat.png" 
    MODEL_PATH      = "pix2seq_best_map_quad.pth"

    #predict_and_draw(TEST_IMAGE_PATH, MODEL_PATH)

    FOLDER_PATH = ""
    if FOLDER_PATH:
        print("FOLDER_PATH active!")
        from glob import glob
        from tqdm import tqdm
        for image in tqdm(glob(os.path.join(FOLDER_PATH, "*"))):
            predict_and_draw(image, MODEL_PATH, folder_mode=True)

    VAL_JSON = "val_split.json"
    if VAL_JSON:
        print("VAL_JSON active!")
        import json
        from tqdm import tqdm

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_seq_len = 1 + (9 * MAX_OBJECTS) + 1
        model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=max_seq_len).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        with open(VAL_JSON) as f:
            val_filenames = json.load(f)["filenames"]

        DATA_DIR = "/home/uygarusta/Oriented-Centernet/ruhsat_detection/dataset/ruhsat_extended/"
        for fname in tqdm(val_filenames):
            full_path = os.path.join(DATA_DIR, fname)
            # FIX: pass device explicitly — was undefined in original
            predict_and_draw_nomodel(full_path, model, device, folder_mode=True)