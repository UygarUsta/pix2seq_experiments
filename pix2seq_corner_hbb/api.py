import os
import io
import base64
import torch
import asyncio
import aiohttp
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel

from training import Pix2SeqModel, VOCAB_SIZE
from pix2seq_decode import MAX_SEQ_LEN, predict, draw_predictions

# --- CONFIGURATION & GLOBALS ---
MODEL_PATH = "pix2seq_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCORE_THRESHOLD = 0.0  # /predict yanıtını score'a göre filtrelemek için > 0 yapılabilir

# Global variables to hold the model and the queue
model = None
request_queue = asyncio.Queue()


class JsonPredictRequest(BaseModel):
    url: str | None = None
    image_base64: str | None = None


# --- CORE INFERENCE LOGIC ---
def process_image_sync(image_bytes: bytes):
    """Synchronous CPU/GPU bound task that runs the actual inference.
    Encoder bir kez çalışır (model.encode), autoregressive decode döngüsü
    sadece model.decode çağırır — bkz. pix2seq_decode.py."""
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    predictions, sequence = predict(model, original_img, DEVICE, score_threshold=SCORE_THRESHOLD)

    drawn_img = draw_predictions(original_img.copy(), predictions)
    buffered = io.BytesIO()
    drawn_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        "predictions": predictions,
        "image_base64": img_str,
        "raw_tokens": sequence,
    }


# --- BACKGROUND QUEUE WORKER ---
async def inference_worker():
    """Continuously pulls requests from the queue and processes them."""
    while True:
        # Wait for a request to be added to the queue
        image_bytes, future = await request_queue.get()
        try:
            # Run the heavy PyTorch task in a separate thread so it doesn't block FastAPI
            result = await asyncio.to_thread(process_image_sync, image_bytes)
            # Send the result back to the specific request that asked for it
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()

# --- FASTAPI LIFECYCLE ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Startup: Load model ONCE into memory
    global model
    print(f"Loading model on {DEVICE}...")
    model = Pix2SeqModel(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")

    model.eval()

    # 2. Startup: Start the queue worker
    worker_task = asyncio.create_task(inference_worker())

    yield # API is now running and accepting requests

    # 3. Shutdown: Clean up
    worker_task.cancel()
    print("Shutting down API...")

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan, title="Pix2Seq Inference API")

# --- ENDPOINTS ---
async def fetch_image_from_url(url: str) -> bytes:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download image from URL: {str(e)}")

@app.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(None),
    url: str = Form(None)
):
    """
    Accepts EITHER an uploaded file (local inference) OR an image URL.
    Places the job in a queue and waits for the worker to process it.
    """
    if not file and not url:
        raise HTTPException(status_code=400, detail="Must provide either a 'file' or a 'url'.")

    # 1. Get Image Bytes
    if file:
        image_bytes = await file.read()
    else:
        image_bytes = await fetch_image_from_url(url)

    # 2. Create a future to listen for the result
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    # 3. Put job in the queue
    await request_queue.put((image_bytes, future))

    # 4. Wait for the background worker to finish this specific job
    try:
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")



@app.post("/predict_json")
async def predict_json(request: JsonPredictRequest):
    """
    Accepts an application/json payload with either a URL or a base64 encoded image.
    """
    if not request.url and not request.image_base64:
        raise HTTPException(status_code=400, detail="Must provide either 'url' or 'image_base64'.")

    # 1. Get Image Bytes
    if request.image_base64:
        try:
            # Strip the "data:image/jpeg;base64," prefix if the client includes it
            base64_str = request.image_base64
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_bytes = base64.b64decode(base64_str)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image string.")
    else:
        image_bytes = await fetch_image_from_url(request.url)

    # 2. Put job in the queue
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await request_queue.put((image_bytes, future))

    # 3. Wait for result
    try:
        result = await future
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
