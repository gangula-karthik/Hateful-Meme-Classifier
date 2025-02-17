import os
from fastapi import FastAPI, HTTPException, Response
import uvicorn
import time
import re
from pydantic import BaseModel
from PIL import Image
import random
import string
import logging
import base64
from io import BytesIO
from typing import List
import asyncio
import numpy as np
import tensorflow as tf
from .utils import image_processing_pipeline, text_preprocessing_pipeline, get_clip_embeddings_from_base64, embedding_fusion, meme_explanation
from starlette.middleware.cors import CORSMiddleware
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    images: List[str]

try:
    tflite_interpreter = tf.lite.Interpreter(
        model_path="./model_checkpoints/best_model.tflite",
        experimental_delegates=[tf.lite.experimental.load_delegate('libtensorflowlite_xnnpack.so')]
    )
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
except Exception as e:
    raise HTTPException(status_code=500, detail="Error loading model")


def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

def insert_judge_output(res, predictions, image):
    """Uploads judge LLM output & image to Supabase, ensuring unique filename."""
    
    def rand_filename(ext="png", length=8):
        return f"{''.join(random.choices(string.ascii_letters + string.digits, k=length))}.{ext}"
    
    res = {
        "judge_model": 1 if res["final_label"] == "harmful" else 0,
        "prod_model": predictions["predicted_class"],
        "judge_explanation": res["explanation"]
    }

    for _ in range(5):  # Max 5 retries for unique filename
        file_name = rand_filename()
        image = re.sub(r"^data:image/\w+;base64,", "", image)
        try:
            meme_upload = supabase.storage.from_("memes").upload(
                path=file_name,
                file=base64.b64decode(image),
                file_options={"cache-control": "3600", "upsert": "false"},
            )
            res["meme_url"] = supabase.storage.from_("memes").get_public_url(meme_upload.path)
            break  # Success
        except Exception as e:
            if "already exists" not in str(e):
                return {"error": str(e)}  # Unexpected error, exit

    return supabase.table("harmful-meme-retraining-loop").insert(res).execute()
    
async def process_text_and_image(image_base64):
    """Run text and image processing in parallel."""
    text_task = asyncio.to_thread(text_preprocessing_pipeline, image_base64)
    image_task = asyncio.to_thread(image_processing_pipeline, image_base64)

    text, processed_image = await asyncio.gather(text_task, image_task)
    return text, processed_image


@app.get("/", tags=["healthCheck"])
def health(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return {"status": "OK", "message": "Welcome to the hateful-meme-classifier microservice. Please refer to /docs for further details."}

@app.get("/fetch_model_performance")
def model_performance(response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response = supabase.table("harmful-meme-retraining-loop").select("*").execute()
    logger.debug(response)
    return response

@app.post("/predict")
async def predict(image_request: ImageRequest, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    processing_time = 0

    start_time = time.perf_counter()

    try:
        text, processed_image = await process_text_and_image(image_request.images[0])
        text_emb, img_emb = get_clip_embeddings_from_base64(processed_image, text)

        tflite_interpreter.set_tensor(input_details[0]['index'], text_emb)  
        tflite_interpreter.set_tensor(input_details[1]['index'], img_emb)

        tflite_interpreter.invoke()

        pred = tflite_interpreter.get_tensor(output_details[0]['index'])
        predicted_class = int(pred[0] > 0.6)
        confidence = float(pred[0])

        predictions = {
            "predicted_class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    end_time = time.perf_counter()
    processing_time = end_time - start_time

    # Explanation
    judge_output = meme_explanation(image_request.images[0], predictions)

    output = {
        "predictions": predictions,
        "time_taken": processing_time,
        "explanation": judge_output['explanation']
    }

    # adding to supbase
    insert_judge_output(judge_output, predictions, image_request.images[0])

    return output


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.getenv('PORT', 8000))
