import os
from fastapi import FastAPI, HTTPException, Response
import uvicorn
import time
from pydantic import BaseModel
from PIL import Image
import logging
import base64
from io import BytesIO
from typing import List
import asyncio
import numpy as np
import tensorflow as tf
from .utils import image_processing_pipeline, text_preprocessing_pipeline, get_clip_embeddings_from_base64, embedding_fusion, meme_explanation
# from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.cors import CORSMiddleware

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
    tflite_interpreter = tf.lite.Interpreter(model_path="./model_checkpoints/best_model.tflite")
    tflite_interpreter.allocate_tensors()

    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
except Exception as e:
    logger.debug(f"Current directory: {os.getcwd()}")
    logger.error(f"Error loading TFLite model: {e}")
    raise HTTPException(status_code=500, detail="Error loading model")


def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    
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


@app.post("/predict")
async def predict(image_request: ImageRequest, response: Response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    predictions = []
    processing_time = 0
    explanation = None

    start_time = time.time()

    try:
        text, processed_image = await process_text_and_image(image_request.images[0])
        text_emb, img_emb = get_clip_embeddings_from_base64(processed_image, text)

        tflite_interpreter.set_tensor(input_details[0]['index'], text_emb)  
        tflite_interpreter.set_tensor(input_details[1]['index'], img_emb)

        tflite_interpreter.invoke()

        pred = tflite_interpreter.get_tensor(output_details[0]['index'])
        predicted_class = int(pred[0] >= 0.5)
        confidence = float(pred[0])

        predictions.append({
            "predicted_class": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    end_time = time.time()
    processing_time = end_time - start_time

    # Explanation (if necessary, can be an additional function)
    explanation = meme_explanation(image_request.images[0], predictions[0])

    logger.debug(predictions)

    return {
        "predictions": predictions[0],
        "time_taken": processing_time,
        "explanation": explanation
    }


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.getenv('PORT', 8000))