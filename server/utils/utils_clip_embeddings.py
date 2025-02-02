from transformers import CLIPProcessor, TFCLIPModel # type: ignore
import numpy as np # type: ignore
import base64
from io import BytesIO
from tensorflow.keras.layers import Dense, Lambda # type: ignore


clip_model_name = "openai/clip-vit-base-patch32"
clip_model = TFCLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)


def get_clip_embeddings_from_base64(images, texts, embedding_dim=1024):
    image_embeds = []
    text_embeds = []
    
    inputs = clip_processor(text=texts, images=images, return_tensors="tf", padding=True)
    outputs = clip_model(**inputs)

    # Get CLIP embeddings
    image_embeds = outputs.image_embeds.numpy()
    text_embeds = outputs.text_embeds.numpy()

    return image_embeds, text_embeds

def embedding_fusion(img_emb, text_emb): 
    return img_emb * text_emb