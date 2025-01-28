from transformers import CLIPProcessor, TFCLIPModel
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.layers import Dense, Lambda


clip_model_name = "openai/clip-vit-base-patch32"
clip_model = TFCLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)


def get_clip_embeddings_from_base64(images, texts, embedding_dim=1024, alpha=0.5):
    image_embeds = []
    text_embeds = []

        # Decode base64 images
        # images = []
        # for base64_img in base64_images:
        #     # try:
        #     #     image_data = base64.b64decode(base64_img)
        #     #     print(image_data)
        #     #     image = Image.open(BytesIO(image_data)).convert("RGB")
        #     #     images.append(image)
        #     # except Exception as e:
        #     #     print(f"Error decoding base64 image: {e}")
        #     #     continue

        # Process images and texts with CLIP
    inputs = clip_processor(text=texts, images=images, return_tensors="tf", padding=True)
    outputs = clip_model(**inputs)

    # Get CLIP embeddings
    image_embeds = outputs.image_embeds.numpy()
    text_embeds = outputs.text_embeds.numpy()

    # Define linear projection layers
    image_proj_layer = Dense(embedding_dim, name="image_projection")
    text_proj_layer = Dense(embedding_dim, name="text_projection")

    # Apply projection
    F_proj_I = image_proj_layer(image_embeds)
    F_proj_T = text_proj_layer(text_embeds)

    # Define feature adapters
    image_adapter = Dense(embedding_dim, activation='relu', name="image_adapter")
    text_adapter = Dense(embedding_dim, activation='relu', name="text_adapter")

    # Apply adapters
    A_I = image_adapter(F_proj_I)
    A_T = text_adapter(F_proj_T)

    # Compute final representations with residual connections
    F_I = alpha * A_I + (1 - alpha) * F_proj_I
    F_T = alpha * A_T + (1 - alpha) * F_proj_T

    return F_I.numpy(), F_T.numpy()

def embedding_fusion(img_emb, text_emb): 
    return img_emb * text_emb

# if __name__ == "__main__": 
#     image_embeds, text_embeds = get_clip_embeddings_from_base64(base64_images, texts, embedding_dim=1024, alpha=0.5)
#     final_feat = embedding_fusion(image_embeds, text_embeds)