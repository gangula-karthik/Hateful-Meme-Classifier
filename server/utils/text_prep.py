import base64
import os
import re
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from tensorflow.keras.layers import Dense, Lambda

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text(image):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all the relevant text from this meme. If in another language, translate text into English. If there is no text found the caption the image in a very short manner. Avoid identifying watermarks.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                },
            ],
        }
    ],
    
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "ocr_output",
            "schema": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string"}
                },
                "required": ["final_answer"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    )

    res = json.loads(response.choices[0].message.content)["final_answer"]
    print(res)
    return res

def masking_usernames(text): 
    username_pattern = r'@\w+'
    return re.sub(username_pattern, '[USERNAME]', text)

def filter_long_text(text): 
    if len(text.split(" ")) > 100: 
        return False
    return True

def text_preprocessing_pipeline(image): 
    res = extract_text(image)
    if filter_long_text(res):
        res = masking_usernames(res)
    else:
        return "Text is too long"
    
    return res


def meme_explanation(image, predictions):
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"The given meme is predicted by the model to be: {predictions}. Do you agree with the model ? Explain your reasoning and the consequences if it is a harmful meme.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image},
                },
            ],
        }
    ],
    
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "explainability_output",
            "schema": {
                "type": "object",
                "properties": {
                    "final_answer": {"type": "string"}
                },
                "required": ["final_answer"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
    )

    res = json.loads(response.choices[0].message.content)["final_answer"]
    return res