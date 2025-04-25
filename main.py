from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import numpy as np
import tensorflow as tf
import asyncio
import cv2
from io import BytesIO
from openai import OpenAI, APIError, AuthenticationError
from PIL import Image
import logging

# Silence TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === Initialize FastAPI ===
app = FastAPI()

# === Constants ===
MODEL_FILE = "skin_disease_model.keras"
IMG_SIZE = 224
CLASS_LABELS = [
    'Acne', "Athlete's Foot", 'Cellulitis', 'Chickenpox',
    'Cutaneous Larva Migrans', 'Impetigo', 'Nail-Fungus',
    'Normal', 'Ringworm', 'Shingles'
]

model = None

# === Load Model Once ===
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")

# === Initialize OpenAI ===
ai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com"
)

# === Utility Functions ===
def is_valid_skin_image(image_bytes: bytes) -> bool:
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        if img_array.shape[0] < 100 or img_array.shape[1] < 100:
            return False
        hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        return skin_ratio > 0.2
    except Exception as e:
        print(f"[Image Check Error] {e}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def getDescription(disease: str) -> str:
    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Give me a description about this {disease}. Only the description. Don't include special characters and don't ask questions at the end"
            }],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except (AuthenticationError, APIError) as e:
        print(f"[OpenAI Description Error] {e}")
        return "Error: Unable to fetch description."
    except Exception as e:
        print(f"[Unexpected Description Error] {e}")
        return "Error: An unexpected error occurred."

def getRemedy(disease: str) -> str:
    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Give me the best non medical remedies about {disease}. Only the remedy and how they use it. Don't include special characters except for numbers, Add point special character for numbers. Don't add any drug medication only the non-medical. And, dont ask questions at the end. Only include 10 non medical remedies."
            }],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except (AuthenticationError, APIError) as e:
        print(f"[OpenAI Remedy Error] {e}")
        return "Error: Unable to fetch remedies."
    except Exception as e:
        print(f"[Unexpected Remedy Error] {e}")
        return "Error: An unexpected error occurred."

# === Main Prediction Route ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        if not await asyncio.to_thread(is_valid_skin_image, image_bytes):
            return JSONResponse(content={"result": "Please capture an image of the affected skin area."})

        img_array = await asyncio.to_thread(preprocess_image, image_bytes)

        predictions = await asyncio.to_thread(model.predict, img_array)
        top_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][top_idx])
        disease = CLASS_LABELS[top_idx]

        if confidence < 0.65:
            return JSONResponse(content={"result": "Skin disease not covered", "confidence": round(confidence, 4)})

        description, remedies = await asyncio.gather(
            asyncio.to_thread(getDescription, disease),
            asyncio.to_thread(getRemedy, disease)
        )

        return JSONResponse(content={
            "result": disease,
            "confidence": round(confidence, 4),
            "description": description,
            "remedies": remedies
        })

    except Exception as e:
        print(f"[Prediction Error] {e}")
        return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)
