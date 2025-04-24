from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import numpy as np
import tensorflow as tf
import uuid
import asyncio
import cv2
from io import BytesIO
from openai import OpenAI, APIError, AuthenticationError
from PIL import Image

# === Initialize FastAPI ===
app = FastAPI()

# === Load Keras Model ===
MODEL_FILE = "mobilenetv2_skin_disease_model.h5"
model = tf.keras.models.load_model(MODEL_FILE)
class_labels = ['Acne', 'Athlete\'s Foot', 'Cellulitis', 'Chickenpox', 'Cutaneous Larva Migrans', 'Impetigo', 'Nail-Fungus', 'Normal', 'Ringworm', 'Shingles']  # Replace with your actual class names
IMG_SIZE = 224

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
        print(f"Image validity check failed: {e}")
        return False

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def getDescription(diseaseName: str) -> str:
    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Give me a description about this {diseaseName}. Only the description. Don't include special characters and don't ask questions at the end"
            }],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except (AuthenticationError, APIError, Exception) as e:
        return f"Error: {e}"

def getRemedy(diseaseName: str) -> str:
    try:
        response = ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"Give me the best non medical remedies about {diseaseName}. Only the remedy and how they use it. Don't include special characters except for numbers, Add point special character for numbers. Don't add any drug medication only the non-medical. And, dont ask questions at the end. Only include 10 non medical remedies."
            }],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except (AuthenticationError, APIError, Exception) as e:
        return f"Error: {e}"

# === Routes ===
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        original_image_bytes = await file.read()

        # Validate image
        if not await asyncio.to_thread(is_valid_skin_image, original_image_bytes):
            return JSONResponse(content={"result": "Please capture an image of the affected skin area."}, status_code=200)

        # Preprocess image for model input
        img_array = await asyncio.to_thread(preprocess_image, original_image_bytes)

        # Predict
        predictions = model.predict(img_array)[0]
        top_index = np.argmax(predictions)
        confidence = float(predictions[top_index])
        predicted_disease = class_labels[top_index]

        if confidence < 0.65:
            return JSONResponse(content={
                "result": "Skin disease not covered",
                "confidence": round(confidence, 4)
            }, status_code=200)

        # Fetch description and remedies in parallel
        description, remedies = await asyncio.gather(
            asyncio.to_thread(getDescription, predicted_disease),
            asyncio.to_thread(getRemedy, predicted_disease)
        )

        return JSONResponse(content={
            "result": predicted_disease,
            "confidence": round(confidence, 4),
            "description": description or "No description available",
            "remedies": remedies or "No remedies available"
        })

    except Exception as e:
        return JSONResponse(content={"error": f"Prediction failed: {e}"}, status_code=500)
