from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
from .inference import predict_image

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ASL API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    result = predict_image(image)
    return JSONResponse({"prediction": result})
