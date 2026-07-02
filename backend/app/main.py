from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from backend.app.inference import predict_uploaded_image
from backend.app.validation import validate_upload
from backend.app.constants.routes import HEALT_ROUTE, PREDICT_ROUTE

app = FastAPI(title="Industrial Defect Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(HEALT_ROUTE)
def health():
    return {
        "status": True,
        "message": "API up n'running",
    }


@app.post(PREDICT_ROUTE)
async def predict(file: UploadFile = File(...)):
    file_bytes = await file.read()

    validate_upload(file, file_bytes)

    result = predict_uploaded_image(
        file_bytes=file_bytes,
        filename=file.filename or "image.png",
    )

    return {
        "status": True,
        "data": result,
    }
