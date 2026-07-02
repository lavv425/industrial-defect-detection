from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from computer_vision.predict import load_model

from backend.app.constants.routes import API_NAMESPACE_PREFIX, HEALTH_ROUTE, PREDICT_ROUTE
from backend.app.inference import analyze_image
from backend.app.schemas.prediction import PredictResponse
from backend.app.validation import validate_upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load model once, not per-request
    app.state.model, app.state.device = load_model()
    yield


app = FastAPI(
    title="Industrial Defect Detection API", version="1.0.0", lifespan=lifespan
)

# TODO: update for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix=API_NAMESPACE_PREFIX)


def get_model(request: Request):
    """Dependency exposing the model loaded at startup."""
    return request.app.state.model, request.app.state.device


@router.get(HEALTH_ROUTE)
def health():
    return {
        "status": True,
        "message": "API up n'running",
        "data": None,
    }


@router.post(PREDICT_ROUTE, response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model_device=Depends(get_model)):
    file_bytes = await file.read()

    validate_upload(file, file_bytes)

    model, device = model_device
    result = analyze_image(model, device, file_bytes)

    return PredictResponse(data=result)


app.include_router(router)
