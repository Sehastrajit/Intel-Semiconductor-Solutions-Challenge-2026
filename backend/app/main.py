from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.model import CLASSES, DEVICE, MODEL_PATH, predict_image_bytes

app = FastAPI(title="Intel Defect Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="app/templates")
app.mount("/assets", StaticFiles(directory="app/assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    metrics = {
        "test_accuracy": "95.56%",
        "test_loss": "0.6153",
        "best_val_accuracy": "97.50%",
        "train_size_per_class": 210,
        "val_size_per_class": 45,
        "test_size_per_class": 45,
        "evaluated_classes": [
            "defect1", "defect2", "defect3", "defect4",
            "defect5", "defect8", "defect9", "defect10"
        ]
    }

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "metrics": metrics,
        }
    )


@app.get("/api")
def api_root():
    return {
        "message": "Intel Defect Classifier API is running",
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "classes": CLASSES
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image_bytes = await file.read()
        result = predict_image_bytes(image_bytes)
        return {
            "filename": file.filename,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return {}