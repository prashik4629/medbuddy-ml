import logging

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers.prediction import router as prediction_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="MedBuddy.ML API",
    description="Heart Disease Prediction API powered by Machine Learning",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


app.include_router(prediction_router)


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "ok",
        "model": "heart_disease_prediction_model",
        "version": "1.0.0",
    }