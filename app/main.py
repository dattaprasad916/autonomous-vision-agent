from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes import router

app = FastAPI(
    title="Object Detection App",
    description="YOLOv8 object detection with visual output",
    version="1.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)
