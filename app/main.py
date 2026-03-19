from fastapi import FastAPI
from app.api.routes.assistant import router as assistant_router

app = FastAPI(
    title="Medical Assistant API",
    version="1.0.0"
)

app.include_router(assistant_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}