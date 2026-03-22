import os
import uvicorn
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)