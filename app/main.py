from fastapi import FastAPI
from app.api import health

app = FastAPI(
    title="LLM Data Prep Evaluation API",
    description="API pour l'évaluation des workflows de préparation de données générés par LLM",
    version="0.1.0",
)

app.include_router(health.router, prefix="/api", tags=["health"])

@app.get("/")
def root():
    return {"message": "LLM Data Prep Evaluation API is running"}