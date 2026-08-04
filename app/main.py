"""
app/main.py
=============
Point d'entrée du serveur FastAPI. Pour l'instant, seule la route /health est
branchée. Les autres routes (profile, prompt, workflow, execution, evaluation,
benchmark, reports) seront ajoutées au fur et à mesure des phases du projet.

Lancer avec :
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from app.api import health

app = FastAPI(
    title="LLM Data Prep Evaluation",
    description="Évaluation multi-objectifs des workflows de data preparation générés par LLM",
    version="0.1.0",
)

app.include_router(health.router, tags=["health"])


@app.get("/")
def root():
    return {"message": "LLM Data Prep Evaluation API — voir /docs pour la liste des routes"}
