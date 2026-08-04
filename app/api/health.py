"""
app/api/health.py
====================
Route de test minimale pour vérifier que le serveur FastAPI et la connexion au LLM
fonctionnent. Utile pour valider l'installation avant de coder le reste.
"""

from fastapi import APIRouter
from app.services.llm import call_llm

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/health/llm")
def health_check_llm():
    """Vérifie que le LLM configuré (Ollama par défaut) répond bien."""
    try:
        result = call_llm("Reply with exactly: OK")
        return {"status": "ok", "llm_response": result["text"][:100],
                "latency_seconds": result["latency_seconds"], "provider": result["provider"]}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
