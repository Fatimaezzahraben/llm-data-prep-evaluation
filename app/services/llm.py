"""
app/services/llm.py
======================
Point d'entrée UNIQUE pour appeler un LLM, quel que soit le fournisseur
(Ollama en local, ou une API payante pour comparaison). Toutes les autres parties du
projet (workflow_generator, notebooks) appellent uniquement call_llm() — jamais
directement requests/openai/mistralai — pour rester interchangeables.
"""

import time
import requests

from app.utils.config import settings


def call_llm(prompt: str, provider: str = None, model: str = None) -> dict:
    """
    Envoie un prompt au LLM choisi et retourne le texte généré + des métadonnées
    utiles pour les métriques de coût/latence (Phase 5).

    Returns
    -------
    dict avec les clés :
        - "text" : la réponse texte du LLM (le script Python généré)
        - "latency_seconds" : temps de la requête
        - "provider" : fournisseur utilisé
        - "model" : modèle utilisé
    """
    provider = provider or settings.DEFAULT_LLM_PROVIDER
    start = time.time()

    if provider == "ollama":
        model = model or settings.OLLAMA_MODEL
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=900,   
        )
        response.raise_for_status()
        text = response.json()["response"]

    elif provider == "mistral_api":
        from mistralai import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        model = model or "mistral-large-latest"
        result = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.choices[0].message.content

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        model = model or "gpt-4o-mini"
        result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        model = model or "claude-sonnet-4-5"
        result = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text

    else:
        raise ValueError(f"Fournisseur LLM inconnu : {provider}")

    latency = time.time() - start
    return {"text": text, "latency_seconds": round(latency, 2),
            "provider": provider, "model": model}
