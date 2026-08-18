"""
app/services/llm.py
======================
Point d'entrée UNIQUE pour appeler un LLM, quel que soit le fournisseur
(Ollama en local, ou une API payante pour comparaison). Toutes les autres parties du
projet (workflow_generator, notebooks) appellent uniquement call_llm() — jamais
directement requests/openai/mistralai — pour rester interchangeables.

IMPORTANT — reproductibilite (cf. cahier des charges, Phase 5) :
Tous les appels utilisent temperature=0 (generation deterministe/greedy). Sans cela,
deux executions du MEME prompt peuvent produire des scripts differents et donc des F1
differents, ce qui rend toute comparaison "avant/apres amelioration du prompt" invalide
-- on ne pourrait plus distinguer un vrai effet du prompt d'un simple bruit d'echantillonnage.

IMPORTANT — system_prompt optionnel :
Quand system_prompt est fourni, les regles/contraintes generales y sont placees (role
"system"), separement du contenu specifique au dataset (role "user"). Les modeles de
chat suivent generalement plus fidelement des regles placees en system que noyees dans
un long message user.
"""

import time
import requests

from app.utils.config import settings


def call_llm(prompt: str, provider: str = None, model: str = None,
              system_prompt: str = None) -> dict:
    """
    Envoie un prompt au LLM choisi et retourne le texte généré + des métadonnées
    utiles pour les métriques de coût/latence (Phase 5).

    Parameters
    ----------
    prompt : str
        Le message "user" (contenu specifique au dataset : schema/profil/exemples).
    system_prompt : str or None
        Le message "system" optionnel (regles/contraintes generales). Si None, seul
        `prompt` est envoye (comportement retrocompatible).

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
        # L'API /api/generate d'Ollama n'a pas de role "system" separe -> on prefixe.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=900,
        )
        response.raise_for_status()
        text = response.json()["response"]

    elif provider == "mistral_api":
        from mistralai.client import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        model = model or "mistral-large-latest"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = result.choices[0].message.content

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        model = model or "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = result.choices[0].message.content

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        model = model or "claude-sonnet-4-5"
        kwargs = {"model": model, "max_tokens": 2000, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            kwargs["system"] = system_prompt
        result = client.messages.create(**kwargs)
        text = result.content[0].text

    else:
        raise ValueError(f"Fournisseur LLM inconnu : {provider}")

    latency = time.time() - start
    return {"text": text, "latency_seconds": round(latency, 2),
            "provider": provider, "model": model}