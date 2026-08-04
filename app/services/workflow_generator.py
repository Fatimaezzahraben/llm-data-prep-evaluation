"""
app/services/workflow_generator.py
=====================================
Orchestration de la Phase 3 : construit un prompt (simple/schema/profile), l'envoie
au LLM configuré (Ollama/Mistral par défaut via app/services/llm.py), extrait le code
Python du texte retourné, et sauvegarde le script généré dans workflows/generated/.
"""

import re
import time
from pathlib import Path

import pandas as pd

from app.services.prompt_builder import build_prompt
from app.services.llm import call_llm

GENERATED_DIR = Path("workflows/generated")


def extract_python_code(llm_text: str) -> str:
    """
    Le LLM répond souvent avec du texte autour d'un bloc ```python ... ```.
    On extrait uniquement le code ; si aucun bloc n'est trouvé, on retourne le texte
    complet tel quel (au cas où le LLM aurait retourné du code brut sans balises).
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)```", llm_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return llm_text.strip()


def generate_workflow(dataset_path: str, dataset_name: str, prompt_type: str,
                       provider: str = None, save: bool = True) -> dict:
    """
    Génère un workflow de nettoyage pour un dataset donné, avec un type de prompt donné.

    Parameters
    ----------
    dataset_path : str
        Chemin du CSV bruité à faire nettoyer par le LLM (ex: noisy_medium.csv).
    dataset_name : str
        Nom du dataset pour le prompt (ex: "hotel_bookings").
    prompt_type : str
        "simple", "schema" ou "profile".
    provider : str or None
        Fournisseur LLM (par défaut celui de .env, "ollama").
    save : bool
        Si True, sauvegarde le script généré dans workflows/generated/.

    Returns
    -------
    dict avec : prompt_text, generated_code, latency_seconds, provider, model,
                saved_path (ou None si save=False)
    """
    df = pd.read_csv(dataset_path, low_memory=False)
    prompt_text = build_prompt(prompt_type, df, dataset_name)

    llm_result = call_llm(prompt_text, provider=provider)
    generated_code = extract_python_code(llm_result["text"])

    result = {
        "dataset_name": dataset_name,
        "prompt_type": prompt_type,
        "prompt_text": prompt_text,
        "generated_code": generated_code,
        "latency_seconds": llm_result["latency_seconds"],
        "provider": llm_result["provider"],
        "model": llm_result["model"],
        "saved_path": None,
    }

    if save:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"workflow_{dataset_name}_{prompt_type}.py"
        out_path = GENERATED_DIR / filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        result["saved_path"] = str(out_path)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Génère un workflow de nettoyage via LLM")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--prompt_type", required=True, choices=["simple", "schema", "profile"])
    parser.add_argument("--provider", default=None)
    args = parser.parse_args()

    result = generate_workflow(args.dataset_path, args.dataset_name,
                                args.prompt_type, provider=args.provider)
    print(f"Généré en {result['latency_seconds']}s via {result['provider']}/{result['model']}")
    print(f"Sauvegardé : {result['saved_path']}")
