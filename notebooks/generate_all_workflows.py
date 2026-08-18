"""
notebooks/generate_all_workflows.py
======================================
Genere les workflows de nettoyage pour les 4 datasets (hotel, titanic, hospital,
flights), avec Mistral API UNIQUEMENT, prompts schema + profile UNIQUEMENT
(decision de l'encadrante : ne plus utiliser Ollama ni le prompt simple).

Usage:
    python notebooks/generate_all_workflows.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.workflow_generator import generate_workflow

# --- Les 4 datasets : (nom, chemin du CSV bruite a envoyer au LLM) ---
DATASETS = {
    "hotel-booking-demand": "datasets/hotel-booking-demand/noisy_medium.csv",
    "titanic":              "datasets/titanic/noisy_medium.csv",
    "hospital":              "datasets/hospital/dirty.csv",
    "flights":               "datasets/flights/dirty.csv",
}

PROMPT_TYPES = ["schema", "profile"]   # "simple" retire (trop faible en pratique)
PROVIDERS = ["mistral_api"]             # ollama retire (decision de l'encadrante)


def main():
    all_results = []

    for dataset_name, dataset_path in DATASETS.items():
        if not Path(dataset_path).exists():
            print(f"SKIP {dataset_name} : fichier introuvable ({dataset_path})")
            continue

        for provider in PROVIDERS:
            for prompt_type in PROMPT_TYPES:
                print(f"--- {dataset_name} | {provider} | {prompt_type} ---")
                try:
                    result = generate_workflow(
                        dataset_path=dataset_path,
                        dataset_name=dataset_name,
                        prompt_type=prompt_type,
                        provider=provider,
                        save=True,
                    )
                    print(f"  OK -> {result['saved_path']} "
                          f"({result['latency_seconds']}s, model={result['model']})")
                    all_results.append(result)

                except Exception as e:
                    print(f"  ECHEC ({dataset_name}/{provider}/{prompt_type}) : {e}")
                    all_results.append({
                        "dataset_name": dataset_name, "prompt_type": prompt_type,
                        "provider": provider, "error": str(e),
                    })

    print("\n=== Resume ===")
    for r in all_results:
        if "error" in r:
            print(f"  {r['dataset_name']:22s} | {r['provider']:12s} | "
                  f"{r['prompt_type']:8s} | ECHEC : {r['error']}")
        else:
            print(f"  {r['dataset_name']:22s} | {r['provider']:12s} | "
                  f"{r['prompt_type']:8s} | OK ({r['latency_seconds']}s) -> {r['saved_path']}")


if __name__ == "__main__":
    main()
