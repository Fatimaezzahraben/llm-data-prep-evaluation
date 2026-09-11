"""
notebooks/run_agent_c.py
=====================================
Script INDEPENDANT pour lancer l'Agent C sur le resultat d'un nettoyage deja
effectue. Ne fait PARTIE d'aucune boucle de generation -- se lance separement,
apres coup, pour inspecter et noter un fichier deja nettoye.

DEMANDE DE L'ENCADRANTE : Agent C evalue maintenant TOUJOURS DEUX fichiers --
le fichier BRUT (dirty, avant nettoyage) ET le fichier NETTOYE (genere par le
LLM), avec les MEMES regles et le MEME calcul de pourcentages pour les deux.
Cela permet de voir la DIFFERENCE directe entre les deux (avant/apres), pas
seulement le score du fichier nettoye isolement.

Usage:
    python notebooks/run_agent_c.py --noisy-file datasets/hospital/dirty.csv \
        --cleaned-file workflows/executed/hospital_profile_mistral_api_iter5_cleaned.csv \
        --reference-file datasets/hospital/clean.csv --skip-semantic

AUCUNE CONFIGURATION SPECIFIQUE AU DATASET N'EST NECESSAIRE : ce script accepte
N'IMPORTE QUEL dataset (deja connu du projet ou totalement nouveau) tant qu'on
lui donne le chemin vers les 3 fichiers CSV (bruite, nettoye, et optionnellement
la reference propre) -- agent_c.py decouvre tout le reste automatiquement.
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app.services.agent_c import compute_cleanliness_percentages, validate
from app.services.agent_c_dataset_hints import get_dataset_hints


def evaluate_file(label: str, df_original: pd.DataFrame, df_to_check: pd.DataFrame,
                   df_reference: pd.DataFrame, provider: str, model: str,
                   skip_semantic: bool, semantic_hints: dict = None,
                   valid_values_map: dict = None, format_patterns_map: dict = None,
                   sample_size: int = 25) -> dict:
    """
    Lance Agent C (regles + semantique + les deux pourcentages) sur UN fichier
    (df_to_check), et affiche un resume court prefixe par `label` (ex: "DIRTY"
    ou "CLEANED") pour distinguer les deux evaluations a l'ecran.

    REMARQUE DE L'ENCADRANTE (corrigee ici) : quand df_original EST df_to_check
    (cas de l'evaluation du fichier brut "seul", contre lui-meme, faute d'un
    "avant" reel a comparer), les controles data_loss et column_preservation
    sont VIDES DE SENS -- ils comparent un fichier a lui-meme et retournent
    TOUJOURS 100% par construction, quelle que soit la qualite reelle des
    donnees. Detecte automatiquement ce cas (identite d'objet) et desactive ces
    deux controles specifiquement pour lui, via skip_before_after_checks.
    """
    is_self_evaluation = df_original is df_to_check
    print(f"\n{'=' * 60}")
    print(f"[Agent C] Evaluation : {label}")
    print(f"{'=' * 60}")

    result = validate(
        df_original, df_to_check,
        provider=provider, model=model,
        skip_semantic_check=skip_semantic,
        skip_before_after_checks=is_self_evaluation,
        semantic_hints=semantic_hints, valid_values_map=valid_values_map,
        format_patterns_map=format_patterns_map, sample_size=sample_size,
    )
    print(f"[Agent C] Valide (aucun probleme detecte) : {result['valid']}")
    if result["feedback_text"]:
        print(f"[Agent C] Problemes detectes :\n{result['feedback_text']}")

    percentages = compute_cleanliness_percentages(
        df_original, df_to_check, df_reference_clean=df_reference,
        provider=provider, model=model, skip_semantic_check=skip_semantic,
        skip_before_after_checks=is_self_evaluation,
        semantic_hints=semantic_hints, valid_values_map=valid_values_map,
        format_patterns_map=format_patterns_map, sample_size=sample_size,
    )

    print(f"\nPourcentage 1 (respect des regles)      : {percentages['rule_based_percentage']}%")
    for cat, score in percentages["rule_based_breakdown"].items():
        print(f"    - {cat}: {score}%")

    if percentages["reference_comparison_percentage"] is not None:
        print(f"Pourcentage 2 (comparaison avec reference) : {percentages['reference_comparison_percentage']}%")
    else:
        print(f"Pourcentage 2 : non calcule (pas de fichier de reference fourni)")

    return {
        "label": label,
        "valid": result["valid"],
        "rule_issues": result["rule_issues"],
        "feedback_text": result["feedback_text"],
        "rule_based_percentage": percentages["rule_based_percentage"],
        "rule_based_breakdown": percentages["rule_based_breakdown"],
        "reference_comparison_percentage": percentages["reference_comparison_percentage"],
        "reference_comparison_per_column": percentages["reference_comparison_per_column"],
    }


def print_comparison(dirty_result: dict, cleaned_result: dict):
    """Affiche un tableau AVANT/APRES cote a cote -- le coeur de la demande de
    l'encadrante : voir la DIFFERENCE, pas juste un score isole."""
    print(f"\n{'=' * 60}")
    print(f"[Agent C] COMPARAISON DIRTY -> CLEANED")
    print(f"{'=' * 60}")

    r1_dirty = dirty_result["rule_based_percentage"]
    r1_clean = cleaned_result["rule_based_percentage"]
    print(f"Pourcentage 1 (regles)      : {r1_dirty}%  ->  {r1_clean}%   "
          f"(delta: {round(r1_clean - r1_dirty, 2):+}%)")

    r2_dirty = dirty_result["reference_comparison_percentage"]
    r2_clean = cleaned_result["reference_comparison_percentage"]
    if r2_dirty is not None and r2_clean is not None:
        print(f"Pourcentage 2 (comparaison) : {r2_dirty}%  ->  {r2_clean}%   "
              f"(delta: {round(r2_clean - r2_dirty, 2):+}%)")

        print(f"\nDetail par colonne (dirty -> cleaned) :")
        all_cols = set(dirty_result["reference_comparison_per_column"]) | \
                   set(cleaned_result["reference_comparison_per_column"])
        for col in sorted(all_cols):
            d = dirty_result["reference_comparison_per_column"].get(col)
            c = cleaned_result["reference_comparison_per_column"].get(col)
            print(f"    - {col}: {d}%  ->  {c}%")


def main():
    parser = argparse.ArgumentParser(description="Lance l'Agent C sur le fichier brut ET le fichier nettoye, pour comparer")
    parser.add_argument("--noisy-file", required=True, help="Chemin vers le CSV bruite (donnee d'origine, evalue comme baseline)")
    parser.add_argument("--cleaned-file", required=True, help="Chemin vers le CSV deja nettoye (sortie a evaluer)")
    parser.add_argument("--reference-file", default=None,
                         help="Chemin vers le CSV propre de reference (clean.csv). Omis = pourcentage 2 non calcule.")
    parser.add_argument("--provider", default="mistral_api", help="Provider LLM pour le controle semantique")
    parser.add_argument("--model", default=None, help="Modele LLM specifique pour Agent C (optionnel, differe de l'Agent B)")
    parser.add_argument("--output", default=None, help="Chemin du rapport JSON de sortie")
    parser.add_argument("--skip-semantic", action="store_true", help="Desactive le controle LLM (rules only)")
    parser.add_argument("--sample-size", type=int, default=25,
                         help="Nombre de valeurs UNIQUES envoyees au LLM par colonne (defaut: 25). "
                              "IMPORTANT : sur une colonne a forte cardinalite (ex: 70+ valeurs distinctes "
                              "apres corruption sur 1000 lignes), 25 ne couvre qu'une petite fraction des "
                              "lignes reelles -- le pourcentage semantique ne reflete alors que ce sous-"
                              "echantillon, pas tout le fichier. Augmenter (ex: 80-100) pour une evaluation "
                              "plus representative, au prix de plus d'appels LLM.")
    parser.add_argument("--dataset-name", default=None,
                         help="Nom du dataset pour charger des semantic_hints/valid_values_map optionnels "
                              "(app/services/agent_c_dataset_hints.py). Par defaut, deduit du dossier parent "
                              "de --noisy-file (ex: datasets/hospital/dirty.csv -> 'hospital'). Un nom absent "
                              "de ce fichier de hints est sans effet : l'auto-decouverte reste utilisee seule.")
    args = parser.parse_args()

    dataset_name = args.dataset_name or Path(args.noisy_file).resolve().parent.name
    hints = get_dataset_hints(dataset_name)
    if hints["semantic_hints"] or hints["valid_values_map"] or hints["format_patterns"]:
        print(f"[Agent C] Hints dataset trouves pour '{dataset_name}' : "
              f"{len(hints['semantic_hints'])} semantic_hints, "
              f"{len(hints['valid_values_map'])} valid_values_map, "
              f"{len(hints['format_patterns'])} format_patterns")
    else:
        print(f"[Agent C] Aucun hint specifique pour '{dataset_name}' -- auto-decouverte seule.")

    print(f"[Agent C] Chargement des fichiers...")
    df_noisy = pd.read_csv(args.noisy_file, low_memory=False)
    df_cleaned = pd.read_csv(args.cleaned_file, low_memory=False)
    df_reference = pd.read_csv(args.reference_file, low_memory=False) if args.reference_file else None

    # Evaluation 1 : le fichier BRUT lui-meme (baseline "avant nettoyage") --
    # df_original=df_noisy ET df_to_check=df_noisy (on evalue le brut comme s'il
    # etait "la sortie a verifier", pour obtenir son propre score de reference).
    dirty_result = evaluate_file("DIRTY (avant nettoyage)", df_noisy, df_noisy,
                                  df_reference, args.provider, args.model, args.skip_semantic,
                                  semantic_hints=hints["semantic_hints"],
                                  valid_values_map=hints["valid_values_map"],
                                  format_patterns_map=hints["format_patterns"],
                                  sample_size=args.sample_size)

    # Evaluation 2 : le fichier NETTOYE genere par le LLM.
    cleaned_result = evaluate_file("CLEANED (apres nettoyage)", df_noisy, df_cleaned,
                                    df_reference, args.provider, args.model, args.skip_semantic,
                                    semantic_hints=hints["semantic_hints"],
                                    valid_values_map=hints["valid_values_map"],
                                    format_patterns_map=hints["format_patterns"],
                                    sample_size=args.sample_size)

    print_comparison(dirty_result, cleaned_result)

    report = {
        "noisy_file": args.noisy_file,
        "cleaned_file": args.cleaned_file,
        "reference_file": args.reference_file,
        "dataset_name": dataset_name,
        "dataset_hints_used": hints,
        "dirty_evaluation": dirty_result,
        "cleaned_evaluation": cleaned_result,
    }

    output_path = args.output or f"results/agent_c/agent_c_{Path(args.cleaned_file).stem}_comparison.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[Agent C] Rapport de comparaison sauvegarde : {output_path}")


if __name__ == "__main__":
    main()