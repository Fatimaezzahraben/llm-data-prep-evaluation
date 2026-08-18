"""
app/services/safe_executor.py
================================
Phase 4 — Exécution contrôlée d'un workflow (script Python/Pandas) généré par LLM.

Problèmes concrets gérés (observés sur les scripts générés) :
  1. Le LLM invente son propre nom de fichier dans pd.read_csv('...') -> on intercepte
     tous les appels à pd.read_csv et on leur substitue le dataset réel.
  2. Le LLM utilise indifféremment 'df', 'data', 'cleaned_data', 'df_clean'... comme nom
     de variable finale -> on cherche, après exécution, la DataFrame la plus probable
     PARMI CELLES QUI ONT RÉELLEMENT ÉTÉ MODIFIÉES (voir point 4 ci-dessous).
  3. Le script peut planter en cours d'exécution -> capture de l'exception complète.
  4. **Le LLM définit parfois une fonction de nettoyage mais ne l'appelle jamais**
     (tout l'appel est en commentaire "# Example usage"). Dans ce cas, aucune des
     variables candidates n'a été modifiée : on détecte ce cas en comparant chaque
     candidate à une copie vierge du dataset bruité, et si un candidat est identique
     à l'original (donc pas nettoyé), on l'écarte et on cherche une fonction
     utilisateur définie dans le script (nom contenant "clean") pour l'appeler
     nous-mêmes avec le dataset bruité.
  5. Le timeout est configurable pour éviter les scripts qui boucleraient indéfiniment.
"""

import time
import re
import traceback
import threading
import inspect
from pathlib import Path

import pandas as pd

EXECUTED_DIR = Path("workflows/executed")
FAILED_DIR = Path("workflows/failed")

CANDIDATE_RESULT_VARS = [
    "cleaned_data", "df_cleaned", "df_clean", "cleaned_df", "df", "data",
]


def _make_patched_read_csv(df_noisy: pd.DataFrame):
    def _patched(*args, **kwargs):
        return df_noisy.copy()
    return _patched


def _find_actually_modified_dataframe(exec_globals: dict, df_noisy_reference: pd.DataFrame):
    """
    Cherche d'abord parmi CANDIDATE_RESULT_VARS (ordre de preference), puis, si rien
    de valide n'y est trouve, parcourt TOUTES les variables de exec_globals a la
    recherche de n'importe quelle DataFrame reellement modifiee (certains scripts
    utilisent des noms de variable inattendus, ex: 'flights' pour le dataset flights).
    Retourne (nom_variable, dataframe) ou (None, None) si rien de valide trouve.
    """
    def _is_modified(candidate):
        if not isinstance(candidate, pd.DataFrame):
            return False
        try:
            return not candidate.equals(df_noisy_reference)
        except Exception:
            return True  # si la comparaison echoue (shape/dtype tres different), on
                         # considere que c'est une DataFrame differente/valide

    for var_name in CANDIDATE_RESULT_VARS:
        candidate = exec_globals.get(var_name)
        if _is_modified(candidate):
            return var_name, candidate

    # Fallback : n'importe quelle autre DataFrame modifiee, meme sous un nom inattendu
    skip = set(CANDIDATE_RESULT_VARS) | {"pd", "np", "re"}
    for var_name, candidate in exec_globals.items():
        if var_name in skip or var_name.startswith("__"):
            continue
        if _is_modified(candidate):
            return var_name, candidate

    return None, None


def _try_call_unused_cleaning_function(exec_globals: dict, df_noisy: pd.DataFrame):
    """
    Cas où le LLM a défini une fonction (ex: def clean_hotel_bookings(df): ...) mais ne
    l'a jamais appelée dans le script (usage laissé en commentaire). On cherche toute
    fonction définie par l'utilisateur dont le nom contient 'clean', prenant un seul
    argument positionnel, et on l'appelle nous-mêmes avec une copie du dataset bruité.
    """
    for name, obj in exec_globals.items():
        if not inspect.isfunction(obj):
            continue
        if "clean" not in name.lower():
            continue
        try:
            sig = inspect.signature(obj)
            params = [p for p in sig.parameters.values()
                      if p.default is inspect._empty
                      and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            if len(params) != 1:
                continue
            result = obj(df_noisy.copy())
            if isinstance(result, pd.DataFrame):
                return name, result
        except Exception:
            continue
    return None, None


def execute_workflow(script_path: str, dataset_path: str, workflow_name: str,
                      timeout: int = 120, dataset_name: str = None) -> dict:
    with open(script_path, "r", encoding="utf-8") as f:
        script_code = f.read()

    df_noisy = pd.read_csv(dataset_path, low_memory=False)
    df_noisy_reference = df_noisy.copy()  # jamais touchée, sert de référence "avant"

    exec_globals = {
        "pd": pd,
        "np": __import__("numpy"),
        "re": __import__("re"),
    }
    # IMPORTANT : pd.read_csv est une fonction du VRAI module pandas (le meme objet
    # que celui importe partout ailleurs dans le programme). La patcher directement
    # la modifierait de facon PERMANENTE pour tout le processus Python, contaminant
    # tous les appels pd.read_csv suivants (y compris dans evaluate_all_workflows.py
    # et dans les prochains appels a execute_workflow pour d'autres datasets).
    # On sauvegarde donc la fonction originale et on la restaure TOUJOURS, meme en
    # cas d'exception ou de timeout, via un bloc try/finally.
    _original_read_csv = pd.read_csv
    pd.read_csv = _make_patched_read_csv(df_noisy)
    exec_globals["read_csv"] = _make_patched_read_csv(df_noisy)
    exec_globals["df"] = df_noisy.copy()
    exec_globals["data"] = df_noisy.copy()
    # Certains scripts generes utilisent le nom du dataset lui-meme comme variable
    # (ex: "flights" pour le dataset flights) sans jamais l'assigner -> on le pre-charge
    # aussi, pour eviter un NameError evitable.
    if dataset_name:
        safe_name = re.sub(r"\W|^(?=\d)", "_", dataset_name)
        if safe_name and safe_name not in ("df", "data"):
            exec_globals[safe_name] = df_noisy.copy()

    result = {
        "workflow_name": workflow_name,
        "success": False,
        "error": None,
        "traceback": None,
        "execution_time_seconds": None,
        "result_variable_used": None,
        "used_unused_function_fallback": False,
        "n_rows_out": None,
        "n_cols_out": None,
        "output_path": None,
    }

    exception_holder = {}

    def _run():
        try:
            exec(script_code, exec_globals)
        except Exception as e:
            exception_holder["error"] = e
            exception_holder["traceback"] = traceback.format_exc()

    start = time.time()
    try:
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
    finally:
        # TOUJOURS restaurer pd.read_csv, meme si le thread a plante ou timeout.
        pd.read_csv = _original_read_csv
    elapsed = round(time.time() - start, 2)
    result["execution_time_seconds"] = elapsed

    if thread.is_alive():
        result["error"] = f"Timeout dépassé ({timeout}s) — le script n'a pas terminé."
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        return result

    if "error" in exception_holder:
        result["error"] = str(exception_holder["error"])
        result["traceback"] = exception_holder["traceback"]
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        fail_log_path = FAILED_DIR / f"{workflow_name}_error.txt"
        with open(fail_log_path, "w", encoding="utf-8") as f:
            f.write(result["traceback"])
        return result

    # Étape 1 : chercher une DataFrame candidate RÉELLEMENT modifiée
    var_name, df_result = _find_actually_modified_dataframe(exec_globals, df_noisy_reference)

    # Étape 2 : si rien n'a été modifié, chercher une fonction de nettoyage jamais appelée
    if df_result is None:
        var_name, df_result = _try_call_unused_cleaning_function(exec_globals, df_noisy)
        if df_result is not None:
            result["used_unused_function_fallback"] = True

    if df_result is None:
        result["error"] = ("Aucune DataFrame modifiée trouvée : le script n'a "
                            "probablement rien nettoyé (fonction définie mais jamais "
                            "appelée, ou variable de résultat introuvable).")
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        return result

    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXECUTED_DIR / f"{workflow_name}_cleaned.csv"
    df_result.to_csv(output_path, index=False)

    result["success"] = True
    result["result_variable_used"] = var_name
    result["n_rows_out"] = df_result.shape[0]
    result["n_cols_out"] = df_result.shape[1]
    result["output_path"] = str(output_path)
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Exécute un workflow généré en environnement contrôlé")
    parser.add_argument("--script_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--workflow_name", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--dataset_name", default=None)
    args = parser.parse_args()

    result = execute_workflow(args.script_path, args.dataset_path,
                               args.workflow_name, timeout=args.timeout,
                               dataset_name=args.dataset_name)
    print(json.dumps(result, indent=2, ensure_ascii=False))
