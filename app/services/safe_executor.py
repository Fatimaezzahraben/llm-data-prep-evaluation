"""
app/services/safe_executor.py
================================
Phase 4 — Exécution contrôlée d'un workflow (script Python/Pandas) généré par LLM.

Problèmes concrets que ce module gère (observés sur les scripts générés) :
  1. Le LLM invente son propre nom de fichier dans pd.read_csv('...') qui ne correspond
     jamais au vrai chemin du dataset -> on intercepte tous les appels à pd.read_csv et
     on leur substitue le dataset réel, quel que soit l'argument utilisé par le script.
  2. Le LLM utilise indifféremment 'df', 'data', 'cleaned_data', 'df_clean'... comme nom
     de variable finale -> on cherche, après exécution, la DataFrame la plus probable.
  3. Le script peut planter en cours d'exécution (colonne inexistante, mauvais typage,
     etc.) -> on capture l'exception avec la trace complète au lieu de laisser planter
     tout le pipeline de benchmark.
  4. Le script peut boucler ou être anormalement lent -> timeout configurable.
"""

import time
import traceback
import threading
from pathlib import Path

import pandas as pd

EXECUTED_DIR = Path("workflows/executed")
FAILED_DIR = Path("workflows/failed")

# Noms de variable, par ordre de préférence, dans lesquels on va chercher le résultat
# final nettoyé une fois le script exécuté.
CANDIDATE_RESULT_VARS = [
    "cleaned_data", "df_cleaned", "df_clean", "cleaned_df",
    "data", "df",
]


def _make_patched_read_csv(df_noisy: pd.DataFrame):
    """Remplace pandas.read_csv : quel que soit le chemin demandé par le script généré,
    on retourne une COPIE du dataset bruité réellement fourni."""
    def _patched(*args, **kwargs):
        return df_noisy.copy()
    return _patched


def execute_workflow(script_path: str, dataset_path: str, workflow_name: str,
                      timeout: int = 120) -> dict:
    """
    Exécute un script de nettoyage généré par LLM sur un dataset donné, dans un
    environnement contrôlé.

    Parameters
    ----------
    script_path : str
        Chemin du script .py généré (ex: workflows/generated/workflow_..._simple.py).
    dataset_path : str
        Chemin du CSV bruité à nettoyer.
    workflow_name : str
        Nom utilisé pour nommer les fichiers de sortie.
    timeout : int
        Temps maximal d'exécution en secondes.

    Returns
    -------
    dict avec :
        - success (bool)
        - error (str ou None) : message d'erreur si échec
        - traceback (str ou None) : trace complète si échec
        - execution_time_seconds (float)
        - result_variable_used (str ou None) : quel nom de variable a été trouvé
        - n_rows_out, n_cols_out (int ou None)
        - output_path (str ou None) : où le résultat nettoyé a été sauvegardé
    """
    with open(script_path, "r", encoding="utf-8") as f:
        script_code = f.read()

    df_noisy = pd.read_csv(dataset_path, low_memory=False)

    exec_globals = {
        "pd": pd,
        "np": __import__("numpy"),
        "re": __import__("re"),
    }
    # Interception : tout appel à pd.read_csv(...) dans le script généré renverra le
    # vrai dataset bruité, peu importe le nom de fichier écrit par le LLM.
    exec_globals["pd"].read_csv = _make_patched_read_csv(df_noisy)
    # Certains scripts appellent read_csv directement (from pandas import read_csv)
    exec_globals["read_csv"] = _make_patched_read_csv(df_noisy)
    # Variable pré-chargée, au cas où le script suppose que df/data existe déjà
    exec_globals["df"] = df_noisy.copy()
    exec_globals["data"] = df_noisy.copy()

    result = {
        "workflow_name": workflow_name,
        "success": False,
        "error": None,
        "traceback": None,
        "execution_time_seconds": None,
        "result_variable_used": None,
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
    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
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

    # Recherche de la DataFrame résultat parmi les noms candidats
    df_result = None
    for var_name in CANDIDATE_RESULT_VARS:
        if var_name in exec_globals and isinstance(exec_globals[var_name], pd.DataFrame):
            df_result = exec_globals[var_name]
            result["result_variable_used"] = var_name
            break

    if df_result is None:
        result["error"] = ("Aucune DataFrame de résultat trouvée parmi "
                            f"{CANDIDATE_RESULT_VARS} après exécution.")
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        return result

    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXECUTED_DIR / f"{workflow_name}_cleaned.csv"
    df_result.to_csv(output_path, index=False)

    result["success"] = True
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
    args = parser.parse_args()

    result = execute_workflow(args.script_path, args.dataset_path,
                               args.workflow_name, timeout=args.timeout)
    print(json.dumps(result, indent=2, ensure_ascii=False))
