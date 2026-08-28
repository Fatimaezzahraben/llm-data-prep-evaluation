"""
app/services/validation_loop.py
==================================
Phase 3, approche 6 du cahier des charges : "Workflow avec validation post-execution".

Genere un script, l'execute, mesure le F1, et si le score est insuffisant, renvoie au
LLM le script precedent + des exemples CONCRETS de corrections manquees (issus de
traceability), en lui demandant de corriger SPECIFIQUEMENT ces cas sans casser ce qui
marche deja. Repete jusqu'a max_iterations ou jusqu'a ce que le F1 cible soit atteint.
Garde et retourne la MEILLEURE iteration (pas forcement la derniere).

IMPORTANT -- FUITE DE VERITE TERRAIN (decouvert en analysant les scripts generes) :
Le prompt de feedback contient explicitement "the correct value should have been 'X'"
pour chaque exemple d'echec. Sans precaution, le LLM peut se contenter de MEMORISER
ces valeurs (on l'a observe : des constantes comme 26.0, 'SC', ou des .replace({6695.1:
185.4}) copiees mot pour mot depuis les exemples de feedback) plutot que d'apprendre
une strategie generalisable. Le score final serait alors artificiellement gonfle sur
les lignes precisement montrees en feedback, sans refleter une vraie amelioration.

Pour eviter cela, error_log est desormais scinde en deux sous-ensembles disjoints,
comme un split train/test classique en ML :
  - feedback_pool (ex: 70%) : seul sous-ensemble dont des exemples peuvent etre montres
    au LLM dans le prompt de feedback.
  - held_out (ex: 30%) : jamais montre au LLM, sert UNIQUEMENT a calculer le F1 rapporte
    a chaque iteration. Une amelioration du F1 sur held_out reflete donc une vraie
    generalisation, pas une memorisation des exemples de feedback.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd

from app.services.prompt_builder import build_prompt_split
from app.services.llm import call_llm
from app.services.safe_executor import execute_workflow
from app.services.evaluation.metrics import evaluate_workflow
from app.services.evaluation.traceability import build_traceability_table
from app.services.workflow_generator import extract_python_code, GENERATED_DIR


# ---------------------------------------------------------------------------
# Patterns de crash DEJA OBSERVES en pratique, avec le rappel de regle precis qui
# aurait du l'empecher. Un message d'erreur brut ("invalid literal for int()...")
# s'est revele insuffisant a lui seul : le LLM a repete EXACTEMENT la meme erreur
# a l'iteration suivante malgre le feedback generique. On pointe donc explicitement
# vers la regle violee, avec le correctif attendu, au lieu de compter sur le LLM
# pour redecouvrir la regle depuis la trace d'erreur seule.
# ---------------------------------------------------------------------------
KNOWN_CRASH_PATTERNS = [
    ("single positional indexer is out-of-bounds",
     "You likely wrote `if not s.mode().any()` (or similar) to check whether "
     "s.mode() is empty before calling .iloc[0] on it. This is WRONG: .any() checks "
     "if any element is truthy, and on an EMPTY Series it returns False (vacuously), "
     "so `not s.mode().any()` becomes True even when mode() has NO elements — you "
     "then call .iloc[0] on an empty Series and crash. The correct check for "
     "emptiness is `.empty`, not `.any()`: always write "
     "`s.mode().iloc[0] if not s.mode().empty else <fallback>`, never "
     "`if not s.mode().any()`."),
    ("invalid literal for int() with base 10",
     "You called Python's built-in int() (or float()) directly on a raw string "
     "value from the dataset. This crashes on any non-numeric character (e.g. 'x', "
     "'2O', '75€'). This violates the NUMERIC COLUMNS rule. Required fix — never call "
     "int(x)/float(x) directly, always use this 3-step pattern instead:\n"
     "  1. df[col] = df[col].astype(str).str.replace(r'[^0-9.\\-]', '', regex=True)\n"
     "  2. df[col] = pd.to_numeric(df[col], errors='coerce')\n"
     "  3. df[col] = df[col].fillna(df[col].median())"),
    ("Cannot perform reduction 'median' with string dtype",
     "You called .median() on a column that is still text/object dtype. This "
     "violates the NUMERIC COLUMNS rule: before calling .median() on any column, "
     "always confirm pd.api.types.is_numeric_dtype(df[col]) is True first; if not, "
     "convert with pd.to_numeric(df[col], errors='coerce') before calling .median()."),
    ("Invalid value for dtype 'str'",
     "You mixed string and Timestamp/datetime values back into the same column "
     "after casting it with .astype(str). Follow the MULTI-FORMAT DATE PARSING "
     "PATTERN rule exactly: parse into a SEPARATE new Series first (never write "
     "datetime objects into a column already cast to str), and assign the final "
     "result back to df[col] only ONCE, as a single consistent type, at the end."),
    ('"value" parameter must be a scalar, dict or Series, but you passed a "ndarray"',
     "You called .fillna(x.mode()) passing the FULL mode() result (a Series) "
     "directly into fillna() instead of a single value. Always take .iloc[0] of the "
     "mode: df[col] = df[col].fillna(df[col].mode().iloc[0] if not "
     "df[col].mode().empty else <fallback>)."),
    ('"value" parameter must be a scalar, dict or Series, but you passed a "Index"',
     "You called .fillna(...) with a pandas Index object instead of a single value — "
     "likely `.fillna(df[col].mode().index)` (forgot to extract one element) or "
     "`.fillna(some_series.index)` by mistake. .fillna() needs ONE concrete value (a "
     "number, a string, or a dict/Series mapping row->value), never an Index. Fix: "
     "extract a single value first, e.g. `df[col].mode().iloc[0]` (the mode VALUE, "
     "not `.index`, not the whole `.mode()` Series) before passing it to .fillna()."),
    ("invalid error value specified",
     "IMPORTANT — this exact message is the VERBATIM error text from "
     "`numpy.seterr(...)` when given an invalid option — it is NOT primarily about "
     "pandas' `errors=` parameter (a previous version of this guidance said it was; "
     "that diagnosis was likely wrong and caused repeated unresolved crashes). Check "
     "FIRST for any call to `np.seterr(...)` in your script (e.g. `np.seterr(all=...)` "
     "or `np.seterr(divide=...)` etc.) — the only valid values for each np.seterr "
     "keyword are the exact strings 'ignore', 'warn', 'raise', 'call', 'print', or "
     "'log'; nothing else, and there is usually no reason to call np.seterr() at all "
     "in a data-cleaning script — simply remove the call if you added one. If there is "
     "genuinely no np.seterr() call anywhere in the script, check instead every "
     "`errors=` argument passed to a pandas function (pd.to_numeric, pd.to_datetime, "
     ".astype()) and make sure it is spelled exactly 'coerce' or 'raise' (never "
     "'ignore', which is deprecated/removed in recent pandas and will raise this exact "
     "error) and never True/False."),
    ("is not defined",
     "You referenced a function or variable name that was never actually defined in "
     "this script (a NameError) — most likely a helper function you intended to write "
     "(e.g. a small time/date conversion helper) but forgot to include the `def "
     "your_function_name(...):` block for, or defined it with a different name/inside "
     "a scope where it isn't visible where you call it. Before submitting, check every "
     "function name you CALL has a matching `def` at the top level of the script, "
     "spelled exactly the same way, defined BEFORE it is first used."),
    ("Unknown format code 'd' for object of type 'float'",
     "You used an integer format spec like `f\"{x:02d}\"` or `f\"{x:d}\"` on a value "
     "that is actually a float, not an int — `:d` only works on int. This commonly "
     "happens with time/minutes arithmetic: `minutes // 60` and `minutes % 60` LOOK "
     "like they produce ints, but if `minutes` itself came from `np.median(...)` of an "
     "EVEN-length list, the median is the average of the two middle values and can be "
     "a float (e.g. median of [10, 15] = 12.5) — that float then silently propagates "
     "through // and % (floor-division/modulo of a float stays a float in Python) and "
     "crashes at the final f-string. Fix: explicitly wrap in int() right before "
     "formatting, e.g. `hh = int(minutes // 60)` and `mm = int(minutes % 60)`, or cast "
     "the source value itself right after computing it (e.g. "
     "`delay = int(np.median(delays))`) so it can never carry a fractional part "
     "forward. Do this everywhere you divide/mod a value that might have come from "
     "median() before formatting it with :d or :02d."),
    ("Too many indexers",
     "You used `.loc[]` or `.iloc[]` with MORE index arguments than a DataFrame "
     "supports (a DataFrame only takes exactly 2: `df.loc[row_selector, "
     "col_selector]`). This usually comes from an accidental extra comma or an "
     "attempt to chain multiple conditions/columns inside a single `.loc[...]` call, "
     "e.g. writing `df.loc[mask1, mask2, 'col']` (3 arguments) instead of combining "
     "the masks first: `df.loc[mask1 & mask2, 'col']`. Check every `.loc[`/`.iloc[` "
     "call in your script for exactly 2 comma-separated arguments, not 3 or more."),
]
# Note: a crash message that is just a bare column name in quotes (e.g. "'act_dep_time'"
# with no other text) is a KeyError — it means you referenced a column that does not
# exist in `df` at that point in the script (misspelled, dropped earlier, or renamed).
# There is no fixed substring to match here since the column name varies, so this is
# not added as a KNOWN_CRASH_PATTERNS entry, but the same general guidance applies:
# before referencing any column, make sure it was not renamed/dropped by an earlier
# line in the same script.


def _match_known_crash(error_message: str):
    """Retourne l'explication ciblee si error_message correspond a un pattern de
    crash deja repertorie, sinon None."""
    if not error_message:
        return None
    for pattern, explanation in KNOWN_CRASH_PATTERNS:
        if pattern in error_message:
            return explanation
    return None


def _split_error_log(error_log: pd.DataFrame, feedback_fraction: float = 0.7,
                      random_state: int = 42):
    """
    Scinde error_log en (feedback_pool, held_out), stratifie par colonne pour que
    chaque colonne soit representee dans les deux sous-ensembles autant que possible.
    """
    rng = np.random.default_rng(random_state)
    feedback_idx = []
    held_out_idx = []

    for col, group in error_log.groupby("column"):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = max(1, int(len(idx) * feedback_fraction)) if len(idx) > 1 else len(idx)
        feedback_idx.extend(idx[:cut])
        held_out_idx.extend(idx[cut:])

    feedback_pool = error_log.loc[feedback_idx].reset_index(drop=True)
    held_out = error_log.loc[held_out_idx].reset_index(drop=True)
    return feedback_pool, held_out


def _build_feedback_prompt(base_prompt: str, previous_code: str,
                            failed_examples: list, report: dict) -> str:
    """Construit un prompt de correction cible a partir des echecs observes.

    NOTE : les exemples ici viennent UNIQUEMENT de feedback_pool (jamais de held_out),
    donc meme si le LLM memorise ces valeurs exactes, cela n'affecte pas le F1 rapporte
    (calcule sur held_out, disjoint)."""
    examples_block = "\n".join(
        f"- column '{ex['column']}': input was '{ex['valeur_bruitee_vue_par_llm']}', "
        f"your script produced '{ex['valeur_produite_par_llm']}', "
        f"but the correct value should have been '{ex['valeur_originale']}'"
        for ex in failed_examples
    )

    worst_families = sorted(
        report.get("by_error_family", {}).items(),
        key=lambda kv: kv[1]["f1_score"]
    )
    worst_block = "\n".join(
        f"- {fam}: F1={stats['f1_score']} ({stats['correct_repairs']}/{stats['total_errors']} correct)"
        for fam, stats in worst_families
    )

    return f"""You previously wrote the following data-cleaning script for this dataset:

```python
{previous_code}
```

When executed and compared to the ground truth, it scored:
{worst_block}

Here are concrete examples where your script's output was WRONG (input -> your wrong
output -> the value it should have produced):
{examples_block}

Rewrite the COMPLETE script, keeping everything that already works correctly, but fix
the GENERAL PATTERN behind these examples (e.g. a better imputation strategy, a missing
mapping rule, a parsing bug) rather than hardcoding these specific values as constants
in your code — your script will be tested on different rows next time, so any fix that
only works for the exact values shown above will not help. Pay special attention to
the error family with the lowest F1 score. Follow all the original constraints below.

{base_prompt}"""


def run_validation_loop(dataset_path: str, dataset_name: str, prompt_type: str,
                         df_clean: pd.DataFrame, error_log: pd.DataFrame,
                         provider: str = "mistral_api", max_iterations: int = 3,
                         target_f1: float = 0.90, n_feedback_examples: int = 8,
                         feedback_fraction: float = 0.7) -> dict:
    """
    Boucle generation -> execution -> evaluation -> feedback, jusqu'a target_f1 ou
    max_iterations. Retourne la MEILLEURE iteration observee.

    Le F1 rapporte a chaque iteration est calcule UNIQUEMENT sur held_out (jamais
    montre au LLM), pour mesurer une vraie generalisation et non de la memorisation.

    Returns
    -------
    dict avec : best_iteration, best_f1, best_code, best_report, history (liste de
    tous les F1 par iteration, pour tracer la courbe de convergence dans le rapport).
    """
    df_noisy = pd.read_csv(dataset_path, low_memory=False)
    system_prompt, base_prompt = build_prompt_split(prompt_type, df_noisy, dataset_name)

    feedback_pool, held_out = _split_error_log(error_log, feedback_fraction=feedback_fraction)
    print(f"  [split] feedback_pool={len(feedback_pool)} erreurs (visibles en feedback) | "
          f"held_out={len(held_out)} erreurs (jamais montrees, sert au score)")

    history = []
    best = {"f1": -1, "code": None, "report": None, "df_cleaned": None, "iteration": None}

    current_prompt = base_prompt

    for iteration in range(1, max_iterations + 1):
        print(f"  [iter {iteration}] generation...")
        llm_result = call_llm(current_prompt, provider=provider, system_prompt=system_prompt)
        code = extract_python_code(llm_result["text"])

        workflow_name = f"{dataset_name}_{prompt_type}_{provider}_iter{iteration}"
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        script_path = GENERATED_DIR / f"workflow_{workflow_name}.py"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        exec_result = execute_workflow(str(script_path), dataset_path, workflow_name,
                                        timeout=120, dataset_name=dataset_name)

        if not exec_result["success"]:
            error_msg = exec_result["error"]
            print(f"  [iter {iteration}] ECHEC EXECUTION : {error_msg}")
            history.append({"iteration": iteration, "f1": None, "error": error_msg})

            known_fix = _match_known_crash(error_msg)
            previous_error = history[-2]["error"] if len(history) >= 2 else None
            repeated_same_pattern = (
                known_fix is not None
                and previous_error is not None
                and _match_known_crash(previous_error) == known_fix
            )

            specific_guidance = ""
            if known_fix:
                specific_guidance = f"\nSPECIFIC RULE VIOLATION DETECTED:\n{known_fix}\n"
                if repeated_same_pattern:
                    specific_guidance += (
                        "\nWARNING: you made this EXACT SAME mistake in the previous "
                        "iteration too, even after being told how to fix it. Re-read "
                        "the fix above very carefully line by line and apply it "
                        "precisely — do not repeat this error a third time.\n"
                    )
            elif error_msg and error_msg.strip().startswith("'") and error_msg.strip().endswith("'") \
                    and " " not in error_msg.strip().strip("'"):
                # Message is JUST a quoted single token (e.g. "'act_dep_time'") with no
                # other text -- the classic signature of a pandas/Python KeyError from
                # referencing a column that doesn't exist in df at that point.
                specific_guidance = (
                    "\nSPECIFIC RULE VIOLATION DETECTED:\nThis error message is just a "
                    "single quoted name with nothing else — that is the signature of a "
                    "KeyError: you referenced a column that does not exist in `df` at "
                    "that exact point in the script (misspelled, dropped by an earlier "
                    "`df.drop(columns=...)`, renamed by an earlier step, or never "
                    "created). Check every column name you reference against the exact "
                    "list of valid column names given in this prompt, and check whether "
                    "any earlier line in your OWN script renamed or dropped it before "
                    "this point.\n"
                )

            current_prompt = f"""Your previous script crashed with this error:
{error_msg}
{specific_guidance}
Here is the script that crashed:
```python
{code}
```

Rewrite the COMPLETE script, fixing this specific error, while following all the
original constraints below.

{base_prompt}"""
            continue

        df_cleaned = pd.read_csv(exec_result["output_path"], low_memory=False)

        # Score UNIQUEMENT sur held_out (jamais montre au LLM) : mesure de generalisation.
        pipeline_time = round(llm_result["latency_seconds"] + exec_result["execution_time_seconds"], 2)
        report = evaluate_workflow(df_clean, None, df_cleaned, held_out,
                                    workflow_name=workflow_name,
                                    pipeline_time_seconds=pipeline_time)
        f1 = report["global_f1"]
        print(f"  [iter {iteration}] F1 (held-out)={f1}")
        history.append({"iteration": iteration, "f1": f1, "error": None})

        improved = f1 > best["f1"]  # True au 1er tour (best["f1"] demarre a -1)
        regressed = f1 < best["f1"]  # STRICTEMENT pire -- une egalite n'est PAS une
        # regression. Avant : `f1 <= best["f1"]` traitait aussi une egalite comme une
        # regression, ce qui declenchait le message "scored WORSE" envoye au LLM alors
        # que le score n'avait pas baisse -- une affirmation fausse dans le prompt.
        if improved:
            best = {"f1": f1, "code": code, "report": report,
                     "df_cleaned": df_cleaned, "iteration": iteration}

        if f1 >= target_f1:
            print(f"  [iter {iteration}] Objectif F1>={target_f1} atteint, arret.")
            break

        if iteration < max_iterations:
            # HILL-CLIMBING : on reconstruit toujours le feedback a partir du
            # MEILLEUR script connu, jamais du dernier essai s'il a regresse. Sans
            # cela, un essai qui degrade le score "pollue" l'iteration suivante, qui
            # tente de corriger un script deja pire que ce qu'on avait -- observe en
            # pratique (ex: flights F1 0.52 -> 0.13 -> 0.00 sur 3 iterations, une
            # cascade de regressions au lieu d'une convergence).
            base_code = best["code"]
            base_df_cleaned = best["df_cleaned"]
            base_report = best["report"]

            traceability = build_traceability_table(df_clean, base_df_cleaned, feedback_pool)
            failed = traceability[traceability["statut"] != "Corrigé"]
            failed_examples = failed.sample(min(n_feedback_examples, len(failed)),
                                             random_state=42).to_dict("records") if len(failed) else []
            current_prompt = _build_feedback_prompt(base_prompt, base_code, failed_examples, base_report)

            if regressed:
                current_prompt = (
                    f"NOTE: your previous attempt actually scored WORSE (F1={f1}) than an "
                    f"earlier attempt (F1={best['f1']}). The script shown below is that "
                    f"EARLIER, BETTER-SCORING attempt — build your next fix on top of THIS "
                    f"version, not your most recent (worse) one, so you don't lose the "
                    f"progress it had already made.\n\n" + current_prompt
                )

    if best["code"] is not None:
        final_path = GENERATED_DIR / f"workflow_{dataset_name}_{prompt_type}_{provider}_BEST.py"
        with open(final_path, "w", encoding="utf-8") as f:
            f.write(best["code"])
        best["saved_path"] = str(final_path)

    # Sauvegarde le rapport complet (format attendu par l'encadrante) de la MEILLEURE
    # iteration -- c'est LE fichier JSON demande, un par (dataset, prompt_type, provider).
    benchmark_json_path = None
    if best["report"] is not None:
        results_dir = Path("results/benchmark_results")
        results_dir.mkdir(parents=True, exist_ok=True)
        benchmark_json_path = results_dir / f"benchmark_results_{dataset_name}_{prompt_type}_{provider}.json"
        with open(benchmark_json_path, "w", encoding="utf-8") as f:
            json.dump(best["report"], f, indent=2, ensure_ascii=False)
        print(f"  [benchmark] Rapport JSON sauvegarde : {benchmark_json_path}")

    return {"best_f1": best["f1"], "best_code": best["code"], "best_report": best["report"],
            "best_iteration": best["iteration"], "history": history,
            "saved_path": best.get("saved_path"),
            "benchmark_json_path": str(benchmark_json_path) if benchmark_json_path else None,
            "feedback_pool_size": len(feedback_pool), "held_out_size": len(held_out)}