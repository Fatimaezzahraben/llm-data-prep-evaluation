# Rapport de traçabilité — titanic_profile_mistral_api

**Total d'erreurs suivies :** 602

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Corrigé | 269 | 44.7% |
| Mal corrigé (valeur différente mais fausse) | 252 | 41.9% |
| Non corrigé (regression) | 81 | 13.5% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 61 | 0 | 206 |
| outliers | 0 | 35 | 33 |
| typos | 208 | 46 | 13 |

## Exemples de corrections réussies

- **Age** (ligne 66) : `nan` → `29.0` (attendu : `29.0`)
- **Age** (ligne 53) : `nan` → `29.0` (attendu : `29.0`)
- **Embarked** (ligne 321) : `nan` → `S` (attendu : `S`)
- **Embarked** (ligne 527) : `nan` → `S` (attendu : `S`)
- **Embarked** (ligne 92) : `nan` → `S` (attendu : `S`)

## Exemples de corrections manquées ou erronées

- **Age** (ligne 322, Mal corrigé (valeur différente mais fausse)) : `nan` → `29.0` (attendu : `30.0`)
- **Age** (ligne 590, Mal corrigé (valeur différente mais fausse)) : ` ` → `29.0` (attendu : `35.0`)
- **Age** (ligne 599, Mal corrigé (valeur différente mais fausse)) : `nan` → `29.0` (attendu : `49.0`)
- **Age** (ligne 305, Mal corrigé (valeur différente mais fausse)) : ` ` → `29.0` (attendu : `0.92`)
- **Age** (ligne 317, Mal corrigé (valeur différente mais fausse)) : `nan` → `29.0` (attendu : `54.0`)