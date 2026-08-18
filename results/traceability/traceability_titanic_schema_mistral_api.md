# Rapport de traçabilité — titanic_schema_mistral_api

**Total d'erreurs suivies :** 602

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Mal corrigé (valeur différente mais fausse) | 333 | 55.3% |
| Corrigé | 152 | 25.2% |
| Non corrigé (regression) | 117 | 19.4% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 57 | 33 | 177 |
| outliers | 0 | 34 | 34 |
| typos | 95 | 50 | 122 |

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