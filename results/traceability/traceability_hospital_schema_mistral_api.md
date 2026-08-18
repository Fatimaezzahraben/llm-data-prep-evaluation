# Rapport de traçabilité — hospital_schema_mistral_api

**Total d'erreurs suivies :** 482

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Non corrigé (regression) | 308 | 63.9% |
| Mal corrigé (valeur différente mais fausse) | 118 | 24.5% |
| Corrigé | 56 | 11.6% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| real_world_error | 56 | 308 | 118 |

## Exemples de corrections réussies

- **state** (ligne 42) : `xl` → `al` (attendu : `al`)
- **state** (ligne 46) : `xl` → `al` (attendu : `al`)
- **state** (ligne 50) : `xl` → `al` (attendu : `al`)
- **state** (ligne 83) : `xl` → `al` (attendu : `al`)
- **state** (ligne 98) : `xl` → `al` (attendu : `al`)

## Exemples de corrections manquées ou erronées

- **provider_number** (ligne 13, Non corrigé (regression)) : `1xx19` → `1xx19` (attendu : `10019`)
- **provider_number** (ligne 45, Non corrigé (regression)) : `x0005` → `x0005` (attendu : `10005`)
- **provider_number** (ligne 81, Non corrigé (regression)) : `1000x` → `1000x` (attendu : `10006`)
- **provider_number** (ligne 213, Non corrigé (regression)) : `x00xx` → `x00xx` (attendu : `10011`)
- **provider_number** (ligne 244, Non corrigé (regression)) : `x00x5` → `x00x5` (attendu : `10015`)