# Rapport de traçabilité — hotel-booking-demand_simple_mistral_api

**Total d'erreurs suivies :** 268623

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Non corrigé (regression) | 244932 | 91.2% |
| Corrigé | 23691 | 8.8% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 2178 | 117212 | 0 |
| format_errors | 0 | 35817 | 0 |
| outliers | 0 | 23876 | 0 |
| typos | 21513 | 68027 | 0 |

## Exemples de corrections réussies

- **country** (ligne 14480) : `nan` → `nan` (attendu : `nan`)
- **country** (ligne 14076) : `nan` → `nan` (attendu : `nan`)
- **country** (ligne 16786) : `nan` → `nan` (attendu : `nan`)
- **country** (ligne 14360) : `nan` → `nan` (attendu : `nan`)
- **country** (ligne 65909) : `nan` → `nan` (attendu : `nan`)

## Exemples de corrections manquées ou erronées

- **country** (ligne 97010, Non corrigé (regression)) : `unknown` → `unknown` (attendu : `FRA`)
- **country** (ligne 46443, Non corrigé (regression)) : `unknown` → `unknown` (attendu : `PRT`)
- **country** (ligne 95424, Non corrigé (regression)) : `nan` → `nan` (attendu : `PRT`)
- **country** (ligne 82915, Non corrigé (regression)) : `nan` → `nan` (attendu : `PRT`)
- **country** (ligne 85547, Non corrigé (regression)) : `nan` → `nan` (attendu : `DEU`)