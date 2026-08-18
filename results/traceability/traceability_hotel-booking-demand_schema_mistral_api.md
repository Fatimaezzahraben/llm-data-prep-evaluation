# Rapport de traçabilité — hotel-booking-demand_schema_mistral_api

**Total d'erreurs suivies :** 134906

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Corrigé | 66070 | 49.0% |
| Mal corrigé (valeur différente mais fausse) | 47304 | 35.1% |
| Non corrigé (regression) | 21532 | 16.0% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 32580 | 0 | 27115 |
| format_errors | 3193 | 0 | 14715 |
| outliers | 3 | 4685 | 4860 |
| typos | 30294 | 16847 | 614 |

## Exemples de corrections réussies

- **country** (ligne 66713) : `nan` → `PRT` (attendu : `PRT`)
- **country** (ligne 30783) : `nan` → `PRT` (attendu : `PRT`)
- **country** (ligne 83042) : `unknown` → `PRT` (attendu : `PRT`)
- **country** (ligne 8054) : `nan` → `PRT` (attendu : `PRT`)
- **country** (ligne 25860) : `nan` → `PRT` (attendu : `PRT`)

## Exemples de corrections manquées ou erronées

- **country** (ligne 110654, Mal corrigé (valeur différente mais fausse)) : `unknown` → `PRT` (attendu : `SVN`)
- **country** (ligne 115315, Mal corrigé (valeur différente mais fausse)) : `nan` → `PRT` (attendu : `BRA`)
- **country** (ligne 14764, Mal corrigé (valeur différente mais fausse)) : `nan` → `PRT` (attendu : `GBR`)
- **country** (ligne 24038, Mal corrigé (valeur différente mais fausse)) : ` ` → `PRT` (attendu : `FRA`)
- **country** (ligne 33356, Mal corrigé (valeur différente mais fausse)) : `unknown` → `PRT` (attendu : `ESP`)