# Rapport de traçabilité — hotel-booking-demand_profile_mistral_api

**Total d'erreurs suivies :** 134906

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Corrigé | 70258 | 52.1% |
| Mal corrigé (valeur différente mais fausse) | 47494 | 35.2% |
| Non corrigé (regression) | 17154 | 12.7% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 35026 | 0 | 24669 |
| format_errors | 13883 | 0 | 4025 |
| outliers | 3 | 290 | 9255 |
| typos | 21346 | 16864 | 9545 |

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