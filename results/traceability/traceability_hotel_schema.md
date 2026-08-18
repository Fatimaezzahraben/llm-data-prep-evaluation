# Rapport de traçabilité — hotel_schema

**Total d'erreurs suivies :** 134906

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Colonne/ligne absente après exécution | 134900 | 100.0% |
| Mal corrigé (valeur différente mais fausse) | 4 | 0.0% |
| Corrigé | 2 | 0.0% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 1 | 0 | 1 |
| format_errors | 0 | 0 | 1 |
| outliers | 0 | 0 | 1 |
| typos | 1 | 0 | 1 |

## Exemples de corrections réussies

- **children** (ligne 0) : ` ` → `0.0` (attendu : `0.0`)
- **adults** (ligne 1) : `2O` → `2.0` (attendu : `2`)

## Exemples de corrections manquées ou erronées

- **country** (ligne 110654, Colonne/ligne absente après exécution) : `unknown` → `nan` (attendu : `SVN`)
- **country** (ligne 66713, Colonne/ligne absente après exécution) : `nan` → `nan` (attendu : `PRT`)
- **country** (ligne 115315, Colonne/ligne absente après exécution) : `nan` → `nan` (attendu : `BRA`)
- **country** (ligne 30783, Colonne/ligne absente après exécution) : `nan` → `nan` (attendu : `PRT`)
- **country** (ligne 14764, Colonne/ligne absente après exécution) : `nan` → `nan` (attendu : `GBR`)