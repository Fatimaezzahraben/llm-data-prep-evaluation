# Rapport de traçabilité — flights_schema_mistral_api

**Total d'erreurs suivies :** 4920

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Mal corrigé (valeur différente mais fausse) | 2814 | 57.2% |
| Non corrigé (regression) | 2051 | 41.7% |
| Corrigé | 55 | 1.1% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 45 | 0 | 2267 |
| real_world_error | 10 | 2051 | 547 |

## Exemples de corrections réussies

- **sched_dep_time** (ligne 50) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)
- **sched_dep_time** (ligne 100) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)
- **sched_dep_time** (ligne 148) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)
- **sched_dep_time** (ligne 149) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)
- **sched_dep_time** (ligne 337) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)

## Exemples de corrections manquées ou erronées

- **sched_dep_time** (ligne 52, Mal corrigé (valeur différente mais fausse)) : `nan` → `7:10 a.m.` (attendu : `6:40 a.m.`)
- **sched_dep_time** (ligne 53, Mal corrigé (valeur différente mais fausse)) : `nan` → `7:10 a.m.` (attendu : `8:00 a.m.`)
- **sched_dep_time** (ligne 54, Mal corrigé (valeur différente mais fausse)) : `nan` → `7:10 a.m.` (attendu : `8:41 a.m.`)
- **sched_dep_time** (ligne 55, Mal corrigé (valeur différente mais fausse)) : `nan` → `7:10 a.m.` (attendu : `7:45 p.m.`)
- **sched_dep_time** (ligne 56, Mal corrigé (valeur différente mais fausse)) : `nan` → `7:10 a.m.` (attendu : `3:27 p.m.`)