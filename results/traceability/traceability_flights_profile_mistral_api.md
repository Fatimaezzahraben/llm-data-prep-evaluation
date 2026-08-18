# Rapport de traçabilité — flights_profile_mistral_api

**Total d'erreurs suivies :** 4920

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Corrigé | 3249 | 66.0% |
| Mal corrigé (valeur différente mais fausse) | 855 | 17.4% |
| Non corrigé (regression) | 816 | 16.6% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| missing_values | 1891 | 0 | 421 |
| real_world_error | 1358 | 816 | 434 |

## Exemples de corrections réussies

- **sched_dep_time** (ligne 50) : `nan` → `7:10 a.m.` (attendu : `7:10 a.m.`)
- **sched_dep_time** (ligne 52) : `nan` → `6:40 a.m.` (attendu : `6:40 a.m.`)
- **sched_dep_time** (ligne 53) : `nan` → `8:00 a.m.` (attendu : `8:00 a.m.`)
- **sched_dep_time** (ligne 54) : `nan` → `8:41 a.m.` (attendu : `8:41 a.m.`)
- **sched_dep_time** (ligne 55) : `nan` → `7:45 p.m.` (attendu : `7:45 p.m.`)

## Exemples de corrections manquées ou erronées

- **sched_dep_time** (ligne 86, Mal corrigé (valeur différente mais fausse)) : `nan` → `12:00 p.m.` (attendu : `12:00 a.m.`)
- **sched_dep_time** (ligne 186, Non corrigé (regression)) : `12:00 p.m.` → `12:00 p.m.` (attendu : `12:00 a.m.`)
- **sched_dep_time** (ligne 281, Non corrigé (regression)) : `12:00 p.m.` → `12:00 p.m.` (attendu : `12:00 a.m.`)
- **sched_dep_time** (ligne 373, Mal corrigé (valeur différente mais fausse)) : `nan` → `12:00 p.m.` (attendu : `12:00 a.m.`)
- **sched_dep_time** (ligne 473, Non corrigé (regression)) : `12:00 p.m.` → `12:00 p.m.` (attendu : `12:00 a.m.`)