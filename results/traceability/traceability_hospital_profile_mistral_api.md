# Rapport de traçabilité — hospital_profile_mistral_api

**Total d'erreurs suivies :** 482

## Répartition des statuts

| Statut | Nombre | % |
|---|---|---|
| Non corrigé (regression) | 265 | 55.0% |
| Corrigé | 134 | 27.8% |
| Mal corrigé (valeur différente mais fausse) | 83 | 17.2% |

## Répartition par famille d'erreur

| Famille | Corrigé | Non corrigé | Mal corrigé |
|---|---|---|---|
| real_world_error | 134 | 265 | 83 |

## Exemples de corrections réussies

- **provider_number** (ligne 13) : `1xx19` → `10019` (attendu : `10019`)
- **provider_number** (ligne 45) : `x0005` → `10005` (attendu : `10005`)
- **provider_number** (ligne 81) : `1000x` → `10006` (attendu : `10006`)
- **provider_number** (ligne 213) : `x00xx` → `10011` (attendu : `10011`)
- **provider_number** (ligne 244) : `x00x5` → `10015` (attendu : `10015`)

## Exemples de corrections manquées ou erronées

- **address_1** (ligne 57, Non corrigé (regression)) : `2505xuxsxhighwayx431xnorth` → `2505xuxsxhighwayx431xnorth` (attendu : `2505 u s highway 431 north`)
- **address_1** (ligne 94, Non corrigé (regression)) : `702xnxmainxst` → `702xnxmainxst` (attendu : `702 n main st`)
- **address_1** (ligne 106, Non corrigé (regression)) : `702 x maix st` → `702 x maix st` (attendu : `702 n main st`)
- **address_1** (ligne 161, Non corrigé (regression)) : `201 pine sxreex norxhwesx` → `201 pine sxreex norxhwesx` (attendu : `201 pine street northwest`)
- **address_1** (ligne 175, Non corrigé (regression)) : `8000 alabama xigxway 69` → `8000 alabama xigxway 69` (attendu : `8000 alabama highway 69`)