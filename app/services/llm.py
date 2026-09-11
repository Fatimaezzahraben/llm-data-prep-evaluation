"""
app/services/llm.py
======================
Point d'entrée UNIQUE pour appeler un LLM, quel que soit le fournisseur
(Ollama en local, ou une API payante pour comparaison). Toutes les autres parties du
projet (workflow_generator, notebooks) appellent uniquement call_llm() — jamais
directement requests/openai/mistralai — pour rester interchangeables.

IMPORTANT — reproductibilite (cf. cahier des charges, Phase 5) :
Tous les appels utilisent temperature=0 (generation deterministe/greedy). Sans cela,
deux executions du MEME prompt peuvent produire des scripts differents et donc des F1
differents, ce qui rend toute comparaison "avant/apres amelioration du prompt" invalide
-- on ne pourrait plus distinguer un vrai effet du prompt d'un simple bruit d'echantillonnage.

IMPORTANT — system_prompt optionnel :
Quand system_prompt est fourni, les regles/contraintes generales y sont placees (role
"system"), separement du contenu specifique au dataset (role "user"). Les modeles de
chat suivent generalement plus fidelement des regles placees en system que noyees dans
un long message user.

IMPORTANT — retry automatique sur erreurs reseau transitoires :
Un appel API peut echouer pour des raisons purement reseau (timeout, connexion
coupee), sans rapport avec le prompt ou le code -- observe en pratique avec
httpx.ReadTimeout sur le dataset le plus volumineux (hotel, prompt le plus long).
Ce n'est PAS un bug a corriger dans le prompt : c'est reessaye automatiquement avec
un backoff exponentiel (2s, 4s, 8s...) avant d'abandonner, comme le ferait un humain
qui relance simplement la requete.
"""

import time
import requests

from app.utils.config import settings


class EmptyResponseError(Exception):
    """Levee quand un provider renvoie un contenu vide/None sans erreur SDK --
    observe en pratique avec openrouter/free (routeur a selection aleatoire de
    modele). Enregistree comme retryable (server_error) dans _classify_retry
    ci-dessous, car une nouvelle tentative tire tres probablement un modele
    different qui repondra normalement."""
    pass


# Erreurs considerees comme transitoires (reseau) -> on reessaie. Toute autre
# exception (ex: ValueError sur un provider inconnu, erreur d'authentification)
# remonte immediatement, sans retry inutile.
_RETRYABLE_EXCEPTION_NAMES = (
    "ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
    "Timeout", "ConnectionError", "APIConnectionError", "APITimeoutError",
    "EmptyResponseError",  # provider a renvoie un contenu vide/None (voir la
                           # classe ci-dessus) -- toujours retryable en tant
                           # que server_error, jamais ambigu contrairement a
                           # SDKError.
    "SDKError",  # ex: mistralai.client.errors.sdkerror.SDKError -- son NOM seul ne
                 # dit rien sur la nature de l'erreur (un 503 transitoire ET un 403
                 # "tier non autorise" ont la MEME classe) -> on inspecte EN PLUS le
                 # message pour ne retenter que les vraies erreurs serveur (5xx),
                 # jamais les erreurs client (4xx, ex: 401/403) qu'un retry ne peut
                 # pas resoudre (observe en pratique : 403 "model not available in
                 # your subscription tier" -- reessayer ne change rien, c'est un
                 # probleme de compte/abonnement Mistral, pas un probleme reseau).
)

# Marqueurs textuels indiquant une erreur SERVEUR transitoire (5xx) -> on peut
# reessayer, le service devrait redevenir disponible.
_RETRYABLE_SERVER_ERROR_MARKERS = (
    "503", "502", "500", "504", "529",
    "service unavailable", "internal_server_error", "server_error",
    "bad gateway", "gateway timeout",
)
# 429 est un code 4xx (erreur "client") mais, contrairement a 401/403/400, C'EST une
# erreur transitoire par definition -- le quota de requetes/minute va se reinitialiser
# tout seul. Traite separement des 5xx car il merite un backoff plus LONG (un quota
# par minute a souvent besoin de bien plus que quelques secondes pour se liberer,
# surtout avec un modele "small"/gratuit dont le rate limit est plus serre).
_RATE_LIMIT_MARKERS = ("429", "rate_limited", "rate limit", "too many requests")
# Marqueurs indiquant explicitement une erreur CLIENT (4xx) -> jamais retryable,
# meme si un marqueur serveur apparaissait par coincidence ailleurs dans le message.
_NON_RETRYABLE_MARKERS = (
    "401", "403", "invalid_api_key", "unauthorized", "authentication",
    "tier_not_allowed", "not available in your subscription",
)


def _classify_retry(exc: Exception) -> str | None:
    """Retourne "rate_limit" (backoff long), "server_error" (backoff normal), ou
    None (pas retryable) -- inspecte la chaine d'exceptions par nom de classe (cas
    simple) et par contenu du message pour les "boites noires" type SDKError."""
    exc_chain = []
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        exc_chain.append(current)
        current = current.__cause__ or current.__context__

    for e in exc_chain:
        name = type(e).__name__

        # Classes SANS AMBIGUITE (SDK openai, utilise aussi par openrouter/groq via
        # le meme client openai pointe sur un autre endpoint) -- le nom seul suffit.
        if name == "RateLimitError":
            return "rate_limit"
        if name == "InternalServerError":
            return "server_error"

        if name not in _RETRYABLE_EXCEPTION_NAMES:
            continue
        if name != "SDKError":
            return "server_error"
        msg = str(e).lower()
        has_client_marker = any(m in msg for m in _NON_RETRYABLE_MARKERS)
        if has_client_marker:
            continue
        if any(m in msg for m in _RATE_LIMIT_MARKERS):
            return "rate_limit"
        if any(m in msg for m in _RETRYABLE_SERVER_ERROR_MARKERS):
            return "server_error"
    return None


def _is_retryable(exc: Exception) -> bool:
    """Conserve pour retrocompatibilite (ex: tests existants) -- utiliser
    _classify_retry() directement pour choisir le delai de backoff."""
    return _classify_retry(exc) is not None


def call_llm(prompt: str, provider: str = None, model: str = None,
              system_prompt: str = None, max_retries: int = 6,
              retry_base_delay: float = 2.0, rate_limit_base_delay: float = 15.0,
              max_delay_cap: float = 120.0) -> dict:
    """
    Envoie un prompt au LLM choisi et retourne le texte généré + des métadonnées
    utiles pour les métriques de coût/latence (Phase 5).

    Parameters
    ----------
    prompt : str
        Le message "user" (contenu specifique au dataset : schema/profil/exemples).
    system_prompt : str or None
        Le message "system" optionnel (regles/contraintes generales). Si None, seul
        `prompt` est envoye (comportement retrocompatible).
    max_retries : int
        Nombre de tentatives supplementaires en cas d'erreur transitoire (reseau OU
        rate limit) (defaut : 6, donc 7 tentatives au total avant d'abandonner --
        augmente par rapport a avant : 4 tentatives se sont averees insuffisantes en
        pratique sur un compte au rate limit tres serre, ou l'echec persistait apres
        ~225s d'attente cumulee).
    retry_base_delay : float
        Delai (secondes) avant la 1ere retentative pour une erreur SERVEUR (5xx) ;
        double a chaque echec suivant (backoff exponentiel : 2s, 4s, 8s, 16s...).
    rate_limit_base_delay : float
        Delai (secondes) avant la 1ere retentative pour un 429 "rate limit exceeded"
        specifiquement -- volontairement plus LONG que retry_base_delay (15s par
        defaut) car un quota par minute a besoin de plus de temps pour se liberer
        qu'un simple blip reseau ; double aussi a chaque echec suivant, jusqu'a
        max_delay_cap.
    max_delay_cap : float
        Plafond (secondes) sur un delai d'attente individuel -- sans ce plafond, un
        backoff exponentiel pur peut demander une attente demesuree (ex: 480s des
        la 5eme tentative). Avec ce plafond a 120s et max_retries=6, l'attente
        cumulee totale peut atteindre ~465s (~7-8 min), bien plus genereuse qu'avant
        (~225s), sans jamais imposer une seule attente exageree.

    Returns
    -------
    dict avec les clés :
        - "text" : la réponse texte du LLM (le script Python généré)
        - "latency_seconds" : temps de la requête
        - "provider" : fournisseur utilisé
        - "model" : modèle utilisé
    """
    provider = provider or settings.DEFAULT_LLM_PROVIDER
    start = time.time()

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            text, resolved_model = _call_provider_once(provider, model, prompt, system_prompt)
            latency = time.time() - start
            return {"text": text, "latency_seconds": round(latency, 2),
                    "provider": provider, "model": resolved_model}
        except Exception as e:
            last_exc = e
            kind = _classify_retry(e)
            if attempt < max_retries and kind is not None:
                base = rate_limit_base_delay if kind == "rate_limit" else retry_base_delay
                delay = min(base * (2 ** attempt), max_delay_cap)
                label = "rate limit (429)" if kind == "rate_limit" else type(e).__name__
                print(f"  [llm] Erreur transitoire ({label}), nouvelle tentative "
                      f"dans {delay:.0f}s (essai {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                continue
            raise

    raise last_exc  # ne devrait jamais etre atteint, garde-fou


def _call_provider_once(provider: str, model: str, prompt: str, system_prompt: str):
    """Un seul essai d'appel au provider choisi. Leve une exception si ca echoue ;
    call_llm() gere le retry autour de cette fonction."""
    if provider == "ollama":
        model = model or settings.OLLAMA_MODEL
        # L'API /api/generate d'Ollama n'a pas de role "system" separe -> on prefixe.
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": full_prompt, "stream": False,
                  "options": {"temperature": 0}},
            timeout=900,
        )
        response.raise_for_status()
        text = response.json()["response"]
        return text, model

    elif provider == "mistral_api":
        from mistralai.client import Mistral
        client = Mistral(api_key=settings.MISTRAL_API_KEY)
        # IMPORTANT : mistral-large-latest necessite un compte Pay-As-You-Go (403
        # "not available in your subscription tier" sans ca). Le modele par defaut
        # se lit maintenant depuis settings.MISTRAL_MODEL (configurable via .env),
        # avec un petit modele gratuit comme valeur par defaut -- teste d'abord avec
        # /health/llm (app/api/health.py) avant de lancer un run complet, pour
        # confirmer que le modele choisi est bien accessible avec votre cle API.
        model = model or settings.MISTRAL_MODEL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = client.chat.complete(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = result.choices[0].message.content
        return text, model

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        model = model or "gpt-4o-mini"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        text = result.choices[0].message.content
        return text, model

    elif provider == "openrouter":
        # OpenRouter expose une API compatible OpenAI -- on reutilise le SDK openai
        # deja installe, juste pointe vers le endpoint d'OpenRouter (pas de nouveau
        # package pip necessaire), meme approche que pour Groq. Modele par defaut :
        # "openrouter/free", un ROUTEUR qui choisit automatiquement un modele
        # GRATUIT disponible parmi tout le catalogue OpenRouter -- plus robuste
        # qu'un modele gratuit specifique nomme en dur, puisque la liste des
        # modeles gratuits change regulierement sans preavis. Free tier au moment
        # de l'ecriture : ~20 requetes/minute, 50-1000 requetes/jour (voir
        # https://openrouter.ai/docs pour les limites a jour).
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        model = model or settings.OPENROUTER_MODEL
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        # IMPORTANT (crash reel observe) : OpenRouter peut repondre HTTP 200 avec
        # un corps "en erreur" plutot que lever une exception SDK -- observe en
        # pratique avec `result.choices` valant None (pas juste une liste vide)
        # quand le modele choisi est indisponible/surcharge/en cours de
        # moderation. `result.choices[0]` plante alors avec un TypeError brut
        # ("'NoneType' object is not subscriptable") qui n'est PAS retryable
        # (TypeError n'est pas dans _RETRYABLE_EXCEPTION_NAMES) -- ca faisait
        # planter tout le run pour un seul appel malchanceux. Verifie choices
        # AVANT d'indexer, et traite ce cas comme n'importe quelle reponse vide
        # (EmptyResponseError, retryable) plutot que de laisser le TypeError
        # remonter brut.
        if not result.choices:
            error_detail = getattr(result, "error", None)
            raise EmptyResponseError(
                f"Provider '{provider}' (model={model}) returned no choices "
                f"(error={error_detail})"
            )
        text = result.choices[0].message.content
        if not text:
            # openrouter/free choisit un modele ALEATOIRE a chaque appel -- un
            # crash reel a ete observe ou le modele tire au sort renvoyait un
            # contenu vide/None (sans lever d'erreur cote SDK). On force une
            # retentative (via une exception reconnue comme "server_error" par
            # _classify_retry) plutot que de propager silencieusement un texte
            # vide -- une nouvelle tentative tirera tres probablement un AUTRE
            # modele, qui repondra correctement.
            raise EmptyResponseError(
                f"Provider '{provider}' (model={model}) returned an empty response"
            )
        return text, model

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        model = model or "claude-sonnet-4-5"
        kwargs = {"model": model, "max_tokens": 2000, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]}
        if system_prompt:
            kwargs["system"] = system_prompt
        result = client.messages.create(**kwargs)
        text = result.content[0].text
        return text, model

    else:
        raise ValueError(f"Fournisseur LLM inconnu : {provider}")