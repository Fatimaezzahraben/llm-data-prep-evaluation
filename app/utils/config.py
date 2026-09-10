"""
app/utils/config.py
=====================
Charge la configuration depuis le fichier .env et l'expose comme un objet unique
importable partout ailleurs dans le projet (app/services/llm.py, app/api/*.py, etc.).
"""

import os
from dotenv import load_dotenv

load_dotenv()  # lit le fichier .env à la racine du projet


class Settings:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    # mistral-large-latest necessite un compte Pay-As-You-Go actif (sinon 403 "not
    # available in your subscription tier"). mistral-small-latest est generalement
    # accessible sur le tier gratuit -- changeable via .env (MISTRAL_MODEL=...) sans
    # toucher au code, si Mistral change un jour quel modele est gratuit.
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    # "openrouter/free" : routeur special qui choisit automatiquement un modele
    # GRATUIT disponible parmi tout le catalogue OpenRouter -- plus robuste qu'un
    # modele gratuit specifique nomme en dur (la liste des modeles gratuits change
    # regulierement, parfois sans preavis). Changeable via .env (OPENROUTER_MODEL=
    # "openai/gpt-oss-120b:free" par exemple) si un modele precis est prefere.
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "ollama")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()