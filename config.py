"""Central configuration for VentureIQ.

All paths are resolved relative to this file, never to a machine-specific location.
Environment variables (or a local ``.env`` file) override the defaults below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = "VentureIQ"
APP_VERSION = "1.0.0"


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from ``.env`` without overriding real environment variables."""
    path = path or BASE_DIR / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(os.environ.get(name, default))))
    except (TypeError, ValueError):
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (BASE_DIR / p).resolve()


@dataclass(frozen=True)
class Settings:
    groq_model: str
    groq_base_url: str
    llm_timeout_s: float
    llm_max_attempts: int       # attempts to obtain schema-valid JSON
    llm_network_retries: int    # extra tries for transient network / 5xx errors
    llm_max_tokens: int
    kb_dir: Path
    index_dir: Path
    models_dir: Path
    data_dir: Path
    db_path: Path
    chunk_words: int
    chunk_overlap: int
    top_k: int
    max_chunks_per_doc: int
    embedding_model: str
    semantic_mode: str          # "auto" or "off"
    seed: int
    risk_lower_min: float       # probability >= this -> "Lower risk"
    risk_moderate_min: float    # probability >= this -> "Moderate risk", else "Higher risk"
    log_level: str


def get_settings() -> Settings:
    """Read settings from the environment at call time (so tests can override them)."""
    return Settings(
        groq_model=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        groq_base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        llm_timeout_s=_env_float("VENTUREIQ_LLM_TIMEOUT", 60.0, 5.0, 300.0),
        llm_max_attempts=_env_int("VENTUREIQ_LLM_MAX_ATTEMPTS", 2, 1, 4),
        llm_network_retries=_env_int("VENTUREIQ_LLM_NETWORK_RETRIES", 1, 0, 3),
        llm_max_tokens=_env_int("VENTUREIQ_LLM_MAX_TOKENS", 3500, 500, 8000),
        kb_dir=_env_path("VENTUREIQ_KB_DIR", BASE_DIR / "knowledge_base"),
        index_dir=_env_path("VENTUREIQ_INDEX_DIR", BASE_DIR / "models" / "rag_index"),
        models_dir=_env_path("VENTUREIQ_MODELS_DIR", BASE_DIR / "models"),
        data_dir=_env_path("VENTUREIQ_DATA_DIR", BASE_DIR / "data"),
        db_path=_env_path("VENTUREIQ_DB_PATH", BASE_DIR / "data" / "ventureiq_history.sqlite3"),
        chunk_words=_env_int("VENTUREIQ_CHUNK_WORDS", 300, 50, 1000),
        chunk_overlap=_env_int("VENTUREIQ_CHUNK_OVERLAP", 60, 0, 400),
        top_k=_env_int("VENTUREIQ_TOP_K", 5, 1, 12),
        max_chunks_per_doc=_env_int("VENTUREIQ_MAX_CHUNKS_PER_DOC", 2, 1, 6),
        embedding_model=os.environ.get(
            "VENTUREIQ_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        semantic_mode=os.environ.get("VENTUREIQ_SEMANTIC", "auto").lower(),
        seed=_env_int("VENTUREIQ_SEED", 42, 0, 2**31 - 1),
        risk_lower_min=_env_float("VENTUREIQ_RISK_LOWER_MIN", 0.65, 0.0, 1.0),
        risk_moderate_min=_env_float("VENTUREIQ_RISK_MODERATE_MIN", 0.40, 0.0, 1.0),
        log_level=os.environ.get("VENTUREIQ_LOG_LEVEL", "INFO").upper(),
    )


PROVIDERS = {
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "OPENROUTER_MODEL",
        "url_env": "OPENROUTER_BASE_URL",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "url": "https://openrouter.ai/api/v1",
        "get_key": "https://openrouter.ai/keys",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "url_env": "GROQ_BASE_URL",
        "model": "llama-3.3-70b-versatile",
        "url": "https://api.groq.com/openai/v1",
        "get_key": "https://console.groq.com/keys",
    },
}


def active_provider(override_key: str | None = None) -> str:
    """Choose the active provider based on override key, explicit LLM_PROVIDER, or available keys."""
    if override_key and override_key.strip():
        k = override_key.strip()
        if k.startswith("gsk_"):
            return "groq"
        if k.startswith("sk-or-"):
            return "openrouter"
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced in PROVIDERS:
        return forced
    # Check if Groq key exists and is non-empty
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    # Check if OpenRouter key exists
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    return "groq"


def provider_config(override_key: str | None = None) -> dict:
    """Return provider, model and base_url (never the key) for the active provider."""
    name = active_provider(override_key)
    p = PROVIDERS[name]
    return {
        "provider": name,
        "model": os.environ.get(p["model_env"], p["model"]),
        "base_url": os.environ.get(p["url_env"], p["url"]),
        "key_env": p["key_env"],
        "get_key": p["get_key"],
    }


def resolve_api_key(override: str | None = None) -> str | None:
    """Return the active AI API key from (in order) a UI override, env var, or Streamlit secrets."""
    if override and override.strip():
        return override.strip()
    name = active_provider()
    p = PROVIDERS[name]
    env = os.environ.get(p["key_env"], "").strip()
    if env:
        return env
    # Fallback to alternate provider key if primary is empty
    alt_name = "openrouter" if name == "groq" else "groq"
    alt_env = os.environ.get(PROVIDERS[alt_name]["key_env"], "").strip()
    if alt_env:
        return alt_env
    try:  # Streamlit secrets are optional
        import streamlit as st
        val = st.secrets.get("GROQ_API_KEY") or st.secrets.get("OPENROUTER_API_KEY")
        if val and str(val).strip():
            return str(val).strip()
    except Exception:
        pass
    return None


def save_api_key_to_env(
    key: str,
    provider: str = "auto",
    model: str | None = None,
    path: Path | None = None,
) -> dict:
    """Safely update or add API credentials to .env and immediately update os.environ.

    Never logs the full secret. Returns metadata with a masked key.
    """
    clean_key = (key or "").strip()
    if not clean_key:
        raise ValueError("API key cannot be empty.")

    # Determine provider
    chosen_provider = provider.lower() if provider else "auto"
    if chosen_provider == "auto":
        if clean_key.startswith("gsk_"):
            chosen_provider = "groq"
        elif clean_key.startswith("sk-or-"):
            chosen_provider = "openrouter"
        else:
            chosen_provider = "groq"

    if chosen_provider not in PROVIDERS:
        chosen_provider = "groq"

    target_env = PROVIDERS[chosen_provider]["key_env"]
    model_env = PROVIDERS[chosen_provider]["model_env"]
    target_model = model or os.environ.get(model_env, PROVIDERS[chosen_provider]["model"])

    # Update runtime environment immediately
    os.environ[target_env] = clean_key
    os.environ["LLM_PROVIDER"] = chosen_provider
    if target_model:
        os.environ[model_env] = target_model

    # Persist to .env file
    env_file = path or (BASE_DIR / ".env")
    lines: list[str] = []
    if env_file.is_file():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated_key = False
    updated_provider = False
    updated_model = False
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{target_env}=") or stripped.startswith(f"#{target_env}="):
            new_lines.append(f"{target_env}={clean_key}")
            updated_key = True
        elif stripped.startswith("LLM_PROVIDER=") or stripped.startswith("#LLM_PROVIDER="):
            new_lines.append(f"LLM_PROVIDER={chosen_provider}")
            updated_provider = True
        elif target_model and (stripped.startswith(f"{model_env}=") or stripped.startswith(f"#{model_env}=")):
            new_lines.append(f"{model_env}={target_model}")
            updated_model = True
        else:
            new_lines.append(line)

    if not updated_key:
        new_lines.append(f"{target_env}={clean_key}")
    if not updated_provider:
        new_lines.append(f"LLM_PROVIDER={chosen_provider}")
    if target_model and not updated_model:
        new_lines.append(f"{model_env}={target_model}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    masked = clean_key[:6] + "..." + clean_key[-4:] if len(clean_key) > 10 else "***"
    return {
        "ok": True,
        "provider": chosen_provider,
        "model": target_model,
        "key_env": target_env,
        "masked_key": masked,
    }
