"""Chat model factory for the three supported providers: OpenAI, Ollama, Hugging Face."""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    """Return a LangChain chat model for the given provider.

    provider: "openai" | "ollama" | "huggingface". Defaults to $PROVIDER (or "openai").
    model: provider-specific model name. Defaults to DEFAULT_MODELS[provider].
    """
    # Explicit args win; otherwise fall back to .env (PROVIDER / OPENAI_MODEL etc.),
    # then to the hardcoded default for that provider.
    provider = (provider or os.environ.get("PROVIDER", "openai")).lower()
    model = model or os.environ.get(f"{provider.upper()}_MODEL") or DEFAULT_MODELS[provider]

    if provider == "openai":
        # Imported lazily so installing/using one provider doesn't require the others' SDKs.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ.get("OPENAI_API_KEY"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "huggingface":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        # HF's endpoint rejects temperature=0.0, so nudge it up to the smallest usable value.
        llm = HuggingFaceEndpoint(
            repo_id=model,
            temperature=temperature or 0.01,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        )
        return ChatHuggingFace(llm=llm)

    raise ValueError(f"Unknown provider: {provider!r}. Expected one of {sorted(DEFAULT_MODELS)}.")
