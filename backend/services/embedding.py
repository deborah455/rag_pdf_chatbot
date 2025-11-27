import os
import requests

# Base URL + model can be overridden via env vars if needed
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")


def embed_text(text: str):
    """
    Generate an embedding for a piece of text using Ollama's /api/embed endpoint.
    Requires: ollama serve + an embedding model (e.g. mxbai-embed-large) pulled.
    """
    url = f"{OLLAMA_BASE_URL}/api/embed"
    payload = {
        "model": OLLAMA_EMBED_MODEL,
        "input": text,
    }

    try:
        resp = requests.post(url, json=payload, timeout=40)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise ValueError("No embeddings returned from Ollama.")
        return embeddings[0]
    except Exception as e:
        # Let the caller see the problem (easier to debug during development)
        raise RuntimeError(f"Ollama embedding error: {e}")
