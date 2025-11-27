import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")


def generate_answer(question: str, chunks):
    """
    Use Ollama's /api/chat endpoint to generate an answer grounded in retrieved context.
    - chunks: result from ChromaDB query (dict with 'documents', 'ids', etc.)
    """
    # Build context from retrieved docs
    context = ""
    if isinstance(chunks, dict) and "documents" in chunks and chunks["documents"]:
        docs = chunks["documents"][0]
        context = "\n\n".join(docs)

    system_prompt = (
        "You are a helpful assistant for a Retrieval-Augmented Generation (RAG) system. "
        "Use ONLY the context provided below to answer. "
        "If the context is not sufficient, say that clearly instead of guessing.\n\n"
        f"Context:\n{context}\n"
    )

    user_prompt = f"Question: {question}\n\nAnswer based only on the context above."

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,  # easier to handle as a single JSON
    }

    try:
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")

        if not content:
            if not context:
                return (
                    "I couldn't find any relevant context for this question. "
                    "Try uploading a document and ingesting it into the RAG store first."
                )
            return (
                "A response could not be generated from the model, "
                "but some context was retrieved. You may want to try again."
            )

        return content

    except Exception as e:
        # Graceful fallback message
        if not context:
            return (
                "The local LLM (Ollama) could not be reached and no context was available. "
                "Please check that Ollama is running and that documents have been ingested."
            )
        return (
            "The local LLM (Ollama) could not be reached, but context was retrieved.\n\n"
            f"Here is the raw context:\n\n{context}\n\n"
            f"(Technical error: {e})"
        )
