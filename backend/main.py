from fastapi import FastAPI
from backend.routers import rag

app = FastAPI(
    title="ContextFlow RAG API",
    description="FastAPI backend for a local RAG system using ChromaDB and Ollama.",
    version="0.1.0",
)

# Register RAG router
app.include_router(rag.router)


@app.get("/")
def root():
    return {"status": "running", "message": "ContextFlow RAG backend is active"}
