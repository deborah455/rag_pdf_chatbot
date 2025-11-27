from fastapi import APIRouter
from pydantic import BaseModel
from backend.services.embedding import embed_text
from backend.services.retrieval import retrieve_chunks, collection
from backend.services.chat import generate_answer

router = APIRouter(prefix="/rag", tags=["RAG"])


class IngestRequest(BaseModel):
    doc_id: str
    text: str


@router.post("/ingest")
def ingest_document(payload: IngestRequest):
    """
    Ingest a plain-text document into the Chroma collection.
    """
    embedding = embed_text(payload.text)
    collection.add(
        documents=[payload.text],
        embeddings=[embedding],
        ids=[payload.doc_id],
    )
    return {"status": "ok", "doc_id": payload.doc_id}


@router.post("/query")
def rag_query(question: str):
    """
    Main RAG query endpoint:
    - embed question
    - retrieve similar chunks from ChromaDB
    - generate answer from context
    """
    q_embed = embed_text(question)
    results = retrieve_chunks(q_embed)
    answer = generate_answer(question, results)

    return {
        "question": question,
        "chunks_used": results,
        "answer": answer,
    }
