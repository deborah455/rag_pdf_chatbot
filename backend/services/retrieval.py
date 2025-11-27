import chromadb
from backend.services.embedding import embed_text

# Initialize Chroma client and collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"}
)

# Optional seed docs so you get something even before uploading
seed_docs = [
    "This system is a Retrieval-Augmented Generation (RAG) chatbot demo built with FastAPI, ChromaDB, Streamlit, and local LLMs via Ollama.",
    "User questions are embedded into vectors using an Ollama embedding model, then used to retrieve similar text chunks from ChromaDB.",
    "The architecture supports ingesting user documents (PDF/TXT), storing them as embeddings, and answering questions grounded in those documents."
]

seed_ids = ["doc1", "doc2", "doc3"]

if collection.count() == 0:
    embeddings = [embed_text(text) for text in seed_docs]
    collection.add(
        documents=seed_docs,
        embeddings=embeddings,
        ids=seed_ids,
    )

def retrieve_chunks(query_embedding, top_k=3):
    """
    Retrieve similar chunks from the vector store.
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results
