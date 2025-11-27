# rag_pdf_chatbot
🧠 RAG PDF Chatbot

A simple chatbot that answers questions from uploaded PDFs using RAG (Retrieval-Augmented Generation) with OpenAI embeddings + FAISS.

🚀 Features

Upload PDF and extract text

Chunk + embed using OpenAI

Store embeddings in FAISS

Retrieve top relevant chunks

Generate grounded answers using an LLM

Clean and simple backend structure

🛠 Tech Stack

Python

OpenAI API

FAISS

pdfplumber / PyPDF

Django or FastAPI (depending on your version)

📦 Setup
git clone https://github.com/deborah455/rag_pdf_chatbot.git
cd rag_pdf_chatbot
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt


Create .env:

OPENAI_API_KEY=your_key_here


Run the app:

python manage.py runserver   # for Django


or

uvicorn main:app --reload    # for FastAPI

▶️ Example Usage

POST /ask

{
  "doc_id": "abc123",
  "question": "Summarize section 2"
}

⭐ Future Improvements

UI interface

Multi-file support

Postgres + pgvector option

Docker version
