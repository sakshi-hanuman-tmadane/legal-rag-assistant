# ⚖️ Legal RAG Assistant

An AI-powered Legal Document Question Answering System built using Retrieval-Augmented Generation (RAG).

## Features

- Upload legal PDF documents
- Automatic text extraction
- Semantic search using Sentence Transformers
- FAISS Vector Database
- Google Gemini API
- Source citation from retrieved chunks
- Streamlit Web Interface

## Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Google Gemini API
- PyMuPDF

## Project Structure

```
legal-rag-assistant/
│
├── utils/
├── streamlit_app.py
├── requirements.txt
├── rebuild_vector_db.py
├── README.md
└── .gitignore
```

## Installation

```bash
git clone https://github.com/yourusername/legal-rag-assistant.git

cd legal-rag-assistant

pip install -r requirements.txt
```

Create a `.env` file

```
GOOGLE_API_KEY=YOUR_API_KEY
```

Run

```bash
streamlit run streamlit_app.py
```

## Author

**Sakshi Hanumant Madane**

B.Sc Data Science

Savitribai Phule Pune University