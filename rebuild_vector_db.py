from utils.pdf_loader import load_pdfs
from utils.chunking import chunk_documents
from utils.embeddings import create_embeddings
from utils.vector_store import save_index

documents = load_pdfs("data/legal_pdfs")

chunks = chunk_documents(documents)

embeddings = create_embeddings(chunks)

save_index(embeddings, chunks)

print("Vector database rebuilt successfully!")