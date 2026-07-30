import time
import streamlit as st
import tempfile
import fitz
import faiss
import numpy as np
import pickle
from pathlib import Path
from utils.chunking import chunk_documents
from utils.embeddings import model
from utils.gemini_api import ask_gemini
# -----------------------------------
# VECTOR DATABASE PATHS
# -----------------------------------

VECTOR_DIR = Path("vector_db")

INDEX_FILE = VECTOR_DIR / "faiss.index"

CHUNKS_FILE = VECTOR_DIR / "chunks.pkl"

VECTOR_DIR.mkdir(exist_ok=True)



# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="⚖️ Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal RAG Assistant")
st.caption("Ask questions from your uploaded legal document")

# ----------------------------------------------------
# SESSION STATE
# ----------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = ""

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Legal PDF")

    uploaded_pdf = st.file_uploader(
        "Choose PDF",
        type=["pdf"]
    )

    if uploaded_pdf is not None:

        if uploaded_pdf.name != st.session_state.uploaded_file:

            with st.spinner("Reading PDF..."):
                start = time.time()

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:

                    tmp.write(uploaded_pdf.read())

                    pdf_path = tmp.name

                pdf = fitz.open(pdf_path)

                documents = []

                for page in range(len(pdf)):
                    st.write("PDF Read:", round(time.time() - start, 2), "seconds")

                    text = pdf.load_page(page).get_text()

                    documents.append(
                        {
                            "text": text,
                            "page": page + 1,
                            "source": uploaded_pdf.name
                        }
                    )

                chunks = chunk_documents(documents)
                st.write("Chunking:", round(time.time() - start, 2), "seconds")

                st.success(f"✅ Total Chunks: {len(chunks)}")

                texts = [c["text"] for c in chunks]

                from utils.embeddings import create_embeddings

                embeddings = create_embeddings(chunks).astype("float32")
                st.success("✅ Embeddings Created")

                embeddings = embeddings.astype("float32")
                st.write("Embeddings:", round(time.time() - start, 2), "seconds")

                dimension = embeddings.shape[1]

                index = faiss.IndexFlatL2(dimension)

                index.add(embeddings)
                st.write("FAISS:", round(time.time() - start, 2), "seconds")
                st.success("✅ FAISS Index Created")

                # Save FAISS index
                faiss.write_index(index, str(INDEX_FILE))
                st.success("✅ FAISS Saved")

                # Save chunks
                with open(CHUNKS_FILE, "wb") as f:
                   pickle.dump(chunks, f)
                   st.success("✅ Chunks Saved")

                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.uploaded_file = uploaded_pdf.name

        

                st.success("✅ PDF Indexed Successfully")

    st.divider()

    if st.session_state.uploaded_file != "":

        st.success(st.session_state.uploaded_file)

    st.divider()

    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# ----------------------------------------------------
# CHAT HISTORY
# ----------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])
        # ----------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------

query = st.chat_input("Ask your legal question...")

if query:

    # Show user message
    st.chat_message("user").markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # No PDF uploaded
    if st.session_state.index is None:

        answer = "⚠ Please upload a legal PDF first."

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    else:

        with st.spinner("Searching document..."):

            # Query Embedding
            query_embedding = model.encode(
                [query],
                convert_to_numpy=True
            ).astype("float32")

            # Search Top 5 Chunks
            distances, indices = st.session_state.index.search(
                query_embedding,
                5
            )

            retrieved_chunks = []

            for idx in indices[0]:

                if idx != -1:

                    retrieved_chunks.append(
                        st.session_state.chunks[idx]
                    )

            # Build Context
            context = "\n\n".join(
                chunk["text"]
                for chunk in retrieved_chunks
            )

            # Ask Gemini
            answer = ask_gemini(
                context=context,
                question=query
            )

        # Assistant Response
        with st.chat_message("assistant"):

            st.markdown(answer)

            st.divider()

            with st.expander("📄 Source Chunks"):

                for chunk in retrieved_chunks:

                    st.markdown(
                        f"### 📄 Page {chunk['page']}"
                    )

                    st.write(chunk["text"])

                    st.divider()

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

            