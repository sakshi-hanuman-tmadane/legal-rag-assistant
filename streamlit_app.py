import time
import streamlit as st
import tempfile
import fitz
import faiss
import pickle
from pathlib import Path

from utils.chunking import chunk_documents
from utils.embeddings import model
from utils.groq_api import ask_groq


# ----------------------------------------------------
# VECTOR DATABASE PATHS
# ----------------------------------------------------

VECTOR_DIR = Path("vector_db")

INDEX_FILE = VECTOR_DIR / "faiss.index"

CHUNKS_FILE = VECTOR_DIR / "chunks.pkl"

VECTOR_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------------------------
# CUSTOM STYLE
# ----------------------------------------------------

st.markdown(
    """
    <style>

    .main-title {
        font-size:42px;
        font-weight:700;
        color:#1f2937;
    }

    .subtitle {
        font-size:18px;
        color:#6b7280;
        margin-bottom:25px;
    }

    .source-card {
        background:#f8fafc;
        padding:15px;
        border-radius:12px;
        border:1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

st.markdown(
    """
    <div class="main-title">
        ⚖️ Legal AI Assistant
    </div>

    <div class="subtitle">
        AI-powered legal document question answering using
        Retrieval-Augmented Generation (RAG)
    </div>
    """,
    unsafe_allow_html=True
)


st.divider()


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

    st.markdown(
        """
        ## ⚖️ Legal AI

        Upload a legal document and ask questions
        using AI-powered semantic search.
        """
    )

    st.divider()

    st.subheader("📂 Upload Document")


    uploaded_pdf = st.file_uploader(
        "Select Legal PDF",
        type=["pdf"]
    )


    if uploaded_pdf is not None:

        if uploaded_pdf.name != st.session_state.uploaded_file:

            with st.spinner("📖 Processing legal document..."):

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

                    text = pdf.load_page(page).get_text()

                    documents.append(
                        {
                            "text": text,
                            "page": page + 1,
                            "source": uploaded_pdf.name
                        }
                    )
                    # -----------------------------
                # CHUNKING
                # -----------------------------

                chunks = chunk_documents(documents)

                st.success(
                    f"✅ Total Chunks Created: {len(chunks)}"
                )


                # -----------------------------
                # EMBEDDINGS
                # -----------------------------

                from utils.embeddings import create_embeddings

                embeddings = create_embeddings(chunks)

                embeddings = embeddings.astype("float32")


                st.success(
                    "✅ Document Embeddings Created"
                )


                # -----------------------------
                # FAISS VECTOR DATABASE
                # -----------------------------

                dimension = embeddings.shape[1]


                index = faiss.IndexFlatL2(
                    dimension
                )


                index.add(embeddings)


                st.success(
                    "✅ FAISS Search Index Created"
                )


                # -----------------------------
                # SAVE VECTOR DATABASE
                # -----------------------------

                faiss.write_index(
                    index,
                    str(INDEX_FILE)
                )


                with open(CHUNKS_FILE, "wb") as f:

                    pickle.dump(
                        chunks,
                        f
                    )


                st.success(
                    "💾 Vector Database Saved"
                )


                # Store in session

                st.session_state.index = index

                st.session_state.chunks = chunks

                st.session_state.uploaded_file = uploaded_pdf.name


                st.success(
                    "🎉 PDF Indexed Successfully"
                )


    # -----------------------------
    # CURRENT DOCUMENT
    # -----------------------------

    st.divider()


    if st.session_state.uploaded_file:

        st.success(
            f"📄 {st.session_state.uploaded_file}"
        )


    st.divider()


    if st.button(
        "🗑 Clear Conversation",
        use_container_width=True
    ):

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

query = st.chat_input(
    "💬 Ask your legal question..."
)


if query:

    # User message

    with st.chat_message("user"):

        st.markdown(query)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    # Check PDF

    if st.session_state.index is None:


        answer = (
            "⚠️ Please upload a legal PDF first."
        )


        with st.chat_message("assistant"):

            st.warning(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    else:


        with st.spinner(
            "🔍 Searching legal documents and generating answer..."
        ):


            # -----------------------------
            # QUERY EMBEDDING
            # -----------------------------

            query_embedding = model.encode(
                [query],
                convert_to_numpy=True
            ).astype("float32")



            # -----------------------------
            # FAISS SEARCH
            # -----------------------------

            distances, indices = (
                st.session_state.index.search(
                    query_embedding,
                    5
                )
            )


            retrieved_chunks = []


            for idx in indices[0]:

                if idx != -1:

                    retrieved_chunks.append(
                        st.session_state.chunks[idx]
                    )



            # -----------------------------
            # BUILD CONTEXT
            # -----------------------------

            context = "\n\n".join(
                chunk["text"]
                for chunk in retrieved_chunks
            )



            # -----------------------------
            # GROQ RESPONSE
            # -----------------------------

            answer = ask_groq(
                context=context,
                question=query
            )



        # -----------------------------
        # ASSISTANT RESPONSE
        # -----------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)


            st.divider()


            with st.expander(
                "📚 Sources Used"
            ):


                for chunk in retrieved_chunks:


                    st.markdown(
                        f"""
                        <div class="source-card">

                        <h4>📄 {chunk['source']}</h4>

                        <b>Page:</b> {chunk['page']}

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    st.write(
                        chunk["text"]
                    )


                    st.divider()



        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )



# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.divider()

st.markdown(
    """
    <div style="
    text-align:center;
    color:#6b7280;
    font-size:14px;
    ">

    ⚖️ Legal AI Assistant<br>

    Built using Streamlit • FAISS • Groq • RAG

    </div>
    """,
    unsafe_allow_html=True
)