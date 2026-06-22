import streamlit as st

from rag.embeddings import EmbeddingClient
from rag.generate import LLMClient
from rag.pipeline import RAGPipeline
from rag.vector_store import VectorStore

import tempfile
from pathlib import Path

from rag.ingest import (
    extract_text_from_pdf,
    make_overlapping_chunks,
    build_index_in_memory,
)

from rag.vector_store import (
    VectorStore,
    StoredChunk,
)

INDEX_DIR = "data/index/faiss_index"

st.set_page_config(
    page_title="FinDoc QA",
    page_icon="📄",
    layout="centered",
)

st.title("📊 FinDoc QA")

st.markdown(
     """
Financial Document Question Answering powered by Retrieval Augmented Generation (RAG).

Ask questions about:
- Annual Reports
- RBI Circulars
- Loan Agreements
- Earnings Reports

Answers are grounded in the document with page-level citations and similarity scores.
"""
)

@st.cache_resource
def load_pipeline():
    embedding_client = EmbeddingClient.from_env()
    vector_store = VectorStore.from_disk(
        INDEX_DIR
    )
    llm_client = LLMClient.from_env()
    pipeline = RAGPipeline(
        embedding_client=embedding_client,
        vector_store=vector_store,
        llm_client=llm_client,
    )
    return pipeline

@st.cache_resource
def build_uploaded_pipeline(file_bytes, file_name):

    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        delete=False
    ) as tmp:

        tmp.write(file_bytes)
        temp_path = Path(tmp.name)

    raw_chunks = extract_text_from_pdf(
        temp_path
    )

    chunks = make_overlapping_chunks(
        raw_chunks
    )

    embedding_client = (
        EmbeddingClient.from_env()
    )

    index, metadata = (
        build_index_in_memory(
            chunks,
            embedding_client,
        )
    )

    stored_chunks = [
        StoredChunk(**m)
        for m in metadata
    ]

    temp_vector_store = VectorStore(
        index=index,
        chunks=stored_chunks,
    )
    temp_llm = LLMClient.from_env()

    return RAGPipeline(
        embedding_client=embedding_client,
        vector_store=temp_vector_store,
        llm_client=temp_llm,
    )

pipeline = load_pipeline()
active_pipeline = pipeline


with st.sidebar:
    st.title("📊 FinDoc QA")
    st.markdown(
        """
FinDoc QA powered by:

- MiniLM Embeddings
- FAISS Vector Search
- Llama 3.1 via Groq

Answers are grounded in retrieved document chunks.
"""
    )
    st.divider()
    st.markdown("### Documents")

    selected_docs = []
    if st.checkbox(
        "Infosys FY2025 Annual Report",
        value=True
    ):
        selected_docs.append("infosys-ar-25.pdf")

    if st.checkbox(
        "RBI Digital Lending Circular",
        value=True
    ):
        selected_docs.append("rbi_digital_lending.pdf")

    if st.checkbox(
        "Loan Agreement",
        value=True
    ):
        selected_docs.append("loan_agreement.pdf")

    if selected_docs:
        st.success(
            f"Searching across {len(selected_docs)} document(s)"
        )
    else:
        st.warning(
            "Select at least one document"
        )

    st.divider()
    st.markdown("### Upload PDF")

    uploaded_file = st.file_uploader(
        "Upload your own financial document",
        type=["pdf"]
    )

    if uploaded_file:
        with st.spinner(
            "Parsing PDF and building vector index..."
        ):
            active_pipeline = (
                build_uploaded_pipeline(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                )
            )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )
        st.info(
            "Searching uploaded document only."
        )

EXAMPLE_QUESTIONS = {
    "infosys-ar-25.pdf": [
        "What was the revenue in FY2025?",
        "What are the company's risk factors?",
        "Summarize the business performance.",
    ],
    "rbi_digital_lending.pdf": [
        "What disclosures are required?",
        "What are the digital lending regulations?",
        "What grievance redressal mechanisms are required?",
    ],
    "loan_agreement.pdf": [
        "What is the interest rate?",
        "What are the repayment terms?",
        "What are the default conditions?",
    ],
}

st.markdown("### Example Questions")

example_questions = []
for doc in selected_docs:
    example_questions.extend(
        EXAMPLE_QUESTIONS.get(doc, [])
    )

# remove duplicates while preserving order
example_questions = list(
    dict.fromkeys(example_questions)
)

cols = st.columns(3)

for i, q in enumerate(
    example_questions[:3]
):
    if cols[i].button(q):
        st.session_state["question"] = q

question = st.text_input(
    "Ask a question",
    value=st.session_state.get("question", "")
)


if st.button("Ask", type="primary"):

    if question.strip():
        with st.spinner("Searching document..."):
            result = active_pipeline.query(
                question=question,
                selected_docs=selected_docs
                if not uploaded_file
                else None
            )

        st.markdown("---")
        st.markdown("## Answer")
        st.info(result["answer"])
        col1, col2 = st.columns(2)
        with col1:
            confidence_pct = int(result["confidence"] * 100)
            st.metric(
                "Confidence",
                f"{confidence_pct}%"
        )

        with col2:
            st.metric(
                "Sources Used",
                len(result["sources"])
            )

        st.markdown("---")
        st.markdown("## Sources")

        for source in result["sources"]:

            title = (
                f"{source['document']}"
                f" | Page {source['page']}"
                f" | Similarity {source.get('score',0):.3f}"
            )

            with st.expander(title):
                preview = source["chunk"][:500]
                if len(source["chunk"]) > 500:
                    preview += "..."
                st.markdown(preview)

st.markdown("---")
st.caption(
    "FinDoc QA • Built with FAISS, MiniLM and Llama 3.1 via Groq"
)