import streamlit as st

from app.main import get_service
from app.rag import ConfigurationError

st.set_page_config(page_title="Atlas RAG", page_icon="📚", layout="wide")
st.markdown("<style>.block-container{max-width:1180px;padding-top:2rem}.source{padding:12px;border:1px solid #3b4252;border-radius:10px;margin:8px 0}</style>", unsafe_allow_html=True)
try:
    service = get_service()
except ConfigurationError as error:
    st.error(str(error)); st.stop()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Atlas RAG")
    st.caption("Document intelligence workspace")
    uploads = st.file_uploader("Add PDF documents", type=["pdf"], accept_multiple_files=True)
    if st.button("Index documents", type="primary", use_container_width=True, disabled=not uploads):
        for upload in uploads:
            with st.spinner(f"Indexing {upload.name}..."):
                try:
                    result = service.ingest_pdf(upload.name, upload.getvalue())
                    st.success(f"{result['filename']}: {result['chunks_indexed']} chunks")
                except Exception as error:
                    st.error(f"{upload.name}: {error}")
    st.divider(); st.subheader("Search scope")
    documents = service.list_documents()
    selected = []
    for document in documents:
        if st.checkbox(f"{document['filename']} ({document['pages_indexed']} pages)", value=True, key=document["document_id"]):
            selected.append(document["document_id"])
    st.caption(f"{len(documents)} indexed document(s)")

st.title("Ask your documents")
st.caption("Answers are grounded in retrieved content and return page-level citations.")
if not documents:
    st.info("Upload PDFs from the sidebar to begin.")
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])
question = st.chat_input("Ask a question about the selected documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try: answer, sources = service.ask(question, selected or None)
            except Exception as error: st.error(str(error)); st.stop()
        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=True):
                for source in sources:
                    st.markdown(f"<div class='source'><b>[{source['citation']}] {source['filename']} - Page {source['page']}</b><br>{source['excerpt']}</div>", unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
