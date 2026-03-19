# Responsável por recuperar trechos relevantes para a pergunta
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.vector_store import create_vector_store


vector_store = None


def get_retriever():
    global vector_store

    if vector_store is None:
        docs = load_documents()
        chunks = split_documents(docs)
        vector_store = create_vector_store(chunks)

    return vector_store.as_retriever()


def retrieve_context(question: str):
    retriever = get_retriever()

    docs = retriever.invoke(question)

    context = "\n".join([doc.page_content for doc in docs])
    sources = ["protocol_documents"]

    return context, sources