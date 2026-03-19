from langchain_openai import OpenAIEmbeddings
from app.core.settings import settings
from langchain_community.vectorstores import FAISS


def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store