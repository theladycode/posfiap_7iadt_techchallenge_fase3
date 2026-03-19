from app.rag.retriever import retrieve_context
from typing import Tuple, List


def get_protocol_context(question: str) -> Tuple[str, List[str]]:
    context, sources = retrieve_context(question)
    return context, sources