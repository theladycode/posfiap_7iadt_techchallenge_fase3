# Responsável por carregar documentos (PDF, TXT, DOCX no futuro)
from langchain_community.document_loaders import TextLoader
import os


def load_documents():
    docs = []
    folder_path = "data/protocols"

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)

            try:
                loader = TextLoader(
                    file_path,
                    encoding="utf-8"
                )
                docs.extend(loader.load())

            except UnicodeDecodeError:
                loader = TextLoader(
                    file_path,
                    encoding="latin-1"
                )
                docs.extend(loader.load())

    return docs