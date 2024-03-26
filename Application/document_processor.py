from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from pathlib import Path
from tqdm import tqdm

root_dir = os.path.abspath(os.getcwd())


class DocumentProcessor:
    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        self.model_name = model_name
        self.file_path = Path(root_dir, "Articles")
        print(f"Using default model name {model_name}")

    # Load text document from file system
    def load_text_document(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def load_multiple_docs(self):
        articles = []
        file_names = [
            file_name
            for file_name in os.listdir(self.file_path)
            if file_name.endswith(".txt")
        ]
        for file_name in tqdm(file_names, desc="Loading files", unit="file"):
            if file_name.endswith(".txt"):
                file_path = os.path.join(self.file_path, file_name)
                content = self.load_text_document(file_path)
                articles.append(content)
        return articles

    def chunkify(self, articles):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
        chunking = [text_splitter.split_text(article) for article in articles]

        # Define a class to represent documents
        class Document:
            def __init__(self, page_content, metadata=None):
                self.page_content = page_content
                self.metadata = metadata

        # Convert dictionaries into objects with 'page_content' attribute
        documents = [
            Document(chunk, metadata=None) for chunks in chunking for chunk in chunks
        ]

        return documents

    def get_embeddings_data(self, documents):
        access_key = os.getenv("HUGGING_FACE")
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=access_key, model_name="BAAI/bge-base-en-v1.5"
        )
        vectorstore = Chroma.from_documents(documents, embeddings)
        return vectorstore
