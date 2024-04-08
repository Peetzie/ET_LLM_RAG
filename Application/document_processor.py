from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

import os
from pathlib import Path
from tqdm import tqdm

root_dir = os.path.abspath(os.getcwd())
vectorstore_dir = Path(root_dir, "chroma")


class DocumentProcessor:
    def __init__(
        self, model_name="BAAI/bge-base-en-v1.5", access_key=os.getenv("HUGGING_FACE")
    ):
        self.model_name = model_name
        self.file_path = Path(root_dir, "Articles")
        self.huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key=access_key, model_name="BAAI/bge-base-en-v1.5"
        )
        self.collection_name = "RAG"
        self.persistent_client = chromadb.PersistentClient(path=str(vectorstore_dir))
        self.collection = self.persistent_client.get_or_create_collection(
            self.collection_name, embedding_function=self.huggingface_ef
        )
        self.access_key = access_key
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

    def get_embeddings_data(self, documents):
        filenames = [
            file_name
            for file_name in os.listdir(self.file_path)
            if file_name.endswith(".txt")
        ]
        self.collection.add(documents=documents, ids=filenames)

    def add_new_articles(self, new_articles):
        articles = []
        for title, file_name in new_articles:
            if file_name.endswith(".txt"):
                content = self.load_text_document(file_name)
                articles.append(content)
        filenames = [file_name for (title, file_name) in new_articles]
        self.collection.add(documents=articles, ids=filenames)

    def load_all_docs_to_vectorstore(self):
        articles = self.load_multiple_docs()
        filenames = [
            file_name for file_name in self.file_path if file_name.endswith(".txt")
        ]
        self.collection.add(documents=articles, ids=filenames)

    def load_new_docs_to_vectorstore(self, new_articles):
        articles = []
        filenames = [file_name for (title, file_name) in new_articles]
        for title, file_name in new_articles:
            if file_name.endswith(".txt"):
                content = self.load_text_document(file_name)
                articles.append(content)
        self.collection.add(documents=articles, ids=filenames)

    def get_retriever(self):
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=self.access_key, model_name=self.model_name
        )

        langchain_chroma = Chroma(
            client=self.persistent_client,
            collection_name=self.collection_name,
            embedding_function=embeddings,
        )
        retriever = langchain_chroma.as_retriever(
            search_type="mmr", search_kwargs={"k": 2}
        )
        return retriever

    def get_docs(self, query):
        results = self.collection(query_texts=[query], n_results=5)
        return results
