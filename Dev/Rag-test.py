from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from pathlib import Path
import os
import requests
import wikipediaapi


# Load text document from file system
def load_text_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


# Function to get Wikipedia page title from Wikidata QID
def get_wikipedia_title(qid, user_agent):
    headers = {"User-Agent": user_agent}
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}&props=sitelinks&format=json"
    response = requests.get(url, headers=headers)
    data = response.json()
    if "sitelinks" in data.get("entities", {}).get(qid, {}):
        sitelinks = data["entities"][qid]["sitelinks"]
        if "enwiki" in sitelinks:
            return sitelinks["enwiki"]["title"]
    return None


# Function to fetch Wikipedia article content based on page title
def fetch_wikipedia_article(title, num_words, user_agent):
    headers = {"User-Agent": user_agent}
    wiki_wiki = wikipediaapi.Wikipedia("en", headers=headers)
    page = wiki_wiki.page(title)
    if page.exists():
        article_content = page.text
        # Extract specified number of words
        words = article_content.split()[:num_words]
        return " ".join(words)
    return None


# Function to download Wikipedia articles
def download_wikipedia_article(qid, user_agent, num_words_to_save):
    title = get_wikipedia_title(qid, user_agent)
    if title:
        content = fetch_wikipedia_article(title, num_words_to_save, user_agent)
        if content:
            # Save article content to disk
            root_dir = Path(".")
            directory = root_dir / "Articles"
            os.makedirs(directory, exist_ok=True)
            filename = directory / f"{title}.txt"
            with open(filename, "w", encoding="utf-8") as file:
                file.write(content)
        else:
            print("Failed to fetch article content.")
    else:
        print(f"No Wikipedia article found for QID: {qid}")


# Define file path
file_path = "/work3/s174159/ET_LLM_RAG/Dev/Articles/Søren Pape Poulsen.txt"

# Load text document
content = load_text_document(file_path)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)
chunking = text_splitter.split_text(content)


# Define a class to represent documents
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata


# Convert dictionaries into objects with 'page_content' attribute
documents = [Document(chunk, metadata=None) for chunk in chunking]

# Embed chunks using the specified embedding model
access_key = os.getenv("HUGGING_FACE")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=access_key, model_name="BAAI/bge-base-en-v1.5"
)
vectorstore = Chroma.from_documents(documents, embeddings)

# Define retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

# Perform retrieval-based question answering task
query = "What political party was Søren Pape Poulsen associated with?"
docs_rel = retriever.get_relevant_documents(query)

# Augment knowledge base with additional information from Wikipedia
qid = "Q12004862"
user_agent = "flarsen"
num_words_to_save = 1000
download_wikipedia_article(qid, user_agent, num_words_to_save)

# Generate response to a given query using augmented knowledge base
model = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.5, "max_new_tokens": 512, "max_length": 64},
    huggingfacehub_api_token=access_key,
)
qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
prompt = f"""
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers
</s>

{query}
</s>
"""
response = qa(prompt)
print(response["result"])
