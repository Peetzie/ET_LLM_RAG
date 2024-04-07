from dotenv import load_dotenv
import torch
import os
from pathlib import Path
import re
from tqdm import tqdm

import requests
import wikipediaapi

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline


from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from pathlib import Path

# Load variables
load_dotenv()
# change dir root (one above)
access_key = os.getenv("HUGGING_FACE")
root_dir = os.getcwd()
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER", cache_dir=model_dir)
model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", cache_dir=model_dir
).to("cuda")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "Give me a resume of Tom Holland's acting career"

ner_results = nlp(example)
print(ner_results)


def extract_entities(entities):
    """
    Extracts persons and locations from a list of entities recognized by an entity recognition system.

    Parameters:
    - entities (list of dicts): The output from an entity recognition system, where each dictionary
      contains details about the recognized entity, including its type (person or location).

    Returns:
    - tuple of two lists: (persons, locations) where each is a list of extracted entity names.
    """
    # Initialize lists to hold persons and locations
    persons = []
    locations = []

    # Temporary variables to construct full names and locations
    current_person = ""
    current_location = ""

    # Iterate over each entity to extract and construct full names and locations
    for entity in entities:
        if entity["entity"].startswith("B-PER"):
            # If there's a current person, append it to persons before starting a new one
            if current_person:
                persons.append(current_person.strip())
                current_person = ""
            current_person += entity["word"].lstrip("##")
        elif entity["entity"].startswith("I-PER") and current_person:
            # Handle token splitting for names correctly
            if entity["word"].startswith("##"):
                current_person += entity["word"].replace("##", "")
            else:
                current_person += " " + entity["word"]
        elif entity["entity"].startswith("B-ORG"):
            # Similarly, for locations
            if current_location:
                locations.append(current_location.strip())
                current_location = ""
            current_location += entity["word"].lstrip("##")
        elif entity["entity"].startswith("I-ORG") and current_location:
            if entity["word"].startswith("##"):
                current_location += entity["word"].replace("##", "")
            else:
                current_location += " " + entity["word"]

    # Append any remaining entities to their respective lists
    if current_person:
        persons.append(current_person.strip())
    if current_location:
        locations.append(current_location.strip())
    words_to_remove = ["like", "the", "and", "or", "but", "so", "for", "in", "at", "on"]
    cleaned_locations = []
    for location in locations:
        cleaned_location = " ".join(
            [word for word in location.split() if word.lower() not in words_to_remove]
        )
        cleaned_locations.append(cleaned_location)
    return persons, cleaned_locations


persons, locations = extract_entities(ner_results)

print(f"Persons: {persons}")
print(f"Locations: {locations}")


from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())


results = []
# Iterate through each person and location
for item in tqdm(persons + locations, desc=f"Querying Wikidata"):
    # Run the command and append the output to the results list
    result = wikidata.run(item)
    results.append(result)


results


qids = []

# Iterate through each result
for result in results:
    # Extract the QIDs from the result
    qids += re.findall(r"Q\d+", result)
qids


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


def download_wikipedia_article_with_progress(user_agent, num_words_to_save):
    qids_list = qids
    num_articles = len(qids_list)

    # Initialize a single progress bar for the overall process
    with tqdm(
        total=num_articles, desc="Downloading Wikipedia Articles", unit=" articles"
    ) as pbar_total:
        for qid in qids_list:
            title = get_wikipedia_title(qid, user_agent)
            if title:
                content = fetch_wikipedia_article(title, num_words_to_save, user_agent)
                if content:
                    # Save article content to disk
                    filename = Path(root_dir, f"Articles/{title}.txt")
                    with open(filename, "w", encoding="utf-8") as file:
                        file.write(content)
                    pbar_total.update(
                        1
                    )  # Update the progress bar for each downloaded article
                else:
                    print(f"Failed to fetch content for article with QID: {qid}")
            else:
                print(f"No Wikipedia article found for QID: {qid}")


length = 5000


download_wikipedia_article_with_progress("flarsen", length)


# Load text document from file system
def load_text_document(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content


# Define folder path
folder_path = "/work3/s174159/ET_LLM_RAG/Articles/"

# Encode articles in the folder
articles = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):
        file_path = os.path.join(folder_path, file_name)
        content = load_text_document(file_path)
        articles.append(content)

# Split text into chunks for each article
text_splitter = RecursiveCharacterTextSplitter(chunk_size=length, chunk_overlap=1)
chunking = [text_splitter.split_text(article) for article in articles]


# Define a class to represent documents
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata


# Convert dictionaries into objects with 'page_content' attribute
documents = [Document(chunk, metadata=None) for chunks in chunking for chunk in chunks]

# Embed chunks using the specified embedding model
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=access_key, model_name="BAAI/bge-base-en-v1.5"
)
vectorstore = Chroma.from_documents(documents, embeddings)

# Define retriever
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})


# Perform retrieval-based question answering task
query = example
docs_rel = retriever.get_relevant_documents(query)

# Generate response to a given query using augmented knowledge base
model = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 1024, "max_length": 512},
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


# Extract context from the provided information
response = qa(prompt)
print(response["result"])
