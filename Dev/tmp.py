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
from dotenv import load_dotenv

# Load variables
load_dotenv()
access_key = os.getenv("HUGGING_FACE")
root_dir = os.path.abspath(os.getcwd())
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")


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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=1)
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
query = "Can you give a list of supervised students of Lars Kai Hansen"
docs_rel = retriever.get_relevant_documents(query)

# Generate response to a given query using augmented knowledge base
model = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-alpha",
    model_kwargs={"temperature": 0.1, "max_new_tokens": 1024, "max_length": 512},
    huggingfacehub_api_token=access_key,
)
qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever)

# Extract context from the provided information
context = """Result Q20984804:\nLabel: Lars Kai Hansen\nDescription: researcher, machine learning, Technical University of Denmark\nAliases: L K Hansen, Lars K. Hansen, L.K. Hansen, L. K. HANSEN, Lars K Hansen, Hansen LK\ninstance of: human\noccupation: professor\nsex or gender: male\ndate of birth: 1957-09-16\nplace of birth: Samsø\neducated at: University of Copenhagen\nemployer: Technical University of Denmark\nfield of work: machine learning, neuroinformatics\nnotable work: Neural network ensembles, Generalizable patterns in neuroimaging: how many principal components?, Plurality and resemblance in fMRI data analysis, Bayesian Averaging is Well-Temperated, The quantitative evaluation of functional neuroimaging experiments: the NPAIRS data analysis framework, The quantitative evaluation of functional neuroimaging experiments: mutual information learning curves, Mean-field approaches to independent component analysis, Independent component analysis of functional MRI: what is signal and what is noise?, Parallel Factor Analysis as an exploratory tool for wavelet transformed event-related EEG, Linear State-Space Models for Blind Source Separation, Bayesian Model Comparison in Nonlinear BOLD fMRI Hemodynamics, A Cure for Variance Inflation in High Dimensional Kernel Principal Component Analysis\ndoctoral student: Peter Toft, Finn Årup Nielsen, Thomas Kolenda, Morten Mørup, Sune Lehmann, Trine Julie Abrahamsen, Lasse Lohilahti Mølgaard, Tue Herlau, Simon Kamronn, Andreas Trier Poulsen, Ling Feng, Kristoffer Hougaard Madsen, Joaquin Quiñonero Candela, Michael Riis Andersen, Martin Christian Axelsen, Tue Lehn-Schiøler, Ulrik Kjems, Claus Svarer, Toke Jansen Hansen, Laura Rieger, Bjarne Ørum Fruergaard, Kaare Brandt Petersen, Sofie Therese Hansen, Philip J. H. Jørgensen, Jeppe Nørregaard, Damian Konrad Kowalczyk\nstudent: Finn Årup Nielsen, Sune Lehmann, Jeppe Nørregaard\naffiliation: Cognitive Systems, Center for Integrated Molecular Brain Imaging, DABAI, Danish Data Science Academy, Pioneer Centre for Artificial Intelligence\n\nResult Q53557020:\nLabel: Variance inflation in high dimensional Support Vector Machines\nDescription: scholarly article by Trine Julie Abrahamsen & Lars Kai Hansen published December 2013 in Pattern Recognition Letters\ninstance of: scholarly article\npublication date: 2013-12\nauthor: Lars Kai Hansen, Trine Julie Abrahamsen"""
response = qa(context + "\n" + query)
print(response["result"])
