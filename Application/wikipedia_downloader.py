import requests
import wikipediaapi
from tqdm import tqdm
import json
import os
from pathlib import Path
root_dir = os.path.abspath(os.getcwd())

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
            filename = Path(root_dir,f"Articles/{title}.txt")
            with open(filename, "w", encoding="utf-8") as file:
                file.write(content)
        else:
            print("Failed to fetch article content.")
    else:
        print(f"No Wikipedia article found for QID: {qid}")


def get_qids():
    # Define the path to the JSONL file
    jsonl_file_path = Path(root_dir,"Tweeki_gold.jsonl")
    qids_list = []

    with open(jsonl_file_path, "r") as file:
        for line in file:
            data = json.loads(line)
            link_list = data.get("link", [])
            qids = [link.split("|")[1] for link in link_list]
            qids_list.extend(qids)
    return qids_list


def create_folder():
    # Create directory to save articles
    if not os.path.exists(Path(root_dir, "Articles")):
        os.makedirs(Path(root_dir, "Articles"))


# Function to delete all files within a folder
def delete_dataset():
    articles_folder = Path(root_dir, "Articles")
    try:
        for filename in os.listdir(articles_folder):
            filepath = os.path.join(articles_folder, filename)
            os.remove(filepath)
        print("Dataset deleted successfully.")
    except Exception as e:
        print(f"Error deleting dataset: {str(e)}")


def download_wikipedia_article_with_progress(user_agent, num_words_to_save):
    qids_list = get_qids()
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
                    filename = Path(root_dir,f"Articles/{title}.txt")
                    with open(filename, "w", encoding="utf-8") as file:
                        file.write(content)
                    pbar_total.update(
                        1
                    )  # Update the progress bar for each downloaded article
                else:
                    print(f"Failed to fetch content for article with QID: {qid}")
            else:
                print(f"No Wikipedia article found for QID: {qid}")
