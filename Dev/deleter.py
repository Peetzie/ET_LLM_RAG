from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np

# Load variables
load_dotenv()
# Load Hugging Face access key from environment variables
access_key = os.getenv("HUGGING_FACE")
# Define directories
root_dir = os.path.abspath("/work3/s174159/ET_LLM_RAG/")
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")


# Get the input argument
num_articles_to_keep = int(input("Enter the number of articles to keep: "))

# Get the list of articles in the directory
articles_list = os.listdir(articles_dir)

# Remove the article "Søren Pape Poulsen" from the list
articles_list.remove("Søren Pape Poulsen.txt")

# Shuffle the articles list
np.random.shuffle(articles_list)

# Delete articles until the desired number is reached
for article in articles_list[num_articles_to_keep:]:
    article_path = Path(articles_dir, article)
    os.remove(article_path)
