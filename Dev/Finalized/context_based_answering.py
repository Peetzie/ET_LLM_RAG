from transformers import pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForTokenClassification,
)

import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Load variables
load_dotenv()
# change dir root (one above)
access_key = os.getenv("HUGGING_FACE")
root_dir = os.getcwd()
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")


# Assuming you have the `access_key` for authentication with Hugging Face API

## NER
ner_tokenizer = AutoTokenizer.from_pretrained(
    "dslim/bert-base-NER", cache_dir=model_dir
)
ner_model = AutoModelForTokenClassification.from_pretrained(
    "dslim/bert-base-NER", cache_dir=model_dir
).to("cuda")

nlp = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer)
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


## Get context

from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())


results = []
# Iterate through each person and location
for item in tqdm(persons + locations, desc=f"Querying Wikidata"):
    # Run the command and append the output to the results list
    result = wikidata.run(item)
    results.append(result)
print(results)
# Load your chosen model
model_name = "HuggingFaceH4/zephyr-7b-alpha"
question_tokenizer = AutoTokenizer.from_pretrained(
    model_name, token=access_key, cache_dir=model_dir
)
question_model = AutoModelForCausalLM.from_pretrained(
    model_name, token=access_key, cache_dir=model_dir
).to(device)

# Define the text generation pipeline
text_generation = pipeline(
    "text-generation",
    model=question_model,
    tokenizer=question_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)

# Define your prompt with the context
context = str(results)
query = example

prompt = f"""
You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers

{context}

{query}
"""

# Generate response
response = text_generation(prompt, max_length=1024, temperature=0.7)[0][
    "generated_text"
]

# Print or process the response as needed
print(response)
