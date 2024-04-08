import re


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


def find_qids(wikidata_results):
    qids = []
    for result in wikidata_results:
        # Extract the QIDs from the result
        qids += re.findall(r"Q\d+", result)
        qids
    return qids
