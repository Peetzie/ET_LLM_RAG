from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
from tqdm import tqdm

wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())


def get_knowledge(persons, locations):
    results = []
    # Iterate through each person and location
    for item in tqdm(persons + locations, desc=f"Querying Wikidata"):
        # Run the command and append the output to the results list
        result = wikidata.run(item)
        results.append(result)
    return results
