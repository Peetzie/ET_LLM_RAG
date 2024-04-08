import sys
from yaspin import yaspin
import LLMS
import document_processor
import wikipedia_downloader
import torch
from subprocess import call
import wikidata
import helper_functions


def initialize_processor(model_name="BAAI/bge-base-en-v1.5"):
    processor = document_processor.DocumentProcessor(model_name=model_name)
    return processor


def initialize_chatbot(choice, processor, run_locally):
    if choice == "1":
        retriever = processor.get_retriever()
        with yaspin(text="Initializing Zephyr-7b...", color="blue") as spinner:
            chatbot = LLMS.Zephyr(
                retriever=retriever,
                run_locally=run_locally,
            )
            spinner.text = "Zephyr initialized"
            spinner.ok("✔")
            return chatbot
    else:
        print("Invalid choice.")
        return None


def chat():
    print("Welcome to the Chat Application!")
    print("Choose a model to use:")
    print("1. RAGchat zephyr [RAG]")
    choice = input("Enter your choice (1/): ")

    # Ask for additional options
    print("Additional options:")
    print("1. Use Wikidata for context")
    print("2. Use QA/RAG (Wikipedia Articles) for context")
    print("3. Use both for context")
    print("4. None (No additional context)")
    additional_choice = input("Enter your choice (1/2/3/4): ")

    use_wikidata_for_context = False
    use_qa_rag = False
    use_both = False
    neither = False

    if additional_choice == "1":
        use_wikidata_for_context = True
    elif additional_choice == "2":
        use_qa_rag = True
    elif additional_choice == "3":
        use_both = True
    elif additional_choice == "4":
        neither = True

    # print values for each of the above
    print(f"Use Wikidata for context: {use_wikidata_for_context}")
    print(f"Use QA/RAG for context: {use_qa_rag}")
    print(f"Use both for context: {use_both}")
    print(f"Use neither for context: {neither}")
    run_locally_choise = input(
        "Run locally (Takes longer - otherwise HuggingFaceInference is used)? (yes/no): "
    )
    if run_locally_choise.lower() == "yes":
        run_locally = True
    else:
        run_locally = False

    processor = initialize_processor()

    chatbot = initialize_chatbot(
        (str(choice)),
        processor,
        run_locally,
    )

    if use_wikidata_for_context or use_qa_rag or use_both:
        print("Additional context enabled.")
        with yaspin(text="Initializing NER model...", color="yellow") as spinner:
            NER = LLMS.NER(model_name="dslim/bert-base-NER")
            spinner.text = "NER model initialized"
            spinner.ok("✔")

    if chatbot is None:
        return
    docs = None  # Bug fix
    print("You can start chatting. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        with yaspin(text="Thinking...", color="green") as spinner:
            if use_both:
                persons, locations = NER.extract_entities(user_input)
                wikidata_results = wikidata.get_knowledge(persons, locations)
                qids = helper_functions.find_qids(wikidata_results=wikidata_results)

                new_articles = (
                    wikipedia_downloader.download_wikipedia_article_with_progress(
                        "flarsen", num_words_to_save=5000, qids=qids
                    )
                )
                print(new_articles)
                if new_articles:
                    processor.add_new_articles(new_articles)
                if run_locally:
                    docs = processor.get_docs(user_input)
                response = chatbot.generate_response_with_retrieval_context(
                    user_input, context=wikidata_results, docs=docs
                )
            elif use_wikidata_for_context:
                persons, locations = NER.extract_entities(user_input)
                wikidata_results = wikidata.get_knowledge(persons, locations)

                response = chatbot.generate_response_context(
                    user_input, context=wikidata_results
                )
            elif use_qa_rag:
                persons, locations = NER.extract_entities(user_input)
                wikidata_results = wikidata.get_knowledge(persons, locations)
                qids = helper_functions.find_qids(wikidata_results=wikidata_results)

                new_articles = (
                    wikipedia_downloader.download_wikipedia_article_with_progress(
                        "flarsen", num_words_to_save=5000, qids=qids
                    )
                )
                print(qids)
                print(new_articles)
                if new_articles:
                    processor.add_new_articles(new_articles)

                if run_locally:
                    docs = processor.get_docs(user_input)
                response = chatbot.generate_response_with_retrieval(
                    user_input, docs=docs
                )
            else:
                response = chatbot.generate_response(user_input)
            spinner.text = "Bot responded"
            spinner.ok("✔")
        print("Bot:", response)


def main():
    print("What would you like to do?")
    print("1. Chat with the Chatbot")
    print("2. Create a dataset")
    print("3. Delete the dataset")
    print("4. Check integrity")
    choice = input("Enter your choice (1/2/3/4): ")

    if choice == "1":
        chat()
    elif choice == "2":
        wikipedia_downloader.create_folder()
        # Ask for username
        user_agent = input("Enter your user_agent (name): ")
        # Ask for how many words to save articles with
        num_words = int(input("Enter the number of words to save articles with: "))
        wikipedia_downloader.download_wikipedia_article_with_progress(
            user_agent=user_agent, num_words_to_save=num_words
        )
        print("Dataset sucessfully downloaded.. Returning..")
        main()
    elif choice == "3":
        confirm = input("Are you sure you want to delete the dataset? (yes/no): ")
        if confirm.lower() == "yes":
            wikipedia_downloader.delete_dataset()
            print("Dataset successfully deleted.. Returning..")
            main()
        else:
            print("Dataset deletion canceled.")
            main()
    elif choice == "4":
        empty_files = wikipedia_downloader.check_non_empty()
        if empty_files:
            print("The following files are empty:")
            for file_name in empty_files:
                print(file_name)
        else:
            print("All .txt files in the folder are non-empty.")
    else:
        print("Invalid choice.")
        main()


if __name__ == "__main__":
    main()
