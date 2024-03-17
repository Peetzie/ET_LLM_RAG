import sys
from yaspin import yaspin
import LLMS
import document_processor
import wikipedia_downloader
import torch
from subprocess import call


def initialize_chatbot(choice):
    if choice == "1":
        with yaspin(text="Initializing Blenderbot...", color="cyan") as spinner:
            chatbot = LLMS.BlenderbotChat()
            spinner.text = "Blenderbot initialized"
            spinner.ok("✔")
            return chatbot
    elif choice == "2":
        GPU = ask_gpu()
        if GPU is not None:
            with yaspin(text="Initializing Gemma-7b...", color="magenta") as spinner:
                chatbot = LLMS.Gemma7bChat(GPU=GPU)
                spinner.text = "Gemma initialized"
                spinner.ok("✔")
                return chatbot
    elif choice == "3":
        GPU = ask_gpu()
        if GPU is not None:
            with yaspin(text="Initializing Gemma-2b...", color="yellow") as spinner:
                chatbot = LLMS.Gemma2bChat(GPU=GPU)
                spinner.text = "Gemma2b initialized"
                spinner.ok("✔")
                return chatbot
    elif choice == "4":
        with yaspin(text="Initializing Llama2-7b...", color="blue") as spinner:
            chatbot = LLMS.Llama7BChat(GPU=GPU)
            spinner.text = "Llama2 initialized"
            spinner.ok("✔")
            return chatbot
    elif choice == "5":

        vectorstore = obtain_vectorstore()
        with yaspin(text="Initializing Zephyr-7b...", color="blue") as spinner:
            chatbot = LLMS.Zephyr(vectorstore=vectorstore)
            spinner.text = "Zephyr initialized"
            spinner.ok("✔")
            return chatbot
    else:
        print("Invalid choice.")
        return None


def ask_gpu():
    print("Do you want to use GPU for inference?")
    print("1. Yes")
    print("2. No")
    choice = input("Enter your choice (1/2): ")
    if choice == "1":
        print("Using GPU for inference.")
        return True
    elif choice == "2":
        print("Using CPU for inference.")
        return False
    else:
        print("Invalid choice. Defaulting to CPU.")
        return False


def chat():
    print("Welcome to the Chat Application!")
    print("Choose a model to use:")
    print("1. Blenderbot [Only CPU]")
    print("2. Gemma-7b")
    print("3. Gemma-2b")
    print("4. Llama2-7b [Requires GPU]")
    print("5. RAGchat zephyr")
    choice = input("Enter your choice (1/2/3/4/5): ")

    chatbot = initialize_chatbot(choice)
    if chatbot is None:
        return

    print("You can start chatting. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break
        with yaspin(text="Thinking...", color="green") as spinner:
            response = chatbot.generate_response(user_input)
            spinner.text = "Bot responded"
            spinner.ok("✔")
        print("Bot:", response)


def obtain_vectorstore():
    processor = document_processor.DocumentProcessor()
    # Load the dataset
    articles = processor.load_multiple_docs()
    documents = processor.chunkify(articles)
    with yaspin(text="Generating embeddings...", color="blue") as spinner:
        vectorstore = processor.get_embeddings_data(documents)
        spinner.text = "Embeddings created"
        spinner.ok("✔")
    return vectorstore


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
