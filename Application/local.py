import sys
from yaspin import yaspin
import LLMS
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
    else:
        print("Invalid choice.")
        return None


# Add the Gemma2bChat class similar to Gemma7bChat and BlenderbotChat


def ask_gpu():
    print("Do you want to use GPU for inference?")
    print("1. Yes")
    print("2. No")
    choice = input("Enter your choice (1/2): ")
    if choice == "1":
        return True
    elif choice == "2":
        return False
    else:
        print("Invalid choice. Defaulting to CPU.")
        return False


def chat():
    print("Welcome to the Chat Application!")
    print("Choose a model to use:")
    print("1. Blenderbot")
    print("2. Gemma-7b")
    print("3. Gemma-2b")
    choice = input("Enter your choice (1/2/3): ")

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


if __name__ == "__main__":
    print("What would you like to do?")
    print("1. Chat with the Chatbot")
    print("2. Create a dataset")
    print("3. Delete the dataset")
    choice = input("Enter your choice (1/2/3): ")

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
    elif choice == "3":
        confirm = input("Are you sure you want to delete the dataset? (yes/no): ")
        if confirm.lower() == "yes":
            wikipedia_downloader.delete_dataset()
        else:
            print("Dataset deletion canceled.")
    else:
        print("Invalid choice.")
