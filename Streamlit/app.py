import streamlit as st
import os
import sys
# Path fix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Application')))
import LLMS as LLMS
import wikipedia_downloader as wikipedia_downloader


# Initialize session_state object
if "messages" not in st.session_state:
    st.session_state.messages = []

response = None
bot = None
first_run = True
st.title("ETLINK-RAG-LLM")
tab1, tab2 = st.tabs(["Wikidata", "ChatBot"])

# Create a session_state object
if st.session_state.get("Stop_button", False):
    st.session_state.disabled = False

with tab1:
    # Set the title
    st.header("Wikidata")
    clear_button = st.button("Clear Articles")
    st.markdown(
        "Based on source code from [Tweeki](https://github.com/ucinlp/tweeki), a dataset that contains entity linking from Twitter posts we can download relevant Wikipedia articles",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Select the length of articles - Note that articles will be cut off after this length"
    )
    user_agent = st.text_input("Enter User Agent Name", value="Your name")
    article_length = st.slider(
        "Select Article Length", min_value=20, max_value=10000, value=1000
    )
    download_button = st.button("Download Articles")
    stop_button = st.button(
        "Stop download",
        key="stop_button",
        disabled=st.session_state.get("disabled", False),
    )
    progress_text = "Operation in progress. Please wait."
    if clear_button:
        # Delete the "Articles" folder and its contents
        if os.path.exists("Articles"):
            import shutil

            shutil.rmtree("Articles")

    if download_button and not st.session_state.stop_button:
        qids_list = wikipedia_downloader.get_qids()

        wikipedia_downloader.create_folder()

        # Initialize progress bar
        progress_bar = st.progress(0, text=progress_text)

        # Calculate total number of articles
        total_articles = len(qids_list)

        # Initialize counter for downloaded articles
        downloaded_articles = 0

        for qid in qids_list:
            wikipedia_downloader.download_wikipedia_article(
                qid, user_agent, article_length
            )
            # Update progress
            downloaded_articles += 1
            progress = downloaded_articles / total_articles
            progress_bar.progress(
                progress,
                f"Operation in progress. Please wait -- Downloading article: {downloaded_articles}/{total_articles}",
            )

            # Check if the stop button is clicked
            if st.session_state.stop_button:
                st.error("Download stopped by user.")
                break
        st.balloons()  # Celebrate finished downloading.. :)

with tab2:
    st.markdown("Please select a model that you would like to utilize?")
    model_selection = st.selectbox("Select Model", ["Other", "Blenderbot"])

    if model_selection == "Blenderbot":
        if "bot" not in st.session_state:
            # Load the model if it's not already loaded
            with st.spinner("Loading model"):
                st.session_state.bot = LLMS.BlenderbotChat()
            bot = st.session_state.bot
            st.success("Model loaded")
        else:
            # Use the model from the session state
            bot = st.session_state.bot

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        input_text = st.text_input(
            label="You: ", placeholder="Your prompt here", key="input"
        )  # Change default prompt to empty string

        if input_text:
            output = bot.generate_response(input_text)
            st.session_state.past.append(input_text)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                with st.container():
                    st.chat_message("assistant").markdown(
                        st.session_state["generated"][i]
                    )
                    st.chat_message("user").markdown(st.session_state["past"][i])
