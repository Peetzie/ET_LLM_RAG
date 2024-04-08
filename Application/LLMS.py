# HuggingFace
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)
from transformers import pipeline

# Local
from pathlib import Path
import torch
import os
from dotenv import load_dotenv

# Langchain
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# progress
from tqdm import tqdm

import helper_functions

# Load variables
load_dotenv()
access_key = os.getenv("HUGGING_FACE")
root_dir = os.path.abspath(os.getcwd())
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")


class BlenderbotChat:
    def __init__(self):
        self.mname = "facebook/blenderbot-400M-distill"
        self.model = BlenderbotForConditionalGeneration.from_pretrained(
            self.mname, token=access_key, cache_dir=model_dir
        )
        self.tokenizer = BlenderbotTokenizer.from_pretrained(
            self.mname, token=access_key, cache_dir=model_dir
        )

    def to_string(self, gen_text):
        return gen_text[0].split("<s>")[1].split("</s>")[0]

    def generate_response(self, utterance):
        inputs = self.tokenizer([utterance], return_tensors="pt")
        reply_ids = self.model.generate(**inputs)
        return self.to_string(self.tokenizer.batch_decode(reply_ids))


class Gemma7bChat:
    def __init__(self, GPU=False):
        self.mname = "google/gemma-7b-it"
        self.GPU = GPU
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mname, cache_dir=model_dir, token=access_key
        )

        if self.GPU:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mname,
                device_map="auto",
                cache_dir=model_dir,
                token=access_key,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mname, cache_dir=model_dir, token=access_key
            )

    def generate_response(self, utterance):
        if self.GPU:
            input_ids = self.tokenizer(utterance, return_tensors="pt").to("cuda:0")
        else:
            input_ids = self.tokenizer(utterance, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0])


class Gemma2bChat:
    def __init__(self, GPU=False):
        self.mname = "google/gemma-2b-it"
        self.GPU = GPU
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mname, cache_dir=model_dir, token=access_key
        )

        if self.GPU:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mname,
                device_map="auto",
                cache_dir=model_dir,
                token=access_key,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.mname, cache_dir=model_dir, token=access_key
            )

    def generate_response(self, utterance):
        if self.GPU:
            input_ids = self.tokenizer(utterance, return_tensors="pt").to("cuda:0")
        else:
            input_ids = self.tokenizer(utterance, return_tensors="pt")
        outputs = self.model.generate(**input_ids, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0])


class Llama7BChat:
    """
    Is only GPU supported
    """

    def __init__(self):
        self.mname = "meta-llama/Llama-2-7b-chat-hf"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.mname,
            torch_dtype=torch.float16,
            cache_dir=model_dir,
            token=access_key,
        )
        self.model.to("cuda:0")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.mname, cache_dir=model_dir, token=access_key
        )

        self.tokenizer.use_default_system_prompt = False

    def generate_response(self, utterance):
        input_ids = self.tokenizer.encode(utterance, return_tensors="pt").to("cuda")
        output = self.model.generate(
            input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


class NER:
    def __init__(self, model_name="dslim/bert-base-NER") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.mname = model_name
        self.ner_tokenizer = AutoTokenizer.from_pretrained(
            self.mname, cache_dir=model_dir
        )
        self.ner_model = AutoModelForTokenClassification.from_pretrained(
            self.mname, cache_dir=model_dir
        ).to(self.device)

        self.nlp = pipeline("ner", model=self.ner_model, tokenizer=self.ner_tokenizer)

    def extract_entities(self, utterance):
        ner_results = self.nlp(utterance)
        persons, locations = helper_functions.extract_entities(ner_results)
        print(f"Persons: {persons}")
        print(f"Locations: {locations}")
        # Allow the user to correct the information in persons and locations via input
        persons_input = input(
            "Enter the correct persons seperated by ','. Leave blank for automatic found individuals: "
        ).split(",")
        locations_input = input(
            "Enter the correct locations seperated by ','Leave blank for automatic found locations:: "
        ).split(",")
        # Only overrite if the user entered anything
        if persons_input:
            persons = persons_input
        if locations_input:
            locations = locations_input

        return persons, locations


class Zephyr:
    def __init__(self, retriever, run_locally=False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mname = "HuggingFaceH4/zephyr-7b-alpha"
        self.model_kwargs = {
            "temperature": 0.7,
            "max_new_tokens": 1024,
            "max_length": 512,
        }
        self.retriever = retriever
        self.model = HuggingFaceHub(
            repo_id=self.mname,
            model_kwargs=self.model_kwargs,
            huggingfacehub_api_token=access_key,
        )
        self.use_local_model = run_locally

        if self.use_local_model:
            print(self.device)
            self.question_tokenizer = AutoTokenizer.from_pretrained(
                self.mname, cache_dir=model_dir, token=access_key
            )
            self.question_model = AutoModelForCausalLM.from_pretrained(
                self.mname, cache_dir=model_dir, token=access_key
            ).to(self.device)
            self.text_generation = pipeline(
                "text-generation",
                model=self.question_model,
                tokenizer=self.question_tokenizer,
                device=self.device,
            )

    def generate_response_with_retrieval(self, utterance, docs=None):
        if self.use_local_model:
            prompt = f"""
            You are an AI Assistant that follows instructions extremely well.
            Please be truthful and give direct answers
            </s>

            {utterance}
            From associated document retrieval {docs}
            </s>
            """
            response = self.text_generation(
                prompt, max_new_tokens=1024, temperature=0.7
            )[0]["generated_text"]
            return response
        else:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                </s>
                """
            qa = RetrievalQA.from_chain_type(llm=self.model, retriever=self.retriever)
            response = qa(prompt)
            return response["result"]

    def generate_response_context(self, utterance, context):
        if self.use_local_model:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                
                I have provided context: {context}
                </s>
                """
            response = self.text_generation(
                prompt, max_new_tokens=1024, temperature=0.7
            )[0]["generated_text"]
            return response
        else:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                
                I have provided context: {context}
                </s>
                """
            response = self.model(prompt)
            filtered_response = response.split(prompt)[-1]
            return filtered_response

    def generate_response(self, utterance):
        if self.use_local_model:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                </s>
                """
            response = self.text_generation(
                prompt, max_new_tokens=1024, temperature=0.7
            )[0]["generated_text"]
            return response
        else:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                </s>
                """
            response = self.model(prompt)
            filtered_response = response.split(prompt)[-1]
            return filtered_response

    def generate_response_with_retrieval_context(self, utterance, context, docs=None):
        if self.use_local_model:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                From associated document retrieval {docs}
                Further elaborated context: {context}
                </s>
                """
            response = self.text_generation(
                prompt, max_new_tokens=1024, temperature=0.7
            )[0]["generated_text"]
            return response
        else:
            prompt = f"""
                You are an AI Assistant that follows instructions extremely well.
                Please be truthful and give direct answers
                </s>

                {utterance}
                </s>
                """
            qa = RetrievalQA.from_chain_type(llm=self.model, retriever=self.retriever)
            context = str(context)
            response = qa(context + "\n" + prompt)
            return response["result"]
