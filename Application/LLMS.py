# HuggingFace
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM


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


class Zephyr:
    def __init__(self, vectorstore) -> None:
        self.mname = "HuggingFaceH4/zephyr-7b-alpha"
        self.model_kwargs = {
            "temperature": 0.5,
            "max_new_tokens": 512,
            "max_length": 64,
        }
        self.vectorstore = vectorstore
        self.retiever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 5}
        )
        self.model = HuggingFaceHub(
            repo_id=self.mname,
            model_kwargs=self.model_kwargs,
            huggingfacehub_api_token=access_key,
        )

    def generate_response(self, utterance):
        prompt = f"""
            You are an AI Assistant that follows instructions extremely well.
            Please be truthful and give direct answers
            </s>

            {utterance}
            </s>
            """
        relevant_docs = self.retiever.get_relevant_documents(utterance)
        qa = RetrievalQA.from_chain_type(llm=self.model, retriever=self.retiever)
        response = qa(prompt)
        result = response["result"]
        return result
