from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path


## Load variables
load_dotenv()
access_key = os.getenv("HUGGING_FACE")
root_dir = os.path.abspath(os.getcwd())
model_dir = Path(root_dir, "models")

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
            "google/gemma-7b-it", cache_dir=model_dir, token=access_key
        )

        if self.GPU:
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-7b-it",
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
            "google/gemma-2b-it", cache_dir=model_dir, token=access_key
        )

        if self.GPU:
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2b-it",
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
