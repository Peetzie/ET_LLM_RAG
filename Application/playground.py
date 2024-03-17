from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch

## Load variables
load_dotenv()
access_key = os.getenv("HUGGING_FACE")
root_dir = os.path.abspath(os.getcwd())
model_dir = Path(root_dir, "models")

if torch.cuda.is_available():
    print("CUDA is available.")
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        cache_dir=model_dir,
        token=access_key,
    )
    print("Model loaded.")
    model.to("cuda:0")
    print("Model moved to GPU.")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=model_dir, token=access_key
    )
    tokenizer.use_default_system_prompt = False
    print("Tokenizer loaded.")


def chat_with_llama(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to("cuda")
    print("Input tokens:", input_ids)
    output = model.generate(
        input_ids, max_length=256, num_beams=4, no_repeat_ngram_size=2
    )
    print("Output tokens:", output)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


prompt = "Hi, explain me the concept of quantum mechanics."
print("Prompt:", prompt)
response = chat_with_llama(prompt=prompt)
print("Llama2-7b:", response)
