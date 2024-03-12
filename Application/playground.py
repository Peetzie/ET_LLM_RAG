import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

## Load variables
load_dotenv()
access_key = os.getenv("HUGGING_FACE")
model_dir = os.getenv("MODEL_DIR")


tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-7b", cache_dir=model_dir, token=access_key
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b", cache_dir=model_dir, token=access_key, device_map="auto"
)


input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
print("Generating text")
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
