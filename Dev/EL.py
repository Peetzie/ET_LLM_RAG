from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path
import numpy as np

import evaluate

from huggingface_hub import login

from transformers import Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from transformers import TrainingArguments

# Load variables
load_dotenv()
# Load Hugging Face access key from environment variables
access_key = os.getenv("HUGGING_FACE")
# Define directories
root_dir = os.path.abspath("/work3/s174159/ET_LLM_RAG/")
model_dir = Path(root_dir, "models")
articles_dir = Path(root_dir, "Articles")

# Load WNUT dataset
wnut = load_dataset("wnut_17", cache_dir=model_dir)

# Get label list from the training set
label_list = wnut["train"].features["ner_tags"].feature.names

# List of tokenizer models to use
tokenizer_models = [
    "distilbert-base-uncased",
]

# Loop through each tokenizer model
for model_name in tokenizer_models:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)

    # Tokenization function to tokenize and align labels
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )

        labels = []

        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Tokenize the dataset using the current tokenizer
    tokenized_wnut = wnut.map(
        tokenize_and_align_labels,
        batched=True,
    )

    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Load evaluation metric
    seqeval = evaluate.load("seqeval", cache_dir=model_dir)

    # Function to compute metrics
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Define label mappings
    id2label = {
        0: "O",
        1: "B-corporation",
        2: "I-corporation",
        3: "B-creative-work",
        4: "I-creative-work",
        5: "B-group",
        6: "I-group",
        7: "B-location",
        8: "I-location",
        9: "B-person",
        10: "I-person",
        11: "B-product",
        12: "I-product",
    }
    label2id = {
        "O": 0,
        "B-corporation": 1,
        "I-corporation": 2,
        "B-creative-work": 3,
        "I-creative-work": 4,
        "B-group": 5,
        "I-group": 6,
        "B-location": 7,
        "I-location": 8,
        "B-person": 9,
        "I-person": 10,
        "B-product": 11,
        "I-product": 12,
    }

    # Load model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=13,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_dir,
    )

    # Login to Hugging Face Hub
    login(token=access_key)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/EntityLinking_{model_name}",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=["none"],
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_wnut["train"],
        eval_dataset=tokenized_wnut["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_args,
    )

    # Train the model
    trainer.train()
