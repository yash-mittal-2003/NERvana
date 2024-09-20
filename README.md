# NERvana

This repository contains code for fine-tuning the `IndicBERT` model on custom datasets for Named Entity Recognition (NER) tasks. The model can be fine-tuned for multiple Indian languages such as Hindi, Malayalam, and others. You can use any dataset for token classification to recognize entities like Persons (PER), Locations (LOC), Organizations (ORG), etc.

## Project Overview

- **Model**: [IndicBERT](https://huggingface.co/ai4bharat/indic-bert) pre-trained on various Indian languages.
- **Task**: Named Entity Recognition (NER).
- **Fine-Tuning**: Token classification to identify named entities in the input text.

## Setup

### Prerequisites

Make sure you have the following Python packages installed:

```bash
pip install transformers datasets torch seqeval scikit-learn
```

### Dataset

You can load any NER dataset in a format supported by the [Hugging Face Datasets library](https://huggingface.co/docs/datasets/), such as [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam).

Example:

```python
from datasets import load_dataset

# Example for loading the Naamapadam dataset
dataset = load_dataset("ai4bharat/naamapadam", 'hi')
```

### Fine-Tuning

1. Preprocess your dataset by tokenizing the input text and aligning it with labels.
2. Set up training parameters using `TrainingArguments` and use the `Trainer` class to handle the training process.

Example for training:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

training_args = TrainingArguments(
    output_dir="/path/to/output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Model Inference

After training, you can load the fine-tuned model and use it to predict named entities in new sentences.

```python
# Load fine-tuned model
model = AutoModelForTokenClassification.from_pretrained("/path/to/output")
tokenizer = AutoTokenizer.from_pretrained("/path/to/output")

# Predict
inputs = tokenizer("Sample sentence", return_tensors="pt")
outputs = model(**inputs)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.


