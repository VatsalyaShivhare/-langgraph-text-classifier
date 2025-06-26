import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from dataset_builder.dataset_builder import IMDBDataset
from tokenizer_setup import get_tokenizer
from data_loader import load_imdb_dataset
import torch

# Step 1: Load the IMDb dataset
texts, labels = load_imdb_dataset("aclImdb", split="train", limit=1000)

# Step 2: Load tokenizer and create tokenized dataset
tokenizer = get_tokenizer()
dataset = IMDBDataset(texts, labels, tokenizer)

# Step 3: Load DistilBERT base model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Step 4: Correct LoRA Configuration for DistilBERT
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],  # ✅ valid modules for DistilBERT
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)

# Step 5: Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Step 6: Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    save_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",  # disable wandb
)

# Step 7: Setup Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# Step 8: Save the model and tokenizer
model.save_pretrained("model/fine_tuned_model")
tokenizer.save_pretrained("model/fine_tuned_model")

