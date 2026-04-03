import tensorflow as tf
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import AdamWeightDecay

# Configuration
model_checkpoint = 'Helsinki-NLP/opus-mt-en-hi'
max_input_length = 128
max_target_length = 128
source_lang = 'en'
target_lang = 'hi'
batch_size = 16
learning_rate = 2e-5
weight_decay = 0.01
num_train_epochs = 1

base_dir = os.path.dirname(os.path.dirname(__file__))
model_save_path = os.path.join(base_dir, "model")

print("Loading dataset...")
raw_datasets = load_dataset('cfilt/iitb-english-hindi')

print("Filtering dataset size for efficient training...")
raw_datasets["train"] = raw_datasets["train"].select(range(50000))
raw_datasets["validation"] = raw_datasets["validation"].select(range(2000))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples['translation']]
    targets = [ex[target_lang] for ex in examples['translation']]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=targets, 
        max_length=max_target_length, 
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True,
    load_from_cache_file=True
)

# CRITICAL FIX: Ensure tensors are correctly formatted for TensorFlow
tokenized_datasets.set_format(
    type="tensorflow",
    columns=["input_ids", "attention_mask", "labels"]
)

print("Loading model...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="tf")

print("Preparing TF datasets...")
train_dataset = model.prepare_tf_dataset(
    tokenized_datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator,
)

validation_dataset = model.prepare_tf_dataset(
    tokenized_datasets["validation"],
    batch_size=batch_size,
    shuffle=False,
    collate_fn=data_collator,
)

optimizer = AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=weight_decay)
model.compile(optimizer=optimizer)

print("Starting training...")
model.fit(train_dataset, validation_data=validation_dataset, epochs=num_train_epochs)

print(f"Saving model to {model_save_path}...")
os.makedirs(model_save_path, exist_ok=True)
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Training finished successfully.")
