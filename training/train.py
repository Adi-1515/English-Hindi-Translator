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
val_size = min(2000, len(raw_datasets["validation"]))
raw_datasets["validation"] = raw_datasets["validation"].select(range(val_size))

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples['translation']]
    targets = [ex[target_lang] for ex in examples['translation']]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding="max_length",
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_length,
        truncation=True,
        padding="max_length",
    )

    # Fix: Ensure labels are strict Python integers and pre-replace pad tokens with -100.
    # This prevents PyArrow/HuggingFace from inferring them as floats (which causes
    # the 'Cannot convert [array([ ... ])] to EagerTensor of dtype int64' error later).
    # Since they are integers, prepare_tf_dataset and DataCollator will correctly handle them as int64.
    pad_token_id = tokenizer.pad_token_id
    model_inputs["labels"] = [
        [-100 if int(t) == pad_token_id else int(t) for t in label]
        for label in labels["input_ids"]
    ]

    return model_inputs

print("Tokenizing datasets...")
tokenized_datasets = raw_datasets.map(
    preprocess_function, 
    batched=True,
    load_from_cache_file=True
)

# Do NOT call set_format(type="tensorflow") here.
# That call silently casts integer label token IDs to float32 TF tensors,
# which causes: TypeError: Cannot convert ... to EagerTensor of dtype int64
# inside prepare_tf_dataset(). Instead, leave the dataset in its native
# Python/NumPy integer format — prepare_tf_dataset() + DataCollatorForSeq2Seq
# (with return_tensors="tf") handle the TF conversion correctly and preserve int64.

print("Loading model...")
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="np")

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
