import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# ==============================
# CONFIG
# ==============================
MODEL_NAME = "facebook/bart-base"
MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LR = 3e-5

DATA_PATH = "./data"

# ==============================
# CREATE TIMESTAMP RUN FOLDER
# ==============================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = f"./checkpoints/run_{timestamp}"

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(f"{RUN_DIR}/logs", exist_ok=True)
os.makedirs(f"{RUN_DIR}/predictions", exist_ok=True)
os.makedirs(f"{RUN_DIR}/final_model", exist_ok=True)

print(f"Run directory: {RUN_DIR}")

# ==============================
# LOAD PARQUET DATASET
# ==============================
dataset = load_dataset(
    "parquet",
    data_files={
        "train": f"{DATA_PATH}/train.parquet",
        "validation": f"{DATA_PATH}/valid.parquet",
        "test": f"{DATA_PATH}/test.parquet"
    }
)

# ==============================
# LOAD TOKENIZER + MODEL
# ==============================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# ==============================
# PREPROCESS FUNCTION
# ==============================
def preprocess(example):
    inputs = example["complex"]
    targets = example["simple"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# ==============================
# METRIC: BERTScore
# ==============================
bertscore = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    results = bertscore.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        lang="en"
    )

    return {
        "bertscore_f1": sum(results["f1"]) / len(results["f1"])
    }

# ==============================
# TRAINING ARGUMENTS
# ==============================
training_args = TrainingArguments(
    output_dir=RUN_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_dir=f"{RUN_DIR}/logs",
    logging_steps=100,

    fp16=torch.cuda.is_available(),

    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True
)

# ==============================
# DATA COLLATOR
# ==============================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ==============================
# TRAINER
# ==============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==============================
# TRAIN
# ==============================
trainer.train()

# ==============================
# SAVE FINAL MODEL
# ==============================
trainer.save_model(f"{RUN_DIR}/final_model")
tokenizer.save_pretrained(f"{RUN_DIR}/final_model")

print("Final model saved.")

# ==============================
# TEST PREDICTIONS
# ==============================
print("Generating test predictions...")

test_dataset = dataset["test"]

pred_file = f"{RUN_DIR}/predictions/test_predictions.txt"

with open(pred_file, "w", encoding="utf-8") as f:
    for item in test_dataset:
        source = item["complex"]
        reference = item["simple"]

        inputs = tokenizer(
            source,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_LENGTH
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_length=MAX_TARGET_LENGTH,
            num_beams=4
        )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        f.write("SOURCE:\n" + source + "\n")
        f.write("REFERENCE:\n" + reference + "\n")
        f.write("PREDICTED:\n" + prediction + "\n")
        f.write("="*80 + "\n")

print(f"Predictions saved at: {pred_file}")

# ==============================
# FINAL TEST METRIC
# ==============================
test_results = trainer.evaluate(tokenized_dataset["test"])
print("Test Results:", test_results)