import pandas as pd
import numpy as np
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import torch
from torch import nn
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
OUTPUT_DIR = "./saved_security_model"
NUM_EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 2e-5

# --- CUSTOM TRAINER FOR CLASS WEIGHTS ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if hasattr(self, 'class_weights'):
            weight_tensor = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
        
        # Ensure labels are integers
        try:
            df['label'] = df['label'].astype(int)
        except ValueError:
            raise ValueError("Labels column contains non-integer values.")

        unique_labels = set(df['label'].unique())
        valid_labels = {0, 1}
        if not unique_labels.issubset(valid_labels):
            raise ValueError(f"CSV contains invalid labels: {unique_labels - valid_labels}")
            
        dataset = Dataset.from_pandas(df)
        
        # --- CRITICAL ADJUSTMENT: STRATIFIED SPLIT ---
        # With a 9:1 imbalance, a random split might result in a test set 
        # with almost zero minority class examples. Stratify fixes this.
        dataset = dataset.class_encode_column("label")
        return dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
        
    except Exception as e:
        print(f"[!] Critical Error loading data: {e}")
        exit(1)

def compute_metrics(eval_pred):
    """
    Calculates Precision, Recall, and F1.
    Accuracy is removed/deprioritized because it is misleading on skewed data.
    """
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")
    
    return {
        "accuracy": accuracy['accuracy'],
        "f1": f1['f1'],
        "precision": precision['precision'],
        "recall": recall['recall']
    }

def train():
    print("="*40)
    print("    SECURITY MODEL TRAINING PIPELINE")
    print("="*40)

    # 1. Load Data
    dataset = load_and_preprocess_data("final_dataset.csv")
    print(f"[*] Train Size: {len(dataset['train'])} | Test Size: {len(dataset['test'])}")

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    print("[*] Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("[*] Formatting dataset columns...")
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 3. Calculate Class Weights
    # Since Label 1 is majority, Label 0 will get a much higher weight (~9.0)
    # and Label 1 will get a lower weight (~0.5).
    print("[*] Calculating Class Weights...")
    train_labels = np.array(tokenized_datasets["train"]["labels"])
    
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[*] Calculated Weights (0 vs 1): {class_weights_tensor}")

    # 4. Model Setup
    id2label = {0: "benign", 1: "jailbreak/malicious"}
    label2id = {"benign": 0, "jailbreak/malicious": 1}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    # 5. Trainer Setup
    training_args = TrainingArguments(
        output_dir="./training_checkpoints", 
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1", # Optimization goal changed from loss to F1
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    trainer.class_weights = class_weights_tensor

    # 6. Execute Training
    print("[*] Starting training loop...")
    trainer.train()

    # 7. Save Final Artifacts
    print(f"[*] Saving final model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[*] Done.")

if __name__ == "__main__":
    train()