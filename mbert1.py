import pandas as pd
import torch
import numpy as np
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
import os

# Paths (Adjust these to your local system)
base_path = "C:/Users/Avanee/Research Multilingual Sentiment Analysis"
checkpoint_dir = os.path.join(base_path, "checkpoints")
log_dir = os.path.join(base_path, "logs")

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels=3)

# Load CSVs
bengali_df = pd.read_csv(os.path.join(base_path, "bengali.csv"))
hindi_df = pd.read_csv(os.path.join(base_path, "hindi sentiment analysis.csv"), names=['Sentence', 'Sentiment'])
punjabi_df = pd.read_csv(os.path.join(base_path, "cleaned-pa-train.csv"))

# Preprocess
punjabi_df.drop_duplicates(inplace=True)
hindi_df.drop_duplicates(inplace=True)
bengali_df.drop_duplicates(inplace=True)

punjabi_df.rename(columns={'sentiment': 'Sentiment', 'translated_text': 'Sentence'}, inplace=True)
bengali_df.rename(columns={'Data': 'Sentence', 'Label': 'Sentiment'}, inplace=True)

label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
for df in [hindi_df, punjabi_df, bengali_df]:
    df.columns = df.columns.str.lower().str.strip()
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower().map(label_map)

df = pd.concat([bengali_df, hindi_df, punjabi_df], ignore_index=True)
df = df[['sentence', 'sentiment']].dropna()
df = df[df['sentiment'].isin([0, 1, 2])].reset_index(drop=True)
df['sentiment'] = df['sentiment'].astype(int)

# Tokenize
dataset = Dataset.from_pandas(df)

def tokenize(batch):
    tokens = tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=128)
    tokens['labels'] = batch['sentiment']
    return tokens

tokenized_datasets = dataset.map(tokenize, batched=True)
train_test = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test['train']
test_dataset = train_test['test']

# Metric Function
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='weighted'),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted')
    }

# Callback
class SaveOnEpochEnd(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f" Epoch {state.epoch} completed. Model checkpoint saved at {args.output_dir}")
        return control

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/mbert_checkpoints",
    do_train=True,
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="/content/drive/MyDrive/mbert_logs",
    logging_steps=50,
    eval_strategy="epoch",        
    save_strategy="epoch",              
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[SaveOnEpochEnd()]
)

# Train
trainer.train()

# Final Evaluation
metrics = trainer.evaluate()
print("Final Evaluation Metrics:")
print(metrics)
