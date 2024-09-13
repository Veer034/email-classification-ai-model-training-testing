import sys
import subprocess
import pkg_resources
import io
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
import numpy as np  # Correctly importing NumPy
import evaluate
from google.cloud import storage
from sklearn.model_selection import train_test_split

# Check for GPU availability
if not torch.cuda.is_available():
    print("GPU is not available. This script requires a GPU to run.")
    sys.exit(1)

# Print available GPU
print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))

# Install required packages
required_packages = ['google-cloud-storage', 'scikit-learn', 'evaluate', 'transformers', 'torch', 'pandas', 'datasets']

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        pkg_resources.require(package)
    except pkg_resources.DistributionNotFound:
        print(f"{package} not found. Installing...")
        install(package)

def download_blob_to_dataframe(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    csv_data = blob.download_as_text()
    return pd.read_csv(io.StringIO(csv_data))

# Define your GCS bucket name and file name
bucket_name = 'email_classification_convonest'
file_name = 'refined_spam_and_ham_combined.csv'

# Load the data from GCS
df = download_blob_to_dataframe(bucket_name, file_name)

# Keep only 'email' and 'type' columns, and convert 'type' to numeric
df = df[['email', 'type']]
df['type'] = df['type'].map({'ham': 0, 'spam': 1})

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split dataset into train and test using scikit-learn with stratification
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['type'], random_state=42)

# Convert the split DataFrames back to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def preprocess_function(examples):
    inputs = tokenizer(examples['email'], padding='max_length', truncation=True, max_length=512)
    inputs['labels'] = examples['type']
    return inputs

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Set up the device (we know it's CUDA at this point)
device = torch.device("cuda")
model.to(device)

print(f"Using device: {device}")

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'gs://{bucket_name}/spam_ham_model_checkpoints',
    num_train_epochs=3,
    per_device_train_batch_size=16,  # Reduced batch size for memory
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=3,
    dataloader_pin_memory=True,
    fp16=True,  # Enable mixed precision training for speed and memory optimization
)

# Define the metric
metric = evaluate.load("accuracy")

# Updated compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Convert logits to PyTorch tensor if they are in NumPy format
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits)
    
    # Apply torch.argmax to get the predicted class
    predictions = torch.argmax(logits, axis=-1)
    
    # Compute accuracy
    return metric.compute(predictions=predictions, references=labels)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Clear cache before training
torch.cuda.empty_cache()

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the final model to GCS
final_model_path = f'gs://{bucket_name}/final_spam_ham_model'
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final model saved at {final_model_path}")

# Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
