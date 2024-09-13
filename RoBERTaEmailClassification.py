import sys
import subprocess
import pkg_resources

required_packages = ['scikit-learn', 'evaluate']

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Check and install required packages
for package in required_packages:
    try:
        pkg_resources.require(package)
    except pkg_resources.DistributionNotFound:
        print(f"{package} not found. Installing...")
        install(package)

import io
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, TrainingArguments, Trainer
import torch
from google.cloud import storage
import pandas as pd
import evaluate
from google.cloud import storage
import pandas as pd

def download_blob_to_dataframe(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    csv_data = blob.download_as_text()
    return pd.read_csv(io.StringIO(csv_data))

# Define your GCS bucket name and file names
bucket_name = 'email_classification_convonest'

# List of CSV files to be processed
csv_files = [
    'digital_healthcare_combined.csv',
    'ecommerce_combined.csv',
    'health_insurance_combined.csv',
    'hospitality_and_accommodation_combined.csv',
    'hospitals_and_healthcare_combined.csv',
    'it_combined.csv',
    'refined_spam_combined.csv',
    'retail_combined.csv',
    'tourism_services_combined.csv',
    'travel_and_transportation_combined.csv'
]

# Load all CSV files into a single DataFrame
df_list = []
for csv_file in csv_files:
    df = download_blob_to_dataframe(bucket_name, csv_file)
     # Add department column, 'spam' for the spam-related file
    if 'spam' in csv_file:
        df['department'] = 'spam'
    df_list.append(df)


# Combine all the data into one DataFrame
df = pd.concat(df_list, ignore_index=True)


# Combine department and email into a single 'text' column
df['text'] = df['department'] + " " + df['email']


# Handle spam as its own class
label_mapping = {"query": 0, "complaint": 1, "suggestion": 2, "spam": 3}
df['type'] = df['type'].map(label_mapping)

# Drop unnecessary columns
df = df[['text', 'type']]

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def preprocess_function(examples):
    inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    inputs['labels'] = examples['type']
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# Split dataset into train and test
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Ensure the tokenized datasets contain the necessary columns
print(train_dataset.column_names)
print(eval_dataset.column_names)


# Load pre-trained RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)  # 4 labels: query, complaint, suggestion, spam

# Move the model to the correct device (MPS for Mac M1 or CPU if MPS is not available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


training_args = TrainingArguments(
    output_dir='gs://email_classification_convonest/checkpoints',  # Store checkpoints in GCS
     
    evaluation_strategy='epoch',  # Evaluate the model after every epoch.
    # This will evaluate the model once at the end of each epoch.
    
    save_strategy='epoch',  # Save based on epochs
    # This will save the model once at the end of each epoch.

    learning_rate=2e-5,  # Learning rate controls how quickly or slowly the model learns. 
    # Lower values (e.g., 1e-5) slow down learning but may lead to more stable convergence. 
    # Higher values (e.g., 3e-5, 5e-5) can lead to faster training but may overshoot the optimal solution.
    
    per_device_train_batch_size=4,  # The number of samples processed at a time on each GPU (or CPU).
    # Lower batch size reduces VRAM usage, which is necessary for limited resources (e.g., Colab T4 GPU with 16 GB VRAM).
    # Larger batch sizes (e.g., 16 or 32) would train faster but require more memory. In Colab Pro, 4 is a safe value.
    
    per_device_eval_batch_size=4,  # Similar to train batch size, but for evaluation. A smaller batch size also helps with memory limits during evaluation.
    # If this is set too low, evaluation will take longer. Higher values (e.g., 8) might speed up evaluation if memory allows.
    
    gradient_accumulation_steps=4,  # Accumulates gradients over several batches before updating the model weights.
    # Here, the model accumulates gradients for 4 batches of size 4 before applying the optimizer step.
    # Effectively simulates a batch size of 16 (4 * 4), allowing us to simulate larger batches without exceeding memory limits.
    # Increasing this value will reduce memory usage but slow down training (since updates happen less frequently).
    
    num_train_epochs=3,  # The number of full passes through the dataset during training.
    # More epochs allow the model to learn better, but may risk overfitting if the number is too high.
    # Common values range from 3 to 5 for RoBERTa fine-tuning. Increasing epochs will also increase total training time.
    
    weight_decay=0.01,  # Weight decay helps prevent overfitting by applying a penalty on large weights.
    # A value of 0.01 is a standard setting to encourage generalization. 
    # Increasing it may lead to better generalization but at the risk of underfitting. Setting it to 0 removes weight decay.
    
    save_total_limit=3,  # The maximum number of checkpoints to keep. Older checkpoints are deleted after this limit is reached.
    # Keeping more checkpoints (e.g., 5) takes up more storage but gives you more options to roll back if needed.
    # A lower value (e.g., 1) would conserve space but give fewer options for restoring models.
    # We've set this to 3 to keep checkpoints for all epochs in our 3-epoch training.
    
    logging_dir='./logs',  # Directory to store logs for training. You can monitor training progress here.
    # Logs can help track the model's loss, accuracy, and other metrics over time.
    
    logging_steps=100,  # Log model performance metrics (like loss and accuracy) every 100 steps.
    # If this value is too high (e.g., 1000), you'll get fewer updates. Lowering it (e.g., 50) provides more frequent feedback but increases logging overhead.
    
    dataloader_pin_memory=True,  # This setting is related to data loading. 
    # When set to True, it speeds up data transfer to GPU. This is beneficial when using a GPU on a Google VM.
    # It allows faster data transfer between CPU and GPU memory, which can improve training speed.
    
    load_best_model_at_end=True  # After training, load the model with the best evaluation score (e.g., lowest loss or highest accuracy).
    # This is important if you're evaluating on the validation set and want to ensure that the best-performing model is used.
)

# Metrics: You can define custom metrics to evaluate your model's performance.
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred  # The model’s raw output (logits) and the true labels.
    
    # torch.argmax(logits, axis=-1) converts the logits into predicted class labels by selecting the highest score.
    predictions = torch.argmax(logits, axis=-1)
    
    # metric.compute compares the predictions to the true labels and computes the accuracy score.
    return metric.compute(predictions=predictions, references=labels)


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model and create checkpoints
trainer.train()

# Save the final model to Google Drive
# Save final model to GCS
final_model_path = 'gs://email_classification_convonest/final_model'
model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)
print(f"Final model saved at {final_model_path}")


