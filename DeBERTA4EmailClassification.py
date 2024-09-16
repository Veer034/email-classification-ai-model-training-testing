import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import EarlyStoppingCallback, get_cosine_schedule_with_warmup
from google.cloud import storage
import pandas as pd
import io

# Configuration
bucket_name = 'email_classification_convonest'  # GCS bucket name where the data is stored
file_name = 'combined-data/refined_spam_and_non_spam_combined.csv'  # File path within the bucket

def load_data():
    """ Download CSV file from Google Cloud Storage """
    client = storage.Client()  # Create a GCS client
    bucket = client.bucket(bucket_name)  # Get the bucket object
    blob = bucket.blob(file_name)  # Get the blob (file) object
    data = blob.download_as_text()  # Download the file content as text
    df = pd.read_csv(io.StringIO(data))  # Read the CSV data into a pandas DataFrame
    return df['email'].tolist(), df['type'].tolist()  # Return the 'email' and 'type' columns as lists

def preprocess_function(examples):
    # Tokenize the 'email' text, truncate to max length, and pad to max length
    return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=512)

def compute_metrics(eval_pred):
    logits, labels = eval_pred  # Get the model's output logits and the true labels
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predicted class labels
    # Return the accuracy metric: (correct predictions / total predictions)
    return {"accuracy": (predictions == labels).astype(np.float32).mean().item()}

# Load the DeBERTa model and tokenizer
model_name = 'microsoft/deberta-v3-base'  # Specify the DeBERTa model name
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)  # Load the tokenizer
model = DebertaV2ForSequenceClassification.from_pretrained(model_name, num_labels=4)  # Load the model with 4 output labels

# Prepare the data for training
emails, types = load_data()  # Load the email text and labels
label_dict = {"spam": 0, "query": 1, "complaint": 2, "suggestion": 3}  # Mapping of label strings to numeric values
numeric_labels = [label_dict[label] for label in types]  # Convert labels to numeric values

# Split the data into train and evaluation sets
train_emails, eval_emails, train_labels, eval_labels = train_test_split(emails, numeric_labels, test_size=0.2, stratify=numeric_labels, random_state=42)

# Create separate datasets for train and evaluation
train_dataset = Dataset.from_dict({"email": train_emails, "label": train_labels})
eval_dataset = Dataset.from_dict({"email": eval_emails, "label": eval_labels})

# Tokenize the datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
# Define training arguments
training_args = TrainingArguments(
    output_dir="./deberta_trained_model",  # Directory to save the trained model
    
    num_train_epochs=10,  # Number of training epochs
    # - Current value: 10
    # - Minimum value: 1
    # - Maximum value: No fixed maximum, but too many epochs may lead to overfitting
    # - Explanation: The number of complete passes through the training dataset. Higher values allow the model to learn more, but too many epochs may cause overfitting. The optimal value depends on the dataset size and complexity.
    
    per_device_train_batch_size=8,  # Batch size per device during training
    # - Current value: 8
    # - Minimum value: 1
    # - Maximum value: Depends on available memory
    # - Explanation: The number of training samples processed on each device (GPU or CPU) in one forward pass. Larger batch sizes can speed up training but require more memory. The optimal value depends on the available hardware and model size.
    
    per_device_eval_batch_size=8,  # Batch size per device during evaluation
    # - Current value: 8
    # - Minimum value: 1
    # - Maximum value: Depends on available memory
    # - Explanation: Similar to per_device_train_batch_size, but for evaluation. Using a larger batch size can speed up evaluation but requires more memory.
    
    learning_rate=2e-5,  # Learning rate for the optimizer
    # - Current value: 2e-5
    # - Minimum value: Close to 0 (e.g., 1e-6)
    # - Maximum value: No fixed maximum, but too high may cause divergence
    # - Explanation: The step size at which the model's weights are updated during optimization. A smaller learning rate leads to slower but more stable learning, while a larger learning rate can speed up training but may cause the model to overshoot the optimal solution. The optimal value depends on the model architecture and dataset.
    
    weight_decay=0.01,  # Weight decay for regularization
    # - Current value: 0.01
    # - Minimum value: 0 (no weight decay)
    # - Maximum value: No fixed maximum, but too high may lead to underfitting
    # - Explanation: A regularization technique that adds a penalty term to the loss function, discouraging large weight values. This helps prevent overfitting. The optimal value depends on the model architecture and dataset.
    
    logging_dir='./logs',  # Directory to save training logs
    logging_steps=5,  # Log every 5 steps
    
    eval_strategy="steps",  # Evaluate every specified number of steps
    # - Other options: "no", "steps", "epoch"
    # - Explanation: Determines when to perform evaluation during training. "no" disables evaluation, "steps" evaluates every eval_steps, and "epoch" evaluates after each epoch. Evaluating more frequently can provide better insight into the training progress but increases the overall training time.
    
    eval_steps=20,  # Evaluate every 20 steps
    save_strategy="steps",  # Save model every specified number of steps
    save_steps=20,  # Save model every 20 steps
    
    load_best_model_at_end=True,  # Load the best model at the end of training
    # - Explanation: If set to True, the model with the best performance on the evaluation set during training will be loaded at the end. This ensures that the final model is the best-performing one.
    
    metric_for_best_model="accuracy",  # Use accuracy as the metric for selecting the best model
    greater_is_better=True,  # Higher accuracy is better
    
    fp16=torch.cuda.is_available(),  # Use 16-bit floating-point precision if GPU is available
    # - Explanation: 16-bit floating-point (FP16) precision can speed up training and reduce memory usage on compatible GPUs. If a GPU is available, FP16 will be used. Otherwise, 32-bit floating-point (FP32) precision will be used.
    
    warmup_steps=50,  # Number of warmup steps for learning rate
    # - Explanation: Warmup steps are used to gradually increase the learning rate from 0 to the specified value over a certain number of steps. This helps stabilize training in the beginning. The optimal value depends on the total number of training steps and the learning rate.
)

# Calculate the number of updates per epoch based on the batch size and dataset size
num_update_steps_per_epoch = len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch


# Create a learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate),
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=max_train_steps,
)
# - Other options: linear schedule, polynomial decay, constant schedule
# - Explanation: The learning rate scheduler adjusts the learning rate during training. The cosine schedule gradually decreases the learning rate following a cosine curve, which can help the model converge to a better solution. Other schedules, such as linear decay or polynomial decay, decrease the learning rate linearly or polynomially. The choice of scheduler depends on the model architecture, dataset, and experimental results.

# Initialize Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # - Explanation: Callbacks are used to add functionality to the training loop. The EarlyStoppingCallback monitors the model's performance on the evaluation set and stops training if the performance doesn't improve for a specified number of epochs (early_stopping_patience). This helps prevent overfitting and saves computational resources.
    
    optimizers=(torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate), lr_scheduler),
)

# Train the model
trainer.train()

# After training is complete
output_dir = "./deberta_cosine"

# Save the trained model
trainer.save_model(output_dir)

# Save the tokenizer
tokenizer.save_pretrained(output_dir)

print("Training completed. Model saved at ./deberta_cosine")
