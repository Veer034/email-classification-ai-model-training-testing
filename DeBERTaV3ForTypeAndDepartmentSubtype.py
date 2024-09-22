import torch
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    DebertaV2Config,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage
import pandas as pd
import io
import os
import pickle
import torch.nn as nn
from transformers.trainer_utils import get_last_checkpoint

# Configuration
bucket_name = 'email_classification_convonest'
folder_path = 'combined-data/'

if torch.cuda.is_available():
    print("Using cuda")
else:
    print("Using cpu")

csv_files = [
    'digital_healthcare_combined.csv',
    'ecommerce_combined.csv',
    'health_insurance_combined.csv',
    'hospitality_and_accommodation_combined.csv',
    'hospitals_and_healthcare_combined.csv',
    'it_combined.csv',
    'retail_combined.csv',
    'tourism_services_combined.csv',
    'travel_and_transportation_combined.csv'
]

def load_data_from_gcs():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    all_data = []
    blobs = bucket.list_blobs(prefix=folder_path)
    
    for blob in blobs:
        if blob.name.split('/')[-1] in csv_files:
            print(f"Processing file: {blob.name}")
            data = blob.download_as_text()
            df = pd.read_csv(io.StringIO(data))
            all_data.append(df)
        else:
            print(f"Skipping file: {blob.name}")
    
    if not all_data:
        raise ValueError("No specified CSV files were found in the GCP bucket.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows in combined dataset: {len(combined_df)}")
    return combined_df

# Load and preprocess the data
df = load_data_from_gcs()

# Fit type_encoder on actual data types
type_encoder = LabelEncoder()
type_encoder.fit(df['type'])

department_subtype_encoders = {}
department_subtype_hierarchies = {}

# First pass: collect all subtypes for each department and type
for _, row in df.iterrows():
    dept = row['department']
    type_ = row['type']
    subtype = row['sub_type']
    if dept not in department_subtype_hierarchies:
        department_subtype_hierarchies[dept] = {}
    if type_ not in department_subtype_hierarchies[dept]:
        department_subtype_hierarchies[dept][type_] = set()
    department_subtype_hierarchies[dept][type_].add(subtype)

# Second pass: fit the encoders with all known subtypes
for dept, type_subtypes in department_subtype_hierarchies.items():
    all_subtypes = set()
    for subtypes in type_subtypes.values():
        all_subtypes.update(subtypes)
    encoder = LabelEncoder()
    encoder.fit(list(all_subtypes))
    department_subtype_encoders[dept] = encoder

# Encode types and subtypes
df['type_encoded'] = type_encoder.transform(df['type'])

def encode_subtype(row):
    dept = row['department']
    subtype = row['sub_type']
    encoder = department_subtype_encoders[dept]
    return encoder.transform([subtype])[0]

df['subtype_encoded'] = df.apply(encode_subtype, axis=1)

# Combine department and email into a single 'text' column
df['text'] = df['department'] + " [SEP] " + df['email']

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'department', 'type_encoded', 'subtype_encoded']])

# Load DeBERTa-v3-base tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')

def preprocess_function(examples):
    inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)
    inputs['department'] = examples['department']
    inputs['type_labels'] = examples['type_encoded']
    inputs['subtype_labels'] = examples['subtype_encoded']
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Split dataset into train and test
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Custom model class
class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2PreTrainedModel):
    def __init__(self, config, num_types, department_subtype_encoders):
        super().__init__(config)
        self.num_types = num_types
        self.department_subtype_encoders = department_subtype_encoders
        self.max_subtypes = max(len(encoder.classes_) for encoder in department_subtype_encoders.values())

        # Load the base model
        self.deberta = DebertaV2Model(config)

        # Additional layers
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.type_classifier = nn.Linear(config.hidden_size, self.num_types)
        self.subtype_classifier = nn.Linear(config.hidden_size, self.max_subtypes)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, department=None, type_labels=None, subtype_labels=None, **kwargs):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output[:, 0])  # Use [CLS] token
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        type_logits = self.type_classifier(pooled_output)
        all_subtype_logits = self.subtype_classifier(pooled_output)

        loss = None
        if type_labels is not None and subtype_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            type_loss = loss_fct(type_logits, type_labels)

            # Compute subtype loss only for valid subtypes
            subtype_loss = 0
            for i, dept in enumerate(department):
                dept_encoder = self.department_subtype_encoders[dept]
                valid_subtype_count = len(dept_encoder.classes_)
                dept_subtype_logits = all_subtype_logits[i, :valid_subtype_count]
                dept_subtype_label = subtype_labels[i]
                subtype_loss += loss_fct(dept_subtype_logits.unsqueeze(0), dept_subtype_label.unsqueeze(0))
            subtype_loss /= len(department)

            loss = type_loss + subtype_loss

        return (loss, type_logits, all_subtype_logits)

    def save_pretrained(self, save_directory, **kwargs):
        # Save the configuration and model weights
        super().save_pretrained(save_directory, **kwargs)

        # Save department_subtype_encoders
        encoder_dir = os.path.join(save_directory, 'encoders')
        os.makedirs(encoder_dir, exist_ok=True)
        with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'wb') as f:
            pickle.dump(self.department_subtype_encoders, f)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base model and configuration
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        # Load department_subtype_encoders
        encoder_dir = os.path.join(pretrained_model_name_or_path, 'encoders')
        with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'rb') as f:
            department_subtype_encoders = pickle.load(f)

        # Set the attribute
        model.department_subtype_encoders = department_subtype_encoders
        model.max_subtypes = max(len(encoder.classes_) for encoder in department_subtype_encoders.values())

        # Adjust the subtype classifier
        model.subtype_classifier = nn.Linear(model.config.hidden_size, model.max_subtypes)

        return model


# Calculate num_types
num_types = len(type_encoder.classes_)

# Load pre-trained DeBERTa-v3-base model configuration
config = DebertaV2Config.from_pretrained('microsoft/deberta-v3-base')
config.num_labels = num_types
config.problem_type = "single_label_classification"

# Initialize the custom model
model = DebertaV3ForTypeAndDepartmentSubtype(config, num_types=num_types, department_subtype_encoders=department_subtype_encoders)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",  # Updated parameter name
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available()
)

def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key == 'department':
            batch[key] = [f[key] for f in features]
        elif key in ['type_labels', 'subtype_labels']:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        else:
            batch[key] = torch.tensor([f[key] for f in features])
    return batch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    type_logits, subtype_logits = logits  # Unpack the logits tuple
    type_labels, subtype_labels = labels  # Unpack the labels tuple

    # Convert tensors to numpy arrays if they aren't already
    type_logits = type_logits.detach().cpu().numpy() if isinstance(type_logits, torch.Tensor) else type_logits
    subtype_logits = subtype_logits.detach().cpu().numpy() if isinstance(subtype_logits, torch.Tensor) else subtype_logits
    type_labels = type_labels.detach().cpu().numpy() if isinstance(type_labels, torch.Tensor) else type_labels
    subtype_labels = subtype_labels.detach().cpu().numpy() if isinstance(subtype_labels, torch.Tensor) else subtype_labels

    # Compute accuracies
    type_preds = np.argmax(type_logits, axis=1)
    type_accuracy = (type_preds == type_labels).mean()

    subtype_preds = np.argmax(subtype_logits, axis=1)
    subtype_accuracy = (subtype_preds == subtype_labels).mean()

    return {
        'type_accuracy': type_accuracy,
        'subtype_accuracy': subtype_accuracy,
        'overall_accuracy': (type_accuracy + subtype_accuracy) / 2
    }

# Initialize the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=custom_data_collator
)

# Check if there's a checkpoint
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

# Start or resume training
if last_checkpoint is not None:
    print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    print("Starting training from scratch")
    trainer.train()

# Save the final model using custom save_pretrained method
model.save_pretrained('./final_model')

# Save the tokenizer
tokenizer.save_pretrained('./final_model')

# Save the type encoder and department_subtype_hierarchies
encoder_dir = os.path.join('./final_model', 'encoders')
os.makedirs(encoder_dir, exist_ok=True)
with open(os.path.join(encoder_dir, 'type_encoder.pkl'), 'wb') as f:
    pickle.dump(type_encoder, f)

with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'wb') as f:
    pickle.dump(department_subtype_hierarchies, f)
