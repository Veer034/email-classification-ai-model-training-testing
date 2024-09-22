import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from transformers import EarlyStoppingCallback
from google.cloud import storage
import pandas as pd
import io
from collections import defaultdict
import os
import pickle

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

# Create department-specific type encoders
department_type_encoders = {dept: LabelEncoder().fit(df[df['department'] == dept]['type']) for dept in df['department'].unique()}

department_subtype_encoders = defaultdict(LabelEncoder)

class DefaultDict(dict):
    def __missing__(self, key):
        self[key] = DefaultDict()
        return self[key]

    def add(self, item):
        self[item] = set()

department_subtype_hierarchies = DefaultDict()

# First pass: collect all subtypes for each department and type
for _, row in df.iterrows():
    dept = row['department']
    type_ = row['type']
    subtype = row['sub_type']
    if subtype not in department_subtype_hierarchies[dept][type_]:
        department_subtype_hierarchies[dept][type_].add(subtype)

# Second pass: fit the encoders with all known subtypes
for dept, type_subtypes in department_subtype_hierarchies.items():
    all_subtypes = set()
    for subtypes in type_subtypes.values():
        all_subtypes.update(subtypes)
    department_subtype_encoders[dept].fit(list(all_subtypes))

# Encode types and subtypes
def encode_type(row):
    dept = row['department']
    type_ = row['type']
    encoder = department_type_encoders[dept]
    return encoder.transform([type_])[0]

df['type_encoded'] = df.apply(encode_type, axis=1)

def encode_subtype(row):
    dept = row['department']
    subtype = row['sub_type']
    encoder = department_subtype_encoders[dept]
    if subtype in encoder.classes_:
        return encoder.transform([subtype])[0]
    else:
        print(f"Warning: Unseen subtype '{subtype}' in department '{dept}'. Assigning a new label.")
        new_classes = np.append(encoder.classes_, subtype)
        encoder.classes_ = new_classes
        return len(encoder.classes_) - 1

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

# Split dataset into train and test using Hugging Face's split method
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2ForSequenceClassification):
    def __init__(self, config, department_type_encoders, department_subtype_encoders):
        super().__init__(config)
        self.department_type_encoders = department_type_encoders
        self.department_subtype_encoders = department_subtype_encoders
        max_types = max(len(encoder.classes_) for encoder in department_type_encoders.values())
        max_subtypes = max(len(encoder.classes_) for encoder in department_subtype_encoders.values())
        
        self.type_classifier = torch.nn.Linear(config.hidden_size, max_types)
        self.subtype_classifier = torch.nn.Linear(config.hidden_size, max_subtypes)

    def forward(self, input_ids=None, attention_mask=None, department=None, type_labels=None, subtype_labels=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)

        type_logits = self.type_classifier(pooled_output)
        subtype_logits = self.subtype_classifier(pooled_output)

        loss = None
        if type_labels is not None and subtype_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            type_loss = 0
            subtype_loss = 0
            for i, dept in enumerate(department):
                dept_type_encoder = self.department_type_encoders[dept]
                dept_subtype_encoder = self.department_subtype_encoders[dept]
                
                valid_type_count = len(dept_type_encoder.classes_)
                valid_subtype_count = len(dept_subtype_encoder.classes_)
                
                dept_type_logits = type_logits[i, :valid_type_count]
                dept_subtype_logits = subtype_logits[i, :valid_subtype_count]
                
                type_loss += loss_fct(dept_type_logits.unsqueeze(0), type_labels[i].unsqueeze(0))
                subtype_loss += loss_fct(dept_subtype_logits.unsqueeze(0), subtype_labels[i].unsqueeze(0))
            
            type_loss /= len(department)
            subtype_loss /= len(department)
            loss = type_loss + subtype_loss

        return {'loss': loss, 'type_logits': type_logits, 'subtype_logits': subtype_logits}

# Load pre-trained DeBERTa-v3-base model
model = DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
    'microsoft/deberta-v3-base',
    department_type_encoders=department_type_encoders,
    department_subtype_encoders=department_subtype_encoders
)

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
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    load_best_model_at_end=True,
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
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    type_logits, all_subtype_logits = logits
    type_labels, subtype_labels = labels
    
    type_preds = np.argmax(type_logits, axis=1)
    type_accuracy = (type_preds == type_labels).mean()
    
    subtype_preds = np.argmax(all_subtype_logits, axis=1)
    subtype_accuracy = (subtype_preds == subtype_labels).mean()
    
    return {
        'type_accuracy': type_accuracy,
        'subtype_accuracy': subtype_accuracy,
        'overall_accuracy': (type_accuracy + subtype_accuracy) / 2
    }

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    data_collator=custom_data_collator
)

# Start training
trainer.train()

# Save the final model
trainer.save_model('./final_model')

# Save the tokenizer
tokenizer.save_pretrained('./final_model')

encoder_dir = './final_model/encoders'
os.makedirs(encoder_dir, exist_ok=True)

with open(os.path.join(encoder_dir, 'department_type_encoders.pkl'), 'wb') as file:
    pickle.dump(department_type_encoders, file)

with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'wb') as file:
    pickle.dump(department_subtype_encoders, file)

with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'wb') as file:
    pickle.dump(department_subtype_hierarchies, file)

# Save the model configuration
model_config = {
    'max_types': max(len(encoder.classes_) for encoder in department_type_encoders.values()),
    'max_subtypes': max(len(encoder.classes_) for encoder in department_subtype_encoders.values())
}

with open(os.path.join(encoder_dir, 'model_config.pkl'), 'wb') as file:
    pickle.dump(model_config, file)

def predict_email_type_subtype(email, department, model, tokenizer, subtype_threshold=0.3):
    input_text = f"{department} [SEP] {email}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs['department'] = [department]
    
    with torch.no_grad():
        outputs = model(**inputs)
        type_logits = outputs['type_logits']
        subtype_logits = outputs['subtype_logits']
        
        dept_type_encoder = model.department_type_encoders[department]
        dept_subtype_encoder = model.department_subtype_encoders[department]
        
        valid_type_count = len(dept_type_encoder.classes_)
        valid_subtype_count = len(dept_subtype_encoder.classes_)
        
        dept_type_logits = type_logits[0, :valid_type_count]
        dept_subtype_logits = subtype_logits[0, :valid_subtype_count]
        
        type_probs = torch.nn.functional.softmax(dept_type_logits, dim=0)
        type_pred = torch.argmax(type_probs).item()
        predicted_type = dept_type_encoder.inverse_transform([type_pred])[0]
        
        subtype_probs = torch.nn.functional.softmax(dept_subtype_logits, dim=0)
        max_subtype_prob = torch.max(subtype_probs).item()
        subtype_pred = torch.argmax(subtype_probs).item()
        
        if max_subtype_prob >= subtype_threshold:
            predicted_subtype = dept_subtype_encoder.inverse_transform([subtype_pred])[0]
            valid_subtypes = department_subtype_hierarchies[department][predicted_type]
            if predicted_subtype not in valid_subtypes:
                valid_subtype_indices = [dept_subtype_encoder.transform([subtype])[0] for subtype in valid_subtypes]
                valid_subtype_probs = subtype_probs[valid_subtype_indices]
                max_valid_subtype_prob = torch.max(valid_subtype_probs).item()
                
                if max_valid_subtype_prob >= subtype_threshold:
                    valid_subtype_pred = valid_subtype_indices[torch.argmax(valid_subtype_probs).item()]
                    predicted_subtype = dept_subtype_encoder.inverse_transform([valid_subtype_pred])[0]
                else:
                    predicted_subtype = None
        else:
            predicted_subtype = None
    
    return predicted_type, predicted_subtype

# Example usage
email = "I have a question about my insurance claim..."
department = "health_insurance"
predicted_type, predicted_subtype = predict_email_type_subtype(email, department, model, tokenizer, subtype_threshold=0.5)
print(f"Predicted Type: {predicted_type}")
print(f"Predicted Subtype: {predicted_subtype if predicted_subtype else 'No confident subtype prediction'}")