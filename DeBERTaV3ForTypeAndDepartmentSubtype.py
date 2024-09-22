# Import necessary libraries
import torch  # PyTorch library for deep learning operations
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, TrainingArguments, Trainer  # Hugging Face transformers library components
from datasets import Dataset  # Hugging Face datasets library for easy dataset handling
import numpy as np  # NumPy library for numerical operations
from sklearn.preprocessing import LabelEncoder  # Scikit-learn's LabelEncoder for encoding categorical variables
from transformers import EarlyStoppingCallback  # Callback for early stopping during training
from google.cloud import storage  # Google Cloud Storage client library
import pandas as pd  # Pandas library for data manipulation and analysis
import io  # Input/output operations library
from collections import defaultdict  # Default dictionary for handling missing keys
import os  # Operating system interface
import pickle  # Python object serialization

# Configuration
bucket_name = 'email_classification_convonest'  # Name of the Google Cloud Storage bucket containing the data
folder_path = 'combined-data/'  # Path to the folder containing the CSV files in the bucket

# Check if CUDA (GPU) is available for faster computations
if torch.cuda.is_available():
    print("Using cuda")  # If CUDA is available, use GPU for computations
else:
    print("Using cpu")  # If CUDA is not available, use CPU for computations

# List of CSV files to process
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
]  # List of CSV files to be processed from the GCS bucket

# Function to load data from Google Cloud Storage
def load_data_from_gcs():
    client = storage.Client()  # Initialize Google Cloud Storage client
    bucket = client.bucket(bucket_name)  # Get the specified bucket
    
    all_data = []  # List to store DataFrames from each CSV file
    blobs = bucket.list_blobs(prefix=folder_path)  # List all blobs (files) in the specified folder
    
    # Iterate through blobs and process relevant CSV files
    for blob in blobs:
        if blob.name.split('/')[-1] in csv_files:  # Check if the file is in the list of CSV files to process
            print(f"Processing file: {blob.name}")  # Print the name of the file being processed
            data = blob.download_as_text()  # Download the file content as text
            df = pd.read_csv(io.StringIO(data))  # Convert the text to a pandas DataFrame
            all_data.append(df)  # Add the DataFrame to the list
        else:
            print(f"Skipping file: {blob.name}")  # Print the name of files being skipped
    
    # Raise an error if no specified CSV files were found
    if not all_data:
        raise ValueError("No specified CSV files were found in the GCP bucket.")
    
    # Combine all DataFrames into one
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows in combined dataset: {len(combined_df)}")  # Print the total number of rows in the combined dataset
    return combined_df  # Return the combined DataFrame

# Load and preprocess the data
df = load_data_from_gcs()  # Call the function to load data from GCS

# Create department-specific type encoders
# This dictionary comprehension creates a LabelEncoder for each unique department in the dataset
department_type_encoders = {dept: LabelEncoder().fit(df[df['department'] == dept]['type']) for dept in df['department'].unique()}

# Initialize department-specific subtype encoders using defaultdict
department_subtype_encoders = defaultdict(LabelEncoder)

# Custom DefaultDict class to create a nested dictionary structure
class DefaultDict(dict):
    def __missing__(self, key):
        self[key] = DefaultDict()  # If a key is missing, create a new DefaultDict for that key
        return self[key]

    def add(self, item):
        self[item] = set()  # Add method to create an empty set for a given item

# Initialize the hierarchical structure for department-subtype relationships
department_subtype_hierarchies = DefaultDict()

# First pass: collect all subtypes for each department and type
for _, row in df.iterrows():  # Iterate through each row in the DataFrame
    dept = row['department']
    type_ = row['type']
    subtype = row['sub_type']
    if subtype not in department_subtype_hierarchies[dept][type_]:
        department_subtype_hierarchies[dept][type_].add(subtype)  # Add subtype to the hierarchy if it's not already present

# Second pass: fit the encoders with all known subtypes
for dept, type_subtypes in department_subtype_hierarchies.items():
    all_subtypes = set()
    for subtypes in type_subtypes.values():
        all_subtypes.update(subtypes)  # Collect all unique subtypes for each department
    department_subtype_encoders[dept].fit(list(all_subtypes))  # Fit the LabelEncoder with all subtypes for each department

# Function to encode types
def encode_type(row):
    dept = row['department']
    type_ = row['type']
    encoder = department_type_encoders[dept]  # Get the appropriate encoder for the department
    return encoder.transform([type_])[0]  # Encode the type and return the encoded value

# Apply type encoding to the DataFrame
df['type_encoded'] = df.apply(encode_type, axis=1)  # Create a new column with encoded type values

# Function to encode subtypes
def encode_subtype(row):
    dept = row['department']
    subtype = row['sub_type']
    encoder = department_subtype_encoders[dept]  # Get the appropriate encoder for the department
    if subtype in encoder.classes_:
        return encoder.transform([subtype])[0]  # Encode the subtype if it's known
    else:
        print(f"Warning: Unseen subtype '{subtype}' in department '{dept}'. Assigning a new label.")
        new_classes = np.append(encoder.classes_, subtype)  # Add the new subtype to the encoder's classes
        encoder.classes_ = new_classes
        return len(encoder.classes_) - 1  # Return the index of the new subtype

# Apply subtype encoding to the DataFrame
df['subtype_encoded'] = df.apply(encode_subtype, axis=1)  # Create a new column with encoded subtype values

# Combine department and email into a single 'text' column
df['text'] = df['department'] + " [SEP] " + df['email']  # Create a new column combining department and email

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df[['text', 'department', 'type_encoded', 'subtype_encoded']])  # Create a Dataset object from the DataFrame

# Load DeBERTa-v3-base tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('microsoft/deberta-v3-base')  # Initialize the DeBERTa tokenizer

# Preprocessing function for tokenization
def preprocess_function(examples):
    inputs = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)  # Tokenize the text
    inputs['department'] = examples['department']  # Add department to the inputs
    inputs['type_labels'] = examples['type_encoded']  # Add type labels to the inputs
    inputs['subtype_labels'] = examples['subtype_encoded']  # Add subtype labels to the inputs
    return inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)  # Apply preprocessing to the entire dataset

# Split dataset into train and test using Hugging Face's split method
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)  # Split the dataset into 80% train and 20% test
train_dataset = split_dataset['train']  # Get the training dataset
eval_dataset = split_dataset['test']  # Get the evaluation dataset

# Custom DeBERTa model for type and subtype classification
class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2ForSequenceClassification):
    def __init__(self, config, department_type_encoders, department_subtype_encoders):
        super().__init__(config)  # Initialize the parent class
        self.department_type_encoders = department_type_encoders  # Store type encoders
        self.department_subtype_encoders = department_subtype_encoders  # Store subtype encoders
        max_types = max(len(encoder.classes_) for encoder in department_type_encoders.values())  # Get the maximum number of types
        max_subtypes = max(len(encoder.classes_) for encoder in department_subtype_encoders.values())  # Get the maximum number of subtypes
        
        # Create linear layers for type and subtype classification
        self.type_classifier = torch.nn.Linear(config.hidden_size, max_types)
        self.subtype_classifier = torch.nn.Linear(config.hidden_size, max_subtypes)

    def forward(self, input_ids=None, attention_mask=None, department=None, type_labels=None, subtype_labels=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)  # Get DeBERTa outputs
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)  # Pool the sequence output

        type_logits = self.type_classifier(pooled_output)  # Compute type logits
        subtype_logits = self.subtype_classifier(pooled_output)  # Compute subtype logits

        loss = None
        if type_labels is not None and subtype_labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()  # Define loss function
            type_loss = 0
            subtype_loss = 0
            for i, dept in enumerate(department):
                dept_type_encoder = self.department_type_encoders[dept]
                dept_subtype_encoder = self.department_subtype_encoders[dept]
                
                valid_type_count = len(dept_type_encoder.classes_)
                valid_subtype_count = len(dept_subtype_encoder.classes_)
                
                dept_type_logits = type_logits[i, :valid_type_count]
                dept_subtype_logits = subtype_logits[i, :valid_subtype_count]
                
                # Compute losses for types and subtypes
                type_loss += loss_fct(dept_type_logits.unsqueeze(0), type_labels[i].unsqueeze(0))
                subtype_loss += loss_fct(dept_subtype_logits.unsqueeze(0), subtype_labels[i].unsqueeze(0))
            
            # Average the losses
            type_loss /= len(department)
            subtype_loss /= len(department)
            loss = type_loss + subtype_loss

        return {'loss': loss, 'type_logits': type_logits, 'subtype_logits': subtype_logits}

# Load pre-trained DeBERTa-v3-base model
model = DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
    'microsoft/deberta-v3-base',
    department_type_encoders=department_type_encoders,
    department_subtype_encoders=department_subtype_encoders
)  # Initialize the custom model with pre-trained weights

# Define training arguments
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    # Directory where the model checkpoints will be saved during training.
    # Choose a path that has sufficient storage space, as multiple checkpoints may be saved.
    # Absolute or relative path can be used.

    num_train_epochs=3,
    # Number of times the entire training dataset will be processed.
    # Range: Typically 1-10, but can be higher for complex tasks or smaller datasets.
    # Higher values may lead to overfitting, while lower values might result in underfitting.
    # Monitor validation performance to determine the optimal number of epochs.

    per_device_train_batch_size=8,
    # Number of training examples used in one iteration on a single device (GPU/CPU).
    # Range: Typically 8-64, depending on model size and available memory.
    # Larger batch sizes can lead to faster training but may require more memory.
    # If you encounter out-of-memory errors, reduce this value.

    per_device_eval_batch_size=8,
    # Number of evaluation examples used in one iteration on a single device during validation.
    # Usually set to the same value as per_device_train_batch_size or higher, as no gradients are stored during evaluation.
    # Can be increased if you have memory available to speed up evaluation.

    warmup_steps=500,
    # Number of steps for the warmup phase of the learning rate scheduler.
    # During warmup, the learning rate increases linearly from 0 to the initial lr set in the optimizer.
    # Helps stabilize training in the early stages.
    # Typically set to 5-10% of the total number of training steps.

    weight_decay=0.01,
    # L2 penalty (regularization) applied to model parameters.
    # Range: Typically 0.0 to 0.1. 0.01 is a common starting point.
    # Helps prevent overfitting by discouraging large weights in the model.
    # Increase for stronger regularization, decrease if the model is underfitting.

    logging_dir='./logs',
    # Directory where training logs will be saved.
    # These logs can be used for monitoring training progress and debugging.
    # TensorBoard can be used to visualize these logs.

    logging_steps=10,
    # Interval (in steps) at which training logs are written.
    # Lower values provide more frequent updates but increase overhead.
    # Adjust based on the total number of training steps and desired granularity of logging.

    evaluation_strategy="steps",
    # Determines when evaluations are run during training.
    # Options: "no" (only evaluate at the end), "steps" (evaluate every eval_steps), "epoch" (evaluate at the end of each epoch).
    # "steps" allows for more frequent evaluation, useful for monitoring training progress.

    eval_steps=500,
    # Number of update steps between two evaluations if evaluation_strategy="steps".
    # Should be a multiple of logging_steps for consistency in logs.
    # Balance between frequent evaluation (for closer monitoring) and training speed.

    save_steps=1000,
    # Number of updates steps before two checkpoint saves.
    # Allows you to resume training from a saved checkpoint if interrupted.
    # Set lower for more frequent saves (safer) or higher to save disk space.

    load_best_model_at_end=True,
    # If True, the best model found during training (based on eval metrics) is loaded at the end.
    # Ensures that the final saved model is the best performing one, not necessarily the last one.
    # Requires evaluations to be run during training (evaluation_strategy != "no").

    # Additional arguments to consider:
    # learning_rate: Set the initial learning rate. Typically ranges from 1e-5 to 5e-5 for transformer models.
    # fp16: Set to True to use mixed precision training if supported by your hardware. Can speed up training.
    # gradient_accumulation_steps: Accumulate gradients over multiple steps. Useful for simulating larger batch sizes.
    # metric_for_best_model: Specify which metric to use for selecting the best model when load_best_model_at_end=True.
)

# Custom data collator to handle department-specific encodings
def custom_data_collator(features):
    batch = {}
    for key in features[0].keys():
        if key == 'department':
            batch[key] = [f[key] for f in features]  # Keep department as a list of strings
        elif key in ['type_labels', 'subtype_labels']:
            batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)  # Convert labels to tensors
        else:
            batch[key] = torch.tensor([f[key] for f in features])  # Convert other features to tensors
    return batch

# Function to compute evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    type_logits, all_subtype_logits = logits
    type_labels, subtype_labels = labels
    
    type_preds = np.argmax(type_logits, axis=1)  # Get predicted types
    type_accuracy = (type_preds == type_labels).mean()  # Compute type accuracy
    
    subtype_preds = np.argmax(all_subtype_logits, axis=1)  # Get predicted subtypes
    subtype_accuracy = (subtype_preds == subtype_labels).mean()  # Compute subtype accuracy
    
    return {
        'type_accuracy': type_accuracy,
        'subtype_accuracy': subtype_accuracy,
        'overall_accuracy': (type_accuracy + subtype_accuracy) / 2  # Compute overall accuracy
    }

# Initialize trainer
trainer = Trainer(
    model=model,  # The instantiated model to be trained
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=eval_dataset,  # Evaluation dataset
    compute_metrics=compute_metrics,  # Function to compute metrics
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping callback
    data_collator=custom_data_collator  # Custom data collator
)

# Start training
trainer.train()

# Save the final model
trainer.save_model('./final_model')  # Save the trained model to the specified directory

# Save the tokenizer
tokenizer.save_pretrained('./final_model')  # Save the tokenizer to the same directory as the model

# Create a directory for saving encoders
encoder_dir = './final_model/encoders'
os.makedirs(encoder_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Save department type encoders
with open(os.path.join(encoder_dir, 'department_type_encoders.pkl'), 'wb') as file:
    pickle.dump(department_type_encoders, file)  # Serialize and save the department type encoders

# Save department subtype encoders
with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'wb') as file:
    pickle.dump(department_subtype_encoders, file)  # Serialize and save the department subtype encoders

# Save department subtype hierarchies
with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'wb') as file:
    pickle.dump(department_subtype_hierarchies, file)  # Serialize and save the department subtype hierarchies

# Save the model configuration
model_config = {
    'max_types': max(len(encoder.classes_) for encoder in department_type_encoders.values()),  # Maximum number of types across all departments
    'max_subtypes': max(len(encoder.classes_) for encoder in department_subtype_encoders.values())  # Maximum number of subtypes across all departments
}

# Save the model configuration
with open(os.path.join(encoder_dir, 'model_config.pkl'), 'wb') as file:
    pickle.dump(model_config, file)  # Serialize and save the model configuration

# Function to predict email type and subtype
def predict_email_type_subtype(email, department, model, tokenizer, subtype_threshold=0.3):
    input_text = f"{department} [SEP] {email}"  # Combine department and email text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)  # Tokenize the input
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the same device as the model
    inputs['department'] = [department]  # Add department to the inputs
    
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)  # Get model outputs
        type_logits = outputs['type_logits']
        subtype_logits = outputs['subtype_logits']
        
        dept_type_encoder = model.department_type_encoders[department]  # Get type encoder for the department
        dept_subtype_encoder = model.department_subtype_encoders[department]  # Get subtype encoder for the department
        
        valid_type_count = len(dept_type_encoder.classes_)  # Number of valid types for the department
        valid_subtype_count = len(dept_subtype_encoder.classes_)  # Number of valid subtypes for the department
        
        dept_type_logits = type_logits[0, :valid_type_count]  # Get logits for valid types
        dept_subtype_logits = subtype_logits[0, :valid_subtype_count]  # Get logits for valid subtypes
        
        type_probs = torch.nn.functional.softmax(dept_type_logits, dim=0)  # Convert type logits to probabilities
        type_pred = torch.argmax(type_probs).item()  # Get the predicted type index
        predicted_type = dept_type_encoder.inverse_transform([type_pred])[0]  # Convert type index to label
        
        subtype_probs = torch.nn.functional.softmax(dept_subtype_logits, dim=0)  # Convert subtype logits to probabilities
        max_subtype_prob = torch.max(subtype_probs).item()  # Get the highest subtype probability
        subtype_pred = torch.argmax(subtype_probs).item()  # Get the predicted subtype index
        
        if max_subtype_prob >= subtype_threshold:  # Check if the subtype probability meets the threshold
            predicted_subtype = dept_subtype_encoder.inverse_transform([subtype_pred])[0]  # Convert subtype index to label
            valid_subtypes = department_subtype_hierarchies[department][predicted_type]  # Get valid subtypes for the predicted type
            if predicted_subtype not in valid_subtypes:  # Check if the predicted subtype is valid for the predicted type
                valid_subtype_indices = [dept_subtype_encoder.transform([subtype])[0] for subtype in valid_subtypes]  # Get indices of valid subtypes
                valid_subtype_probs = subtype_probs[valid_subtype_indices]  # Get probabilities of valid subtypes
                max_valid_subtype_prob = torch.max(valid_subtype_probs).item()  # Get the highest probability among valid subtypes
                
                if max_valid_subtype_prob >= subtype_threshold:  # Check if the highest valid subtype probability meets the threshold
                    valid_subtype_pred = valid_subtype_indices[torch.argmax(valid_subtype_probs).item()]  # Get the index of the highest probability valid subtype
                    predicted_subtype = dept_subtype_encoder.inverse_transform([valid_subtype_pred])[0]  # Convert valid subtype index to label
                else:
                    predicted_subtype = None  # No valid subtype meets the threshold
        else:
            predicted_subtype = None  # Subtype probability doesn't meet the threshold
    
    return predicted_type, predicted_subtype  # Return the predicted type and subtype

# Example usage of the prediction function
email = "I have a question about my insurance claim..."  # Sample email text
department = "health_insurance"  # Sample department
predicted_type, predicted_subtype = predict_email_type_subtype(email, department, model, tokenizer, subtype_threshold=0.5)  # Make prediction
print(f"Predicted Type: {predicted_type}")  # Print the predicted type
print(f"Predicted Subtype: {predicted_subtype if predicted_subtype else 'No confident subtype prediction'}")  # Print the predicted subtype or a message if no confident prediction