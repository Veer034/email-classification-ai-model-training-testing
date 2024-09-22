import os
import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification
import pandas as pd
from google.cloud import storage
import io

def load_model_and_tokenizer(model_path, tokenizer_path):
    if not os.path.isdir(model_path):
        raise ValueError(f"The path {model_path} is not a valid directory.")
    
    model = DebertaV2ForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
    return model, tokenizer

def classify_email(model, tokenizer, email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = round(probabilities[0][predicted_class].item(), 2)
    categories = ['spam', 'query', 'complaint', 'suggestion']
    return categories[predicted_class], confidence

def download_data_from_gcs(bucket_name, file_name):
    """ Download CSV file from Google Cloud Storage """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()
    df = pd.read_csv(io.StringIO(data))
    return df[['email', 'type']]

def test_model(model_path, tokenizer_path, bucket_name, file_name):
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    test_data = download_data_from_gcs(bucket_name, file_name)

    print("\nTesting DeBERTa Model:")
    print("-" * 100)

    correct_predictions = 0
    total_predictions = len(test_data)
    count = 0

    for index, row in test_data.iterrows():
        email, true_label = row['email'], row['type']
        prediction, confidence = classify_email(model, tokenizer, email)
        is_correct = prediction == true_label

        if is_correct:
            correct_predictions += 1
        
        count += 1
        # Print summary after every 100 tests
        if count % 100 == 0 or count == total_predictions:
            print("-" * 100)
            print(f"After {count} tests, Correct Predictions: {correct_predictions}")
    
    print("-" * 100)
    accuracy = correct_predictions / total_predictions
    print(f"Final Accuracy after testing {total_predictions} emails: {accuracy:.2%}")


# Configuration
bucket_name = 'email_classification_convonest'
file_name = 'combined-data/refined_spam_and_non_spam_combined.csv'
path = './deberta_complaint_included_same_spam_ham_hardcoded'

# Run the test
print("Starting model testing...")
test_model(path, path, bucket_name, file_name)
print("\nTesting completed.")
