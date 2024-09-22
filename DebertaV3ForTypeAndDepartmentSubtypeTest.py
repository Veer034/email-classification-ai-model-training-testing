import torch
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, DebertaV2Config
import pickle
import os
from collections import defaultdict

# Define DefaultDict class (needed for loading department_subtype_hierarchies)
class DefaultDict(dict):
    def __missing__(self, key):
        self[key] = DefaultDict()
        return self[key]

    def add(self, item):
        self[item] = set()

# Load model configuration
model_path = './final_model'
encoder_dir = os.path.join(model_path, 'encoders')
with open(os.path.join(encoder_dir, 'model_config.pkl'), 'rb') as file:
    model_config = pickle.load(file)

# Load the base configuration and update it
config = DebertaV2Config.from_pretrained(model_path)
config.max_types = model_config['max_types']
config.max_subtypes = model_config['max_subtypes']

class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2ForSequenceClassification):
    def __init__(self, config, department_type_encoders, department_subtype_encoders):
        super().__init__(config)
        self.department_type_encoders = department_type_encoders
        self.department_subtype_encoders = department_subtype_encoders
        self.type_classifier = torch.nn.Linear(config.hidden_size, config.max_types)
        self.subtype_classifier = torch.nn.Linear(config.hidden_size, config.max_subtypes)

    def forward(self, input_ids=None, attention_mask=None, department=None, type_labels=None, subtype_labels=None, **kwargs):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output)

        type_logits = self.type_classifier(pooled_output)
        subtype_logits = self.subtype_classifier(pooled_output)

        return {'type_logits': type_logits, 'subtype_logits': subtype_logits}

# Load encoders and hierarchies
with open(os.path.join(encoder_dir, 'department_type_encoders.pkl'), 'rb') as file:
    department_type_encoders = pickle.load(file)

with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'rb') as file:
    department_subtype_encoders = pickle.load(file)

with open(os.path.join(encoder_dir, 'department_subtype_hierarchies.pkl'), 'rb') as file:
    department_subtype_hierarchies = pickle.load(file)





# Load the model with the updated configuration
model = DebertaV3ForTypeAndDepartmentSubtype.from_pretrained(
    model_path,
    config=config,
    department_type_encoders=department_type_encoders,
    department_subtype_encoders=department_subtype_encoders
)

# Load the tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

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

# Test the model with some example emails
test_emails = [ ("I'm having trouble logging into my HealthTrack account. I've entered my username and password correctly, but keep getting an 'Invalid credentials' error. I've tried resetting my password twice, but the issue persists. This is preventing me from accessing my health data and using the app's features. I need to check my recent blood pressure readings for my doctor's appointment tomorrow. Please help me regain access to my account as soon as possible. If you need any additional information to verify my identity, I'm happy to provide it.", "digital_healthcare", "complaint", "account_login_complaint"),("Your latest app update has rendered HealthTrack unusable on my smartphone. It crashes every time I try to open the exercise tracking feature and frequently shuts down while I'm viewing my health dashboard. I've tried uninstalling and reinstalling, clearing cache, and even factory resetting my phone, but nothing works. This is severely impacting my ability to monitor my daily health goals and manage my chronic condition. I'm a premium subscriber and expect the app to function properly. Please provide a fix immediately or roll back to the previous stable version.", "digital_healthcare", "complaint", "app_malfunction_complaint"),

("I'm writing about an unexpected charge of $49.99 on my credit card from HealthTrack. I don't recall authorizing this transaction. My account shows I've been upgraded to a premium subscription, but I never requested this change. I've been using the free version and am satisfied with it. Please explain this charge and revert my account to the free tier immediately. I want a full refund of the $49.99 charged without my consent. In the future, please ensure that any account changes or charges are clearly communicated and require explicit approval from users.", "digital_healthcare", "complaint", "billing_complaint"),

("I'm trying to use HealthTrack on my new Huawei P40 Pro, but the app isn't compatible. Your website claims to support all Android devices running version 8.0 and above. My phone meets this requirement, yet I can't install the app from the app store. This is disappointing as I've been a loyal user on my previous device and rely on HealthTrack for managing my diabetes. Can you explain why it's not compatible? Are there plans to support Huawei devices in the future? I'd appreciate a timeline for when I can expect to use HealthTrack on my new phone.", "digital_healthcare", "complaint", "compatibility_complaint"),

("I'm concerned about the privacy of my health data in HealthTrack. I've noticed that the app requests access to my phone's microphone and camera, which seems unnecessary for its core functions. Additionally, I can't find clear information about how my data is encrypted or who has access to it. The recent news about data breaches in healthcare apps has made me wary. Can you provide detailed information about your data protection measures? I'm considering deleting my account if I can't get assurance about the security of my sensitive health information.", "digital_healthcare", "complaint", "data_privacy_complaint"),

("The latest HealthTrack update removed the custom meal planning feature I relied on for my specialized diet. This was a key reason I chose your app over competitors. The new generic meal suggestions don't account for my food allergies and dietary restrictions. This change has made the app significantly less useful for me. I've been a premium subscriber for over a year, and removing features without warning is frustrating. Please bring back the custom meal planning feature or provide an alternative solution that meets the needs of users with specific dietary requirements.", "digital_healthcare", "complaint", "feature_complaint"),

("I've discovered a serious error in my health records on HealthTrack. The app shows I have a penicillin allergy, which is incorrect. I've never been allergic to penicillin. This misinformation could be dangerous if shared with healthcare providers in an emergency. I've tried to edit this information in the app, but the allergy section appears to be locked. Please remove this incorrect allergy information from my profile immediately. Also, explain how this error occurred and what steps you're taking to prevent such critical mistakes in users' health data in the future.", "digital_healthcare", "complaint", "incorrect_health_information_complaint"),

("I'm extremely dissatisfied with HealthTrack's customer support. I've been trying to get help with syncing issues for over a week. I've sent three emails and left two voicemails, but haven't received any response. The chatbot on your website is useless for anything beyond basic questions. This lack of support is unacceptable, especially for a health-related app where timely assistance can be crucial. I'm a premium user and expect better service. Please have someone contact me immediately to resolve my syncing problem and explain why my previous attempts to get help were ignored.", "digital_healthcare", "complaint", "poor_customer_support_complaint"),

("I'm writing to cancel my HealthTrack premium subscription and request a refund for the remaining months. I signed up for an annual plan two months ago, but the app hasn't delivered on its promises. The advanced features I paid for, like personalized workout plans and nutrition tracking, are often buggy or unavailable. Despite multiple attempts to address these issues with your support team, the problems persist. I feel the premium service doesn't provide the value it advertises. Please process my cancellation and refund the unused portion of my subscription as soon as possible.", "digital_healthcare", "complaint", "subscription_complaint"),

("The new interface of HealthTrack is incredibly confusing and has made the app much harder to use. Important features like medication reminders and blood glucose logging are now buried under multiple menus. The font size has decreased, making it difficult for me to read my health data. The color scheme lacks contrast, which is problematic for users with visual impairments. These changes have significantly impacted my ability to manage my health effectively. Please consider reverting to the previous, more user-friendly design or at least provide options to customize the interface for better accessibility.", "digital_healthcare", "complaint", "usability_complaint")
]

print("\nTesting predictions and calculating accuracy:")
correct_type = 0
correct_subtype = 0
total = len(test_emails)

for email, department, true_type, true_subtype in test_emails:
    predicted_type, predicted_subtype = predict_email_type_subtype(email, department, model, tokenizer)
    print(f"\nDepartment: {department}")
    print(f"Email: {email[:100]}...")  # Print first 100 characters of email
    print(f"True Type: {true_type}")
    print(f"Predicted Type: {predicted_type}")
    print(f"True Subtype: {true_subtype}")
    print(f"Predicted Subtype: {predicted_subtype if predicted_subtype else 'No confident subtype prediction'}")
    
    if predicted_type == true_type:
        correct_type += 1
    if predicted_subtype == true_subtype:
        correct_subtype += 1

type_accuracy = correct_type / total * 100
subtype_accuracy = correct_subtype / total * 100

print("\nAccuracy Results:")
print(f"Type Accuracy: {type_accuracy:.2f}%")
print(f"Subtype Accuracy: {subtype_accuracy:.2f}%")

# Print model configuration
print("\nModel Configuration:")
print(f"Max Types: {config.max_types}")
print(f"Max Subtypes: {config.max_subtypes}")

# Print out the available departments, types, and subtypes
print("\nAvailable Departments, Types, and Subtypes:")
for dept in department_type_encoders.keys():
    print(f"- {dept}")
    print("  Types:")
    for type_ in department_type_encoders[dept].classes_:
        print(f"  - {type_}")
        print("    Subtypes:")
        for subtype in department_subtype_hierarchies[dept].get(type_, []):
            print(f"    - {subtype}")


# Print department subtype encoders
print("\nDepartment Subtype Encoders:")
for dept, encoder in department_subtype_encoders.items():
    print(f"Department: {dept}")
    print(f"  Classes: {encoder.classes_}")


print("\nTesting completed.")