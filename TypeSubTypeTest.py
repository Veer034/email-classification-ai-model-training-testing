import torch
from transformers import (
    DebertaV2Tokenizer,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    DebertaV2Config,
)
import torch.nn as nn
import pickle
import os
import numpy as np

# Custom model class with updated from_pretrained method
class DebertaV3ForTypeAndDepartmentSubtype(DebertaV2PreTrainedModel):
    def __init__(self, config, num_types, department_subtype_encoders, **kwargs):
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
        # Load configuration
        config = DebertaV2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load type_encoder to get num_types
        encoder_dir = os.path.join(pretrained_model_name_or_path, 'encoders')
        with open(os.path.join(encoder_dir, 'type_encoder.pkl'), 'rb') as f:
            type_encoder = pickle.load(f)
        num_types = len(type_encoder.classes_)

        # Load department_subtype_encoders
        with open(os.path.join(encoder_dir, 'department_subtype_encoders.pkl'), 'rb') as f:
            department_subtype_encoders = pickle.load(f)

        # Pass num_types and department_subtype_encoders as kwargs to the model constructor
        model_kwargs = {
            'num_types': num_types,
            'department_subtype_encoders': department_subtype_encoders,
        }
        model_kwargs.update(kwargs)

        # Load the model using the parent class method
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            **model_kwargs,
        )

        # Ensure the attributes are set
        model.num_types = num_types
        model.department_subtype_encoders = department_subtype_encoders
        model.max_subtypes = max(len(encoder.classes_) for encoder in department_subtype_encoders.values())

        return model

# Load the model
model = DebertaV3ForTypeAndDepartmentSubtype.from_pretrained('./final_model')
model.eval()

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained('./final_model')

# Load encoders
with open(os.path.join('./final_model/encoders', 'type_encoder.pkl'), 'rb') as f:
    type_encoder = pickle.load(f)

with open(os.path.join('./final_model/encoders', 'department_subtype_encoders.pkl'), 'rb') as f:
    department_subtype_encoders = pickle.load(f)

with open(os.path.join('./final_model/encoders', 'department_subtype_hierarchies.pkl'), 'rb') as f:
    department_subtype_hierarchies = pickle.load(f)

# Ensure the model has the department_subtype_encoders
model.department_subtype_encoders = department_subtype_encoders

# Prediction function
def predict_email_type_subtype(
    email,
    department,
    model,
    tokenizer,
    type_encoder,
    department_subtype_encoders,
    department_subtype_hierarchies,
    subtype_threshold=0.5
):
    # Prepare the input text
    input_text = f"{department} [SEP] {email}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs['department'] = [department]
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        _, type_logits, all_subtype_logits = outputs

        type_probs = torch.nn.functional.softmax(type_logits, dim=1)
        type_pred = torch.argmax(type_probs, dim=1).cpu().numpy()[0]
        predicted_type = type_encoder.inverse_transform([type_pred])[0]

        dept_encoder = department_subtype_encoders.get(department)
        if dept_encoder is None:
            return predicted_type, None, None

        # Get valid subtypes for the predicted type
        valid_subtypes = department_subtype_hierarchies.get(department, {}).get(predicted_type, set())
        if not valid_subtypes:
            return predicted_type, None, None

        valid_subtype_indices = [
            dept_encoder.transform([subtype])[0]
            for subtype in valid_subtypes
            if subtype in dept_encoder.classes_
        ]
        if not valid_subtype_indices:
            return predicted_type, None, None

        # Extract logits for valid subtypes
        dept_subtype_logits = all_subtype_logits[0, valid_subtype_indices]
        subtype_probs = torch.nn.functional.softmax(dept_subtype_logits, dim=0)

        max_subtype_prob = torch.max(subtype_probs).item()
        subtype_pred_idx = torch.argmax(subtype_probs).item()
        subtype_pred = valid_subtype_indices[subtype_pred_idx]
        predicted_subtype = dept_encoder.inverse_transform([subtype_pred])[0]

        # Prepare subtype probabilities
        subtype_probs_dict = {}
        for idx, prob in zip(valid_subtype_indices, subtype_probs.cpu().numpy()):
            subtype = dept_encoder.inverse_transform([idx])[0]
            subtype_probs_dict[subtype] = prob

        if max_subtype_prob < subtype_threshold:
            predicted_subtype = None

    return predicted_type, predicted_subtype, subtype_probs_dict


# Test data array
test_data = [
    (
        "I'm having trouble logging into my HealthTrack account. I've entered my username and password correctly, but keep getting an 'Invalid credentials' error. I've tried resetting my password twice, but the issue persists. This is preventing me from accessing my health data and using the app's features. I need to check my recent blood pressure readings for my doctor's appointment tomorrow. Please help me regain access to my account as soon as possible. If you need any additional information to verify my identity, I'm happy to provide it.",
        "digital_healthcare",
        "complaint",
        "account_login_complaint"
    ),
    (
        "Your latest app update has rendered HealthTrack unusable on my smartphone. It crashes every time I try to open the exercise tracking feature and frequently shuts down while I'm viewing my health dashboard. I've tried uninstalling and reinstalling, clearing cache, and even factory resetting my phone, but nothing works. This is severely impacting my ability to monitor my daily health goals and manage my chronic condition. I'm a premium subscriber and expect the app to function properly. Please provide a fix immediately or roll back to the previous stable version.",
        "digital_healthcare",
        "complaint",
        "app_malfunction_complaint"
    ),
    (
        "I'm writing about an unexpected charge of $49.99 on my credit card from HealthTrack. I don't recall authorizing this transaction. My account shows I've been upgraded to a premium subscription, but I never requested this change. I've been using the free version and am satisfied with it. Please explain this charge and revert my account to the free tier immediately. I want a full refund of the $49.99 charged without my consent. In the future, please ensure that any account changes or charges are clearly communicated and require explicit approval from users.",
        "digital_healthcare",
        "complaint",
        "billing_complaint"
    )
]

# Set the subtype confidence threshold
subtype_threshold = 0.5

# Loop over the test data and make predictions
for i, (email, department, expected_type, expected_subtype) in enumerate(test_data, 1):
    print(f"Test Case {i}:")
    print("Expected Type:", expected_type)
    print("Expected Subtype:", expected_subtype)
    predicted_type, predicted_subtype, subtype_probs = predict_email_type_subtype(
        email,
        department,
        model,
        tokenizer,
        type_encoder,
        department_subtype_encoders,
        department_subtype_hierarchies,
        subtype_threshold=subtype_threshold
    )
    print("Predicted Type:", predicted_type)
    if predicted_subtype:
        print("Predicted Subtype:", predicted_subtype)
    else:
        print("No confident subtype prediction.")

    print("Subtype Probabilities:")
    for subtype, prob in sorted(subtype_probs.items(), key=lambda x: x[1], reverse=True):
        print(f"{subtype}: {prob:.4f}")
    print("-" * 50)
