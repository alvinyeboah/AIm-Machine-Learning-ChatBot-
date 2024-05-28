import json
import re

# Function to load the dataset from JSON file
def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to preprocess dataset
def preprocess_dataset(json_file):
    dataset = load_dataset(json_file)
    intents = dataset['intents']
    
    preprocessed_data = []
    intent_response_map = {}
    
    for intent in intents:
        intent_name = intent['tag']
        patterns = [preprocess_text(pattern) for pattern in intent['patterns']]
        responses = [preprocess_text(response) for response in intent['responses']]
        preprocessed_data.append((intent_name, patterns, responses))
        intent_response_map[intent_name] = responses

    input_texts = []
    target_texts = []
    for intent_name, patterns, responses in preprocessed_data:
        for pattern in patterns:
            input_texts.append(pattern)
            target_texts.append(intent_name)
    
    return input_texts, target_texts, intent_response_map
